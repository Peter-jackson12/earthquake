"""T31: geographic code representation (OH / NAT / NUM) x 5 models on the stratified 5-fold split.

CLI (frozen interpreter, run from the worktree root):
  python -m experiments.geo_code_encoding.run prepare --data <earthquake/data> --out <run dir>
  python -m experiments.geo_code_encoding.run fit-all --out <run dir>
  python -m experiments.geo_code_encoding.run status  --out <run dir>
  python -m experiments.geo_code_encoding.run finalize --out <run dir>
(fit-one is the internal per-fit child process entry point.)

Every fit runs in its own single-threaded child process; the orchestrator samples the child's rss every 20 ms.
Folds come from load_development() only. The holdout is never read. All outputs are private (outside Git).
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psutil
from threadpoolctl import threadpool_info, threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from preprocess import GEO, NUMERIC, TEXT, sha256_file  # noqa: E402
from experiments.age995.run import PARAMS as LR_PARAMS  # noqa: E402
from experiments.geo_group_cv.run import PROBS, load, require, score, write  # noqa: E402
from experiments.lgbm_native_cat.run import (atomic_write, check_saved, event, log, now,  # noqa: E402
                                             read_ledger, read_pred, write_pred)
from experiments.lgbm_partial_native import run as t9  # noqa: E402  (reference() / fold_means())
from experiments.model_families import run as t4  # noqa: E402
from experiments.geo_code_encoding import arms, measure, verdict  # noqa: E402

HERE = Path(__file__).resolve().parent
STUDY_ID = 'T31_geo_code_encoding_v1'
SOURCE = ['preprocess.py', 'baseline.py', 'configs/baseline.json', 'configs/feature_budget.json',
          'configs/data_contract.json', 'scripts/audit_eda_oof.py',
          'experiments/age995/run.py', 'experiments/geo_group_cv/run.py', 'experiments/geo_ablation/audit.py',
          'experiments/model_families/run.py', 'experiments/model_families/config.json',
          'experiments/lgbm_native_cat/__init__.py', 'experiments/lgbm_native_cat/run.py',
          'experiments/lgbm_native_cat/config.json',
          'experiments/lgbm_partial_native/__init__.py', 'experiments/lgbm_partial_native/run.py',
          'experiments/lgbm_partial_native/config.json',
          'experiments/geo_code_encoding/__init__.py', 'experiments/geo_code_encoding/arms.py',
          'experiments/geo_code_encoding/measure.py', 'experiments/geo_code_encoding/verdict.py',
          'experiments/geo_code_encoding/run.py', 'experiments/geo_code_encoding/config.json',
          'docs/T31_GEO_CODE_ENCODING_PLAN.md']


class ControlFailed(RuntimeError):
    pass


# ---------------------------------------------------------------- config / plan
def config(path=HERE/'config.json'):
    c = json.loads(Path(path).read_text(encoding='utf-8'))
    require(c['study_id'] == STUDY_ID and c['scheme'] == 'stratified' and c['n_folds'] == 5, 'Unplanned study')
    p = c['classifier_params']
    base = json.loads((ROOT/'configs/baseline.json').read_text(encoding='utf-8'))['models']
    fb = json.loads((ROOT/'configs/feature_budget.json').read_text(encoding='utf-8'))['models']
    require(p['LR'] == LR_PARAMS and p['DT'] == base['dt'], 'LR/DT parameters differ from the baselines')
    require(p['RF'] == dict(fb['rf'], n_jobs=1) and p['GBC'] == fb['gbc'], 'RF/GBC parameters differ from feature_budget')
    require(p['LGBM'] == dict(t4.load_config()['models']['LGBM'], n_jobs=1), 'LightGBM parameters differ from T4 (n_jobs=1 only)')
    require(c['model_order'] == arms.MODELS and sorted(c['arms_by_model']) == sorted(arms.MODELS), 'Model set changed')
    require(len(arms.all_arms(c)) == 11 and c['fit_plan']['study_fits'] == 55 and c['fit_plan']['max_attempts'] == 59,
            'Unplanned fit plan')
    require(c['threading']['threadpool_limit'] == 1 and c['threading']['outer_parallel_jobs'] == 1, 'Single thread required')
    require(not c['holdout_access'] and not c['automatic_model_replacement'], 'Policy changed')
    return c


def check_plan_sha(c, root=ROOT):
    """The sealed plan document must be exactly the approved one (config plan_sha256)."""
    actual = sha256_file(Path(root)/c['plan_doc'])
    require(actual == c['plan_sha256'], f'Plan document SHA {actual} differs from the approved {c["plan_sha256"]}')
    return actual


def environment():
    names = ['numpy', 'pandas', 'scipy', 'scikit-learn', 'lightgbm', 'joblib', 'threadpoolctl', 'psutil']
    return dict(python=platform.python_version(), **{n: importlib.metadata.version(n) for n in names})


def sources():
    return {p: sha256_file(ROOT/p) for p in SOURCE}


def key(arm, fold):
    return f'{arm}/{fold}'


def study_targets(c):
    return [key(a, f) for a in arms.all_arms(c) for f in range(5)]


def arm_order(model, fold, c):
    """Execution order of the arms of one model within a fold (plan sections 6.1 / 7.1)."""
    reps = list(c['arms_by_model'][model])
    if fold == 0:
        return reps                                      # OH -> (NAT) -> NUM, controls are inserted by the plan
    if model == 'LGBM':
        r = fold % 3
        reps = reps[r:] + reps[:r]
    elif fold % 2 == 1:
        reps = ['NUM', 'OH']
    return [arms.arm_id(model, r) for r in reps]


def execution_plan(c):
    mo = c['fit_plan']['measurement_only']
    plan = [dict(kind='measure', role='warmup', arm=mo['warmup'][0][0], fold=mo['warmup'][0][1], rep=1)]
    for model in c['model_order']:
        for f in range(5):
            ordered = arm_order(model, f, c)
            if f == 0:
                ids = [arms.arm_id(model, r) for r in c['arms_by_model'][model]]
                for a in ids[:-1]:
                    plan.append(dict(kind='fit', arm=a, fold=0))
                plan.append(dict(kind='control', model=model))
                plan.append(dict(kind='fit', arm=ids[-1], fold=0))
            else:
                plan.extend(dict(kind='fit', arm=a, fold=f) for a in ordered)
        if model == 'DT':
            for k in range(1, mo['repeats_each'] + 1):
                for a in mo['repeat_arms']:
                    plan.append(dict(kind='measure', role='repeat', arm=a, fold=mo['fold'], rep=k))
    return plan


def measure_key(role, arm, fold, rep):
    return f'measure/{role}/{arm}/{fold}/{rep}'


def measure_tag(role, arm, fold, rep):
    return f'{role}_{arm}_f{fold}_r{rep}'


def jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def write_json(path, obj):
    write(path, jsonable(obj))


# ---------------------------------------------------------------- machine / git helpers
def ac_connected():
    battery = psutil.sensors_battery()
    return None if battery is None else bool(battery.power_plugged)


def hardware_info(path):
    info = dict(platform=platform.platform(), machine=platform.machine(), processor=platform.processor(),
                logical_cores=psutil.cpu_count(logical=True), physical_cores=psutil.cpu_count(logical=False),
                ram_total_bytes=int(psutil.virtual_memory().total), ac_connected=ac_connected(),
                disk_free_bytes=int(shutil.disk_usage(Path(path)).free))
    try:
        info['power_scheme'] = subprocess.run(['powercfg', '/getactivescheme'], capture_output=True, text=True,
                                              timeout=20).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        info['power_scheme'] = None
    return info


def git_blob_mismatches(paths, runner=None):
    """Compare every sealed source's blob in HEAD with the working-tree file (unrelated untracked files are fine)."""
    runner = runner or (lambda args: subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True))
    bad = []
    for p in paths:
        head = runner(['rev-parse', f'HEAD:{p}'])
        work = runner(['hash-object', p])
        if head.returncode != 0 or work.returncode != 0 or head.stdout.strip() != work.stdout.strip():
            bad.append(p)
    return bad


def git_head(runner=None):
    runner = runner or (lambda args: subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True))
    r = runner(['rev-parse', 'HEAD'])
    require(r.returncode == 0, 'git rev-parse HEAD failed')
    return r.stdout.strip()


# ---------------------------------------------------------------- matrices / equivalence
def build_matrices(arm, X_train, X_valid, threads=1):
    pre = arms.make_preprocessor(arm)
    with threadpool_limits(limits=threads):
        mt = pre.fit_transform(X_train)
        mv = pre.transform(X_valid)
    return pre, mt, mv


def equivalence_report(arm, pre, mt, mv, X_train, X_valid, c, include_train=False):
    """Plan 8.4: non-geo columns equal the baseline arm(s); geo columns equal original integers; train-only stats."""
    model, rep = arms.arm_spec(arm)
    names = arms.feature_names(pre)
    approx = set(NUMERIC) if model == 'LR' else set()
    tol = c['equivalence']['lr_scaled_numeric_columns_max_abs_diff']
    out = {}
    bases = ['OH'] + (['NAT'] if model == 'LGBM' else [])
    for base_rep in bases:
        base = arms.arm_id(model, base_rep)
        bpre, bmt, bmv = build_matrices(base, X_train, X_valid)
        bnames = arms.feature_names(bpre)
        out[f'valid_vs_{base}'] = arms.column_equivalence(bmv, bnames, mv, names, approx, tol)
        if include_train:
            out[f'train_vs_{base}'] = arms.column_equivalence(bmt, bnames, mt, names, approx, tol)
        del bpre, bmt, bmv
    scaler = pre['columns'].named_transformers_['numeric']['scale'] if model == 'LR' else None
    out['geo_values_valid'] = arms.geo_value_check(mv, names, X_valid, scaler)
    if include_train:
        out['geo_values_train'] = arms.geo_value_check(mt, names, X_train, scaler)
    out['train_only'] = arms.train_only_checks(pre, arm, pre['schema'].transform(X_train))
    out['ok'] = bool(all(v['ok'] for v in out.values()))
    return out


def lgbm_text_stats(text, geo_idx):
    """Tree count, total splits and geo-column splits from a LightGBM model text."""
    trees, splits, geo = 0, 0, 0
    geo_set = set(geo_idx)
    for line in text.splitlines():
        if line.startswith('Tree='):
            trees += 1
        elif line.startswith('split_feature='):
            feats = [int(v) for v in line.split('=', 1)[1].split() if v]
            splits += len(feats)
            geo += sum(1 for v in feats if v in geo_set)
    return dict(trees=trees, splits=splits, geo_splits=geo)


def checked_probabilities(p, n_rows):
    p = np.asarray(p, dtype=np.float64)
    require(p.shape == (n_rows, 3) and np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all(), 'Invalid probabilities')
    np.testing.assert_allclose(p.sum(1), 1.0, atol=1e-6, rtol=0)
    return p


def model_internals(arm, clf, pre):
    model, rep = arms.arm_spec(arm)
    names = arms.feature_names(pre)
    gidx = arms.geo_indices(names)
    out = dict(n_features=len(names), geo_columns=len(gidx))
    if model == 'LR':
        coef = np.abs(clf.coef_)
        out.update(n_iter=[int(v) for v in np.asarray(clf.n_iter_).ravel()],
                   geo_abs_coef_share=float(coef[:, gidx].sum() / coef.sum()))
    elif model == 'DT':
        out.update(node_count=int(clf.tree_.node_count), geo_importance_sum=float(clf.feature_importances_[gidx].sum()))
    elif model == 'RF':
        out.update(node_count=int(sum(e.tree_.node_count for e in clf.estimators_)),
                   geo_importance_sum=float(clf.feature_importances_[gidx].sum()))
    elif model == 'GBC':
        out.update(node_count=int(sum(t.tree_.node_count for t in clf.estimators_.ravel())),
                   geo_importance_sum=float(clf.feature_importances_[gidx].sum()))
    else:
        out.update(lgbm_text_stats(clf.booster_.model_to_string(), gidx))
        out['rounds_completed'] = int(clf.booster_.current_iteration())
    return out


# ---------------------------------------------------------------- one fit (child process)
def fit_arm_job(arm, X, y, ids, folds, f, d, stem, c, save=True, clock=time.perf_counter):
    """One fit with its measurement. Runs inside the child process; returns the child record.

    Measured intervals (perf_counter marks): encode = enc_start..fit_start, train = fit_start..fit_end,
    predict = fit_end..pred_end. Everything after pred_end (train-set scoring, equivalence checks, saving,
    reload verification) is outside the measurement."""
    model_name, rep = arms.arm_spec(arm)
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    a, b = np.flatnonzero(folds != f), np.flatnonzero(folds == f)
    require(len(a) + len(b) == len(ids) and len(b) > 0, 'Fold membership invalid')
    Xtr, Xva, ytr, yva = X.iloc[a], X.iloc[b], y.iloc[a], y.iloc[b]
    threads = c['threading']['threadpool_limit']
    peaks = dict(after_load=measure.peak_counters())
    pre, clf = arms.make_preprocessor(arm), arms.make_classifier(arm, c)
    kw = {}
    cf = arms.categorical_feature(arm, c)
    if cf is not None:
        kw['categorical_feature'] = cf
    y_fit = arms.fit_labels(arm, ytr)
    rep_pre = pre[1:]                                      # shares the step objects, so pre is fitted in place
    marks, timings = {}, {}
    with threadpool_limits(limits=threads):
        pools = [dict(api=p.get('internal_api'), num_threads=p.get('num_threads')) for p in threadpool_info()]
        marks['enc_start'] = clock()
        t0 = clock(); Xn_tr = pre['schema'].fit_transform(Xtr); norm = clock() - t0
        t0 = clock(); mt = rep_pre.fit_transform(Xn_tr); repr_s = clock() - t0
        t0 = clock(); Xn_va = pre['schema'].transform(Xva); norm += clock() - t0
        t0 = clock(); mv = rep_pre.transform(Xn_va); repr_s += clock() - t0
        marks['fit_start'] = clock()
        cpu0 = time.process_time()
        with warnings.catch_warnings(record=True) as caught_fit:
            warnings.simplefilter('always')
            clf.fit(mt, y_fit, **kw)
        marks['fit_end'] = clock()
        cpu_fit = time.process_time() - cpu0
        peaks['after_fit'] = measure.peak_counters()
        with warnings.catch_warnings(record=True) as caught_pred:
            warnings.simplefilter('always')
            p_raw = clf.predict_proba(mv)
        marks['pred_end'] = clock()
        peaks['after_predict'] = measure.peak_counters()
        # ---- measurement is over
        timings = dict(normalization_seconds=norm, representation_seconds=repr_s,
                       encode_seconds=marks['fit_start'] - marks['enc_start'],
                       fit_seconds=marks['fit_end'] - marks['fit_start'], fit_cpu_seconds=cpu_fit,
                       predict_seconds=marks['pred_end'] - marks['fit_end'],
                       predict_seconds_per_1000_rows=(marks['pred_end'] - marks['fit_end']) / len(b) * 1000)
        timings['total_seconds'] = timings['encode_seconds'] + timings['fit_seconds'] + timings['predict_seconds']
        timings['fit_cpu_wall_ratio'] = cpu_fit / timings['fit_seconds'] if timings['fit_seconds'] else None
        arms.check_container(arm, mt, c)
        arms.check_container(arm, mv, c)
        p = checked_probabilities(p_raw, len(b))
        require(list(clf.classes_) == arms.expected_class_labels(arm), 'Unexpected class order')
        pred = p.argmax(1) + 1
        p_train = checked_probabilities(clf.predict_proba(mt), len(a))
        if model_name == 'LGBM':
            require(clf.booster_.current_iteration() == c['classifier_params']['LGBM']['n_estimators'],
                    'Full fixed boosting budget not completed')
        equiv = None
        if rep == 'NUM':
            equiv = equivalence_report(arm, pre, mt, mv, Xtr, Xva, c)
            require(equiv['ok'], f'{arm}: NUM columns are not equivalent to the baseline arm(s)')
    unseen, _ = arms.unseen_geo(Xtr, Xva)
    names = arms.feature_names(pre)
    vocab = {col: int(Xtr[col].nunique()) for col in GEO}
    geo_max = {col: int(Xtr[col].astype('int64').max()) for col in GEO}
    record = dict(
        study_id=STUDY_ID, arm=arm, fold=int(f), train_rows=len(a), valid_rows=len(b), marks=marks, timings=timings,
        peaks=peaks, thread_pools=pools, containers=dict(train=arms.container_record(mt), valid=arms.container_record(mv)),
        n_encoded_columns=int(mt.shape[1]), feature_names_sha256=hashlib.sha256('\n'.join(names).encode('utf-8')).hexdigest(),
        metrics=score(yva, pred), train_macro_f1=score(ytr, p_train.argmax(1) + 1)['macro_f1'],
        train_class_counts=[int((ytr == v).sum()) for v in (1, 2, 3)],
        unseen_valid_geo=unseen, geo_train_unique=vocab, geo_train_max=geo_max,
        internals=model_internals(arm, clf, pre), equivalence=equiv,
        warnings=[dict(stage='fit', category=w.category.__name__, message=str(w.message)) for w in caught_fit]
        + [dict(stage='predict', category=w.category.__name__, message=str(w.message)) for w in caught_pred],
        effective_parameters={k: v for k, v in clf.get_params().items()}, pid=os.getpid(),
        env_threads={k: os.environ.get(k) for k in c['threading']['env']}, saved=bool(save), finished_at_utc=now())
    record['probabilities_checked'] = True
    files = {}
    if save:
        pre_path, pred_path = d/f'{stem}_preprocessor.joblib', d/f'{stem}_pred.csv.gz'
        joblib.dump(pre, pre_path, compress=0)
        frame = pd.DataFrame({'building_id': ids.iloc[b].to_numpy(), 'fold': f, 'y_true': yva.to_numpy(),
                              'y_pred': pred, **{n: p[:, i] for i, n in enumerate(PROBS)}})
        write_pred(pred_path, frame)
        loaded_pre = joblib.load(pre_path)                   # freshly written trusted file only
        if model_name == 'LGBM':
            import lightgbm
            model_path = d/f'{stem}_model.txt'
            clf.booster_.save_model(str(model_path))
            joblib.dump(clf, d/f'{stem}_model_sklearn.joblib', compress=0)
            booster = lightgbm.Booster(model_file=str(model_path))
            with threadpool_limits(limits=threads):
                mv2 = loaded_pre.transform(Xva)
                p2 = np.asarray(booster.predict(mv2, num_threads=1), dtype=np.float64)
            files['model'] = model_path.name
            files['model_sklearn'] = f'{stem}_model_sklearn.joblib'
        else:
            model_path = d/f'{stem}_model.joblib'
            joblib.dump(clf, model_path, compress=0)
            loaded = joblib.load(model_path)
            with threadpool_limits(limits=threads):
                mv2 = loaded_pre.transform(Xva)
                p2 = np.asarray(loaded.predict_proba(mv2), dtype=np.float64)
            files['model'] = model_path.name
        arms.check_container(arm, mv2, c)
        diff = float(np.max(np.abs(p2 - p)))
        require(diff <= c['control']['reload_probability_atol'] and np.array_equal(p2.argmax(1) + 1, pred),
                f'Model reload mismatch {diff}')
        back = read_pred(pred_path)
        np.testing.assert_array_equal(back.building_id.to_numpy(), frame.building_id.to_numpy())
        np.testing.assert_array_equal(back.y_pred.to_numpy(), pred)
        np.testing.assert_array_equal(back[PROBS].to_numpy(), p)
        files.update(preprocessor=pre_path.name, pred=pred_path.name)
        record['sizes'] = dict(
            classifier_bytes=int(model_path.stat().st_size),
            lgbm_sklearn_joblib_bytes=int((d/files['model_sklearn']).stat().st_size) if 'model_sklearn' in files else None,
            preprocessor_bytes=int(pre_path.stat().st_size))
        record['sizes']['model_total_bytes'] = record['sizes']['classifier_bytes'] + record['sizes']['preprocessor_bytes']
        record['reload_probability_max_abs_diff'] = diff
        record['csv_round_trip_exact'] = True
    record['files'] = files
    return record


def child_dir_stem(out, arm, role, tag, fold):
    out = Path(out)
    if role == 'main':
        return out/'arms'/arm, f'fold_{fold}'
    return out/'measure_only', tag


def is_infrastructure_error(exc):
    """Out of memory or out of disk space: an infrastructure interruption, not a code/data defect."""
    import errno
    return isinstance(exc, MemoryError) or (isinstance(exc, OSError) and exc.errno in (errno.ENOSPC, errno.EDQUOT))


def child_main(out, arm, fold, role, tag, loader=None):
    out = Path(out)
    d, stem = child_dir_stem(out, arm, role, tag, fold)
    d.mkdir(parents=True, exist_ok=True)
    try:
        X, y, ids, folds, design = verify(out, loader)
        c = design['config']
        for k, v in c['threading']['env'].items():
            require(os.environ.get(k) == v, f'{k} must be {v} before the process starts')
        rec = fit_arm_job(arm, X, y, ids, folds, fold, d, stem, c, save=(role == 'main'))
        rec.update(role=role, tag=tag, child_perf_counter_at_exit=time.perf_counter())
        write_json(d/f'{stem}_child.json', rec)
        return 0
    except BaseException as exc:
        if is_infrastructure_error(exc):          # no *_error.json: the orchestrator records 'interrupted' (retry allowed)
            try:
                write_json(d/f'{stem}_infra_error.json', dict(error=f'{type(exc).__name__}: {exc}', at_utc=now()))
            except OSError:
                pass
            raise
        write_json(d/f'{stem}_error.json', dict(arm=arm, fold=fold, role=role, tag=tag, error=f'{type(exc).__name__}: {exc}',
                                                trace=traceback.format_exc(), at_utc=now()))
        raise


# ---------------------------------------------------------------- design verification
def verify(out, loader=None):
    out = Path(out)
    seal = json.loads((out/'design_seal.json').read_text(encoding='utf-8'))
    require(sha256_file(out/'design.json') == seal['design_sha256'], 'Design seal changed')
    design = json.loads((out/'design.json').read_text(encoding='utf-8'))
    require(design['config'] == config(), 'Frozen config changed')
    check_plan_sha(design['config'])
    require(design['source_sha256'] == sources(), 'Frozen source changed')
    require(design['environment'] == environment(), 'Frozen environment changed')
    X, y, ids, old, contract = (loader or load)(design['data_dir'])
    require(contract == design['data_contract'], 'Data contract changed')
    return X, y, ids, np.asarray(old), design


# ---------------------------------------------------------------- control checks (plan 8.3)
def compare_probabilities(p_new, p_ref, pred_ref, atol, tie_atol):
    """Tree/boosting control: probabilities within atol; label differences only on near-tied reference rows."""
    p_new, p_ref = np.asarray(p_new, float), np.asarray(p_ref, float)
    diff = np.abs(p_new - p_ref)
    pred_new = p_new.argmax(1) + 1
    top2 = np.sort(p_ref, axis=1)[:, -2:]
    gap = top2[:, 1] - top2[:, 0]
    mismatch = pred_new != np.asarray(pred_ref)
    tie_allowed = mismatch & (gap <= tie_atol)
    unexplained = int((mismatch & ~tie_allowed).sum())
    max_diff = float(diff.max()) if diff.size else 0.0
    ok = bool(max_diff <= atol and unexplained == 0)
    return dict(max_abs_diff=max_diff, label_mismatch_rows=int(mismatch.sum()), tie_allowed_rows=int(tie_allowed.sum()),
                unexplained_label_mismatch_rows=unexplained, status='PASS' if ok else 'FAIL')


def compare_lr_control(y_true, pred_new, pred_ref, atol):
    f_new = score(y_true, pred_new)['macro_f1']
    f_ref = score(y_true, pred_ref)['macro_f1']
    d = abs(f_new - f_ref)
    return dict(fold_macro_f1_new=f_new, fold_macro_f1_reference=f_ref, abs_diff=float(d),
                label_mismatch_rows=int((np.asarray(pred_new) != np.asarray(pred_ref)).sum()),
                status='PASS' if d <= atol else 'DIVERGED_KNOWN_ENV')


def load_control_reference(design, name, ids, y, folds):
    ref = design['references'][name]
    return t9.reference(ref['path'], ref['sha256'], ids, y, folds)


def control_check(out, arm, ids, y, folds, design):
    out = Path(out)
    c = design['config']
    d = out/'arms'/arm
    new = read_pred(d/'fold_0_pred.csv.gz')
    ref = load_control_reference(design, arm, ids, y, folds)
    ref0 = ref[ref.fold == 0].set_index('building_id').loc[new.building_id.to_numpy()].reset_index()
    require(np.array_equal(ref0.y_true.to_numpy(), new.y_true.to_numpy()), 'Control: true labels differ')
    p_new, p_ref = new[PROBS].to_numpy(), ref0[PROBS].to_numpy()
    model = arms.arm_spec(arm)[0]
    rec = dict(arm=arm, fold=0, reference=design['references'][arm]['path'], reference_sha256=design['references'][arm]['sha256'],
               new_pred_sha256=sha256_file(d/'fold_0_pred.csv.gz'), at_utc=now())
    if model == 'LR':
        rec['primary'] = compare_lr_control(new.y_true, new.y_pred.to_numpy(), ref0.y_pred.to_numpy(),
                                            c['control']['lr_fold_f1_atol'])
        rec['primary']['max_abs_probability_diff_info'] = float(np.abs(p_new - p_ref).max())
        local = load_control_reference(design, 'LR_OH_LOCAL', ids, y, folds)
        l0 = local[local.fold == 0].set_index('building_id').loc[new.building_id.to_numpy()].reset_index()
        pdiff = float(np.abs(p_new - l0[PROBS].to_numpy()).max())
        rec['secondary_local_refit'] = dict(
            reference=design['references']['LR_OH_LOCAL']['path'], max_abs_probability_diff=pdiff,
            probability_atol=c['control']['lr_secondary_probability_atol'],
            labels_identical=bool(np.array_equal(new.y_pred.to_numpy(), l0.y_pred.to_numpy())),
            within_atol=bool(pdiff <= c['control']['lr_secondary_probability_atol']), required=False)
        rec['status'] = rec['primary']['status']
    else:
        rec['primary'] = compare_probabilities(p_new, p_ref, ref0.y_pred.to_numpy(), c['control']['tree_probability_atol'],
                                               c['control']['tree_tie_gap_atol'])
        rec['status'] = rec['primary']['status']
    write_json(d/'control_check.json', rec)
    return rec


# ---------------------------------------------------------------- orchestrator
class Runner:
    """Sequential orchestrator: one child process per fit, rss sampled by this (separate) process."""

    def __init__(self, out, loader=None, spawn=None, idle=None, retry_reason=None, sleep_between=0.0):
        self.out = Path(out)
        self.loader = loader
        self.spawn = spawn or self._spawn_process
        self.retry_reason = retry_reason
        self.X, self.y, self.ids, self.folds, self.design = verify(self.out, loader)
        self.c = self.design['config']
        m = self.c['measurement']
        self.idle = idle or (lambda: measure.idle_check(m['idle_cpu_threshold_percent'], m['idle_window_seconds'],
                                                        m['idle_max_wait_seconds']))

    # ---- process handling
    def _spawn_process(self, args, logfile, env):
        cmd = [sys.executable, '-m', 'experiments.geo_code_encoding.run', 'fit-one', '--out', str(self.out),
               '--arm', args['arm'], '--fold', str(args['fold']), '--role', args['role'], '--tag', args['tag']]
        return subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=logfile, stderr=subprocess.STDOUT)

    def _env(self):
        env = dict(os.environ)
        env.update(self.c['threading']['env'])
        env.update(PYTHONIOENCODING='utf-8', PYTHONDONTWRITEBYTECODE='1')
        return env

    def _run_child(self, args, logpath):
        idle = self.idle()
        interval = self.c['measurement']['rss_sample_interval_seconds']
        with open(logpath, 'ab') as fh:
            t_start = time.perf_counter()
            proc = self.spawn(args, fh, self._env())
            sampler = measure.RssSampler(proc.pid, interval).start()
            rc = proc.wait()
            t_end = time.perf_counter()
            t, rss = sampler.stop()
        return rc, t, rss, idle, t_start, t_end, sampler.error, sampler.sampled_pids

    def _finish_record(self, d, stem, rec, t, rss, idle, t_start, t_end, sampler_error, sampled_pids):
        mem = measure.memory_intervals(t, rss, rec['marks'], self.c['measurement']['sampling_limited_if_samples_below'])
        pk = rec['peaks']
        wset = dict(encode_train=measure.wset_floor(pk['after_load']['peak_wset'], pk['after_fit']['peak_wset']),
                    predict=measure.wset_floor(pk['after_load']['peak_wset'], pk['after_predict']['peak_wset']))
        rec.update(memory=mem, wset=wset, peak_pagefile_bytes=dict(
            after_load=pk['after_load']['peak_pagefile'], after_fit=pk['after_fit']['peak_pagefile'],
            after_predict=pk['after_predict']['peak_pagefile']),
            idle=idle, noisy=bool(idle['noisy']), rss_sample_count=int(len(t)), rss_sampler_error=sampler_error,
            rss_sampled_pids=sampled_pids,
            process_perf_counter_start=t_start, process_perf_counter_end=t_end,
            memory_floor_rule_min_mib=self.c['measurement']['memory_min_increment_mib'])
        pd.DataFrame({'t': t, 'rss_bytes': rss}).to_csv(d/f'{stem}_rss.csv.gz', index=False, float_format='%.9f',
                                                          compression={'method': 'gzip', 'mtime': 0})
        return rec

    # ---- study fits
    def fit_step(self, arm, fold):
        k = key(arm, fold)
        require(k in self.design['study_targets'], f'{k} is not a pre-registered study fit')
        ledger = read_ledger(self.out)
        entry = ledger['folds'].get(k)
        if entry and entry['status'] == 'complete':
            check_saved(self.out, entry)
            log(self.out, f'{k} already complete; verified hashes, not refit')
            return entry, False
        fp = self.c['fit_plan']
        attempts = sum(e['attempts'] for e in ledger['folds'].values())
        require(attempts < fp['max_attempts'], 'Attempt budget exhausted')
        prior = entry['attempts'] if entry else 0
        reasons = list(entry.get('retry_reasons', [])) if entry else []
        if entry:
            require(entry['status'] in ('running', 'interrupted'), f'{k} failed with a code/data defect; stop and report')
            require(self.retry_reason, f'{k} was interrupted; manual --retry-interrupted --retry-reason is required')
            retries = sum(max(0, e['attempts'] - 1) for e in ledger['folds'].values())
            require(prior < fp['max_attempts_per_fit'] and retries < fp['max_retries_total'], 'Retry budget exhausted')
            reasons.append(self.retry_reason)
            log(self.out, f'{k} previous attempt did not finish; manual retry attempt {prior+1}: {self.retry_reason}')
        entry = dict(dir=f'arms/{arm}', arm=arm, fold=fold, status='running', attempts=prior + 1,
                     retry_reasons=reasons, started_at_utc=now())
        ledger['folds'][k] = entry
        atomic_write(self.out/'ledger.json', ledger)
        d = self.out/entry['dir']
        d.mkdir(parents=True, exist_ok=True)
        for stale in list(d.glob(f'fold_{fold}[._]*')):
            stale.rename(stale.with_name(stale.name + f'.incomplete_attempt{prior}'))
        bd0 = self.out/'backup'/entry['dir']
        if bd0.exists():                           # a partial backup of an unfinished attempt must not block the retry
            for stale in list(bd0.glob(f'fold_{fold}[._]*')):
                stale.rename(stale.with_name(stale.name + f'.incomplete_attempt{prior}'))
        event(self.out, fold=k, action='fit_start', attempt=prior + 1)
        stem = f'fold_{fold}'
        args = dict(arm=arm, fold=fold, role='main', tag='main')
        rc, t, rss, idle, t_start, t_end, serr, spids = self._run_child(args, d/f'{stem}.log')
        err = d/f'{stem}_error.json'
        if rc != 0 or not (d/f'{stem}_child.json').exists():
            entry['status'] = 'failed' if err.exists() else 'interrupted'
            entry['returncode'] = rc
            ledger = read_ledger(self.out)
            ledger['folds'][k] = entry
            atomic_write(self.out/'ledger.json', ledger)
            event(self.out, fold=k, action='fit_failed', returncode=rc, status=entry['status'])
            raise RuntimeError(f'{k} child exited {rc} ({entry["status"]})')
        rec = json.loads((d/f'{stem}_child.json').read_text(encoding='utf-8'))
        require(rec['arm'] == arm and rec['fold'] == fold and rec['role'] == 'main', 'Child record does not match the step')
        rec = self._finish_record(d, stem, rec, t, rss, idle, t_start, t_end, serr, spids)
        write_json(d/f'{stem}.json', rec)
        names = sorted(set(rec['files'].values()) | {f'{stem}.json', f'{stem}_child.json', f'{stem}_rss.csv.gz', f'{stem}.log'})
        shas = {n: sha256_file(d/n) for n in names}
        bd = self.out/'backup'/entry['dir']
        bd.mkdir(parents=True, exist_ok=True)
        for n in names:
            require(not (bd/n).exists(), 'Backup already exists; refusing overwrite')
            shutil.copy2(d/n, bd/n)
        entry.update(status='complete', artifact_sha256=shas, macro_f1=rec['metrics']['macro_f1'],
                     fit_seconds=rec['timings']['fit_seconds'], noisy=rec['noisy'], finished_at_utc=now(),
                     process_perf_counter_start=t_start, process_perf_counter_end=t_end)
        check_saved(self.out, entry)
        ledger = read_ledger(self.out)
        ledger['folds'][k] = entry
        atomic_write(self.out/'ledger.json', ledger)
        event(self.out, fold=k, action='fit_complete', macro_f1=entry['macro_f1'], artifact_sha256=shas)
        log(self.out, f'{k} F1={entry["macro_f1"]:.9f} fit={entry["fit_seconds"]:.1f}s noisy={entry["noisy"]} '
                      f'saved+verified+backed-up')
        return entry, True

    # ---- measurement-only runs
    def measure_step(self, role, arm, fold, rep):
        k = measure_key(role, arm, fold, rep)
        ledger = read_ledger(self.out)
        entry = ledger.setdefault('measure_only', {}).get(k)
        if entry:
            if entry['status'] == 'complete':
                for n, digest in entry['artifact_sha256'].items():
                    require(sha256_file(self.out/'measure_only'/n) == digest, f'Measurement artifact changed: {n}')
            log(self.out, f'{k} already recorded as {entry["status"]}; measurement-only runs are not retried')
            return entry
        tag = measure_tag(role, arm, fold, rep)
        d = self.out/'measure_only'
        d.mkdir(parents=True, exist_ok=True)
        entry = dict(role=role, arm=arm, fold=fold, rep=rep, tag=tag, status='running', started_at_utc=now(),
                     counted_in_study_fits=False)
        ledger['measure_only'][k] = entry
        atomic_write(self.out/'ledger.json', ledger)
        rc, t, rss, idle, t_start, t_end, serr, spids = self._run_child(dict(arm=arm, fold=fold, role=role, tag=tag), d/f'{tag}.log')
        if rc != 0 or not (d/f'{tag}_child.json').exists():
            entry.update(status='failed', returncode=rc, finished_at_utc=now())
        else:
            rec = json.loads((d/f'{tag}_child.json').read_text(encoding='utf-8'))
            rec = self._finish_record(d, tag, rec, t, rss, idle, t_start, t_end, serr, spids)
            write_json(d/f'{tag}.json', rec)
            names = [f'{tag}.json', f'{tag}_child.json', f'{tag}_rss.csv.gz', f'{tag}.log']
            entry.update(status='complete', artifact_sha256={n: sha256_file(d/n) for n in names},
                         finished_at_utc=now(), process_perf_counter_start=t_start, process_perf_counter_end=t_end)
        ledger = read_ledger(self.out)
        ledger['measure_only'][k] = entry
        atomic_write(self.out/'ledger.json', ledger)
        log(self.out, f'{k} {entry["status"]} (measurement only; not a study fit)')
        return entry

    # ---- controls
    def control_step(self, model):
        ledger = read_ledger(self.out)
        controls = ledger.setdefault('controls', {})
        for rep in self.c['arms_by_model'][model]:
            if rep == 'NUM':
                continue
            arm = arms.arm_id(model, rep)
            done = controls.get(arm)
            if done:
                require(done['status'] != 'CONTROL_FAILED', f'{arm} CONTROL_FAILED is recorded; new experiment ID required')
                continue
            entry = ledger['folds'].get(key(arm, 0))
            require(entry and entry['status'] == 'complete', f'{arm} fold 0 must be complete before its control check')
            rec = control_check(self.out, arm, self.ids, self.y, self.folds, self.design)
            status_ = 'CONTROL_FAILED' if rec['status'] == 'FAIL' else rec['status']
            ledger = read_ledger(self.out)
            controls = ledger.setdefault('controls', {})
            controls[arm] = dict(status=status_, sha256=sha256_file(self.out/'arms'/arm/'control_check.json'))
            atomic_write(self.out/'ledger.json', ledger)
            log(self.out, f'control {arm}: {status_}')
            event(self.out, fold=key(arm, 0), action='control_check', status=status_)
            if status_ == 'CONTROL_FAILED':
                raise ControlFailed(f'{arm} control check failed; its NUM arm is not started (new experiment ID required)')

    def run_plan(self):
        for step in execution_plan(self.c):
            if step['kind'] == 'fit':
                self.fit_step(step['arm'], step['fold'])
            elif step['kind'] == 'control':
                self.control_step(step['model'])
            else:
                self.measure_step(step['role'], step['arm'], step['fold'], step['rep'])


def fit_all(out, retry_reason=None, loader=None, runner_cls=Runner, **kw):
    out = Path(out)
    lock = out/'RUNNING.lock'
    require(not lock.exists(), f'Another run holds {lock}; remove only if that process is dead')
    lock.write_text(f'pid={os.getpid()} started={now()}\n')
    try:
        ac = ac_connected()
        require(ac is not False, 'Battery power detected; connect AC power before the measured run')
        runner = runner_cls(out, loader=loader, retry_reason=retry_reason, **kw)
        before = pd.util.hash_pandas_object(runner.X, index=True).to_numpy()
        envdir = out/'run_env'
        envdir.mkdir(exist_ok=True)
        write_json(envdir/f'start_{int(time.time())}.json', dict(
            at_utc=now(), top_cpu_processes=measure.top_cpu_processes(), ac_connected=ac, environment=environment(),
            free_disk_bytes=int(shutil.disk_usage(out).free)))
        require(shutil.disk_usage(out).free >= 3 * 2**30, 'Less than 3 GiB free disk')
        log(out, f'fit-all start (completed fits are hash-verified, not refit); retry_reason={retry_reason!r}')
        t0 = time.perf_counter()
        try:
            runner.run_plan()
        except Exception as exc:
            log(out, f'STOPPED {type(exc).__name__}: {exc}')
            event(out, action='stopped', error=f'{type(exc).__name__}: {exc}', trace=traceback.format_exc())
            raise
        np.testing.assert_array_equal(before, pd.util.hash_pandas_object(runner.X, index=True).to_numpy())
        log(out, f'fit-all finished wall={time.perf_counter()-t0:.1f}s')
    finally:
        lock.unlink(missing_ok=True)


def status(out):
    ledger = read_ledger(out)
    folds = ledger['folds']
    return dict(complete=sum(e['status'] == 'complete' for e in folds.values()), total=55,
                attempts=sum(e['attempts'] for e in folds.values()),
                retries=sum(max(0, e['attempts'] - 1) for e in folds.values()),
                running=[k for k, e in folds.items() if e['status'] == 'running'],
                interrupted=[k for k, e in folds.items() if e['status'] == 'interrupted'],
                failed=[k for k, e in folds.items() if e['status'] == 'failed'],
                controls={k: v['status'] for k, v in ledger.get('controls', {}).items()},
                measure_only={k: v['status'] for k, v in ledger.get('measure_only', {}).items()},
                noisy=[k for k, e in folds.items() if e.get('noisy')])


# ---------------------------------------------------------------- prepare
def preflight_matrices(X, folds, c, log_fn=print):
    """Plan 7.1 step 2: representation matrices for all five folds (no classifier fitted)."""
    result = dict(folds=[], classifiers_fitted=0)
    groups = [('LR', ['LR_OH', 'LR_NUM']), ('TREE', ['DT_OH', 'DT_NUM']), ('LGBM', ['LGBM_OH', 'LGBM_NAT', 'LGBM_NUM'])]
    for f in range(c['n_folds']):
        a, b = np.flatnonzero(folds != f), np.flatnonzero(folds == f)
        Xtr, Xva = X.iloc[a], X.iloc[b]
        unseen, _ = arms.unseen_geo(Xtr, Xva)
        fold = dict(fold=f, train_rows=len(a), valid_rows=len(b), unseen_valid_geo=unseen,
                    geo_train_unique={col: int(Xtr[col].nunique()) for col in GEO},
                    geo_train_max={col: int(Xtr[col].astype('int64').max()) for col in GEO}, groups={})
        for gname, members in groups:
            built = {m: build_matrices(m, Xtr, Xva) for m in members}
            rec = {}
            for m, (pre, mt, mv) in built.items():
                arms.check_container(m, mt, c)
                arms.check_container(m, mv, c)
                rec[m] = dict(train=arms.container_record(mt), valid=arms.container_record(mv),
                              n_columns=int(mt.shape[1]))
                if arms.arm_spec(m)[1] == 'NUM':
                    rec[m]['equivalence'] = equivalence_report(m, pre, mt, mv, Xtr, Xva, c, include_train=True)
                    require(rec[m]['equivalence']['ok'], f'{m} fold {f}: NUM equivalence failed')
            fold['groups'][gname] = rec
            del built
        result['folds'].append(fold)
        log_fn(f'preflight fold {f}: OK')
    mx = max(max(fd['geo_train_max'].values()) for fd in result['folds'])
    require(mx < c['geo_float32_exact_limit'], 'Geographic ids are not exactly representable in float32')
    result['geo_max_value'] = int(mx)
    return result


def prepare(data, out, private_root=None, loader=None, preflight_fn=None, runner=None):
    out = Path(out)
    require(not out.exists(), 'Do not overwrite an existing study')
    c = config()
    check_plan_sha(c)
    require(environment() == c['environment'], f'Environment differs from the pinned environment: {environment()}')
    code_commit = git_head(runner)
    bad = git_blob_mismatches(SOURCE, runner)
    require(not bad, f'Sealed sources must equal their HEAD blobs; mismatched: {bad}')
    private_root = Path(private_root) if private_root else ROOT.parents[1]/'local_reproduction'
    X, y, ids, old, contract = (loader or load)(data)
    require(sha256_file(Path(data)/'split_manifest.csv') == c['manifest_sha256'], 'Split manifest SHA differs')
    folds = np.asarray(old)
    refs = {}
    for name, ref in c['references'].items():
        path = private_root/ref['relpath']
        r = t9.reference(path, ref['sha256'], ids, y, folds)
        mean = t9.fold_means(r)
        require(abs(mean - ref['cv_mean']) <= c['reference_cv_mean_tolerance'],
                f'{name}: reference CV mean {mean} differs from the registered {ref["cv_mean"]}')
        refs[name] = dict(path=str(path.resolve()), sha256=sha256_file(path), cv_mean_recomputed=mean,
                          registered_cv_mean=ref['cv_mean'], kind=ref['kind'])
    hw = hardware_info(out.parent if out.parent.exists() else private_root)
    require(hw['disk_free_bytes'] >= c['measurement']['min_free_disk_gib'] * 2**30, 'Less than 10 GiB free disk')
    clock = measure.clock_common_check()
    require(clock['common'], 'perf_counter is not shared between processes on this machine')
    pre = (preflight_fn or preflight_matrices)(X, folds, c)
    out.mkdir(parents=True)
    (out/'backup').mkdir()
    write_json(out/'preflight.json', dict(study_id=STUDY_ID, matrices=pre, hardware=hw, clock_check=clock,
                                          created_at_utc=now(), classifiers_fitted=0))
    design = dict(study_id=STUDY_ID, config=c, code_commit=code_commit, source_sha256=sources(),
                  environment=environment(), data_dir=str(Path(data).resolve()), data_contract=contract,
                  manifest_sha256=c['manifest_sha256'], references=refs, hardware=hw, clock_check=clock,
                  preflight_sha256=sha256_file(out/'preflight.json'),
                  plan_sha256_sealed=sha256_file(ROOT/c['plan_doc']),
                  study_targets=study_targets(c), execution_plan=execution_plan(c), max_attempts=c['fit_plan']['max_attempts'],
                  holdout_accessed=False, classifiers_fitted=0, created_at_utc=now())
    write_json(out/'design.json', design)
    write_json(out/'design_seal.json', dict(design_sha256=sha256_file(out/'design.json'), code_commit=code_commit,
                                            created_at_utc=now(), classifiers_fitted=0))
    atomic_write(out/'ledger.json', dict(study_id=STUDY_ID, folds={}, controls={}, measure_only={}))
    log(out, f'prepared and sealed design {sha256_file(out/"design.json")} commit={code_commit}')
    return design


# ---------------------------------------------------------------- finalize
def fold_record(out, arm, f):
    return json.loads((Path(out)/'arms'/arm/f'fold_{f}.json').read_text(encoding='utf-8'))


def assemble_arm(out, y, ids, folds, arm):
    out = Path(out)
    ledger = read_ledger(out)
    entries = [ledger['folds'].get(key(arm, f)) for f in range(5)]
    require(all(e and e['status'] == 'complete' for e in entries), f'{arm} incomplete')
    d = out/'arms'/arm
    if (d/'metrics.json').exists():
        return json.loads((d/'metrics.json').read_text(encoding='utf-8'))
    for e in entries:
        check_saved(out, e)
    parts = pd.concat([read_pred(d/f'fold_{f}_pred.csv.gz') for f in range(5)])
    oof = parts.set_index('building_id').loc[ids.to_numpy()].reset_index()
    require(len(oof) == len(ids) and oof.building_id.is_unique, 'OOF coverage changed')
    np.testing.assert_array_equal(oof.fold.to_numpy(), folds)
    np.testing.assert_array_equal(oof.y_true.to_numpy(), y.to_numpy())
    write_pred(d/'oof.csv.gz', oof)
    back = read_pred(d/'oof.csv.gz')
    np.testing.assert_array_equal(back[PROBS].to_numpy(), oof[PROBS].to_numpy())
    values = [e['macro_f1'] for e in entries]
    result = dict(study_id=STUDY_ID, arm=arm, cv_mean=float(np.mean(values)), cv_std=float(np.std(values, ddof=1)),
                  fold_macro_f1=values, oof_sha256=sha256_file(d/'oof.csv.gz'), finished_at_utc=now())
    write_json(d/'metrics.json', result)
    bd = out/'backup'/'arms'/arm
    for name in ['oof.csv.gz', 'metrics.json']:
        require(not (bd/name).exists(), 'Backup already exists; refusing overwrite')
        shutil.copy2(d/name, bd/name)
    return result


def compute_metric_values(rec):
    """Per-fit values used for compute ratios (plan 9.3)."""
    t = rec['timings']
    mem = rec['memory']['encode_train']
    return dict(
        train_seconds=t['fit_seconds'], predict_seconds=t['predict_seconds'], total_seconds=t['total_seconds'],
        encode_seconds=t['encode_seconds'],
        memory_mib=mem['increment_mib'], memory_floor_limited=rec['wset']['encode_train']['floor_limited'],
        model_bytes=rec['sizes']['model_total_bytes'], matrix_storage_bytes=rec['containers']['train']['storage_bytes'])


def compute_comparison(cand_recs, base_recs, v, min_mib, tag_allowed=True):
    metrics = ['train_seconds', 'predict_seconds', 'total_seconds', 'model_bytes', 'matrix_storage_bytes']
    cv_, bv = [compute_metric_values(r) for r in cand_recs], [compute_metric_values(r) for r in base_recs]
    out = {}
    for m in metrics:
        ratios = [verdict.safe_ratio(a[m], b[m]) for a, b in zip(cv_, bv)]
        s = verdict.ratio_summary(ratios, v)
        if not tag_allowed:
            s['tag'], s['boundary'] = None, False
        out[m] = s
    eligible = [verdict.memory_pair_eligible(a['memory_mib'], b['memory_mib'], a['memory_floor_limited'],
                                             b['memory_floor_limited'], min_mib) for a, b in zip(cv_, bv)]
    ratios = [verdict.safe_ratio(a['memory_mib'], b['memory_mib']) if ok else None
              for a, b, ok in zip(cv_, bv, eligible)]
    mem = verdict.ratio_summary(ratios, v)
    if not tag_allowed:
        mem['tag'], mem['boundary'] = None, False
    mem.update(n_eligible_folds=int(sum(eligible)), eligible_folds=[i for i, ok in enumerate(eligible) if ok], excluded_folds=[i for i, ok in enumerate(eligible) if not ok],
               candidate_increment_mib=[a['memory_mib'] for a in cv_], baseline_increment_mib=[b['memory_mib'] for b in bv],
               exclusion_reason='increment < 16 MiB or FLOOR_LIMITED for the pair' if not all(eligible) else None)
    out['memory_mib'] = mem
    out['encode_seconds'] = verdict.ratio_summary([verdict.safe_ratio(a['encode_seconds'], b['encode_seconds'])
                                                   for a, b in zip(cv_, bv)], v)
    out['encode_seconds']['tag'] = None                  # reported, not tagged (plan 6.2: not a pure representation cost)
    return out


def arm_summary(oof, X, folds):
    y, p = oof.y_true.to_numpy(), oof.y_pred.to_numpy()
    fold_scores = [score(y[folds == f], p[folds == f]) for f in range(5)]
    values = [s['macro_f1'] for s in fold_scores]
    age = X.age.to_numpy() == 995
    return dict(cv_mean=float(np.mean(values)), cv_std=float(np.std(values, ddof=1)), fold_macro_f1=values,
                fold_scores=fold_scores, pooled=score(y, p), age995_profile=score(y[age], p[age]))


def finalize_all(out, loader=None):
    from experiments.geo_ablation.audit import turnover
    out = Path(out)
    X, y, ids, folds, design = verify(out, loader)
    require(not (out/'analysis.json').exists(), 'analysis.json exists; refusing overwrite')
    c = design['config']
    v = c['verdict']
    ledger = read_ledger(out)
    tg = design['study_targets']
    complete = sum(ledger['folds'].get(k, {}).get('status') == 'complete' for k in tg)
    require(complete == len(tg) == 55 and len(ledger['folds']) == 55, 'Ledger incomplete; not finalizing')
    controls = ledger.get('controls', {})
    need = [a for a in arms.all_arms(c) if a.split('_', 1)[1] != 'NUM']
    require(all(a in controls and controls[a]['status'] in ('PASS', 'DIVERGED_KNOWN_ENV') for a in need),
            'Control checks missing or failed')
    attempts = sum(e['attempts'] for e in ledger['folds'].values())
    all_arms = arms.all_arms(c)
    metrics = {a: assemble_arm(out, y, ids, folds, a) for a in all_arms}
    oof = {a: read_pred(out/'arms'/a/'oof.csv.gz') for a in all_arms}
    for a in all_arms:
        np.testing.assert_array_equal(oof[a].building_id.to_numpy(), ids.to_numpy())
    recs = {a: [fold_record(out, a, f) for f in range(5)] for a in all_arms}
    summ = {a: dict(arm_summary(oof[a], X, folds), oof_sha256=metrics[a]['oof_sha256']) for a in all_arms}
    support3 = int((y == 3).sum())
    # unseen-geo subgroup (any geo id absent from the training fold)
    unseen_masks = []
    for f in range(5):
        _, m = arms.unseen_geo(X.iloc[folds != f], X.iloc[folds == f])
        unseen_masks.append(m)
    for a in all_arms:
        sub = []
        for f in range(5):
            mask = unseen_masks[f]
            idx = np.flatnonzero(folds == f)
            n = int(mask.sum())
            yy, pp = y.to_numpy()[idx][mask], oof[a].y_pred.to_numpy()[idx][mask]
            sub.append(dict(fold=f, n=n, macro_f1=score(yy, pp)['macro_f1'] if n >= v['unseen_subgroup_min_rows'] else None))
        summ[a]['unseen_geo_subgroup'] = sub
        if a.endswith('_NUM'):
            summ[a]['num_equivalence_per_fold'] = [
                {k: dict(ok=v['ok'], bitwise_equal=v.get('bitwise_equal'),
                         max_abs_diff_approx_columns=v.get('max_abs_diff_approx_columns'))
                 for k, v in r['equivalence'].items() if isinstance(v, dict) and k.startswith(('valid_vs_', 'train_vs_'))}
                for r in recs[a]]
        summ[a]['per_fold_records'] = [dict(
            fold=f, train_macro_f1=r['train_macro_f1'], timings=r['timings'], memory=r['memory'], wset=r['wset'],
            containers=r['containers'], sizes=r['sizes'], internals=r['internals'], warnings=r['warnings'],
            noisy=r['noisy'], unseen_valid_geo=r['unseen_valid_geo'], n_encoded_columns=r['n_encoded_columns'],
            geo_train_unique=r['geo_train_unique']) for f, r in enumerate(recs[a])]

    def pair(cand, base, tag_perf, tag_compute, confounded=False):
        sc, sb = summ[cand], summ[base]
        deltas = [x - z for x, z in zip(sc['fold_macro_f1'], sb['fold_macro_f1'])]
        wins = int(sum(d > 0 for d in deltas))
        delta = sc['cv_mean'] - sb['cv_mean']
        r3c, r3b = sc['pooled']['recall'][2], sb['pooled']['recall'][2]
        n31c, n31b = sc['pooled']['error31'], sb['pooled']['error31']
        flags = verdict.safety_flags(cand, n31c, base, n31b, v)
        res = dict(candidate=cand, baseline=base, mean_delta=float(delta), fold_delta=deltas, fold_wins=wins,
                   recall3_delta=float(r3c - r3b), error31=dict(candidate=n31c, baseline=n31b, delta=n31c - n31b,
                   candidate_rate=n31c / support3, baseline_rate=n31b / support3),
                   error32=dict(candidate=sc['pooled']['error32'], baseline=sb['pooled']['error32'],
                                delta=sc['pooled']['error32'] - sb['pooled']['error32']),
                   safety_3to1_flagged_arms=flags,
                   turnover=turnover(y.to_numpy(), oof[base].y_pred.to_numpy(), oof[cand].y_pred.to_numpy()),
                   performance_tag=verdict.performance_tag(delta, wins, r3c - r3b, v) if tag_perf else None,
                   container_confounded=bool(confounded),
                   compute=compute_comparison(recs[cand], recs[base], v, c['measurement']['memory_min_increment_mib'],
                                              tag_allowed=tag_compute))
        return res

    comparisons = []
    for model in c['model_order']:
        num, oh = arms.arm_id(model, 'NUM'), arms.arm_id(model, 'OH')
        comparisons.append(pair(num, oh, True, True))
        if model == 'LGBM':
            nat = arms.arm_id(model, 'NAT')
            comparisons.append(pair(num, nat, True, False, confounded=True))
            comparisons.append(pair(nat, oh, False, False, confounded=True))   # NAT is dense, OH is CSR
    # measurement-only repeats (never used in the verdict)
    mo = {}
    for k, e in ledger.get('measure_only', {}).items():
        if e['status'] == 'complete':
            r = json.loads((out/'measure_only'/f'{e["tag"]}.json').read_text(encoding='utf-8'))
            mo[k] = dict(role=e['role'], arm=e['arm'], timings=r['timings'], noisy=r['noisy'])
    repeat_cv = {}
    for a in c['fit_plan']['measurement_only']['repeat_arms']:
        reps = [m for m in mo.values() if m['role'] == 'repeat' and m['arm'] == a]
        main0 = recs[a][0]['timings']
        for metric in ['encode_seconds', 'fit_seconds', 'predict_seconds']:
            vals_r = [m['timings'][metric] for m in reps]
            repeat_cv[f'{a}/{metric}'] = dict(
                repeats_only=measure.coefficient_of_variation(vals_r),
                with_main_fold0=measure.coefficient_of_variation(vals_r + [main0[metric]]), values=vals_r + [main0[metric]])
    res = dict(study_id=STUDY_ID, design_sha256=sha256_file(out/'design.json'), code_commit=design['code_commit'],
               ledger=dict(complete=complete, total=55, attempts=attempts, retries=attempts - complete,
                           noisy_fits=[k for k, e in ledger['folds'].items() if e.get('noisy')]),
               holdout_accessed=False, scheme='stratified',
               controls=controls, note='Descriptive comparison on the stratified split; selection on shared folds is optimistic; '
               'no significance or causal claims; measurements are one Windows machine, single thread.',
               arms=summ, comparisons=comparisons, measurement_only=mo, repeat_cv=repeat_cv,
               floor_limited_counts=dict(
                   floor_limited=sum(r['wset']['encode_train']['floor_limited'] for a in all_arms for r in recs[a]),
                   sampling_limited=sum(r['memory']['encode_train']['sampling_limited'] for a in all_arms for r in recs[a])),
               geo_citations=dict(note='geo2 Group values are quoted from T8/T9 reports, not measured here',
                                  lgbm_geo2=dict(O=0.582827, P=0.582545, N=0.545694, LR_C=0.565146)),
               finished_at_utc=now())
    write_json(out/'analysis.json', res)
    require(not (out/'backup'/'analysis.json').exists(), 'Backup already exists; refusing overwrite')
    shutil.copy2(out/'analysis.json', out/'backup'/'analysis.json')
    log(out, 'analysis written: ' + json.dumps({f'{cp["candidate"]}_vs_{cp["baseline"]}': cp['performance_tag']
                                                for cp in comparisons}))
    return res


# ---------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('action', choices=['prepare', 'fit-all', 'fit-one', 'status', 'finalize'])
    ap.add_argument('--data', help='earthquake/data absolute path (prepare)')
    ap.add_argument('--out', required=True, help='private run directory')
    ap.add_argument('--private-root', help='local_reproduction directory holding the reference OOF files')
    ap.add_argument('--retry-interrupted', action='store_true')
    ap.add_argument('--retry-reason')
    ap.add_argument('--arm')
    ap.add_argument('--fold', type=int)
    ap.add_argument('--role', default='main')
    ap.add_argument('--tag', default='main')
    a = ap.parse_args(argv)
    if a.action == 'prepare':
        require(a.data, '--data is required')
        prepare(a.data, a.out, a.private_root)
    elif a.action == 'fit-all':
        require(not a.retry_interrupted or a.retry_reason, '--retry-interrupted requires --retry-reason')
        fit_all(a.out, retry_reason=a.retry_reason if a.retry_interrupted else None)
    elif a.action == 'fit-one':
        return child_main(a.out, a.arm, a.fold, a.role, a.tag)
    elif a.action == 'finalize':
        finalize_all(a.out)
    else:
        print(json.dumps(status(a.out), indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
