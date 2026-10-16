"""T32: sampling-target scenarios (original -> partial relaxation -> equalisation).

Commands (all private outputs live outside Git; the holdout is never touched):
  gidx      G-IDX gate: dict-path vs 'auto'-path sampler indices on all five folds (labels only).
  prepare   seal the design (HEAD/blob match, data+manifest contract, reuse OOF checks, G-IDX, targets).
  fit-all   three pre-fixed stages; per stage: controls first, then study cells; Fold-wise checkpoints.
  status    ledger summary.
  finalize  aggregate analysis (verdicts A-D, Delta 3->1 beside every Delta Macro F1).

Single factor family: how the per-grade training mass is set (sampling vs weights, direction, share of
the gap closed). Models, hyper-parameters, preprocessing, folds and decision rule never change.
Order inside a fold: schema/encoder/scaler fit on the whole training fold -> sampler or weights -> fit.
Validation rows are never resampled, never reweighted and never passed to the sampler.
"""
from __future__ import annotations

import argparse
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
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from preprocess import CATEGORICAL, make_preprocessor as make_lr_preprocessor, sha256_file  # noqa: E402
from experiments.geo_group_cv.run import PROBS, load, require, score, write  # noqa: E402
from experiments.lgbm_native_cat import run as t8  # noqa: E402
from experiments.sampling_targets import targets as tg  # noqa: E402

HERE = Path(__file__).resolve().parent
STUDY_ID = 'T32_sampling_target_scenarios_v1'
SOURCE = ['preprocess.py', 'baseline.py', 'configs/data_contract.json',
          'experiments/age995/run.py', 'experiments/geo_group_cv/run.py',
          'experiments/model_families/run.py', 'experiments/model_families/config.json',
          'experiments/lgbm_native_cat/run.py', 'experiments/lgbm_native_cat/config.json',
          'experiments/sampling_targets/__init__.py', 'experiments/sampling_targets/targets.py',
          'experiments/sampling_targets/run.py', 'experiments/sampling_targets/config.json',
          'docs/T32_SAMPLING_TARGET_SCENARIOS_PLAN.md', 'tests/test_sampling_targets.py']
STAGES = [1, 2, 3]
GRADE3 = 3


class NonConvergence(RuntimeError):
    pass


class ControlFailure(RuntimeError):
    pass


def now():
    return datetime.now(timezone.utc).isoformat()


def py(obj):
    """Recursively turn numpy scalars/arrays into JSON-safe python values."""
    if isinstance(obj, dict):
        return {str(k): py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [py(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return py(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def jwrite(path, obj):
    write(path, py(obj))


def guard_path(path):
    """Any path that touches the final holdout is refused (also for enumeration)."""
    if 'final_holdout' in str(path).replace('\\', '/').lower():
        raise PermissionError(f'Holdout access refused: {path}')
    return Path(path)


# ----------------------------------------------------------------------------- config
def config(path=HERE/'config.json'):
    c = json.loads(Path(path).read_text(encoding='utf-8'))
    require(c['study_id'] == STUDY_ID and c['scheme'] == 'stratified' and c['n_folds'] == 5, 'Unplanned study')
    require(c['scope']['models'] == ['LR', 'LGBMN'] and c['scope']['rf_extension'] is False, 'Model scope changed')
    require(c['automatic_retries'] == 0 and c['stop_on_nonconvergence'] and c['outer_parallel_jobs'] == 1
            and not c['holdout_access'] and not c['automatic_model_replacement'], 'Unplanned run policy')
    ids = [u['id'] for u in c['units']] + [u['id'] for u in c['controls']]
    require(len(ids) == len(set(ids)), 'Duplicate unit ids')
    stage_fits = {s: 0 for s in STAGES}
    for u in c['units']:
        require(u['model'] in ('LR', 'LGBMN') and u['stage'] in STAGES, f'Bad unit {u["id"]}')
        require(u['scenario'] in tg.ARM_SCENARIOS[u['arm']] and u['arm'] != 'ORI', f'Bad arm/scenario {u["id"]}')
        require((u['model'] == 'LR' and u['arm'] in ('RUP', 'RDN', 'WOBJ'))
                or (u['model'] == 'LGBMN' and u['arm'] in ('RUP', 'WNORM')), f'Arm not allowed for model {u["id"]}')
        expected = f'{u["model"]}.{u["arm"]}.{u["scenario"]}' + (f'.s{u["sampler_seed"]}' if u['noise'] else '')
        require(u['id'] == expected and (u['noise'] or u['sampler_seed'] == c['sampler_seed']), f'Bad id/seed {u["id"]}')
        stage_fits[u['stage']] += 5
    require({str(k): v for k, v in stage_fits.items()} == c['stage_fits'] and sum(stage_fits.values()) == c['max_new_fits'] == 110,
            'Fit plan differs from the registered 30/25/55 = 110')
    require(sum(len(x['folds']) for x in c['controls']) == c['max_control_fits'] == 8
            and all(x['folds'] == [0, 2] for x in c['controls']), 'Control plan differs (Fold 0 and 2, 8 fits)')
    require(sorted((x['id'], x['stage']) for x in c['controls']) == [('c1', 1), ('c2', 2), ('c3', 3), ('c4', 1)],
            'Control stages differ')
    require(c['manual_recovery_max'] == 1, 'Manual recovery limit changed')
    for name, pq in c['phi'].items():
        require(tuple(pq) == tg.PHI[name], 'phi table differs from targets.py')
    c['lgbm_params'] = t8.config()['lgbm_params']          # T8 params, unchanged (validated against T4 there)
    return c


def environment():
    names = ['numpy', 'pandas', 'scipy', 'scikit-learn', 'lightgbm', 'joblib', 'threadpoolctl', 'imbalanced-learn']
    return dict(python=platform.python_version(), **{n: importlib.metadata.version(n) for n in names})


def sources():
    return {p: sha256_file(ROOT/p) for p in SOURCE}


def items_from_config(c):
    """Study units (5 folds each) and controls (Fold 0 and 2) as uniform work items."""
    items = []
    for s in STAGES:
        for u in c['controls']:
            if u['stage'] == s:
                items.append(dict(kind='control', id=u['id'], model=u['model'], arm=u['arm'], scenario=u['scenario'],
                                  sampler_seed=c['sampler_seed'], stage=s, folds=list(u['folds']),
                                  reference=u['reference']))
        for u in c['units']:
            if u['stage'] == s:
                items.append(dict(kind='study', id=u['id'], model=u['model'], arm=u['arm'], scenario=u['scenario'],
                                  sampler_seed=u['sampler_seed'], stage=s, folds=list(range(5)), noise=u['noise']))
    return items


def key(item_id, fold):
    return f'{item_id}/{fold}'


def item_dir(item):
    return ('controls' if item['kind'] == 'control' else 'cells') + '/' + item['id']


# ----------------------------------------------------------------------------- plan (targets, rows, weights)
def build_plan(item, y_train):
    """Training labels -> before/target/after, optional row indices, optional per-row weights.

    The validation fold is not an input here (leak-proof by signature).
    """
    arm, scenario, seed = item['arm'], item['scenario'], int(item['sampler_seed'])
    require(arm in tg.ARM_DIRECTION and scenario in tg.ARM_SCENARIOS[arm], f'Bad arm/scenario {arm}/{scenario}')
    y_train = np.asarray(y_train)
    direction = tg.ARM_DIRECTION[arm]
    n = tg.count_vector(y_train)
    t = tg.scenario_targets(n, direction, scenario)
    rec = dict(arm=arm, scenario=scenario, direction=direction, train_rows_before=int(len(y_train)))
    idx = None
    row_weight = None
    if arm in ('RUP', 'RDN'):
        strategy = tg.sampler_strategy(n, t, direction)
        idx = tg.sample_indices(y_train, strategy, direction, seed)
        after = np.bincount(y_train[idx], minlength=4)[1:4].astype(int).tolist()
        require(after == list(t), f'Sampler contract violated: after {after} != target {list(t)}')
        if direction == 'DOWN':
            require(len(np.unique(idx)) == len(idx), 'Under-sampling must not duplicate rows')
        else:
            require(np.array_equal(idx[:len(y_train)], np.arange(len(y_train))), 'Original rows must come first')
        rec.update(kind='sample', sampler='imblearn Random%sSampler' % ('Over' if direction == 'UP' else 'Under'),
                   sampler_seed=seed, sampling_strategy={str(k): int(v) for k, v in strategy.items()},
                   indices_sha256=tg.index_sha256(idx), fit_rows=int(len(idx)), class_weight=None,
                   effective_mass=[float(v) for v in after])
    elif arm in ('WOBJ', 'WNORM'):
        kind = 'obj' if arm == 'WOBJ' else 'norm'
        cw = tg.class_weights(n, t, kind)
        row_weight = cw[y_train.astype(int) - 1]
        mass = tg.effective_mass(y_train, cw)
        target_mass = np.asarray(t, dtype=float) * (1.0 if kind == 'obj' else sum(n) / sum(t))
        require(np.allclose(mass, target_mass, rtol=1e-9, atol=0), 'Effective mass differs from the target')
        after = list(n)
        rec.update(kind='weight', sampler=None, sampler_seed=None, sampling_strategy=None, indices_sha256=None,
                   fit_rows=int(len(y_train)), class_weight=[float(v) for v in cw], effective_mass=mass,
                   effective_mass_target=[float(v) for v in target_mass], weight_kind=kind,
                   weight_sum=float(np.sum(row_weight)))
    else:
        after = list(n)
        rec.update(kind='none', sampler=None, sampler_seed=None, sampling_strategy=None, indices_sha256=None,
                   fit_rows=int(len(y_train)), class_weight=None, effective_mass=[float(v) for v in n])
    rec.update(before=list(n), target=list(t), after=[int(v) for v in after],
               delta=[int(a) - int(b) for a, b in zip(after, n)],
               multiplier=[a / b for a, b in zip(after, n)],
               share_after=[a / sum(after) for a in after], after_equals_target=bool(list(after) == list(t)) or rec['kind'] == 'weight')
    return dict(idx=idx, row_weight=row_weight, record=rec)


# ----------------------------------------------------------------------------- models
def fit_lr(X_train, y_train, X_valid, plan, c):
    p = c['lr']
    pre = make_lr_preprocessor(scale_numeric=p['scale_numeric'])
    with threadpool_limits(limits=p['threadpool_limit']):
        t0 = time.perf_counter()
        Z = pre.fit_transform(X_train, y_train)            # fitted on the whole training fold, before sampling
        Zv = pre.transform(X_valid)
        pre_seconds = time.perf_counter() - t0
    require(sparse.isspmatrix_csr(Z) and np.isfinite(Z.data).all(), 'CSR finite training matrix required')
    yt = np.asarray(y_train)
    idx = plan['idx']
    Zf, yf = (Z, yt) if idx is None else (Z[idx], yt[idx])
    model = LogisticRegression(solver=p['solver'], C=p['C'], max_iter=p['max_iter'], tol=p['tol'],
                               random_state=p['random_state'])
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=p['threadpool_limit']):
        warnings.simplefilter('always')
        t0 = time.perf_counter()
        model.fit(Zf, yf, sample_weight=plan['row_weight'])
        fit_seconds = time.perf_counter() - t0
        proba = np.asarray(model.predict_proba(Zv), dtype=np.float64)
    info = dict(fit_seconds=fit_seconds, preprocessing_seconds=pre_seconds, encoded_columns=int(Z.shape[1]),
                n_iter=np.asarray(model.n_iter_).tolist(), max_iter=p['max_iter'],
                warnings=[dict(category=w.category.__name__, message=str(w.message)) for w in caught])
    info['converged'] = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    if not info['converged']:
        raise NonConvergence(f'LR did not converge: n_iter={info["n_iter"]} max_iter={p["max_iter"]}')
    require(list(model.classes_) == [1, 2, 3], 'Class order must be 1, 2, 3')
    check_proba(proba, len(X_valid))
    return pre, model, proba, info


def check_proba(p, n_rows):
    require(p.shape == (n_rows, 3) and np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all(), 'Invalid probabilities')
    np.testing.assert_allclose(p.sum(1), 1.0, atol=1e-6, rtol=0)


def fit_lgbm(X_train, y_train, X_valid, plan, c, params):
    from lightgbm import LGBMClassifier
    threads = c['lgbm']['threadpool_limit']
    pre = t8.make_preprocessor('N')
    with threadpool_limits(limits=threads):
        t0 = time.perf_counter()
        mt = pre.fit_transform(X_train)
        mv = pre.transform(X_valid)
        pre_seconds = time.perf_counter() - t0
    t8.check_matrix('N', mt)
    t8.check_matrix('N', mv)
    enc = pre['columns'].named_transformers_['categorical']
    schema = pre['schema'].transform(X_train)
    for col, learned in zip(CATEGORICAL, enc.categories_):
        require(set(learned) == set(schema[col]), 'Vocabulary must equal training-fold values')
    yt = np.asarray(y_train)
    idx = plan['idx']
    mf, yf = (mt, yt) if idx is None else (mt[idx], yt[idx])
    kw = dict(categorical_feature=t8.categorical_indices())
    if plan['row_weight'] is not None:
        kw['sample_weight'] = plan['row_weight']
    model = LGBMClassifier(**params)
    with threadpool_limits(limits=threads), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        t0 = time.perf_counter()
        model.fit(mf, yf.astype(np.int32) - 1, **kw)
        fit_seconds = time.perf_counter() - t0
        proba = t8.checked(model, mv)
    require(model.booster_.current_iteration() == params['n_estimators'], 'Full fixed boosting budget not completed')
    info = dict(fit_seconds=fit_seconds, preprocessing_seconds=pre_seconds, encoded_columns=int(mt.shape[1]),
                rounds_completed=int(model.booster_.current_iteration()),
                vocabulary_sizes={col: int(len(v)) for col, v in zip(CATEGORICAL, enc.categories_)},
                valid_unseen_nan={col: int(np.isnan(mv[:, t8.categorical_indices()][:, i]).sum()) for i, col in enumerate(CATEGORICAL)},
                warnings=[dict(category=w.category.__name__, message=str(w.message)) for w in caught])
    require(not np.isnan(mf[:, t8.categorical_indices()]).any(), 'Training codes cannot be missing')
    check_proba(proba, len(X_valid))
    return pre, model, proba, info


# ----------------------------------------------------------------------------- io helpers
def atomic_write(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    jwrite(tmp, obj)
    os.replace(tmp, path)


def read_pred(path):
    return pd.read_csv(path, float_precision='round_trip')


def write_pred(path, frame):
    frame.to_csv(path, index=False, compression={'method': 'gzip', 'mtime': 0}, float_format='%.17g')


def log(out, message):
    line = f'{now()} [pid {os.getpid()}] {message}'
    print(line, flush=True)
    with open(Path(out)/'run.log', 'a', encoding='utf-8') as fh:
        fh.write(line + '\n')


def event(out, **kw):
    with open(Path(out)/'attempts.jsonl', 'a', encoding='utf-8') as fh:
        fh.write(json.dumps(py(dict(at_utc=now(), pid=os.getpid(), **kw)), ensure_ascii=False) + '\n')


def read_ledger(out):
    return json.loads((Path(out)/'ledger.json').read_text(encoding='utf-8'))


def check_saved(out, entry):
    for name, digest in entry['artifact_sha256'].items():
        require(sha256_file(Path(out)/entry['dir']/name) == digest, f'Artifact changed: {entry["dir"]}/{name}')
        require(sha256_file(Path(out)/'backup'/entry['dir']/name) == digest, f'Backup changed: {entry["dir"]}/{name}')


# ----------------------------------------------------------------------------- references (reused OOF / T8 folds)
_REF_CACHE: dict = {}


def reference_spec(design, name):
    for r in design['config']['references']:
        if r['name'] == name:
            return r
    raise KeyError(name)


def reference_frame(design, name):
    """Hash-verified reference predictions indexed by building_id (all folds)."""
    ck = (design['local_root'], name)
    if ck in _REF_CACHE:
        return _REF_CACHE[ck]
    spec, root = reference_spec(design, name), Path(design['local_root'])
    if spec['kind'] == 'oof':
        path = guard_path(root/spec['path'])
        require(sha256_file(path) == spec['sha256'], f'Reference OOF changed: {name}')
        frame = read_pred(path)
    else:
        parts = []
        for f in range(5):
            path = guard_path(Path(root)/spec['dir']/f'fold_{f}_pred.csv.gz')
            require(sha256_file(path) == spec['fold_sha256'][f], f'Reference fold changed: {name} fold {f}')
            parts.append(read_pred(path))
        frame = pd.concat(parts, ignore_index=True)
    require(frame.building_id.is_unique, f'Reference OOF ids not unique: {name}')
    frame = frame.set_index('building_id')
    _REF_CACHE[ck] = frame
    return frame


def compare_control(new, ref, fold, atol):
    """new: fold prediction frame; ref: reference indexed by building_id. Returns the control verdict."""
    r = ref.reindex(new.building_id.to_numpy())
    rf = ref[ref.fold == fold]
    complete = bool(r.y_pred.notna().all())
    diff = float(np.max(np.abs(new[PROBS].to_numpy() - r[PROBS].to_numpy()))) if complete else None
    out = dict(rows=int(len(new)), reference_fold_rows=int(len(rf)), all_ids_in_reference=complete,
               ids_identical=bool(len(rf) == len(new) and set(rf.index) == set(new.building_id)),
               fold_label_identical=bool(complete and (r.fold.to_numpy() == fold).all()),
               y_true_identical=bool(complete and np.array_equal(r.y_true.to_numpy(), new.y_true.to_numpy())),
               predicted_labels_identical=bool(complete and np.array_equal(r.y_pred.to_numpy(), new.y_pred.to_numpy())),
               probability_max_abs_diff=diff, probability_atol=atol)
    out['status'] = 'PASS' if (complete and out['ids_identical'] and out['fold_label_identical'] and out['y_true_identical']
                               and out['predicted_labels_identical'] and diff <= atol) else 'FAIL'
    return out


# ----------------------------------------------------------------------------- one fold
def execute(X, y, ids, folds, item, f, design, d):
    """Fit one (item, fold), write all artifacts into d, verify by reloading. Returns (record, files)."""
    c = design['config']
    a, b = np.flatnonzero(folds != f), np.flatnonzero(folds == f)
    require(len(a) + len(b) == len(ids) and len(b) > 0 and not np.intersect1d(a, b).size, 'Fold membership invalid')
    X_tr, X_va = X.iloc[a], X.iloc[b]
    y_tr, y_va = y.iloc[a].to_numpy(), y.iloc[b].to_numpy()
    plan = build_plan(item, y_tr)
    rec = plan['record']
    exp_n = design['fold_train_counts'][str(f)]
    require(rec['before'] == exp_n, f'Training class counts differ from the sealed design in fold {f}')
    params = c['lgbm_params'] if item['model'] == 'LGBMN' else None
    if item['model'] == 'LR':
        pre, model, proba, info = fit_lr(X_tr, y_tr, X_va, plan, c)
    else:
        pre, model, proba, info = fit_lgbm(X_tr, y_tr, X_va, plan, c, params)
    pred = proba.argmax(1) + 1
    require(len(proba) == len(y_va), 'One prediction per validation row required')
    d.mkdir(parents=True, exist_ok=True)
    pre_path, pred_path, idx_path = d/f'fold_{f}_preprocessor.joblib', d/f'fold_{f}_pred.csv.gz', d/f'fold_{f}_sample_indices.npy'
    joblib.dump(pre, pre_path, compress=3)
    if item['model'] == 'LR':
        model_path = d/f'fold_{f}_model.joblib'
        joblib.dump(model, model_path, compress=3)
    else:
        model_path = d/f'fold_{f}_model.txt'
        model.booster_.save_model(str(model_path))
    frame = pd.DataFrame({'building_id': ids.iloc[b].to_numpy(), 'fold': f, 'y_true': y_va, 'y_pred': pred,
                          **{n: proba[:, i] for i, n in enumerate(PROBS)}})
    write_pred(pred_path, frame)
    files = [model_path, pre_path, pred_path]
    if plan['idx'] is not None:
        np.save(idx_path, plan['idx'].astype(np.int32))
        require(tg.index_sha256(np.load(idx_path)) == rec['indices_sha256'], 'Saved sample indices differ')
        files.append(idx_path)
    # reload verification (trusted, freshly written files only)
    loaded_pre = joblib.load(pre_path)
    if item['model'] == 'LR':
        loaded = joblib.load(model_path)
        with threadpool_limits(limits=c['lr']['threadpool_limit']):
            p2 = np.asarray(loaded.predict_proba(loaded_pre.transform(X_va)), dtype=np.float64)
    else:
        import lightgbm
        booster = lightgbm.Booster(model_file=str(model_path))
        with threadpool_limits(limits=c['lgbm']['threadpool_limit']):
            p2 = np.asarray(booster.predict(loaded_pre.transform(X_va), num_threads=params['n_jobs']), dtype=np.float64)
    reload_diff = float(np.max(np.abs(p2 - proba)))
    require(reload_diff <= c['checkpoint']['model_reload_probability_atol'] and np.array_equal(p2.argmax(1) + 1, pred),
            f'Model reload mismatch {reload_diff}')
    back = read_pred(pred_path)
    np.testing.assert_array_equal(back.building_id.to_numpy(), frame.building_id.to_numpy())
    np.testing.assert_array_equal(back.y_pred.to_numpy(), pred)
    np.testing.assert_array_equal(back[PROBS].to_numpy(), proba)
    record = dict(study_id=STUDY_ID, kind=item['kind'], id=item['id'], model=item['model'], arm=item['arm'],
                  scenario=item['scenario'], fold=f, train_rows=len(a), valid_rows=len(b),
                  train_class_counts=rec['before'],
                  valid_class_counts=[int((y_va == v).sum()) for v in (1, 2, 3)],
                  metrics=score(y_va, pred), sampling=rec, info=info,
                  reload_probability_max_abs_diff=reload_diff, csv_round_trip_exact=True,
                  effective_parameters=model.get_params(),
                  finished_at_utc=now())
    require(record['valid_class_counts'] == [int(v) for v in np.bincount(y.to_numpy()[folds == f], minlength=4)[1:4]],
            'Validation class counts changed')
    control = None
    if item['kind'] == 'control':
        control = compare_control(frame, reference_frame(design, item['reference']), f,
                                  c['control_criteria']['probability_atol'])
        record['control'] = control
    return record, files, control


def fit_unit(X, y, ids, folds, out, item, f, design, execute_fn=None):
    out = Path(out)
    c = design['config']
    k = key(item['id'], f)
    require(k in design['fit_keys'], f'{k} is not a registered fit')
    ledger = read_ledger(out)
    entry = ledger['folds'].get(k)
    if entry and entry['status'] == 'complete':
        check_saved(out, entry)
        log(out, f'{k} already complete; verified hashes, not refit')
        if entry.get('control_status') == 'FAIL':
            raise ControlFailure(f'{k} control FAIL (recorded); reused cell not qualified')
        return entry, False
    require(not (entry and entry['status'] == 'failed'),
            f'{k} failed earlier; no automatic retry. A new pre-registration and approval are required.')
    prior = entry['attempts'] if entry else 0
    if entry:  # status running: process vanished (manual recovery, at most once overall)
        require(len(ledger['recoveries']) < c['manual_recovery_max'], f'{k} unfinished and manual recovery exhausted')
        ledger['recoveries'].append(k)
        log(out, f'{k} previous attempt did not finish; manual recovery attempt {prior+1}')
    cap = c['max_new_fits'] if item['kind'] == 'study' else c['max_control_fits']
    ext = sum(1 for r in ledger['recoveries'] if design['fit_keys'][r] == item['kind'])
    used = sum(e['attempts'] for e in ledger['folds'].values() if e['kind'] == item['kind'])
    require(used < cap + ext, 'Fit budget exhausted')
    d = out/item_dir(item)
    entry = dict(kind=item['kind'], id=item['id'], dir=item_dir(item), fold=f, stage=item['stage'], status='running',
                 attempts=prior+1, started_at_utc=now())
    ledger['folds'][k] = entry
    atomic_write(out/'ledger.json', ledger)
    d.mkdir(parents=True, exist_ok=True)
    for stale in d.glob(f'fold_{f}[._]*'):
        stale.rename(stale.with_name(stale.name + f'.incomplete_attempt{prior}'))
    event(out, fold=k, action='fit_start', attempt=prior+1)
    try:
        record, files, control = (execute_fn or execute)(X, y, ids, folds, item, f, design, d)
        write(d/f'fold_{f}.json', py(record))
        shas = {p.name: sha256_file(p) for p in files + [d/f'fold_{f}.json']}
        bd = out/'backup'/entry['dir']
        bd.mkdir(parents=True, exist_ok=True)
        for name in shas:
            require(not (bd/name).exists(), 'Backup already exists; refusing overwrite')
            shutil.copy2(d/name, bd/name)
        entry.update(status='complete', artifact_sha256=shas, macro_f1=record['metrics']['macro_f1'],
                     fit_seconds=record['info']['fit_seconds'], finished_at_utc=now())
        if control is not None:
            entry.update(control_status=control['status'], control_max_abs_diff=control['probability_max_abs_diff'],
                         control_labels_identical=control['predicted_labels_identical'])
        check_saved(out, entry)
    except Exception as exc:
        ledger = read_ledger(out)
        ledger['folds'][k].update(status='failed', error=f'{type(exc).__name__}: {exc}', failed_at_utc=now(),
                                  nonconvergence=isinstance(exc, NonConvergence))
        atomic_write(out/'ledger.json', ledger)
        event(out, fold=k, action='failed', error=f'{type(exc).__name__}: {exc}', trace=traceback.format_exc())
        log(out, f'STOPPED {k} {type(exc).__name__}: {exc}')
        raise
    ledger = read_ledger(out)
    ledger['folds'][k] = entry
    atomic_write(out/'ledger.json', ledger)
    event(out, fold=k, action='fit_complete', macro_f1=entry['macro_f1'], seconds=entry['fit_seconds'],
          artifact_sha256=shas, control_status=entry.get('control_status'))
    log(out, f'{k} F1={entry["macro_f1"]:.9f} fit={entry["fit_seconds"]:.1f}s '
             f'after={record["sampling"]["after"]} target={record["sampling"]["target"]} '
             f'reload_diff={record["reload_probability_max_abs_diff"]:.3g}'
             + (f' control={entry["control_status"]} diff={entry["control_max_abs_diff"]}' if control else '')
             + ' saved+verified+backed-up')
    if control is not None and control['status'] != 'PASS':
        raise ControlFailure(f'{k} control FAIL: {control}')
    return entry, True


# ----------------------------------------------------------------------------- stage orchestration
def stage_items(design, stage):
    return [it for it in design['items'] if it['stage'] == stage]


def stage_complete(ledger, design, stage):
    for it in stage_items(design, stage):
        for f in it['folds']:
            e = ledger['folds'].get(key(it['id'], f))
            if not e or e['status'] != 'complete':
                return False
            if it['kind'] == 'control' and e.get('control_status') != 'PASS':      # missing status counts as incomplete
                return False
    return True


def preflight_ledger(ledger):
    bad = [k for k, e in ledger['folds'].items() if e['status'] == 'failed']
    require(not bad, f'Failed units present ({bad[:3]}...); no automatic retry or bypass. New pre-registration needed.')
    ctl = [k for k, e in ledger['folds'].items() if e.get('control_status') == 'FAIL']
    require(not ctl, f'Control FAIL recorded ({ctl}); the reused cell is not qualified. Separate approval needed.')


def fit_all(out, stage, unit_fn=None, verify_fn=None):
    out = Path(out)
    lock = out/'RUNNING.lock'
    try:
        with open(lock, 'x') as fh:
            fh.write(f'pid={os.getpid()} started={now()}\n')
    except FileExistsError:
        raise ValueError(f'Another run holds {lock}; remove only if that process is dead')
    try:
        X, y, ids, folds, design = (verify_fn or verify)(out)
        before = pd.util.hash_pandas_object(X, index=True).to_numpy()
        wanted = STAGES if stage == 'all' else [int(stage)]
        require(all(s in STAGES for s in wanted), 'Stage must be 1, 2, 3 or all')
        t0 = time.perf_counter()
        for s in wanted:
            ledger = read_ledger(out)
            preflight_ledger(ledger)
            for prev in STAGES:
                if prev < s:
                    require(stage_complete(ledger, design, prev), f'Stage {prev} must be complete before stage {s}')
            log(out, f'stage {s} start')
            work = stage_items(design, s)
            for it in [i for i in work if i['kind'] == 'control'] + [i for i in work if i['kind'] == 'study']:
                for f in it['folds']:
                    (unit_fn or fit_unit)(X, y, ids, folds, out, it, f, design)
            ledger = read_ledger(out)
            require(stage_complete(ledger, design, s), f'Stage {s} incomplete after run')
            fits_after = sum(1 for e in ledger['folds'].values() if e['kind'] == 'study' and e['status'] == 'complete')
            cumulative = sum(design['config']['stage_fits'][str(k)] for k in STAGES if k <= s)
            log(out, f'stage {s} complete; cumulative study fits completed {fits_after} (registered cumulative {cumulative})')
            require(fits_after == cumulative, 'Completed study fits differ from the registered cumulative plan')
        np.testing.assert_array_equal(before, pd.util.hash_pandas_object(X, index=True).to_numpy())
        log(out, f'fit-all finished wall={time.perf_counter()-t0:.1f}s')
    finally:
        lock.unlink(missing_ok=True)


# ----------------------------------------------------------------------------- prepare / verify
def git_blob_check(paths, code_commit=None):
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    require(len(head) == 40, 'Cannot read HEAD')
    require(code_commit in (None, head), 'code_commit differs from HEAD')
    bad = []
    for p in paths:
        r = subprocess.run(['git', 'rev-parse', f'HEAD:{p}'], cwd=ROOT, capture_output=True, text=True)
        w = subprocess.run(['git', 'hash-object', '--', p], cwd=ROOT, capture_output=True, text=True)
        if r.returncode != 0 or w.returncode != 0 or r.stdout.strip() != w.stdout.strip():
            bad.append(p)
    require(not bad, f'Sealed sources must match HEAD blobs (committed, unmodified): {bad}')
    return head


def gidx_command(data, out):
    data_dir = guard_path(data)
    out = guard_path(out)
    require(not out.exists(), 'Do not overwrite an existing G-IDX result')
    X, y, ids, folds, contract = load(data_dir)
    result = tg.gidx(y.to_numpy(), folds, config()['sampler_seed'])
    result.update(study_id=STUDY_ID, created_at_utc=now(), classifiers_fitted=0, holdout_accessed=False,
                  environment=environment(), data_contract=contract)
    jwrite(out, result)
    print(json.dumps(py({str(f): v['pass'] for f, v in result['folds'].items()}), indent=1))
    require(result['all_pass'], 'G-IDX FAILED: dict path and auto path select different indices')
    return result


def verify_references(c, root, ids, y, folds):
    """Bytes SHA, row/fold/label alignment, argmax==y_pred, recomputed metrics against the pinned values."""
    refs = {}
    for r in c['references']:
        if r['kind'] == 'oof':
            path = guard_path(Path(root)/r['path'])
            require(path.is_file() and sha256_file(path) == r['sha256'], f'Reference SHA mismatch: {r["name"]}')
            frame = read_pred(path)
            files = {str(path): r['sha256']}
        else:
            frames, files = [], {}
            for f in range(5):
                path = guard_path(Path(root)/r['dir']/f'fold_{f}_pred.csv.gz')
                require(path.is_file() and sha256_file(path) == r['fold_sha256'][f], f'Reference fold SHA mismatch: {r["name"]} {f}')
                frames.append(read_pred(path))
                files[str(path)] = r['fold_sha256'][f]
            frame = pd.concat(frames, ignore_index=True)
            dp = guard_path(Path(root)/r['design_path'])
            require(dp.is_file() and sha256_file(dp) == r['design_sha256'], 'T8 design SHA differs')
        require(len(frame) == c['expected_rows'] and frame.building_id.is_unique, f'Reference coverage: {r["name"]}')
        o = frame.set_index('building_id').loc[ids.to_numpy()].reset_index()
        np.testing.assert_array_equal(o.fold.to_numpy(), folds)
        np.testing.assert_array_equal(o.y_true.to_numpy(), y.to_numpy())
        require(np.array_equal(o[PROBS].to_numpy().argmax(1) + 1, o.y_pred.to_numpy()), f'argmax != y_pred: {r["name"]}')
        vals = [score(o.y_true[o.fold == f], o.y_pred[o.fold == f])['macro_f1'] for f in range(5)]
        pooled = score(o.y_true, o.y_pred)
        mean = float(np.mean(vals))
        require(abs(mean - r['cv_mean']) <= c['reference_cv_mean_tolerance'], f'Reference mean differs from pin: {r["name"]} {mean}')
        if r['error31'] is not None:
            require(pooled['error31'] == r['error31'], f'Reference 3->1 count differs from pin: {r["name"]}')
        mj = None
        if r['kind'] == 'oof':
            mp = Path(path).with_name('metrics.json')
            if mp.is_file():
                mj = json.loads(mp.read_text(encoding='utf-8')).get('cv_macro_f1_mean')
                require(mj is None or abs(mj - mean) <= 1e-9, f'metrics.json mean differs from recomputed: {r["name"]}')
        print(f'reference {r["name"]}: recomputed cv_mean={mean:.9f} pin={r["cv_mean"]:.6f} diff={mean - r["cv_mean"]:+.3e} '
              f'(tolerance {c["reference_cv_mean_tolerance"]:.1e})', flush=True)
        refs[r['name']] = dict(pin_diff=mean - r['cv_mean'], role=r['role'], files=files, cv_mean=mean, fold_macro_f1=vals,
                               error31=pooled['error31'], error32=pooled['error32'], metrics_json_mean=mj)
    return refs


def prepare(data, out, local_root=None, code_commit=None):
    out, data_dir = Path(out), guard_path(data)
    guard_path(out)
    require(not out.exists(), 'Do not overwrite an existing study')
    c = config()
    require(environment() == c['environment'], 'Environment differs from the pinned environment')
    require(sha256_file(ROOT/c['plan_doc']) == c['plan_sha256'], 'Plan document SHA differs from the approved design')
    head = git_blob_check(SOURCE, code_commit)
    root = Path(local_root) if local_root else Path(data_dir).resolve().parents[1]
    X, y, ids, folds, contract = load(data_dir)
    require(sha256_file(Path(data_dir)/'split_manifest.csv') == c['manifest_sha256'], 'Split manifest SHA differs')
    require(len(X) == c['expected_rows'], 'Development row count changed')
    fold_counts, target_table = {}, {}
    for f in range(5):
        yt = y.to_numpy()[folds != f]
        yv = y.to_numpy()[folds == f]
        require(len(yt) == c['train_rows_per_fold'] and len(yv) == c['valid_rows_per_fold'], 'Frozen fold size changed')
        n = tg.count_vector(yt)                      # also checks n1 < n3 < n2
        require(list(n) == c['fold_train_counts'][str(f)], f'Fold {f} training counts differ from the registered values')
        fold_counts[str(f)] = list(n)
        target_table[str(f)] = {d: {s: tg.describe_targets(n, tg.scenario_targets(n, d, s))
                                    for s in (tg.ARM_SCENARIOS['RUP'] if d == 'UP' else tg.ARM_SCENARIOS['RDN'])}
                                for d in ('UP', 'DOWN')}
        # W-NORM at EQ must equal sklearn 'balanced' (vector check, no training)
        w = tg.class_weights(n, tg.scenario_targets(n, 'UP', 'EQ'), 'norm')[yt.astype(int) - 1]
        np.testing.assert_allclose(w, compute_sample_weight('balanced', yt), rtol=c['weights']['w_norm_eq_vs_balanced_rtol'], atol=0)
    require(c['grade3_support'] == int((y.to_numpy() == GRADE3).sum()), 'Grade 3 support changed')
    gate = tg.gidx(y.to_numpy(), folds, c['sampler_seed'])
    require(gate['all_pass'], 'G-IDX gate failed: prepare refused')
    refs = verify_references(c, root, ids, y, folds)
    anchor = Path(out).resolve().parent
    while not anchor.exists():
        anchor = anchor.parent
    free_gb = shutil.disk_usage(anchor).free / 1e9
    require(free_gb >= c['min_free_gb'], f'Free disk {free_gb:.1f} GB below the {c["min_free_gb"]} GB floor')
    items = items_from_config(c)
    fit_keys = {key(it['id'], f): it['kind'] for it in items for f in it['folds']}
    require(sum(1 for v in fit_keys.values() if v == 'study') == c['max_new_fits']
            and sum(1 for v in fit_keys.values() if v == 'control') == c['max_control_fits'], 'Fit key count differs')
    out.mkdir(parents=True)
    for sub in ('cells', 'controls', 'backup'):
        (out/sub).mkdir()
    design = dict(study_id=STUDY_ID, config=c, config_sha256=sha256_file(HERE/'config.json'), code_commit=head,
                  source_sha256=sources(), plan_sha256=c['plan_sha256'], environment=environment(),
                  data_contract=contract, data_dir=str(Path(data_dir).resolve()), local_root=str(root.resolve()),
                  manifest_sha256=c['manifest_sha256'],
                  fold_train_counts=fold_counts, target_table=target_table, gidx=gate, references=refs,
                  items=items, fit_keys=fit_keys, free_disk_gb=free_gb, classifiers_fitted=0,
                  holdout_accessed=False, created_at_utc=now())
    jwrite(out/'design.json', design)
    jwrite(out/'design_seal.json', dict(design_sha256=sha256_file(out/'design.json'), code_commit=head,
                                        created_at_utc=now(), classifiers_fitted=0))
    atomic_write(out/'ledger.json', dict(study_id=STUDY_ID, folds={}, recoveries=[]))
    log(out, f'prepared and sealed design {sha256_file(out/"design.json")} commit={head}')
    return design


def verify(out):
    """Sealed design + frozen sources/environment/data; returns (X, y, ids, folds, design)."""
    out = Path(out)
    seal = json.loads((out/'design_seal.json').read_text(encoding='utf-8'))
    require(sha256_file(out/'design.json') == seal['design_sha256'], 'Design seal changed')
    design = json.loads((out/'design.json').read_text(encoding='utf-8'))
    c = config()
    require(design['config'] == c, 'Frozen config changed')
    require(design['source_sha256'] == sources(), 'Frozen source changed')
    require(design['environment'] == environment(), 'Frozen environment changed')
    X, y, ids, folds, contract = load(guard_path(design['data_dir']))
    require(contract == design['data_contract'], 'Data contract changed')
    return X, y, ids, folds, design


# ----------------------------------------------------------------------------- analysis (pure functions)
def summarize_oof(oof, support3):
    y, p, fo = oof.y_true.to_numpy(), oof.y_pred.to_numpy(), oof.fold.to_numpy()
    fold_scores = [score(y[fo == f], p[fo == f]) for f in range(5)]
    vals = [s['macro_f1'] for s in fold_scores]
    pooled = score(y, p)
    cm = pooled['confusion']
    return dict(cv_mean=float(np.mean(vals)), cv_std=float(np.std(vals, ddof=1)), fold_macro_f1=vals,
                fold_error31=[s['error31'] for s in fold_scores], fold_recall3=[s['recall'][2] for s in fold_scores],
                pooled=pooled, error31=pooled['error31'], error32=pooled['error32'], error23=int(cm[1][2]),
                rate31=pooled['error31'] / support3, rate32=pooled['error32'] / support3,
                recall1=pooled['recall'][0], recall3=pooled['recall'][2],
                predicted_share=[float(np.mean(p == v)) for v in (1, 2, 3)])


def classify(delta_f1, wins, delta_rate31, delta_recall3, th):
    """Pre-registered descriptive classes A, B, C1, C2, C3, D (eps absorbs float noise on thresholds)."""
    e = th['eps']
    gain = delta_f1 >= th['T_F1'] - e and wins >= th['T_win']
    if gain:
        return 'A' if delta_rate31 <= th['T_31'] + e else 'B'
    if delta_recall3 >= th['T_R3'] - e:
        if delta_f1 >= th['C_keep'] - e:
            return 'C1'
        if delta_f1 >= th['C_small'] - e:
            return 'C2'
        return 'C3'
    return 'D'


def compare(cell, ori, th):
    """Cell minus ORI on the same folds; always carries the 3->1 columns."""
    deltas = [a - b for a, b in zip(cell['fold_macro_f1'], ori['fold_macro_f1'])]
    row = dict(mean_macro_f1=cell['cv_mean'], sd_macro_f1=cell['cv_std'], delta_macro_f1=cell['cv_mean'] - ori['cv_mean'],
               fold_deltas=deltas, fold_wins=int(sum(d > 0 for d in deltas)),
               pooled_macro_f1=cell['pooled']['macro_f1'],
               delta_pooled_macro_f1=cell['pooled']['macro_f1'] - ori['pooled']['macro_f1'],
               accuracy=cell['pooled']['accuracy'],
               precision=cell['pooled']['precision'], recall=cell['pooled']['recall'], f1=cell['pooled']['f1'],
               support=cell['pooled']['support'], confusion=cell['pooled']['confusion'],
               recall1=cell['recall1'], recall3=cell['recall3'],
               delta_recall1=cell['recall1'] - ori['recall1'], delta_recall3=cell['recall3'] - ori['recall3'],
               error31=cell['error31'], rate31=cell['rate31'], error32=cell['error32'],
               delta_error31=cell['error31'] - ori['error31'], delta_rate31=cell['rate31'] - ori['rate31'],
               delta_error32=cell['error32'] - ori['error32'], predicted_share=cell['predicted_share'])
    row['class'] = classify(row['delta_macro_f1'], row['fold_wins'], row['delta_rate31'], row['delta_recall3'], th)
    return row


def difference_row(a, b, th=None):
    """a - b between two cells (not against ORI): carries Delta 3->1 as the error difference."""
    return dict(delta_macro_f1=a['cv_mean'] - b['cv_mean'], delta_error31=a['error31'] - b['error31'],
                delta_rate31=a['rate31'] - b['rate31'], delta_recall3=a['recall3'] - b['recall3'],
                delta_recall1=a['recall1'] - b['recall1'], delta_error32=a['error32'] - b['error32'])


def make_table(name, rows, columns):
    """Refuses any table showing Delta Macro F1 without Delta 3->1 count and rate beside it."""
    if any('delta_macro_f1' in r for r in rows) or 'delta_macro_f1' in columns:
        require({'delta_error31', 'delta_rate31'} <= set(columns), f'Table {name} must carry Delta 3->1 count and rate')
        for r in rows:
            if 'delta_macro_f1' in r:
                require('delta_error31' in r and 'delta_rate31' in r, f'Table {name}: row without Delta 3->1')
    return dict(name=name, columns=list(columns), rows=[{k: r.get(k) for k in columns} for r in rows])


def sign_changes(values):
    """Sign changes among successive differences of a curve [phi=0, ...]."""
    diffs = [b - a for a, b in zip(values[:-1], values[1:])]
    signs = [np.sign(d) for d in diffs if d != 0]
    return int(sum(1 for a, b in zip(signs[:-1], signs[1:]) if a != b))


SERIES = [('LR', 'RUP', ['P25', 'P50', 'P75', 'EQ']), ('LR', 'RDN', ['P25', 'P50', 'P75', 'EQ']),
          ('LR', 'WOBJ', ['P25', 'P50', 'P75', 'EQ']), ('LGBMN', 'WNORM', ['P25', 'P50', 'P75', 'EQ']),
          ('LGBMN', 'RUP', ['P50', 'EQ'])]
CELL_COLUMNS = ['cell', 'model', 'arm', 'scenario', 'source', 'mean_macro_f1', 'sd_macro_f1', 'delta_macro_f1', 'fold_wins',
                'delta_error31', 'delta_rate31', 'delta_recall3', 'delta_recall1', 'delta_error32', 'pooled_macro_f1',
                'accuracy', 'recall1', 'recall3', 'error31', 'rate31', 'error32', 'class']


def a3_emulation(oof_ori, oof_cells, target_table, th):
    """Zero-training post hoc: LR.ORI log-probabilities + log(t_c/n_c) offset, argmax again (descriptive)."""
    rows = []
    logp = np.log(np.clip(oof_ori[PROBS].to_numpy(), 1e-300, 1.0))
    folds = oof_ori.fold.to_numpy()
    for scen in ('P25', 'P50', 'P75', 'EQ'):
        pred = np.zeros(len(oof_ori), dtype=int)
        for f in range(5):
            tt = target_table[str(f)]['UP'][scen]
            off = np.log(np.asarray(tt['target'], dtype=float) / np.asarray(tt['before'], dtype=float))
            m = folds == f
            pred[m] = (logp[m] + off).argmax(1) + 1
        em = oof_ori.assign(y_pred=pred)
        row = dict(scenario=scen, emulated=summarize_oof(em, int((oof_ori.y_true == GRADE3).sum())))
        for name in (f'LR.WOBJ.{scen}', f'LR.RUP.{scen}'):
            if name in oof_cells:
                row[f'agreement_with_{name}'] = float(np.mean(pred == oof_cells[name].y_pred.to_numpy()))
        rows.append(row)
    return rows


# ----------------------------------------------------------------------------- finalize
def assemble(out, item, y, ids, folds):
    out = Path(out)
    ledger = read_ledger(out)
    entries = [ledger['folds'].get(key(item['id'], f)) for f in range(5)]
    require(all(e and e['status'] == 'complete' for e in entries), f'{item["id"]} incomplete')
    d = out/item_dir(item)
    if (d/'metrics.json').exists():
        meta = json.loads((d/'metrics.json').read_text(encoding='utf-8'))
        return read_pred(d/'oof.csv.gz'), meta
    for e in entries:
        check_saved(out, e)
    parts = pd.concat([read_pred(d/f'fold_{f}_pred.csv.gz') for f in range(5)])
    oof = parts.set_index('building_id').loc[ids.to_numpy()].reset_index()
    require(len(oof) == len(ids) and oof.building_id.is_unique, 'OOF coverage changed')
    np.testing.assert_array_equal(oof.fold.to_numpy(), folds)
    np.testing.assert_array_equal(oof.y_true.to_numpy(), y.to_numpy())
    require(np.array_equal(oof[PROBS].to_numpy().argmax(1) + 1, oof.y_pred.to_numpy()), 'argmax != y_pred')
    write_pred(d/'oof.csv.gz', oof)
    back = read_pred(d/'oof.csv.gz')
    np.testing.assert_array_equal(back[PROBS].to_numpy(), oof[PROBS].to_numpy())
    meta = dict(study_id=STUDY_ID, id=item['id'], oof_sha256=sha256_file(d/'oof.csv.gz'),
                fit_seconds=float(sum(e['fit_seconds'] for e in entries)), finished_at_utc=now())
    jwrite(d/'metrics.json', meta)
    bd = out/'backup'/item_dir(item)
    for name in ('oof.csv.gz', 'metrics.json'):
        require(not (bd/name).exists(), 'Backup already exists; refusing overwrite')
        shutil.copy2(d/name, bd/name)
    return back, meta


def build_analysis(c, design, y, ids, folds, oofs, sources_of, records, ledger_info):
    """oofs: name -> OOF frame aligned to ids. Pure aggregation (no I/O)."""
    th = c['thresholds']
    support3 = c['grade3_support']
    summ = {n: summarize_oof(o, support3) for n, o in oofs.items()}
    result = {'cells': {}}
    ori_for = {'LR': 'LR.ORI', 'LGBMN': 'LGBMN.ORI'}
    require(all(v in summ for v in ori_for.values()), 'ORI cells missing')
    rows, by_name = [], {}
    for name in summ:
        parts = name.split('.')
        model = parts[0]
        if model not in ori_for or name in ori_for.values() or name == 'LR_C':
            continue
        arm, scen = (parts[1], parts[2]) if len(parts) >= 3 else (parts[1], '')
        row = compare(summ[name], summ[ori_for[model]], th)
        row.update(cell=name, model=model, arm=arm, scenario=scen, source=sources_of.get(name, 'study fit'))
        rows.append(row)
        by_name[name] = row
    tables = {}
    tables['cells_vs_ori'] = make_table('cells_vs_ori', rows, CELL_COLUMNS)
    # dose response
    dr = []
    for model, arm, scens in SERIES:
        names = [f'{model}.{arm}.{s}' for s in scens]
        have = [n for n in names if n in summ]
        if not have:
            continue
        seq = [summ[ori_for[model]]] + [summ[n] for n in have]
        f1 = [s['cv_mean'] for s in seq]
        e31 = [s['error31'] for s in seq]
        r3 = [s['recall3'] for s in seq]
        r1 = [s['recall1'] for s in seq]
        for n in have:
            dr.append(dict(series=f'{model}.{arm}', cell=n, scenario=n.split('.')[2], delta_macro_f1=by_name[n]['delta_macro_f1'],
                           delta_error31=by_name[n]['delta_error31'], delta_rate31=by_name[n]['delta_rate31'],
                           delta_recall3=by_name[n]['delta_recall3'], delta_recall1=by_name[n]['delta_recall1']))
        dr.append(dict(series=f'{model}.{arm}', cell='(sign changes of successive differences from ORI)', scenario='-',
                       delta_macro_f1=None, delta_error31=None, delta_rate31=None, delta_recall3=None, delta_recall1=None,
                       sign_changes=dict(macro_f1=sign_changes(f1), error31=sign_changes(e31), recall3=sign_changes(r3),
                                         recall1=sign_changes(r1))))
    tables['dose_response'] = make_table('dose_response', [r for r in dr if r['delta_macro_f1'] is not None],
                                         ['series', 'cell', 'scenario', 'delta_macro_f1', 'delta_error31', 'delta_rate31',
                                          'delta_recall3', 'delta_recall1'])
    result['sign_changes'] = {r['series']: r['sign_changes'] for r in dr if r.get('sign_changes')}
    # decomposition Q3 (LR, UP)
    dec = [by_name[n] for n in ('LR.RUP.ONE1', 'LR.RUP.ONE3', 'LR.RUP.EQ') if n in by_name]
    if len(dec) == 3:
        one1, one3, eq = dec
        dec.append(dict(cell='EQ minus (ONE1 + ONE3)',
                        **{k: eq[k] - one1[k] - one3[k] for k in ('delta_macro_f1', 'delta_error31', 'delta_rate31',
                                                                    'delta_recall3', 'delta_recall1')}))
    tables['decomposition'] = make_table('decomposition', dec, ['cell', 'delta_macro_f1', 'delta_error31', 'delta_rate31',
                                                                 'delta_recall3', 'delta_recall1', 'fold_wins', 'class'])
    # noise (sampler seed spread of ΔF1 at P50)
    noise = {}
    for arm in ('RUP', 'RDN'):
        names = [f'LR.{arm}.P50'] + [f'LR.{arm}.P50.s{sd}' for sd in (101, 202)]
        if all(n in summ for n in names):
            vals = [summ[n]['cv_mean'] for n in names]
            noise[arm] = dict(cells=names, mean_macro_f1=vals, spread=float(max(vals) - min(vals)),
                              error31=[summ[n]['error31'] for n in names],
                              delta_macro_f1=[by_name[n]['delta_macro_f1'] for n in names],
                              delta_error31=[by_name[n]['delta_error31'] for n in names],
                              delta_rate31=[by_name[n]['delta_rate31'] for n in names])
    result['noise'] = noise
    # sampling vs weights (Q4)
    q4 = []
    for model, wname, scens in (('LR', 'WOBJ', ['P25', 'P50', 'P75', 'EQ']), ('LGBMN', 'WNORM', ['P50', 'EQ'])):
        for s in scens:
            a, b = f'{model}.RUP.{s}', f'{model}.{wname}.{s}'
            if a in summ and b in summ:
                r = difference_row(summ[a], summ[b])
                r.update(comparison=f'{a} minus {b}')
                q4.append(r)
    for a, b, label in (('LR.BAL', 'LR.WOBJ.EQ', 'LR.BAL minus LR.WOBJ.EQ (regularisation scale effect, uncontrolled path)'),
                        ('LR.RUP.EQ', 'LR.WOBJ.EQ', 'LR.ROS.EQ minus LR.WOBJ.EQ (replication randomness effect)')):
        if a in summ and b in summ:
            r = difference_row(summ[a], summ[b])
            r.update(comparison=label)
            q4.append(r)
    spread = noise.get('RUP', {}).get('spread')
    for r in q4:
        r['within_sampler_noise'] = (None if spread is None else bool(abs(r['delta_macro_f1']) < spread))
    tables['sampling_vs_weight'] = make_table('sampling_vs_weight', q4, ['comparison', 'delta_macro_f1', 'delta_error31', 'delta_rate31',
                                                                           'delta_recall3', 'delta_recall1', 'delta_error32', 'within_sampler_noise'])
    result['noise_note'] = 'A difference smaller than the three-seed spread of LR R-UP P50 is not distinguishable from sampler noise.'
    # reference columns (local only)
    refrows = []
    for tgt in ('LR_C', 'LGBMN.ORI'):
        if tgt not in summ:
            continue
        for name in summ:
            if name == tgt or name in ('LR.ORI', 'LR_C'):
                continue
            row = compare(summ[name], summ[tgt], th)
            row.update(cell=name, reference=tgt,
                       note='different preprocessing/model: not a single-factor comparison')
            refrows.append(row)
    tables['reference_columns'] = make_table('reference_columns', refrows, ['cell', 'reference', 'mean_macro_f1', 'delta_macro_f1', 'fold_wins',
                                                                              'delta_error31', 'delta_rate31', 'delta_recall3', 'note'])
    # sampling targets and realised counts
    tt = design['target_table']
    target_rows = [dict(fold=int(f), direction=d, scenario=s, **v) for f, dd in tt.items() for d, ss in dd.items() for s, v in ss.items()]
    result['target_counts_table'] = target_rows
    realised = []
    for uid, folds_rec in records.items():
        for f, r in sorted(folds_rec.items()):
            s = r['sampling']
            realised.append(dict(id=uid, fold=int(f), kind=r['kind'], arm=s['arm'], scenario=s['scenario'], before=s['before'],
                                 target=s['target'], after=s['after'], delta=s['delta'], multiplier=s['multiplier'],
                                 fit_rows=s['fit_rows'], after_equals_target=s['after_equals_target'],
                                 effective_mass=s['effective_mass'], class_weight=s['class_weight'],
                                 indices_sha256=s['indices_sha256'], valid_class_counts=r['valid_class_counts']))
    result['realised_counts_table'] = realised
    overview = []
    for d, ss in tt['0'].items():
        for s, v in ss.items():
            overview.append(dict(direction=d, scenario=s, before=v['before'], target=v['target'], delta=v['delta'],
                                 multiplier=v['multiplier'], share_target=v['share_target'],
                                 target_range_over_folds=[[min(tt[str(f)][d][s]['target'][i] for f in range(5)),
                                                           max(tt[str(f)][d][s]['target'][i] for f in range(5))] for i in range(3)]))
    result['scenario_overview_fold0'] = overview
    # A3 (post hoc, zero training)
    a3 = []
    for r in a3_emulation(oofs['LR.ORI'], oofs, tt, th):
        cmp = compare(r['emulated'], summ['LR.ORI'], th)
        row = dict(scenario=r['scenario'], cv_mean=r['emulated']['cv_mean'], error31=r['emulated']['error31'],
                   rate31=r['emulated']['rate31'], recall3=r['emulated']['recall3'],
                   **{k: cmp[k] for k in ('delta_macro_f1', 'fold_wins', 'delta_error31', 'delta_rate31', 'delta_recall3')})
        row.update({k: v for k, v in r.items() if k.startswith('agreement_with_')})
        a3.append(row)
    tables['a3_offset_emulation'] = make_table('a3_offset_emulation', a3, [
        'scenario', 'cv_mean', 'delta_macro_f1', 'fold_wins', 'delta_error31', 'delta_rate31', 'delta_recall3', 'error31', 'rate31',
        'recall3'] + sorted({k for r in a3 for k in r if k.startswith('agreement_with_')}))
    result['a3_note'] = 'Post hoc descriptive statistic on stored LR.ORI probabilities; no training, not a decision-rule study.'
    # per-cell verdicts
    result['verdicts'] = {n: dict(klass=r['class'], delta_macro_f1=r['delta_macro_f1'], fold_wins=r['fold_wins'],
                                  delta_error31=r['delta_error31'], delta_rate31=r['delta_rate31'],
                                  delta_recall3=r['delta_recall3']) for n, r in by_name.items()}
    result['thresholds'] = th
    result['summaries'] = summ
    result['tables'] = tables
    result['ledger'] = ledger_info
    return result


def finalize_all(out):
    out = Path(out)
    X, y, ids, folds, design = verify(out)
    c = design['config']
    require(not (out/'analysis.json').exists(), 'analysis.json exists; refusing overwrite')
    ledger = read_ledger(out)
    preflight_ledger(ledger)
    require(all(stage_complete(ledger, design, s) for s in STAGES), 'Ledger incomplete; not finalizing')
    controls = {k: dict(status=e['control_status'], max_abs_diff=e['control_max_abs_diff'],
                        labels_identical=e['control_labels_identical']) for k, e in ledger['folds'].items() if e['kind'] == 'control'}
    require(len(controls) == c['max_control_fits'] and all(v['status'] == 'PASS' for v in controls.values()),
            'All eight controls must PASS before the reused cells are used as references')
    oofs, srcs, records = {}, {}, {}
    for it in design['items']:
        if it['kind'] == 'control':
            continue
        oof, meta = assemble(out, it, y, ids, folds)
        oofs[it['id']] = oof
        records[it['id']] = {f: json.loads((out/item_dir(it)/f'fold_{f}.json').read_text(encoding='utf-8')) for f in range(5)}
        srcs[it['id']] = 'study fit'
    for it in design['items']:
        if it['kind'] == 'control':
            records[it['id']] = {f: json.loads((out/item_dir(it)/f'fold_{f}.json').read_text(encoding='utf-8')) for f in it['folds']}
    for r in c['references']:
        frame = reference_frame(design, r['name']).loc[ids.to_numpy()].reset_index()
        require(np.array_equal(frame.fold.to_numpy(), folds), 'Reference folds differ')
        oofs[r['name']] = frame
        srcs[r['name']] = f'reused local reference ({r["role"]})'
    attempts = sum(e['attempts'] for e in ledger['folds'].values() if e['kind'] == 'study')
    info = dict(study_fits_complete=sum(1 for e in ledger['folds'].values() if e['kind'] == 'study' and e['status'] == 'complete'),
                study_attempts=attempts, control_fits_complete=len(controls),
                control_attempts=sum(e['attempts'] for e in ledger['folds'].values() if e['kind'] == 'control'),
                manual_recoveries=ledger['recoveries'], automatic_retries=0,
                fit_seconds_sum=float(sum(e['fit_seconds'] for e in ledger['folds'].values())),
                controls=controls)
    res = build_analysis(c, design, y, ids, folds, oofs, srcs, records, info)
    res.update(study_id=STUDY_ID, design_sha256=sha256_file(out/'design.json'), code_commit=design['code_commit'],
               plan_sha256=design['plan_sha256'], gidx_all_pass=design['gidx']['all_pass'], holdout_accessed=False,
               rf_extension='excluded (approved default)', not_run=['DT', 'GBC', 'T14 RF', 'RF W-NORM', 'SMOTE partial relaxation'],
               note='Descriptive single-split comparison on the shared development Stratified 5-fold; local reproduction values only; '
                    'no significance or causal claims; not an independent performance.',
               references_used={n: {k: v for k, v in r.items() if k != 'files'} for n, r in design['references'].items()},
               finished_at_utc=now())
    jwrite(out/'analysis.json', res)
    public = {k: v for k, v in res.items() if k not in ('references_used',)}
    jwrite(out/'aggregate.json', public)
    for name in ('analysis.json', 'aggregate.json'):
        require(not (out/'backup'/name).exists(), 'Backup already exists; refusing overwrite')
        shutil.copy2(out/name, out/'backup'/name)
    log(out, 'analysis written: ' + json.dumps(py({n: v['klass'] for n, v in res['verdicts'].items()})))
    return res


def status(out):
    ledger = read_ledger(out)
    design = json.loads((Path(out)/'design.json').read_text(encoding='utf-8'))
    per_stage = {}
    for s in STAGES:
        its = [i for i in design['items'] if i['stage'] == s]
        total = sum(len(i['folds']) for i in its)
        done = sum(1 for i in its for f in i['folds'] if ledger['folds'].get(key(i['id'], f), {}).get('status') == 'complete')
        per_stage[str(s)] = dict(complete=done, total=total)
    print(json.dumps(py(dict(
        stages=per_stage, study_attempts=sum(e['attempts'] for e in ledger['folds'].values() if e['kind'] == 'study'),
        control_attempts=sum(e['attempts'] for e in ledger['folds'].values() if e['kind'] == 'control'),
        running=[k for k, e in ledger['folds'].items() if e['status'] == 'running'],
        failed=[k for k, e in ledger['folds'].items() if e['status'] == 'failed'],
        control_fail=[k for k, e in ledger['folds'].items() if e.get('control_status') == 'FAIL'],
        recoveries=ledger['recoveries'])), indent=1))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('action', choices=['gidx', 'prepare', 'fit-all', 'status', 'finalize'])
    p.add_argument('--data')
    p.add_argument('--out', required=True)
    p.add_argument('--local-root')
    p.add_argument('--code-commit')
    p.add_argument('--stage', choices=['1', '2', '3', 'all'], help='required for fit-all (no default)')
    a = p.parse_args(argv)
    if a.action == 'gidx':
        require(a.data, '--data required')
        gidx_command(a.data, a.out)
    elif a.action == 'prepare':
        require(a.data, '--data required')
        prepare(a.data, a.out, a.local_root, a.code_commit)
    elif a.action == 'fit-all':
        require(a.stage, 'fit-all requires an explicit --stage 1|2|3|all')
        fit_all(a.out, a.stage)
    elif a.action == 'finalize':
        finalize_all(a.out)
    else:
        status(a.out)


if __name__ == '__main__':
    sys.exit(main())
