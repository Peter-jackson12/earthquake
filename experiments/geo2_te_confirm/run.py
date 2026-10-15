"""T13: confirmation of the T11 LGBM_TE_G geo2 finding on NEW independent geo2 Group splits (no new factor, no tuning).

Splits: StratifiedGroupKFold(5, shuffle=True, random_state=seed) on (geo1, geo2) composite groups, seeds 101/202/303
(same procedure as the T5 geo2 manifest, which used seed 42). Arms (all refit on every new split, configs identical to
their audited originals): LR_C, LGBM_O (T8/T4 one-hot LightGBM), LGBM_TE_G (T11 group-aware inner-fold TE, group_level
geo2, m = 20). 3 arms x 3 seeds x 5 folds = 45 fits. Holdout is never read. Private outputs only.
All model code is imported unchanged from the audited modules; custom classes live in importable modules.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
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
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from preprocess import GEO, ID, sha256_file  # noqa: E402
from experiments.age995.run import PARAMS as LR_PARAMS, make_pipeline as make_C  # noqa: E402
from experiments.geo_group_cv import run as t5  # noqa: E402
from experiments.geo_group_cv.run import PROBS, groups, load, require, score, write  # noqa: E402
from experiments.geo_hier_te import run as t10  # noqa: E402
from experiments.lgbm_native_cat import run as t8  # noqa: E402
from experiments.geo_te_group_inner import run as t11  # noqa: E402

HERE = Path(__file__).resolve().parent
STUDY_ID = 'T13_geo2_te_confirm_v1'
SEEDS = [101, 202, 303]
CONTROL_SEED = 42
ARMS = ['LR_C', 'LGBM_O', 'LGBM_TE_G']
GROUP_COLUMNS = list(GEO[:2])
N_TARGETS = 45
SOURCE = list(dict.fromkeys(t11.SOURCE + [
    'experiments/lgbm_native_cat/__init__.py', 'experiments/lgbm_native_cat/run.py',
    'experiments/lgbm_native_cat/config.json', 'experiments/geo_group_cv/config.json',
    'experiments/geo2_te_confirm/__init__.py', 'experiments/geo2_te_confirm/run.py',
    'experiments/geo2_te_confirm/config.json', 'tests/test_geo2_te_confirm.py',
    'docs/T13_GEO2_TE_CONFIRM_PLAN.md']))
now, environment, log, event = t10.now, t10.environment, t10.log, t10.event
atomic_write, read_pred, write_pred = t10.atomic_write, t10.read_pred, t10.write_pred
read_ledger, check_saved, reference, fold_means = t10.read_ledger, t10.check_saved, t10.reference, t10.fold_means


def scheme_of(seed):
    return f's{seed}'


def key(seed, arm, fold):
    return f'{scheme_of(seed)}/{arm}/{fold}'


def config(path=HERE/'config.json'):
    c = json.loads(Path(path).read_text(encoding='utf-8'))
    t = t11.config()
    require(c['study_id'] == STUDY_ID and c['arms'] == ARMS and c['n_folds'] == 5, 'Unplanned study/arms')
    sp = c['split']
    require(sp['seeds'] == SEEDS and sp['control_seed'] == CONTROL_SEED and sp['n_splits'] == 5 and sp['shuffle']
            and sp['splitter'] == 'StratifiedGroupKFold' and c['group_columns'] == GROUP_COLUMNS, 'Split policy changed')
    require(c['lgbm_params'] == t['lgbm_params'] == t8.config()['lgbm_params'], 'LightGBM params differ from T8/T11')
    require(c['lr_params'] == LR_PARAMS == t['lr_params'], 'LR params differ from LR-C')
    require(c['geo_columns'] == list(GEO) and c['text_categorical_columns'] == t['text_categorical_columns'],
            'Categorical columns changed')
    te = c['te']
    require(te['smoothing_m'] == t['te']['smoothing_m'] == 20.0 and te['group_level'] == 'geo2'
            and te['inner_splits'] == 5 and te['inner_seed'] == 42
            and t['te']['inner_oof']['random_state'] == 42, 'TE design differs from T11')
    require(c['max_new_study_fits'] == N_TARGETS == len(SEEDS) * len(ARMS) * 5, 'Unplanned fit plan')
    require(c['automatic_retries'] == 0 and not c['early_stopping'] and c['class_weight'] is None
            and c['sampling'] is None and not c['tuning'] and not c['threshold_or_decision_rule_change']
            and c['outer_parallel_jobs'] == 1 and c['lr_threadpool_limit'] == 1 and c['lgbm_threadpool_limit'] == 2
            and c['stop_on_lr_nonconvergence'] and not c['holdout_access'], 'Unplanned run policy')
    require(c['environment'] == t['environment'] and c['checkpoint'] == t['checkpoint'], 'Frozen references differ')
    return c


def sources():
    return {p: sha256_file(ROOT/p) for p in SOURCE}


def targets():
    return [(s, a, f) for s in SEEDS for a in ARMS for f in range(5)]


# ---------------------------------------------------------------- split generation
def make_folds(X, y, seed, columns=GROUP_COLUMNS):
    """T5 procedure: StratifiedGroupKFold(5, shuffle=True, random_state=seed) over composite group codes."""
    g, _ = groups(X, columns)
    folds = np.full(len(X), -1, dtype=int)
    coverage = np.zeros(len(X), dtype=int)
    for f, (_, b) in enumerate(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=int(seed)).split(X, y, g)):
        folds[b] = f
        coverage[b] += 1
    require(np.all(coverage == 1), 'OOF coverage must equal one')
    return folds


def manifest_bytes(ids, folds, X, columns=GROUP_COLUMNS):
    frame = pd.DataFrame({ID: ids, 'fold': folds, **{col: X[col] for col in columns}})
    return frame.to_csv(index=False, lineterminator='\n').encode('utf-8')


def write_manifest(path, ids, folds, X, columns=GROUP_COLUMNS):
    data = manifest_bytes(ids, folds, X, columns)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as fh:
        fh.write(data)
    return hashlib.sha256(data).hexdigest()


def reproduces_reference(X, y, ids, reference_path, seed=CONTROL_SEED, columns=GROUP_COLUMNS):
    """True iff the generator output for `seed` is byte-identical to an existing manifest file."""
    return manifest_bytes(ids, make_folds(X, y, seed, columns), X, columns) == Path(reference_path).read_bytes()


def split_gate(X, y, ids, folds, split_config_path=None):
    """T5 gates: group/ID disjointness, all 3 classes >= 30 rows in every train and validation fold, size window."""
    return t5.validate_assignment(X, y, ids, folds, GROUP_COLUMNS, t5.cfg())


def fold_similarity(folds_a, folds_b):
    return float(adjusted_rand_score(folds_a, folds_b))


# ---------------------------------------------------------------- one fit (shared by study and control)
def fit_one(arm, X_train, y_train, X_valid, c):
    """Return dict(p, converged, fit_seconds, pre, model, pipe, mt, mv, p_train, names, info) for a fold."""
    require(arm in ARMS, 'Unknown arm')
    if arm == 'LR_C':
        pipe = make_C('C')
        with threadpool_limits(limits=c['lr_threadpool_limit']), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            t0 = time.perf_counter()
            pipe.fit(X_train, y_train)
            fit_seconds = time.perf_counter() - t0
            p = np.asarray(pipe.predict_proba(X_valid), dtype=np.float64)
        model = pipe['model']
        require(np.array_equal(model.classes_, [1, 2, 3]), 'Unexpected LR class order')
        require(model.get_params()['C'] == 1.0 and model.get_params()['solver'] == 'lbfgs', 'LR params changed')
        require(p.shape == (len(X_valid), 3) and np.isfinite(p).all(), 'Invalid probabilities')
        info = dict(n_iter=model.n_iter_.tolist(), warnings=[dict(category=w.category.__name__, message=str(w.message))
                                                             for w in caught],
                    age995_median=pipe['preprocess']['age995'].median_)
        converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
        return dict(p=p, converged=converged, fit_seconds=fit_seconds, pipe=pipe, pre=pipe['preprocess'], model=model,
                    info=info)
    if arm == 'LGBM_O':
        pre, model, p, p_train, mt, mv, info = t8.fit_arm('O', X_train, y_train, X_valid, c['lgbm_params'],
                                                          c['lgbm_threadpool_limit'])
        return dict(p=p, converged=True, fit_seconds=info['fit_seconds'], pre=pre, model=model, p_train=p_train,
                    mt=mt, mv=mv, info=info)
    pipe, pre, model, p, p_train, mt, mv, names, info = t11.fit_arm('LGBM_TE_G', c['te']['group_level'], X_train,
                                                                     y_train, X_valid, c)
    return dict(p=p, converged=True, fit_seconds=info['fit_seconds'], pre=pre, model=model, p_train=p_train, mt=mt,
                mv=mv, names=names, info=info)


# ---------------------------------------------------------------- control check
def control_check(data, destination):
    c = config()
    dest = Path(destination)
    require(not dest.exists(), 'Do not overwrite an existing control check')
    require(environment() == c['environment'], 'Environment differs from the pinned environment')
    X, y, ids, old, contract = load(data)
    cc = c['control_check']
    refs = cc['references']
    dest.mkdir(parents=True)
    res = dict(study_id=STUDY_ID, kind=cc['kind'], counted_as_study_fit=False, created_at_utc=now(), fold=cc['fold'])
    # (1) generator reproduces T5 geo2 manifest byte-identically
    folds42 = make_folds(X, y, CONTROL_SEED)
    data_bytes = manifest_bytes(ids, folds42, X)
    sha = hashlib.sha256(data_bytes).hexdigest()
    t5_bytes, t11_bytes = Path(refs['manifest_t5']).read_bytes(), Path(refs['manifest_t11']).read_bytes()
    one = dict(generated_sha256=sha, expected_sha256=c['control_manifest_sha256_seed42'],
               equals_T5_file_bytes=data_bytes == t5_bytes, equals_T11_copy_bytes=data_bytes == t11_bytes)
    one['pass_'] = bool(sha == c['control_manifest_sha256_seed42'] and one['equals_T5_file_bytes']
                        and one['equals_T11_copy_bytes'])
    res['1_manifest_seed42_reproduction'] = one
    res['holdout_accessed'] = False
    res['environment'] = environment()
    f0 = cc['fold']
    a, b = np.flatnonzero(folds42 != f0), np.flatnonzero(folds42 == f0)
    # (2) LGBM_O and LGBM_TE_G fold 0 on the seed-42 split vs audited T8 / T11 OOF probabilities
    two = {}
    for arm, ref_key in [('LGBM_O', 'lgbm_o_oof'), ('LGBM_TE_G', 'lgbm_te_g_oof')]:
        r = reference(refs[ref_key]['path'], refs[ref_key]['sha256'], ids, y, folds42)
        rf = r[r.fold == f0]
        np.testing.assert_array_equal(rf.building_id.to_numpy(), ids.iloc[b].to_numpy())
        out = fit_one(arm, X.iloc[a], y.iloc[a], X.iloc[b], c)
        diff = float(np.max(np.abs(out['p'] - rf[PROBS].to_numpy())))
        agree = float(np.mean(out['p'].argmax(1) + 1 == rf.y_pred.to_numpy()))
        two[arm] = dict(max_abs_prob_diff=diff, label_agreement=agree, fit_seconds=out['fit_seconds'],
                        pass_=bool(diff <= cc['prob_atol'] and agree == 1.0))
        log_line = f'control {arm} fold0 max_diff={diff:.3g} agree={agree}'
        print(log_line, flush=True)
    two['pass_'] = bool(all(v['pass_'] for v in two.values()))
    res['2_lgbm_fold0_probabilities_match'] = two
    # (3) LR_C descriptive only
    r = reference(refs['lr_c_oof']['path'], refs['lr_c_oof']['sha256'], ids, y, folds42)
    rf = r[r.fold == f0]
    out = fit_one('LR_C', X.iloc[a], y.iloc[a], X.iloc[b], c)
    pred = out['p'].argmax(1) + 1
    res['3_lr_c_fold0_descriptive'] = dict(
        gating=False, max_abs_prob_diff=float(np.max(np.abs(out['p'] - rf[PROBS].to_numpy()))),
        label_agreement=float(np.mean(pred == rf.y_pred.to_numpy())), converged=out['converged'],
        macro_f1_new=score(y.iloc[b], pred)['macro_f1'], macro_f1_reference=score(rf.y_true, rf.y_pred)['macro_f1'],
        fit_seconds=out['fit_seconds'], note='LR trained in another environment; BLAS-sensitive; not gating')
    res['classifiers_fitted_for_control'] = 3
    res['status'] = 'PASS' if one['pass_'] and two['pass_'] else 'FAIL'
    write(dest/'control_check.json', res)
    print(json.dumps({k: (v.get('pass_') if isinstance(v, dict) else v) for k, v in res.items()
                      if k.startswith(('1_', '2_', 'status'))}, indent=1))
    require(res['status'] == 'PASS', 'Control check failed (gating checks 1 and 2)')
    return res


# ---------------------------------------------------------------- prepare / verify
def prepare(data, output, code_commit, control):
    output = Path(output)
    require(not output.exists(), 'Do not overwrite an existing study')
    require(len(code_commit) == 40 and all(ch in '0123456789abcdef' for ch in code_commit), 'Full commit SHA required')
    c = config()
    require(environment() == c['environment'], 'Environment differs from the pinned environment')
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(['git', 'status', '--porcelain', '--', *SOURCE], cwd=ROOT, capture_output=True, text=True,
                           check=True).stdout.strip()
    require(head == code_commit and not dirty, 'Sealed sources must be committed at code_commit (HEAD) and clean')
    cc = json.loads(Path(control).read_text(encoding='utf-8'))
    require(cc['status'] == 'PASS' and cc['study_id'] == STUDY_ID, 'Control check must PASS before prepare')
    X, y, ids, old, contract = load(data)
    f42 = make_folds(X, y, CONTROL_SEED)
    output.mkdir(parents=True)
    (output/'backup').mkdir()
    split_info, shas, all_folds = {}, {}, {}
    for s in SEEDS:
        folds = make_folds(X, y, s)
        all_folds[s] = folds
        check = split_gate(X, y, ids, folds)  # raises (stop, no seed substitution) if a gate fails
        d = output/'manifests'/f'seed_{s}'
        shas[s] = write_manifest(d/'manifest.csv', ids, folds, X)
        split_info[s] = dict(manifest_sha256=shas[s], validation=check,
                             ari_vs_seed42=fold_similarity(f42, folds),
                             identical_fold_label_fraction_vs_seed42=float(np.mean(f42 == folds)))
    pairs = {f'{p}_vs_{q}': fold_similarity(all_folds[p], all_folds[q]) for p in SEEDS for q in SEEDS if p < q}
    design = dict(study_id=STUDY_ID, config=c, code_commit=code_commit, source_sha256=sources(),
                  data_contract=contract, environment=environment(), manifest_sha256={str(s): shas[s] for s in SEEDS},
                  splits={str(s): split_info[s] for s in SEEDS}, ari_between_new_seeds=pairs,
                  seed42_manifest_sha256=c['control_manifest_sha256_seed42'],
                  control_check=dict(path=str(Path(control).resolve()), sha256=sha256_file(control), status=cc['status']),
                  study_targets=[key(*t) for t in targets()], max_new_study_fits=c['max_new_study_fits'],
                  automatic_retries=0, baseline_refits=0, holdout_accessed=False,
                  created_at_utc=now(), classifiers_fitted=0)
    write(output/'design.json', design)
    write(output/'design_seal.json', dict(design_sha256=sha256_file(output/'design.json'), code_commit=code_commit,
                                          created_at_utc=now(), classifiers_fitted=0))
    atomic_write(output/'ledger.json', dict(study_id=STUDY_ID, folds={}))
    log(output, f'prepared and sealed design {sha256_file(output/"design.json")} commit={code_commit}')
    return design


def verify(data, output):
    output = Path(output)
    seal = json.loads((output/'design_seal.json').read_text(encoding='utf-8'))
    require(sha256_file(output/'design.json') == seal['design_sha256'], 'Design seal changed')
    design = json.loads((output/'design.json').read_text(encoding='utf-8'))
    require(design['config'] == config(), 'Frozen config changed')
    require(design['source_sha256'] == sources(), 'Frozen source changed')
    require(design['environment'] == environment(), 'Frozen environment changed')
    X, y, ids, old, contract = load(data)
    require(contract == design['data_contract'], 'Data contract changed')
    folds = {}
    for s in SEEDS:
        path = output/'manifests'/f'seed_{s}'/'manifest.csv'
        require(sha256_file(path) == design['manifest_sha256'][str(s)], f'Manifest changed: seed {s}')
        mf = pd.read_csv(path)
        np.testing.assert_array_equal(mf.building_id, ids)
        folds[s] = mf.fold.to_numpy()
    return X, y, ids, folds, design


# ---------------------------------------------------------------- one fold with checkpointing
def fit_fold(X, y, ids, folds, out, seed, arm, f, design):
    out = Path(out)
    k = key(seed, arm, f)
    require(k in design['study_targets'], f'{k} is not a pre-registered study fit')
    ledger = read_ledger(out)
    entry = ledger['folds'].get(k)
    if entry and entry['status'] == 'complete':
        check_saved(out, entry)
        log(out, f'{k} already complete; verified hashes, not refit')
        return entry, False
    require(not (entry and entry['status'] == 'complete_nonconverged'), f'{k} did not converge; study stopped')
    attempts = sum(e['attempts'] for e in ledger['folds'].values())
    require(attempts < design['max_new_study_fits'], 'Fit budget exhausted')
    prior = entry['attempts'] if entry else 0
    entry = dict(dir=f'{scheme_of(seed)}/{arm}', seed=seed, arm=arm, fold=f, status='running', attempts=prior+1,
                 started_at_utc=now())
    ledger['folds'][k] = entry
    atomic_write(out/'ledger.json', ledger)
    if prior:
        log(out, f'{k} previous attempt did not finish; manual resume attempt {prior+1}')
    d = out/entry['dir']
    d.mkdir(parents=True, exist_ok=True)
    for stale in d.glob(f'fold_{f}[._]*'):
        stale.rename(stale.with_name(stale.name + f'.incomplete_attempt{prior}'))
    event(out, fold=k, action='fit_start', attempt=prior+1)
    c = design['config']
    fo = folds[seed]
    a, b = np.flatnonzero(fo != f), np.flatnonzero(fo == f)
    require(len(a) + len(b) == len(ids) and len(b) > 0, 'Fold membership invalid')
    Xa, Xb = X.iloc[a], X.iloc[b]
    out_fit = fit_one(arm, Xa, y.iloc[a], Xb, c)
    p, pre, model, info = out_fit['p'], out_fit['pre'], out_fit['model'], out_fit['info']
    pred = p.argmax(1) + 1
    extra = {}
    if arm == 'LGBM_TE_G':
        mt, mv, names = out_fit['mt'], out_fit['mv'], out_fit['names']
        pre_c = make_C('C')['preprocess']
        with threadpool_limits(limits=c['lgbm_threadpool_limit']):
            pre_c.fit(Xa)
            eq_v, n_geo_ohe = t10.same_blocks_as_C(pre_c, pre_c.transform(Xb), mv, names)
            eq_t, _ = t10.same_blocks_as_C(pre_c, pre_c.transform(Xa), mt, names)
        require(eq_v and eq_t, 'Non-geo columns differ from the C preprocessor')
        enc_block = np.asarray(t10.te_encoder(pre).transform(Xb[GEO]))
        require(enc_block.tobytes() == t11.t10_validation_encoding(Xa, y.iloc[a], Xb, c['te']['smoothing_m']).tobytes(),
                'Validation encoding differs from T10')
        extra = dict(geo_onehot_columns_replaced=n_geo_ohe, non_geo_columns_equal_C=True,
                     valid_encoding_identical_to_t10=True, te=info['te'], te_gain_share=info['te_gain_share'],
                     te_gain=info['te_gain'], feature_names_sha256=info['feature_names_sha256'],
                     encoder_module=type(t10.te_encoder(pre)).__module__,
                     rounds_completed=info['rounds_completed'])
        require(extra['encoder_module'] == t11.ENCODER_MODULE, 'Encoder must live in the importable module')
    preds = d/f'fold_{f}_pred.csv.gz'
    if arm == 'LR_C':
        threads = c['lr_threadpool_limit']
        model_paths = [d/f'fold_{f}_model.joblib']
        joblib.dump(out_fit['pipe'], model_paths[0], compress=3)
    else:
        threads = c['lgbm_threadpool_limit']
        model_paths = [d/f'fold_{f}_model.txt', d/f'fold_{f}_preprocessor.joblib']
        model.booster_.save_model(str(model_paths[0]))
        joblib.dump(pre, model_paths[1], compress=3)
        extra.setdefault('rounds_completed', int(model.booster_.current_iteration()))
    frame = pd.DataFrame({'building_id': ids.iloc[b].to_numpy(), 'fold': f, 'y_true': y.iloc[b].to_numpy(),
                          'y_pred': pred, **{n: p[:, i] for i, n in enumerate(PROBS)}})
    write_pred(preds, frame)
    with threadpool_limits(limits=threads):  # reload verification (freshly generated trusted files only)
        if arm == 'LR_C':
            loaded = joblib.load(model_paths[0])
            p2 = np.asarray(loaded.predict_proba(Xb), dtype=np.float64)
            require(np.array_equal(p2, p) and np.array_equal(loaded.predict(Xb), pred), 'LR reload mismatch')
        else:
            import lightgbm
            booster = lightgbm.Booster(model_file=str(model_paths[0]))
            loaded_pre = joblib.load(model_paths[1])
            p2 = np.asarray(booster.predict(loaded_pre.transform(Xb), num_threads=c['lgbm_params']['n_jobs']),
                            dtype=np.float64)
    reload_diff = float(np.max(np.abs(p2 - p)))
    require(reload_diff <= c['checkpoint']['model_reload_probability_atol_lgbm']
            and np.array_equal(p2.argmax(1) + 1, pred), f'Model reload mismatch {reload_diff}')
    back = read_pred(preds)
    np.testing.assert_array_equal(back.building_id.to_numpy(), frame.building_id.to_numpy())
    np.testing.assert_array_equal(back.y_pred.to_numpy(), pred)
    np.testing.assert_array_equal(back[PROBS].to_numpy(), p)
    train_f1 = (score(y.iloc[a], out_fit['p_train'].argmax(1) + 1)['macro_f1'] if 'p_train' in out_fit else None)
    record = dict(study_id=STUDY_ID, seed=seed, arm=arm, fold=f, train_rows=len(a), valid_rows=len(b),
                  train_class_counts=[int((y.iloc[a] == v).sum()) for v in (1, 2, 3)],
                  metrics=score(y.iloc[b], pred), train_macro_f1=train_f1,
                  pred_share=(np.bincount(pred - 1, minlength=3)/len(pred)).tolist(),
                  fit_seconds=out_fit['fit_seconds'], converged=out_fit['converged'],
                  n_iter=info.get('n_iter'), age995_median=info.get('age995_median'),
                  encoded_columns=info.get('encoded_columns'), warnings=info['warnings'],
                  effective_parameters={k2: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                                        for k2, v in model.get_params().items()},
                  reload_probability_max_abs_diff=reload_diff, csv_round_trip_exact=True,
                  finished_at_utc=now(), **extra)
    write(d/f'fold_{f}.json', record)
    shas = {pth.name: sha256_file(pth) for pth in [*model_paths, preds, d/f'fold_{f}.json']}
    bd = out/'backup'/entry['dir']
    bd.mkdir(parents=True, exist_ok=True)
    for name in shas:
        require(not (bd/name).exists(), 'Backup already exists; refusing overwrite')
        shutil.copy2(d/name, bd/name)
    converged = out_fit['converged']
    entry.update(status='complete' if converged else 'complete_nonconverged', artifact_sha256=shas,
                 macro_f1=record['metrics']['macro_f1'], fit_seconds=out_fit['fit_seconds'], converged=converged,
                 finished_at_utc=now())
    check_saved(out, entry)
    ledger = read_ledger(out)
    ledger['folds'][k] = entry
    atomic_write(out/'ledger.json', ledger)
    event(out, fold=k, action='fit_complete', macro_f1=record['metrics']['macro_f1'], seconds=out_fit['fit_seconds'],
          converged=converged, artifact_sha256=shas)
    log(out, f'{k} F1={record["metrics"]["macro_f1"]:.9f} fit={out_fit["fit_seconds"]:.1f}s iter={info.get("n_iter")} '
             f'reload_diff={reload_diff:.3g} saved+verified+backed-up')
    require(converged, 'LR nonconvergence: stop without changing parameters or retrying')
    return entry, True


def fit_all(data, output, only=None):
    out = Path(output)
    lock = out/'RUNNING.lock'
    require(not lock.exists(), f'Another run holds {lock}; remove only if that process is dead')
    lock.write_text(f'pid={os.getpid()} started={now()}\n')
    try:
        X, y, ids, folds, design = verify(data, out)
        before = pd.util.hash_pandas_object(X, index=True).to_numpy()
        todo = [t for t in targets() if only is None or key(*t) in only]
        log(out, f'fit-all start: {len(todo)} targets (completed ones are hash-verified, not refit)')
        t0 = time.perf_counter()
        for s, a, f in todo:
            try:
                fit_fold(X, y, ids, folds, out, s, a, f, design)
            except Exception as exc:
                log(out, f'STOPPED {key(s, a, f)} {type(exc).__name__}: {exc}')
                event(out, fold=key(s, a, f), action='stopped', error=f'{type(exc).__name__}: {exc}',
                      trace=traceback.format_exc())
                raise
        np.testing.assert_array_equal(before, pd.util.hash_pandas_object(X, index=True).to_numpy())
        log(out, f'fit-all finished wall={time.perf_counter()-t0:.1f}s')
    finally:
        lock.unlink(missing_ok=True)


# ---------------------------------------------------------------- rules (pure functions)
def judge(rules, o, lr):
    """Pre-registered confirmation rules. `o` = LGBM_TE_G vs LGBM_O, `lr` = LGBM_TE_G vs LR_C (15-fold aggregates).

    o:  mean_delta, fold_wins, n_folds, pooled_grade3_recall_delta, repair = {seed: (accuracy_diff, agreement)}
    lr: mean_delta, fold_wins, n_folds, te_pooled_grade3_recall, lr_pooled_grade3_recall
    """
    r1, r2 = rules['C1'], rules['C2']
    repair = {str(s): dict(accuracy_diff=a, agreement=g,
                           holds=bool(abs(a) <= r1['repair_every_seed']['abs_accuracy_diff_max']
                                      and g >= r1['repair_every_seed']['prediction_agreement_min']))
              for s, (a, g) in o['repair'].items()}
    c1_parts = dict(mean_delta_ok=bool(o['mean_delta'] >= r1['mean_delta_min']),
                    fold_wins_ok=bool(o['fold_wins'] >= r1['fold_wins_min'] and o['n_folds'] == r1['n_folds']),
                    grade3_recall_ok=bool(o['pooled_grade3_recall_delta'] >= r1['pooled_grade3_recall_delta_min']),
                    repair_every_seed_ok=bool(all(v['holds'] for v in repair.values()) and len(repair) > 0))
    c2_parts = dict(mean_delta_ok=bool(lr['mean_delta'] > r2['mean_delta_gt']),
                    fold_wins_ok=bool(lr['fold_wins'] >= r2['fold_wins_min'] and lr['n_folds'] == r2['n_folds']),
                    grade3_recall_ok=bool(lr['te_pooled_grade3_recall']
                                          >= lr['lr_pooled_grade3_recall'] + r2['pooled_grade3_recall_margin']))
    c1, c2 = all(c1_parts.values()), all(c2_parts.values())
    return dict(C1=dict(holds=bool(c1), parts=c1_parts, repair_by_seed=repair, inputs={k: v for k, v in o.items()
                                                                                      if k != 'repair'}),
                C2=dict(holds=bool(c2), parts=c2_parts, inputs=lr),
                confirmed_geo2_candidate=bool(c1 and c2))


# ---------------------------------------------------------------- finalize / analysis
def assemble(out, y, ids, folds, seed, arm):
    out = Path(out)
    ledger = read_ledger(out)
    entries = [ledger['folds'].get(key(seed, arm, f)) for f in range(5)]
    require(all(e and e['status'] == 'complete' for e in entries), f'{seed}/{arm} incomplete')
    d = out/scheme_of(seed)/arm
    if (d/'metrics.json').exists():
        return json.loads((d/'metrics.json').read_text(encoding='utf-8'))
    for e in entries:
        check_saved(out, e)
    parts = pd.concat([read_pred(d/f'fold_{f}_pred.csv.gz') for f in range(5)])
    oof = parts.set_index('building_id').loc[ids.to_numpy()].reset_index()
    require(len(oof) == len(ids) and oof.building_id.is_unique, 'OOF coverage changed')
    np.testing.assert_array_equal(oof.fold.to_numpy(), folds[seed])
    np.testing.assert_array_equal(oof.y_true.to_numpy(), y.to_numpy())
    write_pred(d/'oof.csv.gz', oof)
    back = read_pred(d/'oof.csv.gz')
    np.testing.assert_array_equal(back[PROBS].to_numpy(), oof[PROBS].to_numpy())
    values = [e['macro_f1'] for e in entries]
    result = dict(study_id=STUDY_ID, seed=seed, arm=arm, cv_mean=float(np.mean(values)),
                  cv_std=float(np.std(values, ddof=1)), fold_macro_f1=values, oof_sha256=sha256_file(d/'oof.csv.gz'),
                  fit_seconds=float(sum(e['fit_seconds'] for e in entries)), finished_at_utc=now())
    write(d/'metrics.json', result)
    bd = out/'backup'/scheme_of(seed)/arm
    bd.mkdir(parents=True, exist_ok=True)
    for name in ['oof.csv.gz', 'metrics.json']:
        require(not (bd/name).exists(), 'Backup already exists; refusing overwrite')
        shutil.copy2(d/name, bd/name)
    log(out, f'{scheme_of(seed)}/{arm} OOF assembled mean={result["cv_mean"]:.9f} sd={result["cv_std"]:.9f}')
    return result


def pooled_block(frames, X, labels=None):
    """Aggregate over concatenated seed OOFs (each row appears once per seed)."""
    y = np.concatenate([f.y_true.to_numpy() for f in frames])
    p = np.concatenate([f.y_pred.to_numpy() for f in frames])
    fold_f1, fold_g3 = [], []
    for f in frames:
        for k in range(5):
            m = f.fold.to_numpy() == k
            s = score(f.y_true.to_numpy()[m], f.y_pred.to_numpy()[m])
            fold_f1.append(s['macro_f1'])
            fold_g3.append(s['recall'][2])
    pooled = score(y, p)
    age = np.tile(X.age.to_numpy() == 995, len(frames))
    geo1 = np.tile(X.geo_level_1_id.astype(str).to_numpy(), len(frames))
    prof = dict(age995=score(y[age], p[age]))
    for g in t10.config()['profiles']['geo1_regions']:
        m = geo1 == str(g)
        prof[f'geo1_{g}'] = score(y[m], p[m])
    return dict(n_folds=len(fold_f1), fold_macro_f1=fold_f1, fold_grade3_recall=fold_g3,
                cv_mean=float(np.mean(fold_f1)), cv_std=float(np.std(fold_f1, ddof=1)), pooled=pooled,
                accuracy=pooled['accuracy'], grade3_recall=pooled['recall'][2],
                pred_share=(np.bincount(p - 1, minlength=3)/len(p)).tolist(),
                error31=pooled['error31'], error32=pooled['error32'], rate31=pooled['rate31'], rate32=pooled['rate32'],
                profiles=prof), y, p


def pooled_compare(a, b, ya, pa, pb):
    from experiments.geo_ablation.audit import turnover
    deltas = [u - v for u, v in zip(a['fold_macro_f1'], b['fold_macro_f1'])]
    return dict(mean_delta=float(np.mean(deltas)), fold_delta=deltas, fold_wins=int(sum(d > 0 for d in deltas)),
                n_folds=len(deltas), pooled_grade3_recall_delta=a['grade3_recall'] - b['grade3_recall'],
                accuracy_diff=a['accuracy'] - b['accuracy'], turnover=turnover(ya, pb, pa))


def finalize_all(data, output):
    out = Path(output)
    X, y, ids, folds, design = verify(data, out)
    require(not (out/'analysis.json').exists(), 'analysis.json exists; refusing overwrite')
    ledger = read_ledger(out)
    tg = design['study_targets']
    complete = sum(ledger['folds'].get(k, {}).get('status') == 'complete' for k in tg)
    attempts = sum(e['attempts'] for e in ledger['folds'].values())
    require(complete == len(tg) == N_TARGETS and len(ledger['folds']) == N_TARGETS, 'Ledger incomplete; not finalizing')
    c = design['config']
    yv = y.to_numpy()
    fit_seconds = {arm: float(sum(e['fit_seconds'] for e in ledger['folds'].values() if e['arm'] == arm)) for arm in ARMS}
    res = dict(study_id=STUDY_ID, design_sha256=sha256_file(out/'design.json'), code_commit=design['code_commit'],
               ledger=dict(complete=complete, total=N_TARGETS, attempts=attempts, retries=attempts-complete,
                           fit_seconds_sum=float(sum(fit_seconds.values())), fit_seconds_by_arm=fit_seconds,
                           all_lr_converged=all(e['converged'] for e in ledger['folds'].values())),
               holdout_accessed=False, rules=c['rules'], splits=design['splits'],
               ari_between_new_seeds=design['ari_between_new_seeds'],
               note='New independent geo2 Group splits; descriptive, no significance or causal claims; random-CV '
                    'performance is the reused T10 stratified result; no holdout.', per_seed={})
    from experiments.geo_ablation.audit import turnover
    oofs = {}
    for s in SEEDS:
        block = {}
        for arm in ARMS:
            mt = assemble(out, y, ids, folds, s, arm)
            oofs[s, arm] = read_pred(out/scheme_of(s)/arm/'oof.csv.gz')
            np.testing.assert_array_equal(oofs[s, arm].building_id.to_numpy(), ids.to_numpy())
            recs = [json.loads((out/scheme_of(s)/arm/f'fold_{f}.json').read_text(encoding='utf-8')) for f in range(5)]
            block[arm] = dict(t11.summary(oofs[s, arm], X, mt['fit_seconds']), oof_sha256=mt['oof_sha256'],
                              fold_fit_seconds=[r['fit_seconds'] for r in recs],
                              converged=[r['converged'] for r in recs], n_iter=[r['n_iter'] for r in recs],
                              pred_share_by_fold=[r['pred_share'] for r in recs])
        comps = {}
        for a_name, b_name in [('LGBM_TE_G', 'LGBM_O'), ('LGBM_TE_G', 'LR_C'), ('LGBM_O', 'LR_C')]:
            comps[f'{a_name}_vs_{b_name}'] = t10.compare(block, a_name, b_name, yv, oofs[s, a_name], oofs[s, b_name])
        block['comparisons'] = comps
        res['per_seed'][str(s)] = block
    # ---- pooled over the 15 new folds
    pooled, ys, ps = {}, {}, {}
    for arm in ARMS:
        pooled[arm], ys[arm], ps[arm] = pooled_block([oofs[s, arm] for s in SEEDS], X)
    pc = {}
    for a_name, b_name in [('LGBM_TE_G', 'LGBM_O'), ('LGBM_TE_G', 'LR_C'), ('LGBM_O', 'LR_C')]:
        pc[f'{a_name}_vs_{b_name}'] = pooled_compare(pooled[a_name], pooled[b_name], ys[a_name], ps[a_name], ps[b_name])
    res['pooled_15_folds'] = dict(arms=pooled, comparisons=pc)
    repair = {s: (res['per_seed'][str(s)]['LGBM_TE_G']['accuracy'] - res['per_seed'][str(s)]['LGBM_O']['accuracy'],
                  res['per_seed'][str(s)]['comparisons']['LGBM_TE_G_vs_LGBM_O']['turnover']['prediction_agreement'])
              for s in SEEDS}
    o, lr = pc['LGBM_TE_G_vs_LGBM_O'], pc['LGBM_TE_G_vs_LR_C']
    res['judgments'] = judge(c['rules'], dict(mean_delta=o['mean_delta'], fold_wins=o['fold_wins'], n_folds=o['n_folds'],
                                              pooled_grade3_recall_delta=o['pooled_grade3_recall_delta'], repair=repair),
                             dict(mean_delta=lr['mean_delta'], fold_wins=lr['fold_wins'], n_folds=lr['n_folds'],
                                  te_pooled_grade3_recall=pooled['LGBM_TE_G']['grade3_recall'],
                                  lr_pooled_grade3_recall=pooled['LR_C']['grade3_recall']))
    res['finished_at_utc'] = now()
    write(out/'analysis.json', res)
    require(not (out/'backup'/'analysis.json').exists(), 'Backup already exists; refusing overwrite')
    shutil.copy2(out/'analysis.json', out/'backup'/'analysis.json')
    j = res['judgments']
    log(out, f'analysis written: C1={j["C1"]["holds"]} C2={j["C2"]["holds"]} confirmed={j["confirmed_geo2_candidate"]}')
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['control-check', 'prepare', 'fit-all', 'finalize', 'status'])
    p.add_argument('--data', default='data')
    p.add_argument('--output')
    p.add_argument('--destination')
    p.add_argument('--code-commit')
    p.add_argument('--control')
    a = p.parse_args()
    if a.action == 'control-check':
        control_check(a.data, a.destination)
    elif a.action == 'prepare':
        prepare(a.data, a.output, a.code_commit, a.control)
    elif a.action == 'fit-all':
        fit_all(a.data, a.output)
    elif a.action == 'finalize':
        finalize_all(a.data, a.output)
    else:
        ledger = read_ledger(a.output)
        print(json.dumps(dict(complete=sum(e['status'] == 'complete' for e in ledger['folds'].values()), total=N_TARGETS,
                              attempts=sum(e['attempts'] for e in ledger['folds'].values()),
                              running=[k for k, e in ledger['folds'].items() if e['status'] == 'running']), indent=1))


if __name__ == '__main__':
    sys.exit(main())
