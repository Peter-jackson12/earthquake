"""T35: stratified 5-fold reproduction benchmark of the teammate (EY) revised notebook pipeline.

NOT an evaluation or ranking of the teammate and NOT a new method.  The revised pipeline (75 features built from a base
frame only, one-hot matrix, class weights with strength 0.5) is re-implemented in `features.py` from cell 28 of the
teammate notebook `지진.ipynb` (SHA-256 633f6e6b3c1c3bdf669af71df29d31581cb97df63a303ab72b0bbc21b8eed33d; read-only)
and run on our development rows with the saved stratified 5-fold `split_manifest.csv`.

Arms (design docs/T35_EY_PIPELINE_V2_BENCHMARK_PLAN.md):
  L : the notebook's final LightGBM setting (800 trees, no early stopping)
  X : the notebook's best XGBoost setting (800 trees, no early stopping)
Both arms use the same per-fold matrices.  Actions: preflight (no study fit), prepare (seal), fit-all (resumable),
status, finalize.  Holdout is never read.  All outputs are private (outside Git).  Budget: 10 fits, cap 12, no retry.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
import sys
import tempfile
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from preprocess import sha256_file  # noqa: E402
from experiments.geo_group_cv import run as t5  # noqa: E402
from experiments.geo_group_cv.run import PROBS, require, score, write  # noqa: E402
from experiments.ey_pipeline_benchmark import run as T33  # noqa: E402  (read-only reuse of sealed helpers)
from experiments.ey_pipeline_benchmark.features import notebook_frame  # noqa: E402
from experiments.ey_pipeline_v2 import features as F  # noqa: E402
from experiments.ey_pipeline_v2 import original_check as OC  # noqa: E402

HERE = Path(__file__).resolve().parent
STUDY_ID = 'T35_ey_pipeline_v2_benchmark_v1'
ARMS = ['L', 'X']
N_FOLDS = 5
MODEL_FILE = {'L': 'fold_{f}_model.txt', 'X': 'fold_{f}_model.json'}
SOURCE = list(dict.fromkeys(t5.SOURCE + [
    'configs/data_contract.json', 'experiments/geo_ablation/audit.py',
    'experiments/ey_pipeline_benchmark/__init__.py', 'experiments/ey_pipeline_benchmark/features.py',
    'experiments/ey_pipeline_benchmark/run.py',
    'experiments/ey_pipeline_v2/__init__.py', 'experiments/ey_pipeline_v2/features.py',
    'experiments/ey_pipeline_v2/original_check.py', 'experiments/ey_pipeline_v2/run.py',
    'experiments/ey_pipeline_v2/config.json', 'tests/test_ey_pipeline_v2.py',
    'docs/T35_EY_PIPELINE_V2_BENCHMARK_PLAN.md']))

now, log, event, atomic_write = T33.now, T33.log, T33.event, T33.atomic_write
read_pred, write_pred, read_ledger, check_saved, backup_files = (T33.read_pred, T33.write_pred, T33.read_ledger,
                                                                 T33.check_saved, T33.backup_files)
git, require_not_holdout, load_inputs, reference, run_lock = (T33.git, T33.require_not_holdout, T33.load_inputs,
                                                              T33.reference, T33.run_lock)


def environment():
    names = ['numpy', 'pandas', 'scipy', 'scikit-learn', 'lightgbm', 'xgboost', 'joblib', 'threadpoolctl']
    return dict(python=platform.python_version(), **{n: importlib.metadata.version(n) for n in names})


def sources():
    return {p: sha256_file(ROOT/p) for p in SOURCE}


def key(arm, fold):
    return f'{arm}/{fold}'


def targets():
    """Fold-major order so each fold's shared matrices are built once for both arms."""
    return [(a, f) for f in range(N_FOLDS) for a in ARMS]


def config(path=HERE/'config.json'):
    c = json.loads(Path(path).read_text(encoding='utf-8'))
    require(c['study_id'] == STUDY_ID and c['arms'] == ARMS and c['n_folds'] == N_FOLDS, 'Unplanned study/arms')
    require(c['features']['n_frame_columns'] == 75 == len(F.feature_columns()) == len(set(F.feature_columns()))
            and c['features']['dummy_columns'] == F.DUMMY_COLS, 'Feature definition differs from the design')
    require(c['lgbm_params'] == dict(random_state=42, n_jobs=4, deterministic=True, verbose=-1, n_estimators=800,
                                     learning_rate=0.05, num_leaves=127, min_child_samples=50, subsample=0.8,
                                     subsample_freq=1, colsample_bytree=0.5), 'LightGBM setting changed')
    require(c['xgb_params'] == dict(tree_method='hist', random_state=42, n_jobs=4, n_estimators=800, learning_rate=0.05,
                                    max_depth=8, subsample=0.8, colsample_bytree=0.7, min_child_weight=10),
            'XGBoost setting changed')
    require(c['weights'] == dict(strength=0.5), 'Weights changed')
    require(c['budget'] == dict(planned_fits=10, max_fits=12, max_recovered_units=2, max_attempts_per_unit=2,
                                automatic_retries=0), 'Unplanned fit budget')
    require(len(ARMS) * N_FOLDS == c['budget']['planned_fits'], 'Budget arithmetic')
    require(c['threads'] == dict(model_n_jobs=4, outer_parallel_jobs=1, feature_building_threads=1), 'Thread plan changed')
    require(c['tolerances'] == dict(comparator_value_atol=5e-7, comparator_recall3_definition='pooled'), 'Tolerances changed')
    require(c['checkpoint']['reload_probability_atol'] == {'L': 1e-12, 'X': 1e-6}, 'Reload tolerance changed')
    require(set(c['comparators']) == {'T33_E1', 'T33_E2', 'T14', 'T8_N', 'LR_C', 'T0', 'LGBM_O'}, 'Comparator set changed')
    require(not c['holdout_access'] and not c['tuning'] and not c['threshold_or_decision_rule_change']
            and not c['automatic_model_replacement'], 'Unplanned run policy')
    require(c['environment'] == environment() or Path(path) != HERE/'config.json', 'Environment differs from pinned')
    return c


# ---------------------------------------------------------------- model helpers
def make_model(c, arm):
    if arm == 'L':
        from lightgbm import LGBMClassifier
        return LGBMClassifier(**c['lgbm_params'])
    from xgboost import XGBClassifier
    return XGBClassifier(**c['xgb_params'])


def checked_proba(model, X):
    require(np.array_equal(np.asarray(model.classes_), [0, 1, 2]), 'Internal class order must be 0,1,2')
    p = np.asarray(model.predict_proba(X), dtype=np.float64)
    require(p.shape == (len(X), 3) and np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all(), 'Invalid probabilities')
    np.testing.assert_allclose(p.sum(1), 1.0, atol=1e-5, rtol=0)
    return p


def fit_model(c, arm, X, y0, w):
    model = make_model(c, arm)
    with threadpool_limits(limits=c['threads']['model_n_jobs']), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        t0 = time.perf_counter()
        model.fit(X, y0, sample_weight=w)
        seconds = time.perf_counter() - t0
    return model, seconds, [dict(category=x.category.__name__, message=str(x.message)[:300]) for x in caught]


def save_model(arm, model, path):
    if arm == 'L':
        model.booster_.save_model(str(path))
    else:
        model.get_booster().save_model(str(path))


def reload_proba(arm, path, X, threads):
    """Re-inference from the saved model file only (text / JSON, no pickle)."""
    if arm == 'L':
        import lightgbm
        b = lightgbm.Booster(model_file=str(path))
        require(b.current_iteration() == 800 and b.num_trees() == 3 * 800, 'Saved LightGBM model is not 800 iterations')
        return np.asarray(b.predict(X.to_numpy(dtype=np.float64), num_threads=threads), dtype=np.float64)
    import xgboost
    b = xgboost.Booster()
    b.load_model(str(path))
    require(b.num_boosted_rounds() == 800, 'Saved XGBoost model is not 800 rounds')
    b.set_param({'nthread': threads})
    dm = xgboost.DMatrix(X.to_numpy(dtype=np.float64), feature_names=[str(x) for x in X.columns])
    return np.asarray(b.predict(dm), dtype=np.float64)


# ---------------------------------------------------------------- shared fold matrices
def matrix_files(f):
    return [f'fold_{f}_valid_matrix.npz', f'fold_{f}_features.json']


def write_valid_matrix(d, f, X_va, valid_ids, train_sha, diagnostics):
    np.savez_compressed(d/f'fold_{f}_valid_matrix.npz', values=X_va.to_numpy(dtype=np.float64),
                        columns=np.array([str(x) for x in X_va.columns], dtype=str),
                        building_id=np.asarray(valid_ids, dtype=np.int64))
    meta = dict(columns=[str(x) for x in X_va.columns], dtypes={str(k): str(v) for k, v in X_va.dtypes.items()},
                rows=int(len(X_va)), valid_matrix_sha256=F.matrix_sha256(X_va), train_matrix_sha256=train_sha,
                sha_convention='features.matrix_sha256: json column names, row count, each column as float64 bytes',
                diagnostics=diagnostics)
    write(d/f'fold_{f}_features.json', meta)


def read_valid_matrix(d, f):
    z = np.load(Path(d)/f'fold_{f}_valid_matrix.npz', allow_pickle=False)
    return pd.DataFrame(z['values'], columns=z['columns'].tolist()), z['building_id']


class Context:
    def __init__(self, X, y, ids, folds, out, design):
        self.X, self.y, self.ids, self.folds = X, y, ids, np.asarray(folds)
        self.out, self.design, self.c = Path(out), design, design['config']
        self.frame = notebook_frame(X, y)
        self._m = {}

    def get(self, f):
        """Matrices of fold f (built once; persisted and reload-verified on first build, hash-checked after)."""
        if f in self._m:
            return self._m[f]
        self._m = {}
        a, b = np.flatnonzero(self.folds != f), np.flatnonzero(self.folds == f)
        require(len(a) + len(b) == len(self.ids) and len(b) > 0, 'Fold membership invalid')
        train = self.frame.iloc[a].reset_index(drop=True)
        valid = self.frame.iloc[b].drop(columns=[F.GRADE]).reset_index(drop=True)
        t0 = time.perf_counter()
        with threadpool_limits(limits=self.c['threads']['feature_building_threads']):
            X_tr, X_va, _, _, diag = F.fold_matrices(train, valid)
        seconds = time.perf_counter() - t0
        diag['build_seconds'] = seconds
        built = dict(a=a, b=b, train=X_tr, valid=X_va, train_sha=F.matrix_sha256(X_tr), valid_sha=F.matrix_sha256(X_va),
                     y0=train[F.GRADE].to_numpy() - 1, diagnostics=diag)
        ledger = read_ledger(self.out)
        entry = ledger['features'].get(f'fold/{f}')
        d = self.out/'shared'
        if entry:
            require(entry['train_matrix_sha256'] == built['train_sha'] and entry['valid_matrix_sha256'] == built['valid_sha'],
                    f'Fold {f} matrices are not reproducible (hash differs from the ledger)')
            check_saved(self.out, entry)
            log(self.out, f'fold {f} matrices rebuilt: hashes equal to the ledger ({seconds:.0f}s)')
        else:
            d.mkdir(parents=True, exist_ok=True)
            for name in matrix_files(f):  # leftovers of an attempt that stopped before its ledger write
                if (d/name).exists():
                    n = 0
                    while (d/f'{name}.incomplete_attempt{n}').exists():
                        n += 1
                    (d/name).rename(d/f'{name}.incomplete_attempt{n}')
            write_valid_matrix(d, f, X_va, self.ids.to_numpy()[b], built['train_sha'], diag)
            back, back_ids = read_valid_matrix(d, f)
            require(F.matrix_sha256(back) == built['valid_sha'] and np.array_equal(back_ids, self.ids.to_numpy()[b]),
                    'Stored validation matrix differs on reload')
            shas = {n: sha256_file(d/n) for n in matrix_files(f)}
            backup_files(self.out, 'shared', shas)
            ledger = read_ledger(self.out)
            ledger['features'][f'fold/{f}'] = dict(dir='shared', artifact_sha256=shas, train_matrix_sha256=built['train_sha'],
                                                   valid_matrix_sha256=built['valid_sha'], diagnostics=diag,
                                                   created_at_utc=now())
            atomic_write(self.out/'ledger.json', ledger)
            event(self.out, action='fold_matrices_saved', fold=f, artifact_sha256=shas,
                  train_matrix_sha256=built['train_sha'], valid_matrix_sha256=built['valid_sha'])
            log(self.out, f'fold {f} matrices built in {seconds:.0f}s ({X_tr.shape[1]} columns), saved and reload-verified')
        self._m[f] = built
        return built


# ---------------------------------------------------------------- one (arm, fold) unit
def fits_started(ledger):
    return sum(e.get('attempts', 0) for e in ledger['units'].values())


def recovered_units(ledger):
    return sum(1 for e in ledger['units'].values() if e.get('attempts', 0) > 1)


def unit_files(arm, f):
    return [MODEL_FILE[arm].format(f=f), f'fold_{f}_pred.csv.gz', f'fold_{f}.json']


def begin_fit(ctx, arm, f):
    """Count the fit before it starts; refuse over budget; move stale files of an unfinished attempt aside."""
    out, cap = ctx.out, ctx.c['budget']
    k = key(arm, f)
    ledger = read_ledger(out)
    entry = ledger['units'].get(k)
    require(fits_started(ledger) < cap['max_fits'], f'Fit budget exhausted ({cap["max_fits"]} fits)')
    prior = entry.get('attempts', 0) if entry else 0
    require(prior < cap['max_attempts_per_unit'], f'{k}: at most {cap["max_attempts_per_unit"]} attempts')
    if prior >= 1:
        require(recovered_units(ledger) < cap['max_recovered_units'], f'At most {cap["max_recovered_units"]} recovered units')
    d = out/arm
    d.mkdir(parents=True, exist_ok=True)
    for name in unit_files(arm, f):
        if (d/name).exists():
            n = 0
            while (d/f'{name}.incomplete_attempt{prior}_{n}').exists():
                n += 1
            (d/name).rename(d/f'{name}.incomplete_attempt{prior}_{n}')
    entry = entry or dict(dir=arm, arm=arm, fold=f, artifact_sha256={}, attempts=0)
    entry.update(attempts=prior + 1, status='running', started_at_utc=now())
    ledger['units'][k] = entry
    atomic_write(out/'ledger.json', ledger)
    if prior:
        log(out, f'{k}: previous attempt did not finish; manual resume attempt {prior + 1}')
    event(out, unit=k, action='fit_start', attempt=prior + 1)
    return entry


def fit_unit(ctx, arm, f):
    out, c = ctx.out, ctx.c
    k = key(arm, f)
    require(arm in ARMS and k in ctx.design['study_targets'], f'{k} is not a pre-registered unit')
    entry = read_ledger(out)['units'].get(k)
    if entry and entry['status'] == 'complete':
        check_saved(out, entry)
        log(out, f'{k} already complete; verified hashes, not refit')
        return entry, False
    m = ctx.get(f)
    begin_fit(ctx, arm, f)
    y0 = m['y0']
    w = F.class_weights(pd.Series(y0), c['weights']['strength'])
    model, seconds, fit_warnings = fit_model(c, arm, m['train'], y0, w)
    p = checked_proba(model, m['valid'])
    pred = p.argmax(1) + 1
    y_true = ctx.y.to_numpy()[m['b']]
    d = out/arm
    model_path, pred_path = d/MODEL_FILE[arm].format(f=f), d/f'fold_{f}_pred.csv.gz'
    save_model(arm, model, model_path)
    frame = pd.DataFrame({'building_id': ctx.ids.to_numpy()[m['b']], 'fold': f, 'y_true': y_true, 'y_pred': pred,
                          **{n: p[:, i] for i, n in enumerate(PROBS)}})
    write_pred(pred_path, frame)
    stored, stored_ids = read_valid_matrix(out/'shared', f)
    require(F.matrix_sha256(stored) == m['valid_sha'] and np.array_equal(stored_ids, frame.building_id.to_numpy()),
            'Stored validation matrix differs from the one used')
    p2 = reload_proba(arm, model_path, stored, c['threads']['model_n_jobs'])
    reload_diff = float(np.max(np.abs(p2 - p)))
    require(reload_diff <= c['checkpoint']['reload_probability_atol'][arm] and np.array_equal(p2.argmax(1) + 1, pred),
            f'{k}: model reload mismatch {reload_diff}')
    back = read_pred(pred_path)
    np.testing.assert_array_equal(back.building_id.to_numpy(), frame.building_id.to_numpy())
    np.testing.assert_array_equal(back.y_pred.to_numpy(), pred)
    np.testing.assert_array_equal(back[PROBS].to_numpy(), p)
    metrics = score(y_true, pred)
    record = dict(study_id=STUDY_ID, arm=arm, fold=f, train_rows=len(m['a']), valid_rows=len(m['b']),
                  train_class_counts=[int((y0 == v).sum()) for v in (0, 1, 2)], metrics=metrics,
                  pred_share=(np.bincount(pred - 1, minlength=3) / len(pred)).tolist(),
                  matrices=dict(train_matrix_sha256=m['train_sha'], valid_matrix_sha256=m['valid_sha'],
                                columns=int(m['train'].shape[1]), diagnostics=m['diagnostics']),
                  fit_seconds=seconds,
                  effective_parameters={k2: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                                        for k2, v in model.get_params().items()},
                  reload_probability_max_abs_diff=reload_diff, csv_round_trip_exact=True, warnings=fit_warnings,
                  finished_at_utc=now())
    write(d/f'fold_{f}.json', record)
    shas = {n: sha256_file(d/n) for n in unit_files(arm, f)}
    backup_files(out, arm, shas)
    ledger = read_ledger(out)
    entry = ledger['units'][k]
    entry['artifact_sha256'] = shas
    entry.update(status='complete', macro_f1=metrics['macro_f1'], fit_seconds=seconds, finished_at_utc=now())
    atomic_write(out/'ledger.json', ledger)
    check_saved(out, entry)
    event(out, unit=k, action='unit_complete', macro_f1=metrics['macro_f1'], seconds=seconds, artifact_sha256=shas)
    log(out, f'{k} complete: F1={metrics["macro_f1"]:.6f} fit={seconds:.0f}s reload_diff={reload_diff:.3g} '
             'saved+verified+backed-up')
    return entry, True


# ---------------------------------------------------------------- preflight (no study fit)
def probe_models(c):
    """Synthetic 400-row fits (NOT study fits, outside the 10/12 budget): save/reload exactness and repeat determinism."""
    rng = np.random.default_rng(0)
    n = 400
    X = pd.DataFrame({'a': rng.normal(size=n), 'b': rng.integers(0, 5, n).astype(np.int64), 'c': rng.integers(0, 2, n)})
    y0 = ((X.a + X.c * 0.7 + rng.normal(0, .5, n) > 0.5).astype(int) + (X.a > 1).astype(int)).to_numpy()
    w = F.class_weights(pd.Series(y0), 0.5)
    out = {}
    small = dict(c, lgbm_params=dict(c['lgbm_params'], n_estimators=20, min_child_samples=5),
                 xgb_params=dict(c['xgb_params'], n_estimators=20))
    for arm in ARMS:
        m1, _, _ = fit_model(small, arm, X, y0, w)
        m2, _, _ = fit_model(small, arm, X, y0, w)
        p1, p2 = checked_proba(m1, X), checked_proba(m2, X)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/MODEL_FILE[arm].format(f='probe')
            save_model(arm, m1, path)
            if arm == 'L':
                import lightgbm
                pr = np.asarray(lightgbm.Booster(model_file=str(path)).predict(X.to_numpy(dtype=np.float64)))
            else:
                import xgboost
                b = xgboost.Booster()
                b.load_model(str(path))
                pr = np.asarray(b.predict(xgboost.DMatrix(X.to_numpy(dtype=np.float64), feature_names=list(X.columns))))
        out[arm] = dict(repeat_identical=bool(np.array_equal(p1, p2)), reload_max_abs_diff=float(np.max(np.abs(pr - p1))),
                        reload_within_tolerance=bool(np.max(np.abs(pr - p1)) <= c['checkpoint']['reload_probability_atol'][arm]))
    return dict(arms=out, counted_as_study_fit=False, probe_fits=4,
                note='synthetic 400-row probe with 20 trees, outside the study budget; not used for any decision')


def notebook_evidence(source_copy, c):
    nb = Path(source_copy)/c['notebook']['file_name']
    require(nb.is_file(), f'Private notebook copy not found: {nb}')
    require(sha256_file(nb) == c['notebook']['sha256'], 'Notebook copy SHA differs from the registered SHA')
    text = OC.cell_text(nb, c['notebook']['cell_index'])
    require(hashlib.sha256(text.encode('utf-8')).hexdigest() == c['notebook']['cell_source_sha256'], 'Cell 28 text SHA differs')
    ns, ev = OC.extract(text)
    require(ev['extract_text_sha256'] == c['notebook']['extract_text_sha256'], 'AST extract SHA differs from the registered SHA')
    require(sorted(ev['functions']) == sorted(c['notebook']['extract_functions'])
            and sorted(ev['assignments']) == sorted(c['notebook']['extract_assignments']), 'Extracted definitions differ')
    return ns, dict(notebook_sha256=c['notebook']['sha256'], cell_source_sha256=c['notebook']['cell_source_sha256'], **ev)


def preflight(data, destination, source_copy=None, local_root=None):
    """No study classifier is fitted.  Real-data checks on fold 0 plus bit-identity against the notebook definitions."""
    c = config()
    dest = Path(destination)
    require(not dest.exists(), 'Do not overwrite an existing preflight (after a failed one, use a new folder name)')
    require_not_holdout(data)
    source_copy = Path(source_copy) if source_copy else dest.resolve().parent/'source_copy'
    local_root = Path(local_root) if local_root else dest.resolve().parents[1]
    require(environment() == c['environment'], 'Environment differs from the pinned environment')
    plan_sha = sha256_file(ROOT/c['plan']['path'])
    require(plan_sha == c['plan']['sha256'], 'Plan document SHA differs from the registered SHA')
    ns, nb_evidence = notebook_evidence(source_copy, c)
    X, y, ids, folds, contract = load_inputs(data)
    res = dict(study_id=STUDY_ID, kind='preflight; no study classifier fitted', classifiers_fitted=0, holdout_accessed=False,
               created_at_utc=now(), environment=environment(), data_contract=contract, plan_sha256=plan_sha,
               notebook=nb_evidence, source_sha256=sources())
    res['inputs'] = T33.input_checks(X, y, ids, folds, contract, c)
    res['comparators'] = T33.comparator_check(c, local_root, ids, y, folds)
    frame = notebook_frame(X, y)
    a, b = np.flatnonzero(folds != 0), np.flatnonzero(folds == 0)
    train = frame.iloc[a].reset_index(drop=True)
    valid_labeled = frame.iloc[b].reset_index(drop=True)
    t0 = time.perf_counter()
    with threadpool_limits(limits=1):
        X_tr, X_va, fr_tr, fr_va, diag = F.fold_matrices(train, valid_labeled.drop(columns=[F.GRADE]))
    build_seconds = time.perf_counter() - t0
    require(list(fr_tr.columns) == F.feature_columns() and fr_tr.shape[1] == 75, '75-column frame order differs')
    require(diag['nan_cells_train'] == 0 and diag['nan_cells_valid'] == 0, f'Unexpected missing cells: {diag}')
    res['matrix_fold0'] = dict(diag, rows=dict(train=len(X_tr), valid=len(X_va)), build_seconds=build_seconds,
                               valid_columns_equal_train=bool(list(X_va.columns) == list(X_tr.columns)))
    t0 = time.perf_counter()
    with threadpool_limits(limits=1):
        res['original_function_check'] = dict(OC.compare(ns, train, valid_labeled, c['weights']['strength']),
                                              seconds=time.perf_counter() - t0,
                                              note='notebook definitions (AST extract of the private copy) run in the frozen '
                                                   'environment; equality here does not prove equality in the notebook environment')
        res['label_probe_fold0'] = F.label_probe(train, valid_labeled)
    res['model_probe'] = probe_models(c)
    ok = (res['original_function_check']['passes'] and res['label_probe_fold0']['passes']
          and res['matrix_fold0']['valid_columns_equal_train']
          and all(v['repeat_identical'] and v['reload_within_tolerance'] for v in res['model_probe']['arms'].values()))
    res['status'] = 'PASS' if ok else 'FAIL'
    write(dest/'preflight.json', res)
    print(json.dumps({k2: res[k2] for k2 in ('status', 'original_function_check', 'label_probe_fold0', 'model_probe')},
                     indent=1, default=str))
    require(ok, 'Preflight failed')
    return res


# ---------------------------------------------------------------- prepare / verify / fit-all / status
def prepare(data, out, preflight_path, code_commit=None, source_copy=None, local_root=None):
    out = Path(out)
    require(not out.exists(), 'Do not overwrite an existing study')
    c = config()
    require_not_holdout(data)
    source_copy = Path(source_copy) if source_copy else out.resolve().parent/'source_copy'
    local_root = Path(local_root) if local_root else out.resolve().parents[1]
    require(environment() == c['environment'], 'Environment differs from the pinned environment')
    require(shutil.disk_usage(out.resolve().parent).free / 2**30 >= c['min_free_disk_gb_at_prepare'], 'Not enough free disk')
    head = git('rev-parse', 'HEAD')
    require(code_commit is None or code_commit == head, 'code_commit must equal HEAD')
    require(len(head) == 40, 'Full commit SHA required')
    for p in SOURCE:
        require(git('hash-object', '--', p) == git('rev-parse', f'HEAD:{p}'), f'Sealed source differs from HEAD (commit it first): {p}')
    plan_sha = sha256_file(ROOT/c['plan']['path'])
    require(plan_sha == c['plan']['sha256'], 'Plan document SHA differs from the registered SHA')
    _, nb_evidence = notebook_evidence(source_copy, c)
    pf = json.loads(Path(preflight_path).read_text(encoding='utf-8'))
    require(pf['status'] == 'PASS' and pf['study_id'] == STUDY_ID and pf['classifiers_fitted'] == 0, 'Preflight must PASS first')
    require(pf['source_sha256'] == sources() and pf['plan_sha256'] == plan_sha and pf['environment'] == environment(),
            'Sources changed since the preflight; rerun it')
    X, y, ids, folds, contract = load_inputs(data)
    require(contract == pf['data_contract'], 'Data contract differs from the preflight')
    inputs = T33.input_checks(X, y, ids, folds, contract, c)
    comparators = T33.comparator_check(c, local_root, ids, y, folds)
    out.mkdir(parents=True)
    (out/'backup').mkdir()
    design = dict(study_id=STUDY_ID, config=c, code_commit=head, source_sha256=sources(), plan_sha256=plan_sha,
                  notebook=nb_evidence, source_copy_dir=str(source_copy.resolve()), local_root=str(local_root.resolve()),
                  data_contract=contract, inputs=inputs, environment=environment(), comparators=comparators,
                  preflight=dict(path=str(Path(preflight_path).resolve()), sha256=sha256_file(preflight_path),
                                 status=pf['status'], label_probe_fold0=pf['label_probe_fold0'],
                                 original_function_check=pf['original_function_check']['passes'],
                                 matrix_columns=pf['matrix_fold0']['columns']),
                  study_targets=[key(*t) for t in targets()], max_fits=c['budget']['max_fits'],
                  planned_fits=c['budget']['planned_fits'], automatic_retries=0, holdout_accessed=False,
                  created_at_utc=now(), classifiers_fitted=0)
    write(out/'design.json', design)
    write(out/'design_seal.json', dict(design_sha256=sha256_file(out/'design.json'), code_commit=head, created_at_utc=now(),
                                       classifiers_fitted=0))
    atomic_write(out/'ledger.json', dict(study_id=STUDY_ID, units={}, features={}))
    log(out, f'prepared and sealed design {sha256_file(out/"design.json")} commit={head}')
    return design


def verify(data, out):
    out = Path(out)
    seal = json.loads((out/'design_seal.json').read_text(encoding='utf-8'))
    require(sha256_file(out/'design.json') == seal['design_sha256'], 'Design seal changed')
    design = json.loads((out/'design.json').read_text(encoding='utf-8'))
    require(design['config'] == config(), 'Frozen config changed')
    require(design['source_sha256'] == sources(), 'Frozen source changed')
    require(design['environment'] == environment(), 'Frozen environment changed')
    X, y, ids, folds, contract = load_inputs(data)
    require(contract == design['data_contract'], 'Data contract changed')
    return X, y, ids, folds, design


def fit_all(data, out, only=None):
    out = Path(out)
    with run_lock(out):
        X, y, ids, folds, design = verify(data, out)
        ctx = Context(X, y, ids, folds, out, design)
        before = pd.util.hash_pandas_object(X, index=True).to_numpy()
        todo = [t for t in targets() if only is None or key(*t) in only]
        log(out, f'fit-all start: {len(todo)} units (complete ones are hash-verified, not refit)')
        t0 = time.perf_counter()
        for arm, f in todo:
            try:
                fit_unit(ctx, arm, f)
            except Exception as exc:
                log(out, f'STOPPED {key(arm, f)} {type(exc).__name__}: {exc}')
                event(out, unit=key(arm, f), action='stopped', error=f'{type(exc).__name__}: {exc}', trace=traceback.format_exc())
                raise
        np.testing.assert_array_equal(before, pd.util.hash_pandas_object(X, index=True).to_numpy())
        log(out, f'fit-all finished wall={time.perf_counter() - t0:.1f}s')


def status(out):
    ledger = read_ledger(out)
    units = ledger['units']
    return dict(complete=sum(e['status'] == 'complete' for e in units.values()), total_units=len(ARMS) * N_FOLDS,
                fits_started=fits_started(ledger), planned_fits=len(ARMS) * N_FOLDS,
                max_fits=json.loads((Path(out)/'design.json').read_text(encoding='utf-8'))['max_fits'],
                recovered_units=recovered_units(ledger), running=[k for k, e in units.items() if e['status'] == 'running'],
                fold_matrices_saved=sorted(ledger['features']))


# ---------------------------------------------------------------- finalize (design 6)
def d4_label(cv_mean, reference, rule):
    return T33.d4_label(cv_mean, reference, rule)


def assemble(out, y, ids, folds, arm):
    out = Path(out)
    ledger = read_ledger(out)
    entries = [ledger['units'].get(key(arm, f)) for f in range(N_FOLDS)]
    require(all(e and e['status'] == 'complete' for e in entries), f'{arm} incomplete')
    d = out/arm
    for e in entries:
        check_saved(out, e)
    oof = pd.concat([read_pred(d/f'fold_{f}_pred.csv.gz') for f in range(N_FOLDS)]).set_index('building_id').loc[ids.to_numpy()].reset_index()
    require(len(oof) == len(ids) and oof.building_id.is_unique, 'OOF coverage changed')
    np.testing.assert_array_equal(oof.fold.to_numpy(), folds)
    np.testing.assert_array_equal(oof.y_true.to_numpy(), y.to_numpy())
    write_pred(d/'oof.csv.gz', oof)
    back = read_pred(d/'oof.csv.gz')
    np.testing.assert_array_equal(back[PROBS].to_numpy(), oof[PROBS].to_numpy())
    values = [e['macro_f1'] for e in entries]
    result = dict(study_id=STUDY_ID, arm=arm, cv_mean=float(np.mean(values)), cv_std=float(np.std(values, ddof=1)),
                  fold_macro_f1=values, oof_sha256=sha256_file(d/'oof.csv.gz'),
                  fit_seconds=float(sum(e['fit_seconds'] for e in entries)), finished_at_utc=now())
    write(d/'metrics.json', result)
    backup_files(out, arm, ['oof.csv.gz', 'metrics.json'])
    log(out, f'{arm} OOF assembled mean={result["cv_mean"]:.9f} sd={result["cv_std"]:.9f}')
    return result


def finalize_all(data, out):
    from experiments.geo_ablation.audit import turnover
    out = Path(out)
    X, y, ids, folds, design = verify(data, out)
    require(not (out/'analysis.json').exists(), 'analysis.json exists; refusing overwrite')
    ledger = read_ledger(out)
    tg = design['study_targets']
    complete = sum(ledger['units'].get(k, {}).get('status') == 'complete' for k in tg)
    require(complete == len(tg) == len(ARMS) * N_FOLDS and len(ledger['units']) == len(tg), 'Ledger incomplete; not finalizing')
    c = design['config']
    started = fits_started(ledger)
    res = dict(study_id=STUDY_ID, design_sha256=sha256_file(out/'design.json'), code_commit=design['code_commit'],
               ledger=dict(complete=complete, total=len(tg), fits_started=started, planned_fits=c['budget']['planned_fits'],
                           max_fits=c['budget']['max_fits'], extra_fits=started - c['budget']['planned_fits'],
                           recovered_units=recovered_units(ledger),
                           fit_seconds_sum=float(sum(e['fit_seconds'] for e in ledger['units'].values()))),
               holdout_accessed=False, rules=c['rules'],
               note='Reproduction benchmark of the EY revised notebook pipeline on our common stratified folds. Descriptive '
                    'labels only; no significance, no causal or single-factor claims, no adoption or ranking of the teammate; '
                    'not independent performance (design 7).', arms={}, comparators={}, comparisons={})
    oofs = {}
    for arm in ARMS:
        m = assemble(out, y, ids, folds, arm)
        oofs[arm] = read_pred(out/arm/'oof.csv.gz')
        np.testing.assert_array_equal(oofs[arm].building_id.to_numpy(), ids.to_numpy())
        rec = [json.loads((out/arm/f'fold_{f}.json').read_text(encoding='utf-8')) for f in range(N_FOLDS)]
        res['arms'][arm] = dict(T33.summarize(oofs[arm], X, c), oof_sha256=m['oof_sha256'],
                                fit_seconds=[r['fit_seconds'] for r in rec],
                                feature_build_seconds=[r['matrices']['diagnostics']['build_seconds'] for r in rec],
                                matrix_columns=[r['matrices']['columns'] for r in rec],
                                pred_share_by_fold=[r['pred_share'] for r in rec])
    yv = y.to_numpy()
    comps = {}
    for name, spec in c['comparators'].items():
        r = reference(Path(design['local_root'])/spec['path'], spec['sha256'], ids, y, folds)
        comps[name] = r
        res['comparators'][name] = dict(T33.summarize(r, X, c), label=spec['label'], oof_sha256=spec['sha256'],
                                        judged=spec['judged'])
    for arm in ARMS:
        res['comparisons'][arm] = {}
        for name, spec in c['comparators'].items():
            a, b = res['arms'][arm], res['comparators'][name]
            d1 = T33.d1_label(a['fold_macro_f1'], b['fold_macro_f1'], c['rules']['D1'])
            if not spec['judged']:
                d1['label'] = 'REFERENCE_ONLY'
            d1['safety_indicators_no_verdict'] = dict(
                recall3=dict(arm=a['recall3'], comparator=b['recall3'], delta=a['recall3'] - b['recall3']),
                error31=dict(arm=a['error31'], comparator=b['error31'], delta=a['error31'] - b['error31']),
                error32=dict(arm=a['error32'], comparator=b['error32'], delta=a['error32'] - b['error32']),
                rate31=dict(arm=a['rate31'], comparator=b['rate31']), rate32=dict(arm=a['rate32'], comparator=b['rate32']))
            d1['turnover'] = turnover(yv, comps[name].y_pred.to_numpy(), oofs[arm].y_pred.to_numpy())
            res['comparisons'][arm][name] = d1
    L, Xa = res['arms']['L'], res['arms']['X']
    d3 = T33.d1_label(L['fold_macro_f1'], Xa['fold_macro_f1'], c['rules']['D3'])
    d3['turnover_X_to_L'] = turnover(yv, oofs['X'].y_pred.to_numpy(), oofs['L'].y_pred.to_numpy())
    res['lgbm_vs_xgb'] = d3
    nb = c['notebook_reference']
    res['notebook_distance'] = dict(
        L_label_vs_test=d4_label(L['cv_mean'], nb['L_test_macro_f1'], c['rules']['D4']),
        numbers_only=dict(L_minus_test=L['cv_mean'] - nb['L_test_macro_f1'],
                          L_minus_validation=L['cv_mean'] - nb['L_validation_macro_f1'],
                          X_minus_validation=Xa['cv_mean'] - nb['X_validation_macro_f1']),
        notebook_reference=nb)
    res['preflight_label_probe_fold0'] = design['preflight']['label_probe_fold0']
    res['judgments'] = dict(D1={arm: {n: res['comparisons'][arm][n]['label'] for n in c['comparators']} for arm in ARMS},
                            D3=d3['label'], D4_L=res['notebook_distance']['L_label_vs_test']['label'],
                            representative_line='L (the notebook final model); X reported alongside')
    res['finished_at_utc'] = now()
    write(out/'analysis.json', res)
    backup_files(out, '.', ['analysis.json'])
    log(out, 'analysis written: ' + json.dumps(res['judgments'], ensure_ascii=False))
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('action', choices=['preflight', 'prepare', 'fit-all', 'status', 'finalize'])
    p.add_argument('--data', help='absolute path of earthquake/data (development.csv + split_manifest.csv)')
    p.add_argument('--out', help='preflight: new destination folder; others: the run folder')
    p.add_argument('--preflight', help='prepare: path of a PASS preflight.json')
    p.add_argument('--code-commit', help='optional; must equal HEAD')
    p.add_argument('--source-copy', help='private notebook copy folder (default: <out>/../source_copy)')
    p.add_argument('--local-root', help='local_reproduction root (default: <out>/../..)')
    p.add_argument('--only', nargs='*', help='fit-all: restrict to units such as L/0')
    a = p.parse_args()
    if a.action == 'preflight':
        preflight(a.data, a.out, a.source_copy, a.local_root)
    elif a.action == 'prepare':
        prepare(a.data, a.out, a.preflight, a.code_commit, a.source_copy, a.local_root)
    elif a.action == 'fit-all':
        fit_all(a.data, a.out, a.only)
    elif a.action == 'status':
        print(json.dumps(status(a.out), indent=1))
    else:
        r = finalize_all(a.data, a.out)
        print(json.dumps(r['judgments'], indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
