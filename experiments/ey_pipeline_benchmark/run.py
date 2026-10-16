"""T33: stratified 5-fold reproduction benchmark of the teammate (EY) notebook pipeline.

NOT an evaluation or ranking of the teammate and NOT a new method.  The pipeline (75 features: 38 raw inputs, material and
structure derivations, region composition, combination damage rates, gaps; LightGBM with the notebook's single chosen
setting) is re-implemented in `features.py` from cell 25 of `earthquake/data/지진.ipynb`
(SHA-256 480d63cd6adc3f68f70142ff159d229654db5d17a27d3af093baa38b831ac07b; read-only; the notebook is never modified)
and run on our development rows with the saved stratified 5-fold `split_manifest.csv`.

Arms (design docs/T33_EY_PIPELINE_BENCHMARK_PLAN.md):
  E1 faithful order  : features A (whole training fold) -> inner tr/val split -> early stopping -> refit on all A rows.
  E2 ES split        : inner tr/val split of the raw rows first -> features B inside tr -> early stopping -> refit on A.
Both refit on the same matrix, so their outer-fold results differ only through the tree count n_trees.

Actions: preflight (no study fit), prepare (seal), fit-all (resumable), status, finalize.
Holdout is never read.  All outputs are private (outside Git).  Study budget: 20 fits, hard cap 24, no automatic retry.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from preprocess import GEO, ID, sha256_file  # noqa: E402
from experiments.geo_group_cv import run as t5  # noqa: E402
from experiments.geo_group_cv.run import PROBS, load, require, score, write  # noqa: E402
from experiments.ey_pipeline_benchmark import features as F  # noqa: E402
from experiments.ey_pipeline_benchmark import original_check as OC  # noqa: E402

HERE = Path(__file__).resolve().parent
STUDY_ID = 'T33_ey_pipeline_benchmark_v1'
ARMS = ['E1', 'E2']
N_FOLDS = 5
SEED = 42
HOLDOUT_MARKER = 'final_holdout'
CAT_PARAMS = ['max_cat_to_onehot', 'cat_smooth', 'cat_l2', 'max_cat_threshold', 'min_data_per_group']
SOURCE = list(dict.fromkeys(t5.SOURCE + [
    'configs/data_contract.json', 'experiments/geo_group_cv/audit.py', 'experiments/geo_ablation/audit.py',
    'experiments/ey_pipeline_benchmark/__init__.py', 'experiments/ey_pipeline_benchmark/features.py',
    'experiments/ey_pipeline_benchmark/original_check.py', 'experiments/ey_pipeline_benchmark/run.py',
    'experiments/ey_pipeline_benchmark/config.json', 'tests/test_ey_pipeline_benchmark.py',
    'docs/T33_EY_PIPELINE_BENCHMARK_PLAN.md']))


# ---------------------------------------------------------------- small helpers
def now():
    return datetime.now(timezone.utc).isoformat()


def environment():
    names = ['numpy', 'pandas', 'scipy', 'scikit-learn', 'lightgbm', 'joblib', 'threadpoolctl']
    return dict(python=platform.python_version(), **{n: importlib.metadata.version(n) for n in names})


def require_not_holdout(path):
    require(HOLDOUT_MARKER not in Path(path).as_posix().lower(), f'Holdout path refused: {path}')


def sources():
    return {p: sha256_file(ROOT/p) for p in SOURCE}


def key(arm, fold):
    return f'{arm}/{fold}'


def targets():
    """Fold-major order so the fold's shared features A are built once and reused by both arms."""
    return [(a, f) for f in range(N_FOLDS) for a in ARMS]


def config(path=HERE/'config.json'):
    c = json.loads(Path(path).read_text(encoding='utf-8'))
    require(c['study_id'] == STUDY_ID and c['arms'] == ARMS and c['n_folds'] == N_FOLDS, 'Unplanned study/arms')
    require(c['features']['n'] == 75 and c['features']['columns'] == F.feature_columns()
            and c['features']['categorical_columns'] == F.CAT_COLS
            and [c['features']['columns'].index(x) + 1 for x in F.CAT_COLS] == c['features']['categorical_positions_1based']
            and len(set(c['features']['columns'])) == 75, 'Feature list/order differs from design 3.1')
    require(c['lgbm_params'] == dict(learning_rate=0.03, num_leaves=31, min_child_samples=20, subsample=1.0,
                                     subsample_freq=1, colsample_bytree=0.5, random_state=42, n_jobs=4, verbose=-1,
                                     deterministic=True, force_col_wise=True), 'LightGBM setting changed')
    require(c['weights']['strength'] == 0.3 and not c['weights']['early_stopping_metric_weighted'], 'Weights changed')
    require(c['es']['max_rounds'] == 2000 and c['es']['patience'] == 50 and c['es']['metric'] == 'multi_logloss',
            'Early stopping setting changed')
    require(c['budget'] == dict(planned_fits=20, max_fits=24, max_recovered_units=2, max_attempts_per_stage=2,
                                automatic_retries=0, fit_stages_per_unit=2), 'Unplanned fit budget')
    require(c['tolerances'] == dict(comparator_value_atol=5e-7, comparator_recall3_definition='pooled'),
            'Comparator tolerance policy changed')
    require(len(ARMS) * N_FOLDS * c['budget']['fit_stages_per_unit'] == c['budget']['planned_fits'], 'Budget arithmetic')
    require(c['threads'] == dict(lgbm_n_jobs=4, outer_parallel_jobs=1, feature_building_threads=1), 'Thread plan changed')
    require(not c['holdout_access'] and not c['tuning'] and not c['threshold_or_decision_rule_change']
            and not c['automatic_model_replacement'], 'Unplanned run policy')
    require(c['inner_split'] == dict(test_size=0.2, random_state=42, stratified=True,
                                     expected_rows=dict(train=133427, val=33357)), 'Inner split changed')
    require(set(c['comparators']) == {'T8_N', 'T14', 'LR_C', 'T0', 'LGBM_O'}, 'Comparator set changed')
    require(c['environment'] == environment() or Path(path) != HERE/'config.json', 'Environment differs from pinned')
    return c


def log(out, message):
    line = f'{now()} [pid {os.getpid()}] {message}'
    print(line, flush=True)
    with open(Path(out)/'run.log', 'a', encoding='utf-8') as fh:
        fh.write(line + '\n')


def event(out, **kw):
    with open(Path(out)/'attempts.jsonl', 'a', encoding='utf-8') as fh:
        fh.write(json.dumps(dict(at_utc=now(), pid=os.getpid(), **kw), ensure_ascii=False) + '\n')


def atomic_write(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    write(tmp, obj)
    os.replace(tmp, path)


def read_pred(path):
    return pd.read_csv(path, float_precision='round_trip')


def write_pred(path, frame):
    frame.to_csv(path, index=False, compression={'method': 'gzip', 'mtime': 0}, float_format='%.17g')


def read_ledger(out):
    return json.loads((Path(out)/'ledger.json').read_text(encoding='utf-8'))


def check_saved(out, entry):
    for name, digest in entry['artifact_sha256'].items():
        require(sha256_file(Path(out)/entry['dir']/name) == digest, f'Artifact changed: {entry["dir"]}/{name}')
        require(sha256_file(Path(out)/'backup'/entry['dir']/name) == digest, f'Backup changed: {entry["dir"]}/{name}')


def backup_files(out, directory, names):
    bd = Path(out)/'backup'/directory
    bd.mkdir(parents=True, exist_ok=True)
    for name in names:
        if (bd/name).exists():  # left by an attempt that stopped before its ledger write: keep it aside, never overwrite
            n = 0
            while (bd/f'{name}.incomplete_attempt{n}').exists():
                n += 1
            (bd/name).rename(bd/f'{name}.incomplete_attempt{n}')
        shutil.copy2(Path(out)/directory/name, bd/name)


def git(*args):
    r = subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True)
    require(r.returncode == 0, f'git {" ".join(args)} failed: {r.stderr.strip()}')
    return r.stdout.strip()


def model_text_parameters(path):
    text = Path(path).read_text(encoding='utf-8')
    return dict(re.findall(r'^\[([a-z_0-9]+): (.*)\]$', text, flags=re.M))


def categorical_effective(path):
    params = model_text_parameters(path)
    return {k: float(params[k]) for k in CAT_PARAMS}


# ---------------------------------------------------------------- inputs
def load_inputs(data):
    require_not_holdout(data)
    X, y, ids, old, contract = load(data)  # load_development() + frozen contract; sorted by building_id
    folds = np.asarray(old)
    require(set(folds.tolist()) == set(range(N_FOLDS)), 'Expected the five stored folds')
    return X, y, ids, folds, contract


def reference(path, digest, ids, y, folds):
    require(sha256_file(path) == digest, f'Reference OOF changed: {path}')
    r = read_pred(path).set_index('building_id').loc[ids.to_numpy()].reset_index()
    require(len(r) == len(ids) and r.building_id.is_unique, 'Reference OOF coverage')
    np.testing.assert_array_equal(r.fold.to_numpy(), folds)
    np.testing.assert_array_equal(r.y_true.to_numpy(), y.to_numpy())
    return r


def comparator_values(r, y_true):
    fo = r.fold.to_numpy()
    fold_scores = [score(r.y_true[fo == f], r.y_pred[fo == f]) for f in range(N_FOLDS)]
    pooled = score(y_true, r.y_pred.to_numpy())
    return dict(cv_mean=float(np.mean([s['macro_f1'] for s in fold_scores])), pooled_recall3=pooled['recall'][2],
                fold_mean_recall3=float(np.mean([s['recall'][2] for s in fold_scores])), error31=pooled['error31'])


def comparator_check(c, local_root, ids, y, folds):
    """Every comparator OOF: SHA, alignment, labels, and the registered CV mean / Recall3 / 3->1."""
    tol = c['tolerances']['comparator_value_atol']
    seen = {}
    for name, spec in c['comparators'].items():
        path = Path(local_root)/spec['path']
        require_not_holdout(path)
        r = reference(path, spec['sha256'], ids, y, folds)
        v = comparator_values(r, y.to_numpy())
        require(abs(v['cv_mean'] - spec['cv_mean']) <= tol, f'{name}: CV Macro F1 {v["cv_mean"]} != registered {spec["cv_mean"]}')
        require(v['error31'] == spec['error31'], f'{name}: 3->1 {v["error31"]} != registered {spec["error31"]}')
        require(abs(v['pooled_recall3'] - spec['recall3']) <= tol,
                f'{name}: pooled Recall3 {v["pooled_recall3"]} != registered {spec["recall3"]}')
        seen[name] = dict(path=str(path), sha256=spec['sha256'], rows=len(r), observed=v,
                          recall3_definition=c['tolerances']['comparator_recall3_definition'])
    return seen


def input_checks(X, y, ids, folds, contract, c):
    """Fold sizes, disjointness, integer round trip of the three region IDs, binary values."""
    sp = c['split']
    require(sp['manifest_sha256'] == contract['files']['split_manifest.csv']
            and sp['development_sha256'] == contract['files']['development.csv'],
            'Registered split/development SHA differs from the data contract')
    sizes = [int((folds != f).sum()) for f in range(N_FOLDS)], [int((folds == f).sum()) for f in range(N_FOLDS)]
    require(all(s == sp['train_rows_per_fold'] for s in sizes[0]) and all(s == sp['valid_rows_per_fold'] for s in sizes[1]),
            f'Fold sizes differ from the frozen 166,784 / 41,696: {sizes}')
    for f in range(N_FOLDS):
        a, b = np.flatnonzero(folds != f), np.flatnonzero(folds == f)
        require(not set(ids.iloc[a]) & set(ids.iloc[b]) and len(a) + len(b) == len(ids), f'Fold {f} train/valid overlap')
    frame = F.notebook_frame(X, y)  # raises if str(int(x)) != x, binaries not 0/1, labels invalid
    return dict(fold_train_rows=sizes[0], fold_valid_rows=sizes[1], region_id_round_trip=True, binary_values_01=True,
                frame_columns=int(frame.shape[1]), frame_rows=int(len(frame)),
                development_sha256=contract['files']['development.csv'], manifest_sha256=contract['files']['split_manifest.csv'])


# ---------------------------------------------------------------- model helpers
def inner_split(y0):
    """The notebook's tr/val split: stratified 80/20, seed 42, positions in the order the splitter returns them."""
    y0 = np.asarray(y0)
    tr, va = train_test_split(np.arange(len(y0)), test_size=0.2, stratify=y0, random_state=SEED)
    return tr, va


def lgbm_params(c, n_estimators):
    return dict(c['lgbm_params'], n_estimators=int(n_estimators))


def checked_proba(model, X):
    require(np.array_equal(model.classes_, [0, 1, 2]), 'Internal class order must be 0,1,2')
    p = np.asarray(model.predict_proba(X), dtype=np.float64)
    require(p.shape == (len(X), 3) and np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all(), 'Invalid probabilities')
    np.testing.assert_allclose(p.sum(1), 1.0, atol=1e-6, rtol=0)
    return p


def fit_lgbm(c, X, y0, w, n_estimators, eval_set=None):
    from lightgbm import LGBMClassifier, early_stopping, log_evaluation
    model = LGBMClassifier(**lgbm_params(c, n_estimators))
    threads = c['threads']['lgbm_n_jobs']
    with threadpool_limits(limits=threads), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        t0 = time.perf_counter()
        if eval_set is None:
            model.fit(X, y0, sample_weight=w)
        else:
            model.fit(X, y0, sample_weight=w, eval_set=[eval_set],
                      callbacks=[early_stopping(c['es']['patience'], verbose=False), log_evaluation(0)])
        seconds = time.perf_counter() - t0
    return model, seconds, [dict(category=x.category.__name__, message=str(x.message)) for x in caught]


def logloss(y0, p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-15, 1.0)
    return float(-np.mean(np.log(p[np.arange(len(y0)), np.asarray(y0)])))


def cross_check_trees(booster, n_trees, X, p_default):
    """L1: a saved/reloaded model holds exactly n_trees iterations (3 classes x n_trees trees) in LightGBM 4.6.0, for the
    early-stopping model (best_iteration) as for the refit; `predict(num_iteration=n_trees)` equals the default prediction."""
    trees, iters = booster.num_trees(), booster.current_iteration()
    require(trees == 3 * iters, f'Tree/iteration mismatch: {trees} vs 3 x {iters}')
    require(iters == n_trees, f'Tree count {iters} != n_trees {n_trees}')
    prefix = np.asarray(booster.predict(X, num_iteration=n_trees), dtype=np.float64)
    require(np.array_equal(prefix, p_default) or float(np.max(np.abs(prefix - p_default))) <= 1e-12,
            'predict(num_iteration=n_trees) differs from the stored prediction')
    return dict(iterations=int(iters), trees=int(trees))


# ---------------------------------------------------------------- run context (data + fold feature cache)
class Context:
    def __init__(self, X, y, ids, folds, out, design):
        self.X, self.y, self.ids, self.folds = X, y, ids, np.asarray(folds)
        self.out, self.design = Path(out), design
        self.c = design['config']
        self.frame = F.notebook_frame(X, y)
        self._A = {}

    def split(self, f):
        a, b = np.flatnonzero(self.folds != f), np.flatnonzero(self.folds == f)
        require(len(a) + len(b) == len(self.ids) and len(b) > 0, 'Fold membership invalid')
        train = self.frame.iloc[a].reset_index(drop=True)
        valid = self.frame.iloc[b].drop(columns=[F.GRADE]).reset_index(drop=True)
        return a, b, train, valid

    def get_A(self, f):
        """Features A of fold f (whole training fold -> train matrix, outer-validation matrix); persisted once."""
        if f in self._A:
            return self._A[f]
        self._A = {}
        a, b, train, valid = self.split(f)
        t0 = time.perf_counter()
        with threadpool_limits(limits=self.c['threads']['feature_building_threads']):
            A_tr, A_va, tables = F.build_features(train, valid)
        seconds = time.perf_counter() - t0
        built = dict(train=A_tr, valid=A_va, tables=tables, seconds=seconds, a=a, b=b, train_frame=train,
                     valid_frame=valid, train_sha=F.matrix_sha256(A_tr), valid_sha=F.matrix_sha256(A_va))
        built['diagnostics'] = dict(
            valid_rows_with_unseen_region=[int((~valid[g].isin(train[g])).sum()) for g in GEO],
            nan_cells_train_numeric=int(A_tr.drop(columns=F.CAT_COLS).isna().sum().sum()),
            nan_cells_valid_numeric=int(A_va.drop(columns=F.CAT_COLS).isna().sum().sum()),
            valid_unseen_category_cells=int(A_va[F.CAT_COLS].isna().sum().sum()),
            tables=F.table_summary(tables), build_seconds=seconds)
        ledger = read_ledger(self.out)
        entry = ledger['features'].get(f'A/{f}')
        if entry:
            require(entry['train_matrix_sha256'] == built['train_sha'] and entry['valid_matrix_sha256'] == built['valid_sha'],
                    f'Features A of fold {f} are not reproducible (matrix hash differs from the ledger)')
            check_saved(self.out, entry)
            log(self.out, f'features A fold {f} rebuilt: matrix hashes equal to the ledger ({seconds:.0f}s)')
        else:
            self._persist_A(f, built)
        self._A[f] = built
        return built

    def _persist_A(self, f, built):
        out = self.out
        d = out/'shared'
        d.mkdir(parents=True, exist_ok=True)
        prefix = f'fold_{f}_features_A'
        state_path, valid_path = d/f'{prefix}_state.joblib', d/f'{prefix}_valid.joblib'
        text_names = F.state_text_names(prefix)
        for name in [state_path.name, valid_path.name] + text_names:  # unregistered leftovers of an interrupted attempt
            if (d/name).exists():
                n = 0
                while (d/f'{name}.incomplete_attempt{n}').exists():
                    n += 1
                (d/name).rename(d/f'{name}.incomplete_attempt{n}')
        joblib.dump(dict(tables=built['tables'], columns=list(built['train'].columns), train_matrix_sha256=built['train_sha'],
                         valid_matrix_sha256=built['valid_sha']), state_path, compress=3)
        joblib.dump(built['valid'], valid_path, compress=3)
        F.write_state_text(d, prefix, built['tables'], built['valid'])      # pickle-free copy for the independent audit
        state = joblib.load(state_path)  # freshly generated trusted file
        rebuilt = F.apply_tables(built['valid_frame'], state['tables'])
        ok, why = OC.frames_identical(rebuilt, built['valid'])
        require(ok, f'Stored tables do not reproduce the outer-validation matrix: {why}')
        require(F.matrix_sha256(joblib.load(valid_path)) == built['valid_sha'], 'Stored validation matrix changed on reload')
        text_rebuilt = F.apply_tables(built['valid_frame'], F.read_tables_text(d, prefix))
        ok, why = OC.frames_identical(text_rebuilt, built['valid'])
        require(ok, f'Text-format tables do not reproduce the outer-validation matrix: {why}')
        require(F.matrix_sha256(F.read_matrix_text(d, prefix)) == built['valid_sha'], 'Text-format validation matrix differs')
        shas = {n: sha256_file(d/n) for n in [state_path.name, valid_path.name] + text_names}
        backup_files(out, 'shared', shas)
        ledger = read_ledger(out)
        ledger['features'][f'A/{f}'] = dict(dir='shared', artifact_sha256=shas, train_matrix_sha256=built['train_sha'],
                                            valid_matrix_sha256=built['valid_sha'], rows=dict(train=len(built['train']),
                                            valid=len(built['valid'])), diagnostics=built['diagnostics'], created_at_utc=now(),
                                            text_format=dict(state=f'{prefix}_state.json', matrix=f'{prefix}_matrix.npz'))
        atomic_write(out/'ledger.json', ledger)
        event(out, action='features_A_saved', fold=f, artifact_sha256=shas, train_matrix_sha256=built['train_sha'],
              valid_matrix_sha256=built['valid_sha'])
        log(out, f'features A fold {f} built in {built["seconds"]:.0f}s, saved (joblib + npz/csv.gz) and reload-verified')


# ---------------------------------------------------------------- one (arm, fold) unit
UNIT_STAGE_FILES = {
    'es': lambda f: ([f'fold_{f}_es_model.txt', f'fold_{f}_es.json', f'fold_{f}_features_B_state.joblib']
                     + F.state_text_names(f'fold_{f}_features_B')),
    'final': lambda f: [f'fold_{f}_final_model.txt', f'fold_{f}_pred.csv.gz', f'fold_{f}.json']}


def fits_started(ledger):
    return sum(e.get('es_attempts', 0) + e.get('final_attempts', 0) for e in ledger['units'].values())


def recovered_units(ledger):
    return sum(1 for e in ledger['units'].values() if e.get('es_attempts', 0) > 1 or e.get('final_attempts', 0) > 1)


def begin_fit(ctx, arm, f, stage):
    """Count the fit before it starts; refuse over budget; move stale files of an unfinished attempt aside."""
    out, design = ctx.out, ctx.design
    k = key(arm, f)
    ledger = read_ledger(out)
    entry = ledger['units'].get(k)
    cap = design['config']['budget']
    require(fits_started(ledger) < cap['max_fits'], f'Fit budget exhausted ({cap["max_fits"]} fits)')
    attempt_key = f'{stage}_attempts'
    prior = entry.get(attempt_key, 0) if entry else 0
    require(prior < cap['max_attempts_per_stage'], f'{key(arm, f)} {stage}: at most {cap["max_attempts_per_stage"]} attempts per stage')
    if prior >= 1:  # a repeat of the same stage = manual recovery of this unit
        already = entry.get('es_attempts', 0) > 1 or entry.get('final_attempts', 0) > 1
        require(already or recovered_units(ledger) < cap['max_recovered_units'],
                f'At most {cap["max_recovered_units"]} units may be recovered manually')
    d = out/arm
    d.mkdir(parents=True, exist_ok=True)
    for name in UNIT_STAGE_FILES[stage](f):
        stale = d/name
        if stale.exists():
            n = 0
            while (d/f'{name}.incomplete_attempt{prior}_{n}').exists():
                n += 1
            stale.rename(d/f'{name}.incomplete_attempt{prior}_{n}')
    if entry is None:
        entry = dict(dir=arm, arm=arm, fold=f, artifact_sha256={}, es_attempts=0, final_attempts=0)
    entry[attempt_key] = prior + 1
    entry['status'] = f'{stage}_running'
    entry[f'{stage}_started_at_utc'] = now()
    ledger['units'][k] = entry
    atomic_write(out/'ledger.json', ledger)
    if prior:
        log(out, f'{k} {stage}: previous attempt did not finish; manual resume attempt {prior + 1}')
    event(out, unit=k, action=f'{stage}_start', attempt=prior + 1)
    return entry


def inner_id_shas(ctx, A, tr_pos, va_pos):
    ids = ctx.ids.to_numpy()[A['a']]
    return F.ids_sha256(ids[tr_pos]), F.ids_sha256(ids[va_pos])


def es_stage(ctx, arm, f, A, tr_pos, va_pos, y_train0):
    """Early-stopping fit.  E1 uses rows of A; E2 builds features B inside tr (validation rows are looked up)."""
    out, c = ctx.out, ctx.c
    d = out/arm
    entry = begin_fit(ctx, arm, f, 'es')
    if arm == 'E1':
        Xtr, Xva, B_tables = A['train'].iloc[tr_pos], A['train'].iloc[va_pos], None
    else:
        with threadpool_limits(limits=c['threads']['feature_building_threads']):
            Xtr, Xva, B_tables = F.features_B(A['train_frame'], tr_pos, va_pos)
    y_tr, y_va = y_train0[tr_pos], y_train0[va_pos]
    w = F.class_weights(pd.Series(y_tr), c['weights']['strength'])
    model, seconds, fit_warnings = fit_lgbm(c, Xtr, y_tr, w, c['es']['max_rounds'], eval_set=(Xva, y_va))
    best = int(model.best_iteration_)
    require(best >= 1, 'Early stopping returned no best iteration')
    curve = [float(v) for v in model.evals_result_['valid_0'][c['es']['metric']]]
    require(int(np.argmin(curve)) + 1 == best, 'best_iteration_ is not the minimum of the validation curve')
    p_va = checked_proba(model, Xva)
    inner = score(y_va + 1, p_va.argmax(1) + 1)
    recomputed = logloss(y_va, p_va)
    require(abs(recomputed - curve[best - 1]) <= c['es']['logloss_recompute_atol'],
            f'Validation multi_logloss mismatch: {recomputed} vs {curve[best - 1]}')
    cap_hit = bool(len(curve) >= c['es']['max_rounds'])  # the run ended at the round cap; read together with n_trees (L9)
    es_model = d/f'fold_{f}_es_model.txt'
    model.booster_.save_model(str(es_model))
    names = [es_model.name]
    if B_tables is not None:
        b_path = d/f'fold_{f}_features_B_state.joblib'
        joblib.dump(dict(tables=B_tables, train_matrix_sha256=F.matrix_sha256(Xtr), val_matrix_sha256=F.matrix_sha256(Xva),
                         columns=list(Xtr.columns)), b_path, compress=3)
        names.append(b_path.name)
        prefix = f'fold_{f}_features_B'      # pickle-free copy: tables + the inner-validation matrix used for stopping
        names += F.write_state_text(d, prefix, B_tables, Xva)
        require(F.matrix_sha256(F.read_matrix_text(d, prefix)) == F.matrix_sha256(Xva), 'Text-format B matrix differs')
        B_va_rebuilt = F.apply_tables(A['train_frame'].iloc[va_pos], F.read_tables_text(d, prefix))
        ok, why = OC.frames_identical(B_va_rebuilt, Xva)
        require(ok, f'Text-format B tables do not reproduce the inner-validation matrix: {why}')
    # reload verification of the early-stopping model (same-model prefix relation)
    import lightgbm
    booster = lightgbm.Booster(model_file=str(es_model))
    p2 = np.asarray(booster.predict(Xva, num_threads=c['threads']['lgbm_n_jobs']), dtype=np.float64)
    reload_diff = float(np.max(np.abs(p2 - p_va)))
    require(reload_diff <= c['checkpoint']['reload_probability_atol'], f'ES model reload mismatch {reload_diff}')
    trees = cross_check_trees(booster, best, Xva, p2)   # exactly best_iteration iterations are saved (LightGBM 4.6.0)
    id_shas = inner_id_shas(ctx, A, tr_pos, va_pos)
    record = dict(study_id=STUDY_ID, arm=arm, fold=f, n_trees=best, cap_hit=cap_hit, max_rounds=c['es']['max_rounds'],
                  patience=c['es']['patience'], curve_length=len(curve), best_validation_multi_logloss=curve[best - 1],
                  multi_logloss_curve=curve, internal_validation_macro_f1=inner['macro_f1'], internal_validation=inner,
                  es_train_rows=len(tr_pos), es_val_rows=len(va_pos),
                  es_train_class_counts=[int((y_tr == v).sum()) for v in (0, 1, 2)],
                  tr_building_id_sha256=id_shas[0], val_building_id_sha256=id_shas[1],
                  cap_hit_definition=c['es']['cap_hit_definition'],
                  es_train_matrix_sha256=F.matrix_sha256(Xtr), es_val_matrix_sha256=F.matrix_sha256(Xva),
                  features_B_tables=F.table_summary(B_tables) if B_tables is not None else None,
                  model_iterations_saved=trees['iterations'], model_trees_saved=trees['trees'],
                  reload_probability_max_abs_diff=reload_diff, fit_seconds=seconds, warnings=fit_warnings,
                  finished_at_utc=now())
    write(d/f'fold_{f}_es.json', record)
    names.append(f'fold_{f}_es.json')
    shas = {n: sha256_file(d/n) for n in names}
    backup_files(out, arm, shas)
    ledger = read_ledger(out)
    entry = ledger['units'][key(arm, f)]
    entry['artifact_sha256'].update(shas)
    entry.update(status='es_done', n_trees=best, es_seconds=seconds, internal_validation_macro_f1=inner['macro_f1'],
                 es_finished_at_utc=now())
    atomic_write(out/'ledger.json', ledger)
    event(out, unit=key(arm, f), action='es_done', n_trees=best, cap_hit=cap_hit, seconds=seconds, artifact_sha256=shas)
    log(out, f'{key(arm, f)} early stopping: n_trees={best} cap_hit={cap_hit} internal F1={inner["macro_f1"]:.6f} '
             f'fit={seconds:.0f}s saved+verified+backed-up')
    return record


def final_stage(ctx, arm, f, A, y_train0, es_record):
    out, c = ctx.out, ctx.c
    d = out/arm
    begin_fit(ctx, arm, f, 'final')
    n_trees = int(es_record['n_trees'])
    w = F.class_weights(pd.Series(y_train0), c['weights']['strength'])
    model, seconds, fit_warnings = fit_lgbm(c, A['train'], y_train0, w, n_trees)
    require(model.booster_.current_iteration() == n_trees and model.booster_.num_trees() == 3 * n_trees,
            'Refit did not complete exactly n_trees iterations')
    p = checked_proba(model, A['valid'])
    pred = p.argmax(1) + 1
    y_true = ctx.y.to_numpy()[A['b']]
    model_path, pred_path = d/f'fold_{f}_final_model.txt', d/f'fold_{f}_pred.csv.gz'
    model.booster_.save_model(str(model_path))
    frame = pd.DataFrame({'building_id': ctx.ids.to_numpy()[A['b']], 'fold': f, 'y_true': y_true, 'y_pred': pred,
                          **{n: p[:, i] for i, n in enumerate(PROBS)}})
    write_pred(pred_path, frame)
    # reload verification: stored model + stored validation matrix -> same probabilities, same CSV
    import lightgbm
    booster = lightgbm.Booster(model_file=str(model_path))
    loaded_valid = joblib.load(out/'shared'/f'fold_{f}_features_A_valid.joblib')  # freshly generated trusted file
    require(F.matrix_sha256(loaded_valid) == A['valid_sha'], 'Stored validation matrix differs from the one used')
    p2 = np.asarray(booster.predict(loaded_valid, num_threads=c['threads']['lgbm_n_jobs']), dtype=np.float64)
    reload_diff = float(np.max(np.abs(p2 - p)))
    require(reload_diff <= c['checkpoint']['reload_probability_atol'] and np.array_equal(p2.argmax(1) + 1, pred),
            f'Model reload mismatch {reload_diff}')
    cross_check_trees(booster, n_trees, loaded_valid, p2)
    back = read_pred(pred_path)
    np.testing.assert_array_equal(back.building_id.to_numpy(), frame.building_id.to_numpy())
    np.testing.assert_array_equal(back.y_pred.to_numpy(), pred)
    np.testing.assert_array_equal(back[PROBS].to_numpy(), p)
    metrics = score(y_true, pred)
    record = dict(study_id=STUDY_ID, arm=arm, fold=f, n_trees=n_trees, cap_hit=es_record['cap_hit'],
                  train_rows=len(A['a']), valid_rows=len(A['b']),
                  train_class_counts=[int((y_train0 == v).sum()) for v in (0, 1, 2)],
                  metrics=metrics, pred_share=(np.bincount(pred - 1, minlength=3) / len(pred)).tolist(),
                  internal_validation_macro_f1=es_record['internal_validation_macro_f1'],
                  internal_minus_outer=es_record['internal_validation_macro_f1'] - metrics['macro_f1'],
                  features_A=dict(train_matrix_sha256=A['train_sha'], valid_matrix_sha256=A['valid_sha'],
                                  build_seconds=A['seconds'], diagnostics=A['diagnostics']),
                  es_fit_seconds=es_record['fit_seconds'], fit_seconds=seconds,
                  effective_parameters={k2: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                                        for k2, v in model.get_params().items()},
                  lightgbm_categorical_effective=categorical_effective(model_path),
                  reload_probability_max_abs_diff=reload_diff, csv_round_trip_exact=True, warnings=fit_warnings,
                  finished_at_utc=now())
    require(record['lightgbm_categorical_effective'] == {k2: float(v) for k2, v in
                                                         ctx.design['lightgbm_categorical_effective_defaults'].items()},
            'Categorical parameters differ from the preflight record')
    write(d/f'fold_{f}.json', record)
    names = [model_path.name, pred_path.name, f'fold_{f}.json']
    shas = {n: sha256_file(d/n) for n in names}
    backup_files(out, arm, shas)
    ledger = read_ledger(out)
    entry = ledger['units'][key(arm, f)]
    entry['artifact_sha256'].update(shas)
    entry.update(status='complete', macro_f1=metrics['macro_f1'], n_trees=n_trees, fit_seconds=seconds + es_record['fit_seconds'],
                 final_seconds=seconds, finished_at_utc=now())
    atomic_write(out/'ledger.json', ledger)
    check_saved(out, entry)
    event(out, unit=key(arm, f), action='unit_complete', macro_f1=metrics['macro_f1'], n_trees=n_trees, seconds=seconds,
          artifact_sha256=shas)
    log(out, f'{key(arm, f)} complete: F1={metrics["macro_f1"]:.6f} n_trees={n_trees} refit={seconds:.0f}s '
             f'reload_diff={reload_diff:.3g} saved+verified+backed-up')
    return entry


def fit_unit(ctx, arm, f):
    """Run (arm, fold) through its checkpoints.  Complete units are only hash-verified; `es_done` units run the refit only."""
    out = ctx.out
    k = key(arm, f)
    require(arm in ARMS and k in ctx.design['study_targets'], f'{k} is not a pre-registered unit')
    ledger = read_ledger(out)
    entry = ledger['units'].get(k)
    if entry and entry['status'] == 'complete':
        check_saved(out, entry)
        log(out, f'{k} already complete; verified hashes, not refit')
        return entry, False
    A = ctx.get_A(f)
    y_train0 = ctx.frame[F.GRADE].to_numpy()[A['a']] - 1
    tr_pos, va_pos = inner_split(y_train0)
    require(not set(tr_pos) & set(va_pos) and len(tr_pos) + len(va_pos) == len(y_train0), 'Inner split must partition the fold')
    if entry and entry['status'] in ('es_done', 'final_running'):
        check_saved(out, {**entry, 'artifact_sha256': {n: s for n, s in entry['artifact_sha256'].items()
                                                       if n in UNIT_STAGE_FILES['es'](f)}})
        es_record = json.loads((out/arm/f'fold_{f}_es.json').read_text(encoding='utf-8'))
        require(inner_id_shas(ctx, A, tr_pos, va_pos) == (es_record['tr_building_id_sha256'], es_record['val_building_id_sha256']),
                f'{k}: inner tr/val membership differs from the early-stopping checkpoint')
        log(out, f'{k} early-stopping checkpoint verified (n_trees={es_record["n_trees"]}); refit only')
    else:
        es_record = es_stage(ctx, arm, f, A, tr_pos, va_pos, y_train0)
    return final_stage(ctx, arm, f, A, y_train0, es_record), True


# ---------------------------------------------------------------- preflight (no study fit)
def probe_lightgbm(c):
    """Synthetic 400-row fit (NOT a study fit, outside the 20/24 budget): effective categorical defaults and the
    behavior of early stopping / saved trees in the pinned LightGBM."""
    from lightgbm import LGBMClassifier, early_stopping, log_evaluation
    import lightgbm
    rng = np.random.default_rng(0)
    n = 400
    X = pd.DataFrame({'a': rng.normal(size=n), 'b': rng.integers(0, 5, n),
                      'c': pd.Categorical(rng.choice(list('uvwxyz'), n))})
    y = (X.a + (X.c.cat.codes % 3) * 0.5 + rng.normal(0, .5, n) > 0.8).astype(int) + (X.a > 1).astype(int)
    tr, va = np.arange(0, 320), np.arange(320, n)
    model = LGBMClassifier(**lgbm_params(c, 300))
    model.fit(X.iloc[tr], y.iloc[tr], eval_set=[(X.iloc[va], y.iloc[va])],
              callbacks=[early_stopping(c['es']['patience'], verbose=False), log_evaluation(0)])
    with tempfile.TemporaryDirectory() as td:
        path = Path(td)/'probe.txt'
        model.booster_.save_model(str(path))
        eff = categorical_effective(path)
        booster = lightgbm.Booster(model_file=str(path))
        diff = float(np.max(np.abs(booster.predict(X.iloc[va]) - model.predict_proba(X.iloc[va]))))
        trees = booster.num_trees()
    best = int(model.best_iteration_)
    curve = model.evals_result_['valid_0'][c['es']['metric']]
    return dict(categorical_effective=eff, best_iteration=best, curve_length=len(curve), saved_trees=int(trees),
                saved_iterations_equal_best=bool(trees == 3 * best), reload_max_abs_diff=diff, counted_as_study_fit=False,
                probe_fits=1, note='synthetic 400-row probe outside the 20/24 fit budget; not used for any decision')


def notebook_evidence(source_copy, c):
    """Private notebook copy: file SHA, cell text SHA, AST extract SHA; returns (namespace, evidence)."""
    nb = Path(source_copy)/c['notebook']['file_name']
    require(nb.is_file(), f'Private notebook copy not found: {nb}')
    require(sha256_file(nb) == c['notebook']['sha256'], 'Notebook copy SHA differs from the registered SHA')
    text = OC.cell_text(nb, c['notebook']['cell_index'])
    require(hashlib.sha256(text.encode('utf-8')).hexdigest() == c['notebook']['cell_source_sha256'], 'Cell 25 text SHA differs')
    extracted = Path(source_copy)/'cell25.py'
    if extracted.exists():
        require(sha256_file(extracted) == c['notebook']['cell_source_sha256'], 'Stored cell extract SHA differs')
    ns, ev = OC.extract(text)
    require(ev['extract_text_sha256'] == c['notebook']['extract_text_sha256'], 'AST extract SHA differs from the registered SHA')
    require(ev['functions'] == c['notebook']['extract_functions'] or sorted(ev['functions']) == sorted(c['notebook']['extract_functions']),
            'Extracted function set differs')
    return ns, dict(notebook_sha256=c['notebook']['sha256'], cell_source_sha256=c['notebook']['cell_source_sha256'],
                    **ev)


def matrix_checks(A_tr, A_va, tables, c):
    cols = list(A_tr.columns)
    require(cols == c['features']['columns'] and list(A_va.columns) == cols and len(cols) == 75, '75-column order differs')
    require([cols.index(x) + 1 for x in F.CAT_COLS] == c['features']['categorical_positions_1based'], 'Categorical positions')
    numeric = [x for x in cols if x not in F.CAT_COLS]
    nan_train = int(A_tr[numeric].isna().sum().sum())
    nan_valid = int(A_va[numeric].isna().sum().sum())
    require(nan_train == 0 and nan_valid == 0, f'Unexpected missing numeric cells: {nan_train}, {nan_valid}')
    require(int(A_tr[F.CAT_COLS].isna().sum().sum()) == 0, 'Training categorical cells cannot be missing')
    return dict(columns=75, nan_numeric_train=nan_train, nan_numeric_valid=nan_valid,
                valid_unseen_category_cells=int(A_va[F.CAT_COLS].isna().sum().sum()), tables=F.table_summary(tables))


def preflight(data, destination, source_copy=None, local_root=None):
    """No study classifier is fitted.  Real-data checks on fold 0 plus bit-identity against the notebook functions.

    The destination folder is created by the first write; a failed preflight leaves `preflight.json` (status FAIL) there and
    a rerun needs a new folder name (existing results are never overwritten)."""
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
    res['inputs'] = input_checks(X, y, ids, folds, contract, c)
    res['comparators'] = comparator_check(c, local_root, ids, y, folds)
    frame = F.notebook_frame(X, y)
    a, b = np.flatnonzero(folds != 0), np.flatnonzero(folds == 0)
    train, valid = frame.iloc[a].reset_index(drop=True), frame.iloc[b].drop(columns=[F.GRADE]).reset_index(drop=True)
    y0 = train[F.GRADE].to_numpy() - 1
    tr_pos, va_pos = inner_split(y0)
    exp = c['inner_split']['expected_rows']
    res['inner_split_fold0'] = dict(train=len(tr_pos), val=len(va_pos), matches_registered=bool(len(tr_pos) == exp['train'] and len(va_pos) == exp['val']))
    t0 = time.perf_counter()
    with threadpool_limits(limits=1):
        A_tr, A_va, tables = F.build_features(train, valid)
        B_tr, B_va, B_tables = F.features_B(train, tr_pos, va_pos)
    res['matrix_A_fold0'] = dict(matrix_checks(A_tr, A_va, tables, c), rows=dict(train=len(A_tr), valid=len(A_va)),
                                 build_and_B_seconds=time.perf_counter() - t0)
    res['matrix_B_fold0'] = dict(matrix_checks(B_tr, B_va, B_tables, c), rows=dict(train=len(B_tr), valid=len(B_va)))
    t0 = time.perf_counter()
    res['original_function_check'] = dict(
        A_order=OC.compare(ns, train, valid),
        B_order=OC.compare(ns, train.iloc[tr_pos].reset_index(drop=True), train.iloc[va_pos].drop(columns=[F.GRADE]).reset_index(drop=True)),
        seconds=time.perf_counter() - t0,
        note='notebook function objects (AST extract of the private copy) run in the frozen environment; equality of two '
             'implementations here does not prove equality in the notebook environment')
    with threadpool_limits(limits=1):
        res['label_probe_fold0'] = F.label_probe(train, frame.iloc[b].reset_index(drop=True), tr_pos, va_pos)  # outer frame WITH real labels (shuffled inside)
    res['lightgbm_probe'] = probe_lightgbm(c)
    exp_cat = {k2: float(v) for k2, v in c['lightgbm_categorical_defaults_expected'].items()}
    ok = (res['inner_split_fold0']['matches_registered'] and res['original_function_check']['A_order']['passes']
          and res['original_function_check']['B_order']['passes'] and res['label_probe_fold0']['passes']
          and res['lightgbm_probe']['categorical_effective'] == exp_cat and res['lightgbm_probe']['reload_max_abs_diff'] <= 1e-12
          and res['lightgbm_probe']['saved_iterations_equal_best'])
    res['status'] = 'PASS' if ok else 'FAIL'
    write(dest/'preflight.json', res)
    print(json.dumps({k2: res[k2] for k2 in ('status', 'inner_split_fold0', 'label_probe_fold0')}, indent=1, default=str))
    require(ok, 'Preflight failed')
    return res


# ---------------------------------------------------------------- prepare / verify
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
    for p in SOURCE:  # sealed sources must equal their committed blobs; unrelated untracked files are allowed
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
    inputs = input_checks(X, y, ids, folds, contract, c)
    comparators = comparator_check(c, local_root, ids, y, folds)
    out.mkdir(parents=True)
    (out/'backup').mkdir()
    design = dict(study_id=STUDY_ID, config=c, code_commit=head, source_sha256=sources(), plan_sha256=plan_sha,
                  notebook=nb_evidence, source_copy_dir=str(source_copy.resolve()), local_root=str(local_root.resolve()),
                  data_contract=contract, inputs=inputs, environment=environment(), comparators=comparators,
                  lightgbm_categorical_effective_defaults=pf['lightgbm_probe']['categorical_effective'],
                  preflight=dict(path=str(Path(preflight_path).resolve()), sha256=sha256_file(preflight_path), status=pf['status'],
                                 label_probe_fold0=pf['label_probe_fold0'], original_function_check={
                                     k2: pf['original_function_check'][k2]['passes'] for k2 in ('A_order', 'B_order')}),
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


@contextlib.contextmanager
def run_lock(out):
    """One writer at a time; a stale lock must be removed by hand after checking that its process is dead."""
    lock = Path(out)/'RUNNING.lock'
    require(not lock.exists(), f'Another run holds {lock}; remove only if that process is dead')
    with open(lock, 'x') as fh:
        fh.write(f'pid={os.getpid()} started={now()}\n')
    try:
        yield lock
    finally:
        lock.unlink(missing_ok=True)


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
    max_fits = json.loads((Path(out)/'design.json').read_text(encoding='utf-8'))['max_fits']
    units = ledger['units']
    return dict(complete=sum(e['status'] == 'complete' for e in units.values()), total_units=len(ARMS) * N_FOLDS,
                fits_started=fits_started(ledger), planned_fits=len(ARMS) * N_FOLDS * 2, max_fits=max_fits,
                recovered_units=recovered_units(ledger),
                running=[k for k, e in units.items() if e['status'].endswith('running')],
                es_done_waiting=[k for k, e in units.items() if e['status'] == 'es_done'],
                features_A_saved=sorted(ledger['features']))


# ---------------------------------------------------------------- pure decision rules (design 8.2)
def d1_label(arm_f1, ref_f1, rule):
    deltas = [float(a) - float(b) for a, b in zip(arm_f1, ref_f1)]
    mean = float(np.mean(deltas))
    wins, losses = int(sum(d > 0 for d in deltas)), int(sum(d < 0 for d in deltas))
    if mean >= rule['mean_delta'] and wins >= rule['fold_wins_min']:
        label = rule['advantage']
    elif mean <= -rule['mean_delta'] and losses >= rule['fold_wins_min']:
        label = rule['disadvantage']
    else:
        label = rule['otherwise']
    return dict(label=label, mean_delta=mean, fold_delta=deltas, fold_wins=wins, fold_losses=losses)


def d3_label(e1_f1, e2_f1, rule):
    deltas = [float(a) - float(b) for a, b in zip(e1_f1, e2_f1)]
    mean_abs, max_abs = abs(float(np.mean(deltas))), float(max(abs(d) for d in deltas))
    small = mean_abs <= rule['abs_mean_delta_max'] and max_abs <= rule['abs_fold_delta_max']
    return dict(label=rule['small'] if small else rule['noticeable'], mean_delta_e1_minus_e2=float(np.mean(deltas)),
                abs_mean_delta=mean_abs, max_abs_fold_delta=max_abs, fold_delta=deltas)


def d4_label(cv_mean, reference, rule):
    gap = abs(float(cv_mean) - float(reference))
    label = rule['labels'][0] if gap <= rule['near_max'] else rule['labels'][1] if gap <= rule['moderate_max'] else rule['labels'][2]
    return dict(label=label, abs_gap=gap, signed_gap=float(cv_mean) - float(reference), reference=float(reference))


# ---------------------------------------------------------------- finalize
def assemble(out, y, ids, folds, arm):
    out = Path(out)
    ledger = read_ledger(out)
    entries = [ledger['units'].get(key(arm, f)) for f in range(N_FOLDS)]
    require(all(e and e['status'] == 'complete' for e in entries), f'{arm} incomplete')
    d = out/arm
    if (d/'metrics.json').exists():
        m = json.loads((d/'metrics.json').read_text(encoding='utf-8'))
        require(sha256_file(d/'oof.csv.gz') == m['oof_sha256'] == sha256_file(out/'backup'/arm/'oof.csv.gz'),
                f'{arm}: assembled OOF changed since it was written')
        return m
    for e in entries:
        check_saved(out, e)
    parts = pd.concat([read_pred(d/f'fold_{f}_pred.csv.gz') for f in range(N_FOLDS)])
    oof = parts.set_index('building_id').loc[ids.to_numpy()].reset_index()
    require(len(oof) == len(ids) and oof.building_id.is_unique, 'OOF coverage changed')
    np.testing.assert_array_equal(oof.fold.to_numpy(), folds)
    np.testing.assert_array_equal(oof.y_true.to_numpy(), y.to_numpy())
    write_pred(d/'oof.csv.gz', oof)
    back = read_pred(d/'oof.csv.gz')
    np.testing.assert_array_equal(back[PROBS].to_numpy(), oof[PROBS].to_numpy())
    values = [e['macro_f1'] for e in entries]
    result = dict(study_id=STUDY_ID, arm=arm, cv_mean=float(np.mean(values)), cv_std=float(np.std(values, ddof=1)),
                  fold_macro_f1=values, n_trees=[e['n_trees'] for e in entries], oof_sha256=sha256_file(d/'oof.csv.gz'),
                  fit_seconds=float(sum(e['fit_seconds'] for e in entries)), finished_at_utc=now())
    write(d/'metrics.json', result)
    backup_files(out, arm, ['oof.csv.gz', 'metrics.json'])
    log(out, f'{arm} OOF assembled mean={result["cv_mean"]:.9f} sd={result["cv_std"]:.9f}')
    return result


def summarize(oof, X, c):
    y, p, fo = oof.y_true.to_numpy(), oof.y_pred.to_numpy(), oof.fold.to_numpy()
    fold_scores = [score(y[fo == f], p[fo == f]) for f in range(N_FOLDS)]
    values = [s['macro_f1'] for s in fold_scores]
    pooled = score(y, p)
    age = X.age.to_numpy() == 995
    geo1 = X.geo_level_1_id.astype(str).to_numpy()
    prof = dict(age995=score(y[age], p[age]) if age.any() else None)
    for g in c['profiles']['geo1_regions']:
        m = geo1 == str(g)
        prof[f'geo1_{g}'] = score(y[m], p[m]) if m.any() else None
    return dict(cv_mean=float(np.mean(values)), cv_std=float(np.std(values, ddof=1)), fold_macro_f1=values,
                fold_grade3_recall=[s['recall'][2] for s in fold_scores],
                fold_class_precision=[s['precision'] for s in fold_scores], fold_class_recall=[s['recall'] for s in fold_scores],
                fold_class_f1=[s['f1'] for s in fold_scores], fold_confusion=[s['confusion'] for s in fold_scores],
                pooled=pooled, accuracy=pooled['accuracy'], recall3=pooled['recall'][2], error31=pooled['error31'],
                error32=pooled['error32'], rate31=pooled['rate31'], rate32=pooled['rate32'],
                pred_share=(np.bincount(p - 1, minlength=3) / len(p)).tolist(), profiles=prof)


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
               note='Reproduction benchmark of the EY notebook pipeline on our common stratified folds. Descriptive labels only; '
                    'no significance, no causal or single-factor claims, no adoption or ranking of the teammate; not independent '
                    'performance (design 9, 12).', arms={}, comparators={}, comparisons={})
    oofs = {}
    for arm in ARMS:
        m = assemble(out, y, ids, folds, arm)
        oofs[arm] = read_pred(out/arm/'oof.csv.gz')
        np.testing.assert_array_equal(oofs[arm].building_id.to_numpy(), ids.to_numpy())
        es = [json.loads((out/arm/f'fold_{f}_es.json').read_text(encoding='utf-8')) for f in range(N_FOLDS)]
        rec = [json.loads((out/arm/f'fold_{f}.json').read_text(encoding='utf-8')) for f in range(N_FOLDS)]
        block = dict(summarize(oofs[arm], X, c), oof_sha256=m['oof_sha256'], n_trees=m['n_trees'],
                     cap_hit=[e['cap_hit'] for e in es], internal_validation_macro_f1=[e['internal_validation_macro_f1'] for e in es],
                     internal_minus_outer=[r['internal_minus_outer'] for r in rec],
                     es_fit_seconds=[e['fit_seconds'] for e in es], refit_seconds=[r['fit_seconds'] for r in rec],
                     feature_build_seconds=[r['features_A']['build_seconds'] for r in rec],
                     pred_share_by_fold=[r['pred_share'] for r in rec])
        block['mean_internal_validation_macro_f1'] = float(np.mean(block['internal_validation_macro_f1']))
        res['arms'][arm] = block
    yv = y.to_numpy()
    comps = {}
    for name, spec in c['comparators'].items():
        r = reference(Path(design['local_root'])/spec['path'], spec['sha256'], ids, y, folds)
        comps[name] = r
        res['comparators'][name] = dict(summarize(r, X, c), label=spec['label'], oof_sha256=spec['sha256'], judged=spec['judged'])
    for arm in ARMS:
        res['comparisons'][arm] = {}
        for name, spec in c['comparators'].items():
            a, b = res['arms'][arm], res['comparators'][name]
            d1 = d1_label(a['fold_macro_f1'], b['fold_macro_f1'], c['rules']['D1'])
            if not spec['judged']:
                d1['label'] = 'REFERENCE_ONLY'
            d1['safety_indicators_no_verdict'] = dict(
                recall3=dict(arm=a['recall3'], comparator=b['recall3'], delta=a['recall3'] - b['recall3']),
                error31=dict(arm=a['error31'], comparator=b['error31'], delta=a['error31'] - b['error31']),
                error32=dict(arm=a['error32'], comparator=b['error32'], delta=a['error32'] - b['error32']),
                rate31=dict(arm=a['rate31'], comparator=b['rate31']), rate32=dict(arm=a['rate32'], comparator=b['rate32']))
            d1['turnover'] = turnover(yv, comps[name].y_pred.to_numpy(), oofs[arm].y_pred.to_numpy())
            res['comparisons'][arm][name] = d1
    e1, e2 = res['arms']['E1'], res['arms']['E2']
    d3 = d3_label(e1['fold_macro_f1'], e2['fold_macro_f1'], c['rules']['D3'])
    same = [int(a == b) for a, b in zip(e1['n_trees'], e2['n_trees'])]
    bit = []
    for f in range(N_FOLDS):
        mask = folds == f
        bit.append(bool(e1['n_trees'][f] == e2['n_trees'][f]
                        and np.array_equal(oofs['E1'][PROBS].to_numpy()[mask], oofs['E2'][PROBS].to_numpy()[mask])))
    res['es_protocol_effect'] = dict(d3, n_trees_e1=e1['n_trees'], n_trees_e2=e2['n_trees'],
                                     n_trees_ratio_e2_over_e1=[b / a for a, b in zip(e1['n_trees'], e2['n_trees'])],
                                     folds_with_equal_n_trees=same,
                                     bit_equal_probabilities_where_n_trees_equal={f: bit[f] for f in range(N_FOLDS) if same[f]},
                                     internal_validation_macro_f1=dict(E1=e1['internal_validation_macro_f1'], E2=e2['internal_validation_macro_f1']),
                                     attribution_note='sum of label mixing, validation-feature construction and table base size; '
                                                      'not attributed to a single cause (design 3.4)')
    nb = c['notebook_reference']
    res['notebook_distance'] = dict(
        reference_single_split_test_macro_f1=nb['single_split_test_macro_f1'],
        reference_xgboost_75_baseline=nb['xgboost_75_baseline_test_macro_f1'],
        E1_label_vs_single_split=d4_label(e1['cv_mean'], nb['single_split_test_macro_f1'], c['rules']['D4']),
        numbers_only=dict(E1_minus_single_split=e1['cv_mean'] - nb['single_split_test_macro_f1'],
                          E2_minus_single_split=e2['cv_mean'] - nb['single_split_test_macro_f1'],
                          E1_minus_xgboost_baseline=e1['cv_mean'] - nb['xgboost_75_baseline_test_macro_f1'],
                          E2_minus_xgboost_baseline=e2['cv_mean'] - nb['xgboost_75_baseline_test_macro_f1']),
        notebook_recall_by_grade=nb['single_split_recall_by_grade'], comparability=nb['comparability'])
    res['preflight_label_probe_fold0'] = design['preflight']['label_probe_fold0']
    res['judgments'] = dict(
        D1={arm: {n: res['comparisons'][arm][n]['label'] for n in c['comparators']} for arm in ARMS},
        D3=d3['label'], D4_E1=res['notebook_distance']['E1_label_vs_single_split']['label'],
        representative_line='E2 (design 8.2); E1 and E2 are reported side by side')
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
    p.add_argument('--only', nargs='*', help='fit-all: restrict to units such as E1/0')
    a = p.parse_args()
    if a.action == 'preflight':
        preflight(a.data, a.out, a.source_copy, a.local_root)
    elif a.action == 'prepare':
        prepare(a.data, a.out, a.preflight, a.code_commit, a.source_copy, a.local_root)
    elif a.action == 'fit-all':
        fit_all(a.data, a.out, set(a.only) if a.only else None)
    elif a.action == 'finalize':
        finalize_all(a.data, a.out)
    else:
        print(json.dumps(status(a.out), indent=1))


if __name__ == '__main__':
    sys.exit(main())
