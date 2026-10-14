"""T10: hierarchical geo target encoding (single factor = representation of the 3 geo columns).

Each family keeps its one-hot baseline exactly except the geo one-hot blocks, which are replaced by
9 hierarchical target-encoding features (te1, te3, cnt for geo1, geo2, geo3):
  te_L = (n_L * rate_L + m * parent_L) / (n_L + m), m = 20; parent(geo1) = training global class rate,
  parent(geo2) = smoothed geo1 value, parent(geo3) = smoothed geo2 value; cnt_L = log1p(n_L).
  Composite keys (geo1, geo1|geo2, geo1|geo2|geo3). Unseen -> parent value, cnt 0.
Leakage control: training rows are encoded by inner 5-fold OOF on the outer training fold
(StratifiedKFold(5, shuffle=True, random_state=42)); validation rows use statistics from the
full outer training fold. Validation labels are never passed to the encoder.
  LR_TE   = LR-C pipeline, TE features -> StandardScaler (training fold), LogisticRegression lbfgs C=1.
  LGBM_TE = T4/T8 arm O preprocessor + LightGBM params, TE features unscaled.
Baselines (LR-C, LGBM one-hot O) are read-only pinned OOF files; nothing is refit.
The holdout is never read. Each fold is saved, hashed, reloaded and backed up immediately;
completed folds are never refit; automatic retries = 0. All outputs are private (outside Git).
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
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from preprocess import BINARY, CATEGORICAL, GEO, NUMERIC, TEXT, sha256_file  # noqa: E402
from experiments.age995.run import FLAG, PARAMS as LR_PARAMS, make_pipeline as make_C  # noqa: E402
from experiments.geo_group_cv.run import PROBS, load, require, score, write  # noqa: E402
from experiments.model_families import run as t4  # noqa: E402

HERE = Path(__file__).resolve().parent
STUDY_ID = 'T10_geo_hier_te_v1'
SCHEMES = ['stratified', 'geo2', 'geo1']
ARMS = ['LR_TE', 'LGBM_TE']
BASELINE_OF = {'LR_TE': 'LR_C', 'LGBM_TE': 'LGBM_O'}
FAMILY = {'LR_TE': 'LR', 'LGBM_TE': 'LGBM'}
LEVELS = ['geo1', 'geo2', 'geo3']
TE_NAMES = [f'{k}_{lvl}' for lvl in LEVELS for k in ('te1', 'te3', 'cnt')]
HEAD = len(NUMERIC) + len(BINARY) + 1  # numeric + binary + age_is_995 = 28 columns
SEP = '|'
N_TARGETS = 30
SOURCE = ['preprocess.py', 'baseline.py', 'scripts/audit_eda_oof.py', 'configs/data_contract.json',
          'experiments/age995/run.py', 'experiments/geo_group_cv/run.py', 'experiments/geo_ablation/audit.py',
          'experiments/model_families/run.py', 'experiments/model_families/config.json',
          'experiments/geo_hier_te/__init__.py', 'experiments/geo_hier_te/run.py',
          'experiments/geo_hier_te/config.json', 'docs/T10_GEO_HIER_TE_PLAN.md']


def now():
    return datetime.now(timezone.utc).isoformat()


def config(path=HERE/'config.json'):
    c = json.loads(Path(path).read_text(encoding='utf-8'))
    require(c['study_id'] == STUDY_ID and c['schemes'] == SCHEMES and c['n_folds'] == 5, 'Unplanned study/schemes')
    require(c['lgbm_params'] == t4.load_config()['models']['LGBM'], 'LightGBM params differ from T4 LGBM_C_v1')
    require(c['lr_params'] == LR_PARAMS, 'LR params differ from LR-C')
    require(c['geo_columns'] == list(GEO) and c['text_categorical_columns'] == list(TEXT)
            and list(GEO) + list(TEXT) == list(CATEGORICAL), 'Categorical column order changed')
    require(c['te']['feature_names'] == TE_NAMES and c['te']['smoothing_m'] == 20.0
            and c['te']['inner_oof'] == {'splitter': 'StratifiedKFold', 'n_splits': 5, 'shuffle': True,
                                         'random_state': 42, 'on': c['te']['inner_oof']['on']}
            and c['te']['classes_encoded'] == [1, 3] and not c['te']['raw_geo_ids_in_matrix'], 'TE design changed')
    require(c['study_fits'] == {a: SCHEMES for a in ARMS} and c['max_new_study_fits'] == N_TARGETS,
            'Unplanned fit plan')
    require(c['automatic_retries'] == 0 and not c['early_stopping'] and c['class_weight'] is None
            and c['sampling'] is None and not c['tuning'] and not c['threshold_or_decision_rule_change']
            and c['outer_parallel_jobs'] == 1 and c['lr_threadpool_limit'] == 1 and c['lgbm_threadpool_limit'] == 2
            and c['stop_on_lr_nonconvergence'], 'Unplanned run policy')
    return c


def environment():
    names = ['numpy', 'pandas', 'scipy', 'scikit-learn', 'lightgbm', 'joblib', 'threadpoolctl']
    return dict(python=platform.python_version(), **{n: importlib.metadata.version(n) for n in names})


def sources():
    return {p: sha256_file(ROOT/p) for p in SOURCE}


def targets():
    c = config()
    return [(s, a, f) for a in ARMS for s in c['study_fits'][a] for f in range(5)]


# ---------------------------------------------------------------- hierarchical target encoding
def geo_keys(G):
    """Composite keys per level: geo1, geo1|geo2, geo1|geo2|geo3 (object arrays)."""
    require(isinstance(G, pd.DataFrame) and list(G.columns) == list(GEO), 'TE input must be exactly the 3 geo columns')
    parts = [G[c].astype(str) for c in GEO]
    for p in parts:
        require(not p.str.contains(SEP, regex=False).any(), 'Separator inside a geo value')
    k1 = parts[0]
    k2 = k1 + SEP + parts[1]
    k3 = k2 + SEP + parts[2]
    return [k.to_numpy(dtype=object) for k in (k1, k2, k3)]


def fit_stats(G, y, m):
    """Smoothed hierarchical rates from (G, y) only. Returns prior and one table per level."""
    y = np.asarray(y).astype(np.int64)
    require(len(y) == len(G) and len(y) > 0 and np.isin(y, [1, 2, 3]).all(), 'Invalid TE labels')
    prior = np.array([np.mean(y == 1), np.mean(y == 3)], dtype=np.float64)
    keys = geo_keys(G)
    levels, prev = [], None
    for lvl, k in enumerate(keys):
        frame = pd.DataFrame({'k': k, 's1': (y == 1).astype(np.float64), 's3': (y == 3).astype(np.float64)})
        g = frame.groupby('k', sort=True).agg(n=('s1', 'size'), s1=('s1', 'sum'), s3=('s3', 'sum'))
        if lvl == 0:
            parent = np.tile(prior, (len(g), 1))
        else:
            pk = np.array([key.rsplit(SEP, 1)[0] for key in g.index], dtype=object)
            idx = prev.index.get_indexer(pk)
            require((idx >= 0).all(), 'Child key without parent in the same data')
            parent = prev[['te1', 'te3']].to_numpy()[idx]
        n = g['n'].to_numpy(dtype=np.float64)
        table = pd.DataFrame({'n': g['n'].to_numpy(dtype=np.int64),
                              'te1': (g['s1'].to_numpy() + m*parent[:, 0])/(n + m),
                              'te3': (g['s3'].to_numpy() + m*parent[:, 1])/(n + m)}, index=g.index)
        levels.append(table)
        prev = table
    return dict(prior=prior, levels=levels, m=float(m), n=int(len(y)))


def encode(stats, G, return_unseen=False):
    """9 float64 features per row; unseen key at a level -> parent value and count 0."""
    keys = geo_keys(G)
    n = len(G)
    out = np.empty((n, len(TE_NAMES)), dtype=np.float64)
    parent = np.tile(stats['prior'], (n, 1))
    unseen = {}
    for lvl, k in enumerate(keys):
        table = stats['levels'][lvl]
        idx = table.index.get_indexer(k)
        seen = idx >= 0
        te = parent.copy()
        te[seen] = table[['te1', 'te3']].to_numpy()[idx[seen]]
        cnt = np.zeros(n, dtype=np.float64)
        cnt[seen] = table['n'].to_numpy()[idx[seen]]
        out[:, 3*lvl] = te[:, 0]
        out[:, 3*lvl + 1] = te[:, 1]
        out[:, 3*lvl + 2] = np.log1p(cnt)
        unseen[LEVELS[lvl]] = int((~seen).sum())
        parent = te
    return (out, unseen) if return_unseen else out


def inner_fold_ids(y, n_splits=5, seed=42):
    y = np.asarray(y)
    folds = np.full(len(y), -1, dtype=np.int64)
    for i, (_, v) in enumerate(StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(np.zeros(len(y)), y)):
        folds[v] = i
    require((folds >= 0).all(), 'Inner fold assignment incomplete')
    return folds


def oof_encode(G, y, folds, m, return_unseen=False):
    """Each row encoded from statistics of the other inner folds only (prior and counts included)."""
    y = np.asarray(y)
    out = np.empty((len(G), len(TE_NAMES)), dtype=np.float64)
    unseen = {lvl: 0 for lvl in LEVELS}
    for i in np.unique(folds):
        tr, va = folds != i, folds == i
        enc, u = encode(fit_stats(G.iloc[np.flatnonzero(tr)], y[tr], m), G.iloc[np.flatnonzero(va)], True)
        out[va] = enc
        for lvl in LEVELS:
            unseen[lvl] += u[lvl]
    return (out, unseen) if return_unseen else out


class HierGeoTargetEncoder(BaseEstimator, TransformerMixin):
    """fit_transform(X, y) -> inner-OOF encodings of the training rows; transform(X) -> full-fit statistics.

    Requires y at fit time; never sees validation labels (transform takes no y).
    """
    def __init__(self, m=20.0, inner_splits=5, inner_seed=42):
        self.m = m
        self.inner_splits = inner_splits
        self.inner_seed = inner_seed

    def fit(self, X, y=None):
        require(y is not None, 'Target encoding needs training labels')
        self.stats_ = fit_stats(X, y, self.m)
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        self.inner_folds_ = inner_fold_ids(y, self.inner_splits, self.inner_seed)
        out, unseen = oof_encode(X, np.asarray(y), self.inner_folds_, self.m, return_unseen=True)
        self.train_oof_unseen_ = unseen
        self.train_oof_mean_ = out.mean(0).tolist()
        self.train_oof_std_ = out.std(0).tolist()
        return out.astype(np.float32)

    def transform(self, X):
        check_is_fitted(self, 'stats_')
        return encode(self.stats_, X).astype(np.float32)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(TE_NAMES, dtype=object)


def reference_encode(G_train, y_train, G_query, m):
    """Brute-force dictionary reference (independent of the vectorized code) for checks."""
    y_train = np.asarray(y_train).astype(int)
    rows = [tuple(map(str, r)) for r in G_train[GEO].itertuples(index=False, name=None)]
    n = len(rows)
    prior = (sum(v == 1 for v in y_train)/n, sum(v == 3 for v in y_train)/n)
    acc = [{}, {}, {}]
    for r, v in zip(rows, y_train):
        for lvl in range(3):
            s = acc[lvl].setdefault(r[:lvl + 1], [0, 0, 0])
            s[0] += 1
            s[1] += v == 1
            s[2] += v == 3
    out = []
    for r in (tuple(map(str, q)) for q in G_query[GEO].itertuples(index=False, name=None)):
        parent, row = prior, []
        for lvl in range(3):
            cnt, s1, s3 = acc[lvl].get(r[:lvl + 1], [0, 0, 0])
            te = ((s1 + m*parent[0])/(cnt + m), (s3 + m*parent[1])/(cnt + m))
            row += [te[0], te[1], float(np.log1p(cnt))]
            parent = te
        out.append(row)
    return np.asarray(out, dtype=np.float64)


def tiny_case():
    """Tiny synthetic hierarchy with hand-computed expected encodings (m = 20)."""
    G = pd.DataFrame({GEO[0]: ['1', '1', '1', '1', '2', '2'], GEO[1]: ['10', '10', '11', '11', '20', '20'],
                      GEO[2]: ['100', '100', '110', '111', '200', '201']})
    y = np.array([1, 3, 2, 1, 3, 3])
    Q = pd.DataFrame({GEO[0]: ['1', '1', '1', '9', '2'], GEO[1]: ['11', '11', '19', '90', '10'],
                      GEO[2]: ['111', '999', '190', '900', '100']})
    m = 20.0
    p1, p3 = 2/6, 3/6
    g1 = {'1': ((2 + m*p1)/(4 + m), (1 + m*p3)/(4 + m)), '2': ((0 + m*p1)/(2 + m), (2 + m*p3)/(2 + m))}
    g2_11 = ((1 + m*g1['1'][0])/(2 + m), (0 + m*g1['1'][1])/(2 + m))
    g3_111 = ((1 + m*g2_11[0])/(1 + m), (0 + m*g2_11[1])/(1 + m))
    L = np.log1p
    expected = np.array([
        [*g1['1'], L(4), *g2_11, L(2), *g3_111, L(1)],          # fully seen
        [*g1['1'], L(4), *g2_11, L(2), *g2_11, 0.0],            # unseen geo3 -> geo2 value
        [*g1['1'], L(4), *g1['1'], 0.0, *g1['1'], 0.0],         # unseen geo2 -> geo1 value
        [p1, p3, 0.0, p1, p3, 0.0, p1, p3, 0.0],                # unseen geo1 -> global prior
        [*g1['2'], L(2), *g1['2'], 0.0, *g1['2'], 0.0],         # child ids of another parent: composite key unseen
    ])
    return G, y, Q, m, expected


# ---------------------------------------------------------------- arms
def te_step(arm, m):
    enc = HierGeoTargetEncoder(m=m, inner_splits=5, inner_seed=42)
    if arm == 'LR_TE':
        return Pipeline([('te', enc), ('scale', StandardScaler())])
    return enc


def patch(pre, arm, m):
    """Replace only the C geo one-hot with the TE step; text categoricals keep the same OneHotEncoder."""
    require(arm in ARMS, 'Unknown arm')
    columns = pre['columns']
    require([n for n, _, _ in columns.transformers] == ['numeric', 'binary', 'categorical'], 'Unexpected C paths')
    require(columns.transformers[0][2] == NUMERIC and columns.transformers[1][2] == BINARY+[FLAG],
            'Noncategorical inputs changed')
    name, ohe, cols = columns.transformers[2]
    require(cols == CATEGORICAL and isinstance(ohe, OneHotEncoder), 'C categorical path changed')
    columns.transformers = columns.transformers[:2] + [('geo_te', te_step(arm, m), list(GEO)), (name, ohe, list(TEXT))]
    return pre


def make_lr_pipeline(m=20.0):
    pipe = make_C('C')
    patch(pipe['preprocess'], 'LR_TE', m)
    return pipe


def make_lgbm_preprocessor(m=20.0):
    return patch(make_C('C')['preprocess'], 'LGBM_TE', m)


def te_encoder(pre):
    step = pre['columns'].named_transformers_['geo_te']
    return step['te'] if isinstance(step, Pipeline) else step


def check_matrix(pre, matrix):
    """CSR float32, finite; TE block at columns 28..36; no raw geo id or geo one-hot column."""
    names = pre['columns'].get_feature_names_out().tolist()
    require(sparse.isspmatrix_csr(matrix) and matrix.dtype == np.float32, 'Matrix must be float32 CSR')
    require(np.isfinite(matrix.data).all(), 'Nonfinite feature')
    require(matrix.shape[1] == len(names), 'Names/columns mismatch')
    require(names[HEAD:HEAD + len(TE_NAMES)] == ['geo_te__' + n for n in TE_NAMES], 'TE block position changed')
    require(not any('geo_level' in n for n in names), 'Raw geo column present')
    require(all(n.startswith('categorical__') for n in names[HEAD + len(TE_NAMES):]), 'Unexpected tail columns')
    return names


def same_blocks_as_C(pre_c, m_c, m_te, names_te):
    """Head (numeric/binary/flag) and text one-hot blocks equal to the C matrix (sparse compare, no .toarray())."""
    names_c = pre_c['columns'].get_feature_names_out().tolist()
    start = next(i for i, n in enumerate(names_c) if n.startswith('categorical__' + TEXT[0]))
    require(names_c[start:] == names_te[HEAD + len(TE_NAMES):], 'Text one-hot vocabulary differs from C')
    head = (m_c[:, :HEAD] != m_te[:, :HEAD]).nnz == 0
    text = (m_c[:, start:] != m_te[:, HEAD + len(TE_NAMES):]).nnz == 0
    return bool(head and text), int(start - HEAD)


def proba(model, matrix, arm):
    expected = [1, 2, 3] if arm == 'LR_TE' else [0, 1, 2]
    require(np.array_equal(model.classes_, expected), 'Unexpected class order')
    p = np.asarray(model.predict_proba(matrix), dtype=np.float64)
    require(p.shape == (matrix.shape[0], 3) and np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all(),
            'Invalid probabilities')
    np.testing.assert_allclose(p.sum(1), 1.0, atol=1e-6, rtol=0)
    return p


def fit_arm(arm, X_train, y_train, X_valid, c):
    """Fit encoder/preprocessor/model on the training fold only. Validation labels are not an argument."""
    from lightgbm import LGBMClassifier
    m = c['te']['smoothing_m']
    if arm == 'LR_TE':
        pipe = make_lr_pipeline(m)
        pre, model, threads = pipe['preprocess'], pipe['model'], c['lr_threadpool_limit']
        require(model.get_params()['C'] == 1.0 and model.get_params()['solver'] == 'lbfgs', 'LR params changed')
    else:
        pre = make_lgbm_preprocessor(m)
        model, threads = LGBMClassifier(**c['lgbm_params']), c['lgbm_threadpool_limit']
        pipe = Pipeline([('preprocess', pre), ('model', model)])
    with threadpool_limits(limits=threads):
        t0 = time.perf_counter()
        mt = pre.fit_transform(X_train, y_train)
        mv = pre.transform(X_valid)
        pre_seconds = time.perf_counter() - t0
    names = check_matrix(pre, mt)
    check_matrix(pre, mv)
    yt = np.asarray(y_train)
    with threadpool_limits(limits=threads), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        t0 = time.perf_counter()
        if arm == 'LR_TE':
            model.fit(mt, yt)
        else:
            model.fit(mt, yt.astype(np.int32) - 1)  # No eval_set / weights / callbacks.
        fit_seconds = time.perf_counter() - t0
        p = proba(model, mv, arm)
        p_train = proba(model, mt, arm)
    enc = te_encoder(pre)
    _, valid_unseen = encode(enc.stats_, X_valid[GEO], return_unseen=True)
    valid_te = enc.transform(X_valid[GEO]).astype(np.float64)
    info = dict(fit_seconds=fit_seconds, preprocessing_seconds=pre_seconds, encoded_columns=int(mt.shape[1]),
                feature_names_sha256=hashlib.sha256('\n'.join(names).encode()).hexdigest(),
                warnings=[dict(category=w.category.__name__, message=str(w.message)) for w in caught],
                age995_median=pre['age995'].median_,
                te=dict(prior=enc.stats_['prior'].tolist(), m=enc.stats_['m'], fit_rows=enc.stats_['n'],
                        levels_seen={lvl: int(len(t)) for lvl, t in zip(LEVELS, enc.stats_['levels'])},
                        inner_fold_sizes=np.bincount(enc.inner_folds_).tolist(),
                        train_oof_unseen=enc.train_oof_unseen_, train_oof_mean=enc.train_oof_mean_,
                        train_oof_std=enc.train_oof_std_, valid_unseen=valid_unseen,
                        valid_mean=valid_te.mean(0).tolist(), valid_std=valid_te.std(0).tolist()))
    if arm == 'LR_TE':
        info['converged'] = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
        info['n_iter'] = model.n_iter_.tolist()
        info['te_coef'] = model.coef_[:, HEAD:HEAD + len(TE_NAMES)].tolist()
        info['te_scaler_mean'] = pre['columns'].named_transformers_['geo_te']['scale'].mean_.tolist()
        np.testing.assert_allclose(info['te_scaler_mean'], enc.train_oof_mean_, rtol=0, atol=1e-6)
    else:
        require(model.booster_.current_iteration() == c['lgbm_params']['n_estimators'],
                'Full fixed boosting budget not completed')
        info['rounds_completed'] = int(model.booster_.current_iteration())
        gain = model.booster_.feature_importance(importance_type='gain')
        info['te_gain_share'] = float(gain[HEAD:HEAD + len(TE_NAMES)].sum() / gain.sum()) if gain.sum() else None
        info['te_gain'] = dict(zip(TE_NAMES, map(float, gain[HEAD:HEAD + len(TE_NAMES)])))
    return pipe, pre, model, p, p_train, mt, mv, names, info


# ---------------------------------------------------------------- io helpers
def atomic_write(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    write(tmp, obj)
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
        fh.write(json.dumps(dict(at_utc=now(), pid=os.getpid(), **kw), ensure_ascii=False) + '\n')


def read_ledger(out):
    return json.loads((Path(out)/'ledger.json').read_text(encoding='utf-8'))


def key(scheme, arm, fold):
    return f'{scheme}/{arm}/{fold}'


def check_saved(out, entry):
    for name, digest in entry['artifact_sha256'].items():
        require(sha256_file(Path(out)/entry['dir']/name) == digest, f'Artifact changed: {entry["dir"]}/{name}')
        require(sha256_file(Path(out)/'backup'/entry['dir']/name) == digest, f'Backup changed: {entry["dir"]}/{name}')


def reference(path, digest, ids, y, folds):
    require(sha256_file(path) == digest, f'Reference OOF changed: {path}')
    r = read_pred(path).set_index('building_id').loc[np.asarray(ids)].reset_index()
    require(len(r) == len(ids) and r.building_id.is_unique, 'Reference OOF coverage')
    np.testing.assert_array_equal(r.fold.to_numpy(), folds)
    np.testing.assert_array_equal(r.y_true.to_numpy(), np.asarray(y))
    return r


def fold_means(r):
    return [score(r.y_true[r.fold == f], r.y_pred[r.fold == f])['macro_f1'] for f in range(5)]


# ---------------------------------------------------------------- control check
def control_check(data, destination):
    """Not a study fit: TE correctness/leakage checks on a ~5k-row stratified subsample + tiny synthetic case."""
    c = config()
    cc = c['control_check']
    dest = Path(destination)
    require(not dest.exists(), 'Do not overwrite an existing control check')
    require(environment() == c['environment'], 'Environment differs from the pinned environment')
    m, atol, seed = c['te']['smoothing_m'], cc['reference_atol'], cc['subsample_seed']
    res = dict(study_id=STUDY_ID, kind=cc['kind'], counted_as_study_fit=False, created_at_utc=now())
    # (a1) tiny hand-computed hierarchy
    G, yy, Q, mm, expected = tiny_case()
    got = encode(fit_stats(G, yy, mm), Q)
    res['a_hand_computed_tiny'] = dict(max_abs_diff=float(np.max(np.abs(got - expected))),
                                       pass_=bool(np.max(np.abs(got - expected)) <= atol))
    # subsample
    X, y, ids, old, contract = load(data)
    sub, _ = train_test_split(np.arange(len(y)), train_size=cc['subsample_rows'], stratify=y, random_state=seed)
    sub = np.sort(sub)
    Xs, ys = X.iloc[sub].reset_index(drop=True), y.iloc[sub].reset_index(drop=True)
    a, b = train_test_split(np.arange(len(ys)), test_size=0.2, stratify=ys, random_state=seed)
    a, b = np.sort(a), np.sort(b)
    Ga, Gb, ya = Xs.iloc[a][GEO].reset_index(drop=True), Xs.iloc[b][GEO].reset_index(drop=True), ys.iloc[a].to_numpy()
    # (a2) vectorized vs brute-force reference
    stats = fit_stats(Ga, ya, m)
    folds = inner_fold_ids(ya)
    v_diff = float(np.max(np.abs(encode(stats, Gb) - reference_encode(Ga, ya, Gb, m))))
    base = oof_encode(Ga, ya, folds, m)
    ref_oof = np.empty_like(base)
    for i in range(5):
        tr, va = np.flatnonzero(folds != i), np.flatnonzero(folds == i)
        ref_oof[va] = reference_encode(Ga.iloc[tr], ya[tr], Ga.iloc[va], m)
    t_diff = float(np.max(np.abs(base - ref_oof)))
    enc = HierGeoTargetEncoder(m=m).fit(Ga, ya)
    tr32 = enc.fit_transform(Ga, ya)
    cast_ok = bool(np.array_equal(tr32, base.astype(np.float32)) and np.array_equal(enc.inner_folds_, folds)
                   and np.array_equal(enc.transform(Gb), encode(stats, Gb).astype(np.float32)))
    res['a_bruteforce_reference'] = dict(subsample_rows=len(sub), train_rows=len(a), valid_rows=len(b),
                                         valid_max_abs_diff=v_diff, train_oof_max_abs_diff=t_diff,
                                         transformer_float32_cast_consistent=cast_ok,
                                         pass_=bool(v_diff <= atol and t_diff <= atol and cast_ok))
    # (b) inner-OOF rows never use their own label (fixed inner assignment)
    rng = np.random.default_rng(seed)
    rows = rng.choice(len(a), 25, replace=False)
    own_same, others_changed, single_same = [], [], []
    for i in rows:
        mask = folds == folds[i]
        y2 = ya.copy()
        y2[mask] = (y2[mask] % 3) + 1  # change every label in the row's inner fold
        e2 = oof_encode(Ga, y2, folds, m)
        own_same.append(bool(np.array_equal(e2[mask], base[mask])))
        others_changed.append(bool(not np.array_equal(e2[~mask], base[~mask])))
        y3 = ya.copy()
        y3[i] = (y3[i] % 3) + 1
        single_same.append(bool(np.array_equal(oof_encode(Ga, y3, folds, m)[i], base[i])))
    res['b_inner_oof_no_self_label'] = dict(rows_checked=len(rows), own_fold_unchanged=all(own_same),
                                            single_row_label_flip_own_unchanged=all(single_same),
                                            nonvacuous_other_folds_changed=all(others_changed),
                                            pass_=bool(all(own_same) and all(single_same) and all(others_changed)))
    # (c) validation encodings independent of validation labels (permute them in the full label vector)
    yperm = ys.copy()
    yperm.iloc[b] = rng.permutation(ys.iloc[b].to_numpy())
    require(not np.array_equal(yperm.iloc[b].to_numpy(), ys.iloc[b].to_numpy()), 'Permutation must change labels')
    c_out = {}
    for arm, build in [('LR_TE', lambda: make_lr_pipeline(m)['preprocess']), ('LGBM_TE', lambda: make_lgbm_preprocessor(m))]:
        mats = []
        for labels in (ys, yperm):
            pre = build()
            with threadpool_limits(limits=1):
                mt = pre.fit_transform(Xs.iloc[a], labels.iloc[a])
                mv = pre.transform(Xs.iloc[b])
            check_matrix(pre, mt)
            names = check_matrix(pre, mv)
            mats.append((mt, mv))
        same_valid = (mats[0][1] != mats[1][1]).nnz == 0
        same_train = (mats[0][0] != mats[1][0]).nnz == 0
        c_out[arm] = dict(valid_matrix_unchanged=bool(same_valid), train_matrix_unchanged=bool(same_train),
                          columns=len(names), no_raw_geo_columns=True)
    res['c_validation_label_independence'] = dict(arms=c_out, pass_=all(v['valid_matrix_unchanged'] and
                                                                        v['train_matrix_unchanged'] for v in c_out.values()))
    # smoke fits on the subsample (not study fits) + block equality with C
    smoke = {}
    for arm in ARMS:
        pipe, pre, model, p, p_train, mt, mv, names, info = fit_arm(arm, Xs.iloc[a], ys.iloc[a], Xs.iloc[b], c)
        pre_c = make_C('C')['preprocess']
        with threadpool_limits(limits=1):
            pre_c.fit(Xs.iloc[a])
            eq_v, _ = same_blocks_as_C(pre_c, pre_c.transform(Xs.iloc[b]), mv, names)
            eq_t, _ = same_blocks_as_C(pre_c, pre_c.transform(Xs.iloc[a]), mt, names)
        pred = p.argmax(1) + 1
        smoke[arm] = dict(sample_macro_f1=score(ys.iloc[b], pred)['macro_f1'], columns=int(mt.shape[1]),
                          blocks_equal_C=bool(eq_v and eq_t), converged=info.get('converged'),
                          fit_seconds=info['fit_seconds'], warnings=info['warnings'], te=info['te'])
    res['smoke_fits'] = dict(note='sample fits on 4k subsample rows; not study fits', arms=smoke,
                             pass_=all(v['blocks_equal_C'] and v['converged'] is not False for v in smoke.values()))
    res['classifiers_fitted_sample_only'] = 2
    res['environment'] = environment()
    res['holdout_accessed'] = False
    res['status'] = 'PASS' if all(v['pass_'] for v in res.values() if isinstance(v, dict) and 'pass_' in v) else 'FAIL'
    dest.mkdir(parents=True)
    write(dest/'control_check.json', res)
    print(json.dumps({k: (v['pass_'] if isinstance(v, dict) and 'pass_' in v else v) for k, v in res.items()
                      if k not in ('smoke_fits',)}, indent=1, default=str))
    print(json.dumps({k: dict(f1=v['sample_macro_f1'], cols=v['columns'], conv=v['converged']) for k, v in smoke.items()}))
    require(res['status'] == 'PASS', 'Control check failed')
    return res


# ---------------------------------------------------------------- prepare/verify
def baseline_paths(t8_run, t4_oof, lr_oof):
    return {'LR_C': {s: str(Path(lr_oof[s]).resolve()) for s in SCHEMES},
            'LGBM_O': {'stratified': str(Path(t4_oof).resolve()),
                       **{s: str((Path(t8_run)/s/'O'/'oof.csv.gz').resolve()) for s in ['geo2', 'geo1']}}}


def unseen_counts(X, folds):
    """Validation rows whose geo key (composite) is absent from the outer training fold; labels not used."""
    keys = geo_keys(X[GEO])
    unseen = {}
    for s, fo in folds.items():
        unseen[s] = []
        for f in range(5):
            tr, va = fo != f, fo == f
            unseen[s].append({lvl: int((~np.isin(k[va], np.unique(k[tr]))).sum()) for lvl, k in zip(LEVELS, keys)}
                             | {'valid_rows': int(va.sum())})
    return unseen


def prepare(data, t8_run, t4_oof, lr_oof, output, code_commit, control):
    output = Path(output)
    require(not output.exists(), 'Do not overwrite an existing study')
    require(len(code_commit) == 40 and all(ch in '0123456789abcdef' for ch in code_commit), 'Full commit SHA required')
    c = config()
    require(environment() == c['environment'], 'Environment differs from the pinned environment')
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(['git', 'status', '--porcelain', '--', *SOURCE], cwd=ROOT, capture_output=True, text=True,
                           check=True).stdout.strip()
    require(head == code_commit and not dirty, 'Sealed sources must be committed at code_commit (HEAD) and clean')
    X, y, ids, old, contract = load(data)
    t8_run = Path(t8_run)
    shas = {'stratified': sha256_file(Path(data)/'split_manifest.csv')}
    for s in ['geo2', 'geo1']:
        shas[s] = sha256_file(t8_run/s/'manifest.csv')
    require(shas == c['manifest_sha256'], 'Split manifest SHA differs')
    cc = json.loads(Path(control).read_text(encoding='utf-8'))
    require(cc['status'] == 'PASS' and cc['study_id'] == STUDY_ID, 'Control check must PASS before prepare')
    paths = baseline_paths(t8_run, t4_oof, lr_oof)
    for s in ['geo2', 'geo1']:  # T8 O OOF must equal its audited metrics.json anchor
        meta = json.loads((t8_run/s/'O'/'metrics.json').read_text(encoding='utf-8'))
        require(meta['oof_sha256'] == c['baselines']['LGBM_O']['oof_sha256'][s], 'T8 O metrics anchor differs')
    output.mkdir(parents=True)
    folds = {'stratified': old}
    for s in ['geo2', 'geo1']:
        (output/s).mkdir()
        shutil.copyfile(t8_run/s/'manifest.csv', output/s/'manifest.csv')
        require(sha256_file(output/s/'manifest.csv') == shas[s], 'Copied manifest SHA differs')
        mf = pd.read_csv(output/s/'manifest.csv')
        np.testing.assert_array_equal(mf.building_id, ids)
        folds[s] = mf.fold.to_numpy()
    (output/'backup').mkdir()
    base = {}
    for name in ['LR_C', 'LGBM_O']:
        base[name] = {}
        for s in SCHEMES:
            r = reference(paths[name][s], c['baselines'][name]['oof_sha256'][s], ids, y, folds[s])
            vals = fold_means(r)
            base[name][s] = dict(path=paths[name][s], sha256=c['baselines'][name]['oof_sha256'][s],
                                 cv_mean=float(np.mean(vals)), fold_macro_f1=vals,
                                 pooled_grade3_recall=score(r.y_true, r.y_pred)['recall'][2])
    ref = c['lr_c_reference']
    require(abs(base['LR_C']['stratified']['cv_mean'] - ref['stratified_mean']) <= ref['tolerance_mean']
            and abs(base['LR_C']['geo2']['cv_mean'] - ref['geo2_mean']) <= ref['tolerance_mean']
            and abs(base['LR_C']['geo2']['pooled_grade3_recall'] - ref['geo2_pooled_grade3_recall']) <= ref['tolerance_recall'],
            'LR-C reference values differ from the registered thresholds')
    unseen = unseen_counts(X, folds)
    design = dict(study_id=STUDY_ID, config=c, code_commit=code_commit, source_sha256=sources(),
                  data_contract=contract, environment=environment(), manifest_sha256=shas,
                  t8_run=str(t8_run.resolve()), baselines=base,
                  control_check=dict(path=str(Path(control).resolve()), sha256=sha256_file(control), status=cc['status']),
                  valid_unseen_geo_keys=unseen,
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
    folds = {'stratified': old}
    for s in ['geo2', 'geo1']:
        path = output/s/'manifest.csv'
        require(sha256_file(path) == design['manifest_sha256'][s], 'Copied split changed')
        mf = pd.read_csv(path)
        np.testing.assert_array_equal(mf.building_id, ids)
        folds[s] = mf.fold.to_numpy()
    return X, y, ids, folds, design


# ---------------------------------------------------------------- one fold
def fit_fold(X, y, ids, folds, out, scheme, arm, f, design):
    out = Path(out)
    k = key(scheme, arm, f)
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
    entry = dict(dir=f'{scheme}/{arm}', scheme=scheme, arm=arm, fold=f, status='running', attempts=prior+1,
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
    a, b = np.flatnonzero(folds[scheme] != f), np.flatnonzero(folds[scheme] == f)
    require(len(a) + len(b) == len(ids) and len(b) > 0, 'Fold membership invalid')
    pipe, pre, model, p, p_train, mt, mv, names, info = fit_arm(arm, X.iloc[a], y.iloc[a], X.iloc[b], c)
    pred = p.argmax(1) + 1
    threads = c['lr_threadpool_limit'] if arm == 'LR_TE' else c['lgbm_threadpool_limit']
    # Same numeric/binary/flag and text one-hot values as the C preprocessor fitted on the same fold.
    pre_c = make_C('C')['preprocess']
    with threadpool_limits(limits=threads):
        pre_c.fit(X.iloc[a])
        eq_v, n_geo_ohe = same_blocks_as_C(pre_c, pre_c.transform(X.iloc[b]), mv, names)
        eq_t, _ = same_blocks_as_C(pre_c, pre_c.transform(X.iloc[a]), mt, names)
    require(eq_v and eq_t, 'Non-geo columns differ from the C preprocessor')
    require(info['te']['valid_unseen'] == {lvl: design['valid_unseen_geo_keys'][scheme][f][lvl] for lvl in LEVELS},
            'Unseen counts differ from design')
    preds = d/f'fold_{f}_pred.csv.gz'
    if arm == 'LR_TE':
        np.testing.assert_array_equal(pipe.predict(X.iloc[b]), pred)  # argmax == LR predict
        model_paths = [d/f'fold_{f}_model.joblib']
        joblib.dump(pipe, model_paths[0], compress=3)
    else:
        model_paths = [d/f'fold_{f}_model.txt', d/f'fold_{f}_preprocessor.joblib']
        model.booster_.save_model(str(model_paths[0]))
        joblib.dump(pre, model_paths[1], compress=3)
    frame = pd.DataFrame({'building_id': ids.iloc[b].to_numpy(), 'fold': f, 'y_true': y.iloc[b].to_numpy(),
                          'y_pred': pred, **{n: p[:, i] for i, n in enumerate(PROBS)}})
    write_pred(preds, frame)
    # reload verification (freshly generated trusted files only)
    with threadpool_limits(limits=threads):
        if arm == 'LR_TE':
            loaded = joblib.load(model_paths[0])
            p2 = np.asarray(loaded.predict_proba(X.iloc[b]), dtype=np.float64)
            require(np.array_equal(p2, p) and np.array_equal(loaded.predict(X.iloc[b]), pred), 'LR reload mismatch')
        else:
            import lightgbm
            booster = lightgbm.Booster(model_file=str(model_paths[0]))
            loaded_pre = joblib.load(model_paths[1])
            p2 = np.asarray(booster.predict(loaded_pre.transform(X.iloc[b]), num_threads=c['lgbm_params']['n_jobs']),
                            dtype=np.float64)
    reload_diff = float(np.max(np.abs(p2 - p)))
    require(reload_diff <= c['checkpoint']['model_reload_probability_atol_lgbm']
            and np.array_equal(p2.argmax(1) + 1, pred), f'Model reload mismatch {reload_diff}')
    back = read_pred(preds)
    np.testing.assert_array_equal(back.building_id.to_numpy(), frame.building_id.to_numpy())
    np.testing.assert_array_equal(back.y_pred.to_numpy(), pred)
    np.testing.assert_array_equal(back[PROBS].to_numpy(), p)
    record = dict(study_id=STUDY_ID, scheme=scheme, arm=arm, fold=f, train_rows=len(a), valid_rows=len(b),
                  train_class_counts=[int((y.iloc[a] == v).sum()) for v in (1, 2, 3)],
                  metrics=score(y.iloc[b], pred), train_macro_f1_oof_encoded=score(y.iloc[a], p_train.argmax(1)+1)['macro_f1'],
                  fit_seconds=info['fit_seconds'], preprocessing_seconds=info['preprocessing_seconds'],
                  encoded_columns=info['encoded_columns'], feature_names_sha256=info['feature_names_sha256'],
                  geo_onehot_columns_replaced=n_geo_ohe, non_geo_columns_equal_C=True,
                  age995_median=info['age995_median'], te=info['te'],
                  converged=info.get('converged'), n_iter=info.get('n_iter'), te_coef=info.get('te_coef'),
                  rounds_completed=info.get('rounds_completed'), te_gain_share=info.get('te_gain_share'),
                  te_gain=info.get('te_gain'), effective_parameters={k2: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                                                                      for k2, v in model.get_params().items()},
                  reload_probability_max_abs_diff=reload_diff, csv_round_trip_exact=True,
                  warnings=info['warnings'], finished_at_utc=now())
    write(d/f'fold_{f}.json', record)
    shas = {pth.name: sha256_file(pth) for pth in [*model_paths, preds, d/f'fold_{f}.json']}
    bd = out/'backup'/entry['dir']
    bd.mkdir(parents=True, exist_ok=True)
    for name in shas:
        require(not (bd/name).exists(), 'Backup already exists; refusing overwrite')
        shutil.copy2(d/name, bd/name)
    converged = info.get('converged', True)
    entry.update(status='complete' if converged else 'complete_nonconverged', artifact_sha256=shas,
                 macro_f1=record['metrics']['macro_f1'], fit_seconds=info['fit_seconds'], converged=converged,
                 finished_at_utc=now())
    check_saved(out, entry)
    ledger = read_ledger(out)
    ledger['folds'][k] = entry
    atomic_write(out/'ledger.json', ledger)
    event(out, fold=k, action='fit_complete', macro_f1=record['metrics']['macro_f1'], seconds=info['fit_seconds'],
          converged=converged, artifact_sha256=shas)
    log(out, f'{k} F1={record["metrics"]["macro_f1"]:.9f} fit={info["fit_seconds"]:.1f}s '
             f'iter={info.get("n_iter")} reload_diff={reload_diff:.3g} unseen={info["te"]["valid_unseen"]} '
             f'saved+verified+backed-up')
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


# ---------------------------------------------------------------- finalize/analysis
def assemble(out, y, ids, folds, scheme, arm):
    out = Path(out)
    ledger = read_ledger(out)
    entries = [ledger['folds'].get(key(scheme, arm, f)) for f in range(5)]
    require(all(e and e['status'] == 'complete' for e in entries), f'{scheme}/{arm} incomplete')
    d = out/scheme/arm
    if (d/'metrics.json').exists():
        return json.loads((d/'metrics.json').read_text(encoding='utf-8'))
    for e in entries:
        check_saved(out, e)
    parts = pd.concat([read_pred(d/f'fold_{f}_pred.csv.gz') for f in range(5)])
    oof = parts.set_index('building_id').loc[ids.to_numpy()].reset_index()
    require(len(oof) == len(ids) and oof.building_id.is_unique, 'OOF coverage changed')
    np.testing.assert_array_equal(oof.fold.to_numpy(), folds[scheme])
    np.testing.assert_array_equal(oof.y_true.to_numpy(), y.to_numpy())
    write_pred(d/'oof.csv.gz', oof)
    back = read_pred(d/'oof.csv.gz')
    np.testing.assert_array_equal(back[PROBS].to_numpy(), oof[PROBS].to_numpy())
    values = [e['macro_f1'] for e in entries]
    result = dict(study_id=STUDY_ID, scheme=scheme, arm=arm, cv_mean=float(np.mean(values)),
                  cv_std=float(np.std(values, ddof=1)), fold_macro_f1=values, oof_sha256=sha256_file(d/'oof.csv.gz'),
                  fit_seconds=float(sum(e['fit_seconds'] for e in entries)), finished_at_utc=now())
    write(d/'metrics.json', result)
    bd = out/'backup'/scheme/arm
    for name in ['oof.csv.gz', 'metrics.json']:
        require(not (bd/name).exists(), 'Backup already exists; refusing overwrite')
        shutil.copy2(d/name, bd/name)
    log(out, f'{scheme}/{arm} OOF assembled mean={result["cv_mean"]:.9f} sd={result["cv_std"]:.9f}')
    return result


def summary(oof, X, fit_seconds=None):
    y, p, fo = oof.y_true.to_numpy(), oof.y_pred.to_numpy(), oof.fold.to_numpy()
    fold_scores = [score(y[fo == f], p[fo == f]) for f in range(5)]
    values = [s['macro_f1'] for s in fold_scores]
    age = X.age.to_numpy() == 995
    geo1 = X.geo_level_1_id.astype(str).to_numpy()
    prof = dict(age995=score(y[age], p[age]))
    for g in config()['profiles']['geo1_regions']:
        msk = geo1 == str(g)
        prof[f'geo1_{g}'] = score(y[msk], p[msk])
    return dict(cv_mean=float(np.mean(values)), cv_std=float(np.std(values, ddof=1)), fold_macro_f1=values,
                fold_grade3_recall=[s['recall'][2] for s in fold_scores],
                fold_confusion=[s['confusion'] for s in fold_scores],
                pooled=score(y, p), profiles=prof, fit_seconds=fit_seconds)


def compare(block, a_name, b_name, y, a_oof, b_oof):
    from experiments.geo_ablation.audit import turnover
    deltas = [u - v for u, v in zip(block[a_name]['fold_macro_f1'], block[b_name]['fold_macro_f1'])]
    return dict(mean_delta=block[a_name]['cv_mean'] - block[b_name]['cv_mean'], fold_delta=deltas,
                fold_wins=int(sum(dd > 0 for dd in deltas)),
                pooled_grade3_recall_delta=block[a_name]['pooled']['recall'][2] - block[b_name]['pooled']['recall'][2],
                turnover=turnover(y, b_oof.y_pred.to_numpy(), a_oof.y_pred.to_numpy()))


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
    res = dict(study_id=STUDY_ID, design_sha256=sha256_file(out/'design.json'), code_commit=design['code_commit'],
               ledger=dict(complete=complete, total=N_TARGETS, attempts=attempts, retries=attempts-complete,
                           fit_seconds_sum=float(sum(e['fit_seconds'] for e in ledger['folds'].values()))),
               baseline_refits=0, holdout_accessed=False, rules=c['rules'],
               note='Two TE arms on the same folds (optimistic if selecting); descriptive, no significance or causal claims.',
               schemes={})
    yv = y.to_numpy()
    for s in SCHEMES:
        oofs = {name: reference(design['baselines'][name][s]['path'], design['baselines'][name][s]['sha256'],
                                ids, y, folds[s]) for name in ['LR_C', 'LGBM_O']}
        block = {name: dict(summary(oofs[name], X), oof_sha256=design['baselines'][name][s]['sha256'],
                            source='pinned baseline OOF (not refit)') for name in oofs}
        for arm in ARMS:
            mt = assemble(out, y, ids, folds, s, arm)
            oofs[arm] = read_pred(out/s/arm/'oof.csv.gz')
            np.testing.assert_array_equal(oofs[arm].building_id.to_numpy(), oofs['LR_C'].building_id.to_numpy())
            recs = [json.loads((out/s/arm/f'fold_{f}.json').read_text(encoding='utf-8')) for f in range(5)]
            block[arm] = dict(summary(oofs[arm], X, mt['fit_seconds']), oof_sha256=mt['oof_sha256'],
                              valid_unseen=[r['te']['valid_unseen'] for r in recs],
                              converged=[r['converged'] for r in recs] if arm == 'LR_TE' else None,
                              n_iter=[r['n_iter'] for r in recs] if arm == 'LR_TE' else None,
                              te_gain_share=[r['te_gain_share'] for r in recs] if arm == 'LGBM_TE' else None)
        block['valid_unseen_geo_keys'] = design['valid_unseen_geo_keys'][s]
        block['comparisons'] = {
            'LR_TE_vs_LR_C': compare(block, 'LR_TE', 'LR_C', yv, oofs['LR_TE'], oofs['LR_C']),
            'LGBM_TE_vs_LGBM_O': compare(block, 'LGBM_TE', 'LGBM_O', yv, oofs['LGBM_TE'], oofs['LGBM_O']),
            'LGBM_TE_vs_LR_C': compare(block, 'LGBM_TE', 'LR_C', yv, oofs['LGBM_TE'], oofs['LR_C']),
        }
        res['schemes'][s] = block
    st, g2 = res['schemes']['stratified'], res['schemes']['geo2']
    r1, r2 = c['rules']['generalization_candidate'], c['rules']['baseline_challenger']
    judg = {}
    for arm in ARMS:
        base = BASELINE_OF[arm]
        cs, cg = st['comparisons'][f'{arm}_vs_{base}'], g2['comparisons'][f'{arm}_vs_{base}']
        inputs1 = dict(geo2_mean_delta=cg['mean_delta'], geo2_fold_wins=cg['fold_wins'],
                       stratified_mean_delta=cs['mean_delta'], geo2_pooled_grade3_recall_delta=cg['pooled_grade3_recall_delta'])
        gen = bool(cg['mean_delta'] >= r1['geo2_mean_delta_min'] and cg['fold_wins'] >= r1['geo2_fold_wins_min']
                   and cs['mean_delta'] >= r1['stratified_mean_delta_min']
                   and cg['pooled_grade3_recall_delta'] >= r1['geo2_pooled_grade3_recall_delta_min'])
        vs = f'{arm}_vs_LR_C'
        inputs2 = dict(stratified_mean=st[arm]['cv_mean'], stratified_fold_wins_vs_LR_C=st['comparisons'][vs]['fold_wins'],
                       geo2_mean=g2[arm]['cv_mean'], geo2_fold_wins_vs_LR_C=g2['comparisons'][vs]['fold_wins'],
                       geo2_pooled_grade3_recall=g2[arm]['pooled']['recall'][2])
        chal = bool(inputs2['stratified_mean'] > r2['stratified_mean_gt']
                    and inputs2['stratified_fold_wins_vs_LR_C'] >= r2['stratified_fold_wins_vs_LR_C_min']
                    and inputs2['geo2_mean'] >= r2['geo2_mean_ge']
                    and inputs2['geo2_fold_wins_vs_LR_C'] >= r2['geo2_fold_wins_vs_LR_C_min']
                    and inputs2['geo2_pooled_grade3_recall'] >= r2['geo2_pooled_grade3_recall_ge'])
        judg[FAMILY[arm]] = dict(generalization_candidate=gen, generalization_inputs=inputs1,
                                 baseline_challenger_candidate=chal, challenger_inputs=inputs2)
    judg['geo1_scheme'] = 'descriptive only'
    res['judgments'] = judg
    res['finished_at_utc'] = now()
    write(out/'analysis.json', res)
    require(not (out/'backup'/'analysis.json').exists(), 'Backup already exists; refusing overwrite')
    shutil.copy2(out/'analysis.json', out/'backup'/'analysis.json')
    log(out, 'analysis written: ' + json.dumps({f: {k: v for k, v in j.items() if not k.endswith('inputs')}
                                                for f, j in judg.items() if isinstance(j, dict)}))
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['control-check', 'prepare', 'fit-all', 'finalize', 'status'])
    p.add_argument('--data', default='data')
    p.add_argument('--output')
    p.add_argument('--destination')
    p.add_argument('--t8-run')
    p.add_argument('--t4-oof')
    p.add_argument('--code-commit')
    p.add_argument('--control')
    for s in SCHEMES:
        p.add_argument(f'--lr-oof-{s}')
    a = p.parse_args()
    if a.action == 'control-check':
        control_check(a.data, a.destination)
    elif a.action == 'prepare':
        lr = {s: getattr(a, f'lr_oof_{s}') for s in SCHEMES}
        prepare(a.data, a.t8_run, a.t4_oof, lr, a.output, a.code_commit, a.control)
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
