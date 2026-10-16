"""T31 arm definitions: only the representation of geo_level_1/2/3_id differs between arms.

OH  = categorical one-hot (v1.0.1 OneHotEncoder; LGBM: T9 arm P = text native + geo one-hot)
NAT = LightGBM native categorical for geo and text (T8 arm N, dense), LGBM only
NUM = original integer value as float32 numeric columns (no vocabulary, no one-hot, no ordinal code)

Existing code (preprocess.py, age995, T8, T9) is imported unchanged. No full .toarray() anywhere.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from preprocess import BINARY, CATEGORICAL, GEO, NUMERIC, TEXT, make_preprocessor as make_v1  # noqa: E402
from experiments.age995.run import FLAG, make_pipeline as make_C  # noqa: E402
from experiments.lgbm_native_cat import run as t8  # noqa: E402
from experiments.lgbm_partial_native import run as t9  # noqa: E402

N_HEAD = len(NUMERIC) + len(BINARY) + 1          # 28: numeric 5 + binary 22 + age_is_995
EXACT_LIMIT = 2 ** 24
MODELS = ['LR', 'DT', 'GBC', 'RF', 'LGBM']
REPS = ['OH', 'NAT', 'NUM']
SKLEARN_MODELS = ['LR', 'DT', 'GBC', 'RF']


def arm_id(model, rep):
    return f'{model}_{rep}'


def arm_spec(arm):
    model, rep = arm.split('_', 1)
    if model not in MODELS or rep not in REPS or (rep == 'NAT' and model != 'LGBM'):
        raise ValueError(f'Unknown arm {arm}')
    return model, rep


def all_arms(config):
    return [arm_id(m, r) for m in config['model_order'] for r in config['arms_by_model'][m]]


# ------------------------------------------------------------------ stateless converters (picklable)
def geo_float32(values):
    """Geographic id (string or integer) -> exact float32 integer value. Stateless, no statistics."""
    frame = values if isinstance(values, pd.DataFrame) else pd.DataFrame(np.asarray(values))
    cols = [pd.to_numeric(frame.iloc[:, i], errors='raise').to_numpy(dtype=np.float64) for i in range(frame.shape[1])]
    arr = np.column_stack(cols) if cols else np.empty((len(frame), 0))
    if not (np.isfinite(arr).all() and (arr == np.floor(arr)).all() and (arr >= 0).all() and (arr < EXACT_LIMIT).all()):
        raise ValueError('Geographic ids must be nonnegative integers below 2**24 for exact float32 storage.')
    return arr.astype(np.float32)


def numeric_with_geo_float32(values):
    """LR NUM numeric block: the five numerics exactly as v1.0.1 (float32 cast) plus geo ids as exact integers."""
    frame = values if isinstance(values, pd.DataFrame) else pd.DataFrame(np.asarray(values))
    parts = []
    for col in frame.columns:
        if col in GEO:
            parts.append(geo_float32(frame[[col]]))
        else:
            parts.append(np.asarray(frame[[col]], dtype=np.float32))
    return np.hstack(parts)


def to_csr_float32(values):
    """Wrap an already-built small dense matrix as CSR (never a densification of a sparse matrix)."""
    return sparse.csr_matrix(np.asarray(values, dtype=np.float32))


# ------------------------------------------------------------------ preprocessors
def _blocks(pre):
    cols = pre['columns']
    return cols, {n: (t, c) for n, t, c in cols.transformers}


def _ohe_of(cols_dict, name='categorical'):
    ohe, cols = cols_dict[name]
    if not (isinstance(ohe, OneHotEncoder) and cols == CATEGORICAL and ohe.handle_unknown == 'ignore'
            and ohe.sparse_output and ohe.dtype == np.float32):
        raise ValueError('v1.0.1 categorical path changed')
    return ohe


def make_preprocessor(arm):
    """Fresh unfitted preprocessor for the arm. The pipeline steps are [schema, (age995), columns, (to_csr)]."""
    model, rep = arm_spec(arm)
    if model == 'LR':
        pre = make_C('C')['preprocess']
        if rep == 'OH':
            return pre
        cols, d = _blocks(pre)
        ohe = _ohe_of(d)
        num_pipe = d['numeric'][0]
        cols.transformers = [
            ('numeric', Pipeline([('float32', FunctionTransformer(numeric_with_geo_float32, feature_names_out='one-to-one')),
                                  ('scale', clone(num_pipe['scale']))]), NUMERIC + GEO),
            ('binary', d['binary'][0], d['binary'][1]),
            ('categorical', clone(ohe), list(TEXT))]
        return pre
    if model in ('DT', 'RF', 'GBC'):
        pre = make_v1(False)
        if rep == 'OH':
            return pre
        cols, d = _blocks(pre)
        ohe = _ohe_of(d)
        cols.transformers = [
            ('numeric', d['numeric'][0], d['numeric'][1]),
            ('binary', d['binary'][0], d['binary'][1]),
            ('categorical', clone(ohe), list(TEXT)),
            ('geo_numeric', FunctionTransformer(geo_float32, feature_names_out='one-to-one'), list(GEO))]
        return pre
    # LGBM
    if rep == 'OH':
        return t9.make_preprocessor()
    if rep == 'NAT':
        return t8.make_preprocessor('N')
    pre = t8.make_preprocessor('N')
    cols, d = _blocks(pre)
    ordinal, ocols = d['categorical']
    if not (isinstance(ordinal, OrdinalEncoder) and ocols == CATEGORICAL):
        raise ValueError('T8 arm N categorical path changed')
    cols.transformers = [
        ('numeric', d['numeric'][0], d['numeric'][1]),
        ('binary', d['binary'][0], d['binary'][1]),
        ('geo_numeric', FunctionTransformer(geo_float32, feature_names_out='one-to-one'), list(GEO)),
        ('text_codes', clone(ordinal), list(TEXT))]
    pre.steps.append(('to_csr', FunctionTransformer(to_csr_float32, accept_sparse=True, feature_names_out='one-to-one')))
    return pre


def make_classifier(arm, config):
    model, _ = arm_spec(arm)
    p = dict(config['classifier_params'][model])
    if model == 'LR':
        return LogisticRegression(**p)
    if model == 'DT':
        return DecisionTreeClassifier(**p)
    if model == 'RF':
        return RandomForestClassifier(**p)
    if model == 'GBC':
        return GradientBoostingClassifier(**p)
    from lightgbm import LGBMClassifier
    return LGBMClassifier(**p)


def categorical_feature(arm, config):
    return config['lightgbm_categorical_feature'].get(arm)


def fit_labels(arm, y):
    model, _ = arm_spec(arm)
    y = np.asarray(y)
    return y.astype(np.int32) - 1 if model == 'LGBM' else y


def class_offset(arm):
    return 1 if arm_spec(arm)[0] == 'LGBM' else 0


def expected_class_labels(arm):
    return [0, 1, 2] if arm_spec(arm)[0] == 'LGBM' else [1, 2, 3]


# ------------------------------------------------------------------ containers
def container_kind(matrix):
    if sparse.isspmatrix_csr(matrix):
        return 'csr'
    if sparse.issparse(matrix):
        return matrix.format
    return 'dense' if isinstance(matrix, np.ndarray) else type(matrix).__name__


def container_record(matrix):
    rows, cols = matrix.shape
    if sparse.issparse(matrix):
        nnz = int(matrix.nnz)
        storage = int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)
    else:
        nnz = int(np.count_nonzero(matrix))
        storage = int(matrix.nbytes)
    return dict(format=container_kind(matrix), dtype=str(matrix.dtype), rows=int(rows), columns=int(cols), nnz=nnz,
                density=float(nnz / (rows * cols)) if rows * cols else 0.0, storage_bytes=storage,
                dense_float32_estimate_bytes=int(rows) * int(cols) * 4)


def check_container(arm, matrix, config):
    expected = config['containers'][arm]
    kind = container_kind(matrix)
    if kind != expected or matrix.dtype != np.float32:
        raise ValueError(f'{arm}: expected {expected} float32 container, got {kind} {matrix.dtype}')
    values = matrix.data if sparse.issparse(matrix) else matrix
    model, rep = arm_spec(arm)
    if model == 'LGBM':            # unseen text/geo categories may be explicit NaN codes
        finite_ok = bool(np.isfinite(values[~np.isnan(values)]).all())
    else:
        finite_ok = bool(np.isfinite(values).all())
    if not finite_ok:
        raise ValueError(f'{arm}: nonfinite feature value')


# ------------------------------------------------------------------ names and comparisons
def feature_names(pre):
    return [str(n) for n in pre['columns'].get_feature_names_out()]


def suffix(name):
    return name.split('__', 1)[1] if '__' in name else name


def geo_indices(names):
    return [i for i, n in enumerate(names) if suffix(n).startswith('geo_level_')]


def other_indices(names):
    return [i for i, n in enumerate(names) if not suffix(n).startswith('geo_level_')]


def canonical_csr(matrix):
    m = sparse.csr_matrix(matrix, copy=True)
    m.sum_duplicates()
    m.eliminate_zeros()
    m.sort_indices()
    return m


def csr_equal(a, b):
    """Exact equality of two matrices through canonical CSR, NaN-aware, never densifying."""
    if a.shape != b.shape:
        return False
    a, b = canonical_csr(a), canonical_csr(b)
    return bool(np.array_equal(a.indptr, b.indptr) and np.array_equal(a.indices, b.indices)
                and np.array_equal(a.data, b.data, equal_nan=True))


def select_columns(matrix, idx):
    if sparse.issparse(matrix):
        return sparse.csr_matrix(matrix.tocsr()[:, list(idx)])
    return sparse.csr_matrix(np.asarray(matrix)[:, list(idx)])


def column_equivalence(base_matrix, base_names, num_matrix, num_names, approx_suffixes=(), approx_tol=0.0):
    """Non-geo columns of NUM must equal the same-named columns of the baseline arm.

    Columns whose suffix is in approx_suffixes (LR standardized numerics) are compared with a tolerance and the
    bitwise result is reported; all others must be bitwise equal. Returns a JSON-able dict."""
    b_sfx = {suffix(base_names[i]): i for i in other_indices(base_names)}
    n_sfx = {suffix(num_names[i]): i for i in other_indices(num_names)}
    same_set = set(b_sfx) == set(n_sfx)
    out = dict(same_non_geo_column_set=bool(same_set), n_compared=len(b_sfx), bitwise_equal=False,
               max_abs_diff_approx_columns=0.0, ok=False)
    if not same_set:
        return out
    exact = [s for s in b_sfx if s not in approx_suffixes]
    approx = [s for s in b_sfx if s in approx_suffixes]
    exact_ok = (not exact) or csr_equal(select_columns(base_matrix, [b_sfx[s] for s in exact]),
                                        select_columns(num_matrix, [n_sfx[s] for s in exact]))
    approx_bitwise, max_diff = True, 0.0
    if approx:
        a = select_columns(base_matrix, [b_sfx[s] for s in approx]).toarray()   # at most 5 columns
        b = select_columns(num_matrix, [n_sfx[s] for s in approx]).toarray()
        approx_bitwise = bool(np.array_equal(a, b))
        max_diff = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))) if a.size else 0.0
    out.update(bitwise_equal=bool(exact_ok and approx_bitwise), exact_columns_equal=bool(exact_ok),
               approx_columns_bitwise_equal=approx_bitwise, max_abs_diff_approx_columns=max_diff,
               ok=bool(exact_ok and (approx_bitwise or max_diff <= approx_tol)))
    return out


def geo_value_check(num_matrix, num_names, X_frame, scaler=None):
    """NUM geo columns must equal the original integer values (LR: after StandardScaler.inverse_transform)."""
    idx = geo_indices(num_names)
    order = [suffix(num_names[i]) for i in idx]
    if sorted(order) != sorted(GEO):
        return dict(ok=False, reason='geo columns missing', order=order)
    block = select_columns(num_matrix, idx).toarray()                      # (n, 3) only
    ints = np.column_stack([X_frame[c].astype('int64').to_numpy() for c in order]).astype(np.float64)
    vals = block.astype(np.float64)
    if scaler is not None:
        # scaler covers NUMERIC + GEO in that order; geo is the tail block
        n_num = len(NUMERIC)
        pos = [n_num + GEO.index(c) for c in order]
        vals = vals * scaler.scale_[pos] + scaler.mean_[pos]
        dev = float(np.max(np.abs(vals - ints))) if vals.size else 0.0
        ok = bool(dev < 0.25 and np.array_equal(np.rint(vals), ints))
    else:
        dev = float(np.max(np.abs(vals - ints))) if vals.size else 0.0
        ok = bool(np.array_equal(vals, ints))
    return dict(ok=ok, max_abs_deviation=dev, order=order)


def unseen_geo(X_train, X_valid):
    out, any_mask = {}, np.zeros(len(X_valid), dtype=bool)
    for c in GEO:
        known = set(X_train[c].astype(str))
        miss = ~X_valid[c].astype(str).isin(known).to_numpy()
        out[c] = int(miss.sum())
        any_mask |= miss
    out['any'] = int(any_mask.sum())
    return out, any_mask


def train_only_checks(pre, arm, Xn_train):
    """Fitted statistics must come from the training fold only (vocabularies and StandardScaler moments)."""
    model, rep = arm_spec(arm)
    res = {}
    cols = pre['columns']
    fitted = cols.named_transformers_
    if 'categorical' in fitted and isinstance(fitted['categorical'], OneHotEncoder):
        names = TEXT if (model != 'LGBM' and rep == 'NUM') else CATEGORICAL
        res['onehot_vocabulary_train_only'] = all(
            set(v) == set(Xn_train[c]) for c, v in zip(names, fitted['categorical'].categories_))
    if 'geo_onehot' in fitted:
        res['geo_onehot_vocabulary_train_only'] = all(
            set(v) == set(Xn_train[c]) for c, v in zip(GEO, fitted['geo_onehot'].categories_))
    for key in ('text_codes',):
        if key in fitted:
            res['text_vocabulary_train_only'] = all(
                set(v) == set(Xn_train[c]) for c, v in zip(TEXT, fitted[key].categories_))
    if 'categorical' in fitted and isinstance(fitted['categorical'], OrdinalEncoder):
        res['ordinal_vocabulary_train_only'] = all(
            set(v) == set(Xn_train[c]) for c, v in zip(CATEGORICAL, fitted['categorical'].categories_))
    numeric = fitted.get('numeric')
    if isinstance(numeric, Pipeline) and 'scale' in numeric.named_steps:
        scaler = numeric['scale']
        want = NUMERIC + (GEO if (model == 'LR' and rep == 'NUM') else [])
        sub = Xn_train.copy()
        if 'age995' in pre.named_steps:
            sub = pre['age995'].transform(sub)
        data = np.column_stack([pd.to_numeric(sub[c]).to_numpy(dtype=np.float64) for c in want])
        data = data.astype(np.float32).astype(np.float64)
        res['scaler_moments_train_only'] = bool(
            np.allclose(scaler.mean_, data.mean(0), atol=1e-6, rtol=1e-9)
            and np.allclose(scaler.var_, data.var(0), atol=1e-4, rtol=1e-6)
            and int(scaler.n_samples_seen_) == len(Xn_train))
    res['ok'] = bool(all(res.values()))
    return res
