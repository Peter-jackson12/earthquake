"""T9: LightGBM partial native categorical (P) = geo one-hot + text native codes.

Single factor relative to T8 arm O: the 8 non-geo text categoricals reach LightGBM as
train-fold OrdinalEncoder codes declared via categorical_feature (unseen -> NaN), while
geo_level_1/2/3_id stay one-hot exactly as in O. Everything else (C age policy, scaled
numeric/binary/flag 28 columns, LightGBM params, threads) is T8 unchanged.
The matrix stays float32 CSR (geo one-hot ~12.5k columns); implicit zeros are code 0
(zero_as_missing=false) and unseen NaN codes are stored explicitly. Never a full .toarray().
O/N/LR-C are read-only reference OOFs (not refit). The holdout is never read.
Each fold is saved, hashed, reloaded and backed up immediately; completed folds are
never refit; automatic retries = 0. All outputs are private (outside Git).
"""
from __future__ import annotations
import argparse
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
from scipy import sparse
from sklearn.base import clone
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from preprocess import BINARY, CATEGORICAL, GEO, NUMERIC, TEXT, sha256_file  # noqa: E402
from experiments.age995.run import FLAG  # noqa: E402
from experiments.geo_group_cv.run import PROBS, load, require, score, write  # noqa: E402
from experiments.model_families import run as t4  # noqa: E402
from experiments.lgbm_native_cat import run as t8  # noqa: E402  (T8 code reused unchanged)
from experiments.lgbm_native_cat.run import (CAT_PARAMS, atomic_write, check_saved, checked,  # noqa: E402
                                             environment, event, key, log, model_text_parameters, now,
                                             read_ledger, read_pred, write_pred)

HERE = Path(__file__).resolve().parent
STUDY_ID = 'T9_lgbm_partial_native_v1'
SCHEMES = ['stratified', 'geo2', 'geo1']
ARM = 'P'
REFS = ['O', 'N', 'LR_C']
N_HEAD = len(NUMERIC) + len(BINARY) + 1
SOURCE = ['preprocess.py', 'baseline.py', 'scripts/audit_eda_oof.py', 'configs/data_contract.json',
          'experiments/age995/run.py', 'experiments/geo_group_cv/run.py', 'experiments/geo_ablation/audit.py',
          'experiments/model_families/run.py', 'experiments/model_families/config.json',
          'experiments/lgbm_native_cat/run.py', 'experiments/lgbm_native_cat/config.json',
          'experiments/lgbm_partial_native/__init__.py', 'experiments/lgbm_partial_native/run.py',
          'experiments/lgbm_partial_native/config.json', 'docs/T9_LGBM_PARTIAL_NATIVE_PLAN.md']


def config(path=HERE/'config.json'):
    c = json.loads(Path(path).read_text(encoding='utf-8'))
    require(c['study_id'] == STUDY_ID and c['schemes'] == SCHEMES and c['n_folds'] == 5, 'Unplanned study/schemes')
    require(c['lgbm_params'] == t4.load_config()['models']['LGBM'], 'LightGBM params differ from T4 LGBM_C_v1')
    require(c['lgbm_params'] == t8.config()['lgbm_params'], 'LightGBM params differ from T8')
    require(c['categorical_columns'] == list(CATEGORICAL) and c['geo_onehot_columns'] == list(GEO)
            and c['text_code_columns'] == list(TEXT) and len(TEXT) == 8, 'Categorical column split changed')
    require(c['study_fits'] == {'P': SCHEMES} and c['max_new_study_fits'] == 15, 'Unplanned fit plan')
    require(c['automatic_retries'] == 0 and not c['early_stopping'] and c['class_weight'] is None
            and c['sampling'] is None and not c['tuning'] and not c['threshold_or_decision_rule_change']
            and c['outer_parallel_jobs'] == 1 and c['threadpool_limit'] == 2, 'Unplanned run policy')
    require(c['n_head_columns'] == N_HEAD and c['categorical_feature_indices_P'] == categorical_indices(),
            'Categorical indices inconsistent')
    require(not c['holdout_access'], 'Holdout access is not allowed')
    return c


def sources():
    return {p: sha256_file(ROOT/p) for p in SOURCE}


def targets():
    return [(s, ARM, f) for s in config()['study_fits'][ARM] for f in range(5)]


# ---------------------------------------------------------------- arm P
def categorical_indices():
    return list(range(N_HEAD, N_HEAD + len(TEXT)))


def make_preprocessor():
    """Fresh unfitted P preprocessor: T8 O (= C) with the categorical path split into
    text -> train-fold OrdinalEncoder codes (cols 28..35) and geo -> the same OneHotEncoder as O."""
    pre = t8.make_preprocessor('O')
    columns = pre['columns']
    require([n for n, _, _ in columns.transformers] == ['numeric', 'binary', 'categorical'], 'Unexpected C paths')
    require(columns.transformers[0][2] == NUMERIC and columns.transformers[1][2] == BINARY+[FLAG],
            'Noncategorical inputs changed')
    _, ohe, cols = columns.transformers[2]
    require(cols == CATEGORICAL and isinstance(ohe, OneHotEncoder) and ohe.handle_unknown == 'ignore'
            and ohe.sparse_output and ohe.dtype == np.float32, 'C categorical path changed')
    require(columns.sparse_threshold == 1.0, 'C must produce sparse output')
    ordinal = OrdinalEncoder(categories='auto', handle_unknown='use_encoded_value', unknown_value=np.nan,
                             encoded_missing_value=np.nan, dtype=np.float32)
    columns.transformers = columns.transformers[:2] + [('text_codes', ordinal, list(TEXT)),
                                                       ('geo_onehot', clone(ohe), list(GEO))]
    return pre


def code_block(matrix):
    """Only the 8 code columns as a small dense (n, 8) array; never the one-hot block."""
    return matrix[:, categorical_indices()].toarray()


def check_matrix(matrix):
    require(sparse.isspmatrix_csr(matrix) and matrix.dtype == np.float32, 'P must be float32 CSR')
    head = matrix[:, :N_HEAD]
    require(np.isfinite(head.data).all(), 'Nonfinite P numeric/binary feature')
    geo = matrix[:, N_HEAD + len(TEXT):]
    require(np.isfinite(geo.data).all() and (geo.data == 1).all(), 'Geo one-hot block must be 0/1')
    require((np.asarray(geo.sum(axis=1)).ravel() <= len(GEO)).all(), 'At most one active level per geo column')
    codes = code_block(matrix)
    known = codes[~np.isnan(codes)]
    require((known >= 0).all() and (known == np.floor(known)).all(), 'Codes must be nonnegative integers')


def expected_unseen(pre, X_train, X_valid):
    """Per text column: number of valid rows whose value is not in the training-fold vocabulary."""
    a, b = pre['schema'].transform(X_train), pre['schema'].transform(X_valid)
    return {c: int((~b[c].isin(set(a[c]))).sum()) for c in TEXT}


def o_blocks_equal(o_matrix, p_matrix, n_geo_columns):
    """P head (28) and geo one-hot block must equal O's head and O's geo columns (sparse compare only)."""
    a_head, b_head = o_matrix[:, :N_HEAD].tocsr(), p_matrix[:, :N_HEAD].tocsr()
    a_geo = o_matrix[:, N_HEAD:N_HEAD + n_geo_columns].tocsr()
    b_geo = p_matrix[:, N_HEAD + len(TEXT):].tocsr()
    head = a_head.shape == b_head.shape and (a_head != b_head).nnz == 0
    geo = a_geo.shape == b_geo.shape and (a_geo != b_geo).nnz == 0
    return bool(head), bool(geo)


def fit_arm(X_train, y_train, X_valid, params, threads):
    """Fit P preprocessor + LightGBM on the training fold only."""
    from lightgbm import LGBMClassifier
    pre = make_preprocessor()
    with threadpool_limits(limits=threads):
        t0 = time.perf_counter()
        mt = pre.fit_transform(X_train)
        mv = pre.transform(X_valid)
        pre_seconds = time.perf_counter() - t0
    check_matrix(mt)
    check_matrix(mv)
    enc = pre['columns'].named_transformers_['text_codes']
    geo_enc = pre['columns'].named_transformers_['geo_onehot']
    schema_train = pre['schema'].transform(X_train)
    for col, learned in zip(TEXT, enc.categories_):
        require(set(learned) == set(schema_train[col]), 'Text vocabulary must equal training values')
    for col, learned in zip(GEO, geo_enc.categories_):
        require(set(learned) == set(schema_train[col]), 'Geo vocabulary must equal training values')
    n_geo = int(sum(len(v) for v in geo_enc.categories_))
    require(mt.shape[1] == N_HEAD + len(TEXT) + n_geo == mv.shape[1], 'P column count inconsistent')
    vt, vv = code_block(mt), code_block(mv)
    require(not np.isnan(vt).any(), 'Training codes cannot be missing')
    unseen = expected_unseen(pre, X_train, X_valid)
    valid_nan = {c: int(np.isnan(vv[:, i]).sum()) for i, c in enumerate(TEXT)}
    require(valid_nan == unseen, 'Unseen text categories must be explicit NaN codes')
    stored_nan = int(np.isnan(mv.data).sum())
    require(stored_nan == sum(unseen.values()), 'NaN codes must be stored explicitly in CSR')
    model = LGBMClassifier(**params)
    with threadpool_limits(limits=threads), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        t0 = time.perf_counter()
        model.fit(mt, np.asarray(y_train).astype(np.int32) - 1, categorical_feature=categorical_indices())
        fit_seconds = time.perf_counter() - t0
        p = checked(model, mv)
        p_train = checked(model, mt)
    require(model.booster_.current_iteration() == params['n_estimators'], 'Full fixed boosting budget not completed')
    info = dict(fit_seconds=fit_seconds, preprocessing_seconds=pre_seconds, encoded_columns=int(mt.shape[1]),
                geo_onehot_columns=n_geo, train_nnz=int(mt.nnz), valid_nnz=int(mv.nnz),
                warnings=[dict(category=w.category.__name__, message=str(w.message)) for w in caught],
                vocabulary_sizes={c: int(len(v)) for c, v in zip(TEXT, enc.categories_)},
                geo_vocabulary_sizes={c: int(len(v)) for c, v in zip(GEO, geo_enc.categories_)},
                valid_unseen_nan=valid_nan, valid_nan_stored=stored_nan,
                age995_median=pre['age995'].median_)
    return pre, model, p, p_train, mt, mv, info


def categorical_effective(path):
    params, infos = model_text_parameters(path)
    eff = {k: float(params[k]) for k in CAT_PARAMS}
    retained = {c: (len(infos[i].split(':')) if infos[i] != 'none' else 0)
                for c, i in zip(TEXT, categorical_indices())}
    return eff, retained, params.get('categorical_feature')


# ---------------------------------------------------------------- sample check (not a study fit)
def sample_check(data, destination):
    """Real-data subsample (training rows of stratified fold != 0 only): CSR-with-codes must give
    identical LightGBM predictions to the dense equivalent of the same small matrix."""
    from lightgbm import LGBMClassifier
    c = config()
    sc = c['sample_check']
    dest = Path(destination)
    require(not dest.exists(), 'Do not overwrite an existing sample check')
    require(environment() == c['environment'], 'Environment differs from the pinned environment')
    X, y, ids, old, contract = load(data)
    pool = np.flatnonzero(old != 0)
    rng = np.random.default_rng(sc['seed'])
    rows = np.sort(rng.choice(pool, size=sc['n_rows'], replace=False))
    perm = rng.permutation(rows)
    a, b = np.sort(perm[:sc['n_train']]), np.sort(perm[sc['n_train']:])
    params = c['lgbm_params']
    pre, model_csr, p_csr, _, mt, mv, info = fit_arm(X.iloc[a], y.iloc[a], X.iloc[b], params, c['threadpool_limit'])
    dt, dv = mt.toarray(), mv.toarray()  # subsample only (<= 5000 rows); never the study matrix
    # Explicit-NaN path: mark some code entries unseen in both representations.
    dv_nan = dv.copy()
    dv_nan[:25, N_HEAD + 2] = np.nan
    dv_nan[25:40, N_HEAD + 7] = np.nan
    mv_nan = sparse.csr_matrix(dv_nan)
    nan_stored = int(np.isnan(mv_nan.data).sum())
    names = [n.split('__', 1)[1] for n in pre['columns'].get_feature_names_out()]
    with threadpool_limits(limits=c['threadpool_limit']), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model_dense = LGBMClassifier(**params).fit(dt, np.asarray(y.iloc[a]).astype(np.int32) - 1,
                                                   categorical_feature=categorical_indices())
        pd_dense = checked(model_dense, dv)
        cross = checked(model_csr, dv)
        pn_csr, pn_dense = checked(model_csr, mv_nan), checked(model_dense, dv_nan)
    dest.mkdir(parents=True)
    m1, m2 = dest/'P_csr.txt', dest/'P_dense.txt'
    model_csr.booster_.save_model(str(m1))
    model_dense.booster_.save_model(str(m2))
    _, _, cat1 = categorical_effective(m1)
    _, _, cat2 = categorical_effective(m2)
    diffs = dict(csr_vs_dense=float(np.max(np.abs(p_csr - pd_dense))),
                 csr_model_dense_input=float(np.max(np.abs(p_csr - cross))),
                 explicit_nan_csr_vs_dense=float(np.max(np.abs(pn_csr - pn_dense))))
    labels = dict(csr_vs_dense=bool(np.array_equal(p_csr.argmax(1), pd_dense.argmax(1))),
                  explicit_nan=bool(np.array_equal(pn_csr.argmax(1), pn_dense.argmax(1))))
    idx_names = [names[i] for i in categorical_indices()]
    out = dict(study_id=STUDY_ID, kind=sc['kind'], counted_as_study_fit=False, classifiers_fitted=2,
               source_rows=sc['source_rows'], seed=sc['seed'], train_rows=len(a), valid_rows=len(b),
               matrix_columns=int(mt.shape[1]), geo_onehot_columns=info['geo_onehot_columns'],
               train_nnz=int(mt.nnz), dense_train_shape=list(dt.shape),
               probability_max_abs_diff=diffs, probability_atol=sc['probability_atol'],
               predicted_labels_identical=labels, explicit_nan_entries_stored=nan_stored,
               model_text_identical=bool(sha256_file(m1) == sha256_file(m2)),
               model_sha256=dict(csr=sha256_file(m1), dense=sha256_file(m2)),
               categorical_feature_model_text=dict(csr=cat1, dense=cat2),
               categorical_feature_indices=categorical_indices(), categorical_feature_names=idx_names,
               valid_unseen_nan=info['valid_unseen_nan'], warnings=info['warnings'],
               environment=environment(), holdout_accessed=False, created_at_utc=now())
    ok = (max(diffs.values()) <= sc['probability_atol'] and all(labels.values())
          and nan_stored == int(np.isnan(dv_nan).sum()) >= 40
          and idx_names == list(TEXT) and cat1 == cat2 == ','.join(map(str, categorical_indices())))
    out['status'] = 'PASS' if ok else 'FAIL'
    write(dest/'sample_check.json', out)
    print(json.dumps(out, indent=1))
    require(out['status'] == 'PASS', 'Sample check failed')
    return out


def probe_categorical_defaults(destination):
    """Synthetic 400-row probe (not a study fit) reading LightGBM effective categorical defaults."""
    from lightgbm import LGBMClassifier
    rng = np.random.default_rng(0)
    Xp = np.c_[rng.normal(size=400), rng.integers(0, 6, 400)].astype(np.float32)
    yy = rng.integers(0, 3, 400)
    params = dict(config()['lgbm_params'], n_estimators=2, verbosity=-1)
    m = LGBMClassifier(**params).fit(Xp, yy, categorical_feature=[1])
    path = Path(destination)/'categorical_defaults_probe.txt'
    m.booster_.save_model(str(path))
    params, _ = model_text_parameters(path)
    eff = {k: float(params[k]) for k in CAT_PARAMS}
    path.unlink()
    return eff


# ---------------------------------------------------------------- references
def reference(path, digest, ids, y, folds):
    require(sha256_file(path) == digest, f'Reference OOF changed: {path}')
    r = read_pred(path).set_index('building_id').loc[np.asarray(ids)].reset_index()
    require(len(r) == len(ids) and r.building_id.is_unique, 'Reference OOF coverage')
    np.testing.assert_array_equal(r.fold.to_numpy(), folds)
    np.testing.assert_array_equal(r.y_true.to_numpy(), np.asarray(y))
    return r


def fold_means(r):
    return float(np.mean([score(r.y_true[r.fold == f], r.y_pred[r.fold == f])['macro_f1'] for f in range(5)]))


# ---------------------------------------------------------------- prepare/verify
def prepare(data, t8_run, ref_paths, output, code_commit, sample):
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
    shas = {'stratified': sha256_file(Path(data)/'split_manifest.csv')}
    for s in ['geo2', 'geo1']:
        shas[s] = sha256_file(Path(t8_run)/s/'manifest.csv')
    require(shas == c['manifest_sha256'], 'Split manifest SHA differs')
    sc = json.loads(Path(sample).read_text(encoding='utf-8'))
    require(sc['status'] == 'PASS' and sc['study_id'] == STUDY_ID and not sc['counted_as_study_fit'],
            'Sample check must PASS before prepare')
    output.mkdir(parents=True)
    folds = {'stratified': old}
    for s in ['geo2', 'geo1']:
        (output/s).mkdir()
        shutil.copyfile(Path(t8_run)/s/'manifest.csv', output/s/'manifest.csv')
        require(sha256_file(output/s/'manifest.csv') == shas[s], 'Copied manifest SHA differs')
        m = pd.read_csv(output/s/'manifest.csv')
        np.testing.assert_array_equal(m.building_id, ids)
        folds[s] = m.fold.to_numpy()
    (output/'backup').mkdir()
    ref_sha, ref_means = {}, {}
    for arm in REFS:
        ref_sha[arm], ref_means[arm] = {}, {}
        for s in SCHEMES:
            path = ref_paths[arm][s]
            r = reference(path, c['reference_oof_sha256'][arm][s], ids, y, folds[s])
            ref_sha[arm][s] = sha256_file(path)
            ref_means[arm][s] = fold_means(r)
            require(abs(ref_means[arm][s] - c['reference_cv_means'][arm][s]) <= c['reference_cv_means']['tolerance'],
                    f'{arm} {s} reference mean differs from registered value')
            if arm == 'LR_C' and s == 'geo2':
                rec = score(r.y_true, r.y_pred)['recall'][2]
                reg = c['lr_c_geo2_pooled_grade3_recall']
                require(abs(rec - reg['value']) <= reg['tolerance'], 'LR-C geo2 grade-3 recall differs')
    eff = probe_categorical_defaults(output)
    require(eff == {k: float(v) for k, v in c['lightgbm_categorical_defaults_expected'].items()},
            f'LightGBM categorical defaults differ from expectation: {eff}')
    design = dict(study_id=STUDY_ID, config=c, code_commit=code_commit, source_sha256=sources(),
                  data_contract=contract, environment=environment(), manifest_sha256=shas,
                  manifest_source=str(Path(t8_run).resolve()),
                  reference_oof_path={a: {s: str(Path(ref_paths[a][s]).resolve()) for s in SCHEMES} for a in REFS},
                  reference_oof_sha256=ref_sha, reference_cv_means=ref_means,
                  sample_check=dict(path=str(Path(sample).resolve()), sha256=sha256_file(sample), status=sc['status'],
                                    probability_max_abs_diff=sc['probability_max_abs_diff']),
                  lightgbm_categorical_effective_defaults=eff, categorical_feature_indices_P=categorical_indices(),
                  study_targets=[key(*t) for t in targets()], max_new_study_fits=c['max_new_study_fits'],
                  automatic_retries=0, reference_refits=0, holdout_accessed=False,
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
        m = pd.read_csv(path)
        np.testing.assert_array_equal(m.building_id, ids)
        folds[s] = m.fold.to_numpy()
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
    pre, model, p, p_train, mt, mv, info = fit_arm(X.iloc[a], y.iloc[a], X.iloc[b], c['lgbm_params'],
                                                   c['threadpool_limit'])
    pred = p.argmax(1) + 1
    # P head (numeric/binary/flag) and geo one-hot must equal O built from the same training fold.
    o_pre = t8.make_preprocessor('O')
    with threadpool_limits(limits=c['threadpool_limit']):
        o_pre.fit(X.iloc[a])
        head_equal, geo_equal = o_blocks_equal(o_pre.transform(X.iloc[b]), mv, info['geo_onehot_columns'])
    require(head_equal and geo_equal, 'P numeric/binary or geo one-hot columns differ from O')
    del o_pre
    model_path, pre_path = d/f'fold_{f}_model.txt', d/f'fold_{f}_preprocessor.joblib'
    preds = d/f'fold_{f}_pred.csv.gz'
    model.booster_.save_model(str(model_path))
    joblib.dump(pre, pre_path, compress=3)
    frame = pd.DataFrame({'building_id': ids.iloc[b].to_numpy(), 'fold': f, 'y_true': y.iloc[b].to_numpy(),
                          'y_pred': pred, **{n: p[:, i] for i, n in enumerate(PROBS)}})
    write_pred(preds, frame)
    # reload verification
    import lightgbm
    booster = lightgbm.Booster(model_file=str(model_path))
    loaded_pre = joblib.load(pre_path)  # freshly generated trusted file only
    with threadpool_limits(limits=c['threadpool_limit']):
        mv2 = loaded_pre.transform(X.iloc[b])
        p2 = np.asarray(booster.predict(mv2, num_threads=c['lgbm_params']['n_jobs']), dtype=np.float64)
    require(sparse.isspmatrix_csr(mv2), 'Reloaded preprocessor must emit CSR')
    reload_diff = float(np.max(np.abs(p2 - p)))
    require(reload_diff <= c['checkpoint']['model_reload_probability_atol']
            and np.array_equal(p2.argmax(1) + 1, pred), f'Model reload mismatch {reload_diff}')
    back = read_pred(preds)
    np.testing.assert_array_equal(back.building_id.to_numpy(), frame.building_id.to_numpy())
    np.testing.assert_array_equal(back.y_pred.to_numpy(), pred)
    np.testing.assert_array_equal(back[PROBS].to_numpy(), p)
    eff, retained, cat_text = categorical_effective(model_path)
    require(cat_text == ','.join(map(str, categorical_indices())), 'Model categorical_feature differs')
    record = dict(study_id=STUDY_ID, scheme=scheme, arm=arm, fold=f, train_rows=len(a), valid_rows=len(b),
                  train_class_counts=[int((y.iloc[a] == v).sum()) for v in (1, 2, 3)],
                  metrics=score(y.iloc[b], pred), train_macro_f1=score(y.iloc[a], p_train.argmax(1)+1)['macro_f1'],
                  fit_seconds=info['fit_seconds'], preprocessing_seconds=info['preprocessing_seconds'],
                  encoded_columns=info['encoded_columns'], geo_onehot_columns=info['geo_onehot_columns'],
                  train_nnz=info['train_nnz'], valid_nnz=info['valid_nnz'], age995_median=info['age995_median'],
                  rounds_completed=int(model.booster_.current_iteration()),
                  vocabulary_sizes=info['vocabulary_sizes'], geo_vocabulary_sizes=info['geo_vocabulary_sizes'],
                  valid_unseen_nan=info['valid_unseen_nan'], valid_nan_stored=info['valid_nan_stored'],
                  numeric_binary_columns_equal_O=head_equal, geo_onehot_columns_equal_O=geo_equal,
                  lightgbm_categorical_effective=eff, lightgbm_retained_categories=retained,
                  categorical_feature=categorical_indices(), categorical_feature_model_text=cat_text,
                  effective_parameters=model.get_params(), reload_probability_max_abs_diff=reload_diff,
                  csv_round_trip_exact=True, warnings=info['warnings'], finished_at_utc=now())
    require(eff == design['lightgbm_categorical_effective_defaults'], 'Categorical params differ from design')
    write(d/f'fold_{f}.json', record)
    shas = {pth.name: sha256_file(pth) for pth in [model_path, pre_path, preds, d/f'fold_{f}.json']}
    bd = out/'backup'/entry['dir']
    bd.mkdir(parents=True, exist_ok=True)
    for name in shas:
        require(not (bd/name).exists(), 'Backup already exists; refusing overwrite')
        shutil.copy2(d/name, bd/name)
    entry.update(status='complete', artifact_sha256=shas, macro_f1=record['metrics']['macro_f1'],
                 fit_seconds=info['fit_seconds'], finished_at_utc=now())
    check_saved(out, entry)
    ledger = read_ledger(out)
    ledger['folds'][k] = entry
    atomic_write(out/'ledger.json', ledger)
    event(out, fold=k, action='fit_complete', macro_f1=record['metrics']['macro_f1'], seconds=info['fit_seconds'],
          artifact_sha256=shas)
    log(out, f'{k} F1={record["metrics"]["macro_f1"]:.9f} fit={info["fit_seconds"]:.1f}s '
             f'reload_diff={reload_diff:.3g} saved+verified+backed-up')
    return entry, True


def fit_all(data, output):
    out = Path(output)
    lock = out/'RUNNING.lock'
    require(not lock.exists(), f'Another run holds {lock}; remove only if that process is dead')
    lock.write_text(f'pid={os.getpid()} started={now()}\n')
    try:
        X, y, ids, folds, design = verify(data, out)
        before = pd.util.hash_pandas_object(X, index=True).to_numpy()
        todo = targets()
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
def assemble(out, y, ids, folds, scheme):
    out = Path(out)
    ledger = read_ledger(out)
    entries = [ledger['folds'].get(key(scheme, ARM, f)) for f in range(5)]
    require(all(e and e['status'] == 'complete' for e in entries), f'{scheme}/{ARM} incomplete')
    d = out/scheme/ARM
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
    result = dict(study_id=STUDY_ID, scheme=scheme, arm=ARM, cv_mean=float(np.mean(values)),
                  cv_std=float(np.std(values, ddof=1)), fold_macro_f1=values, oof_sha256=sha256_file(d/'oof.csv.gz'),
                  fit_seconds=float(sum(e['fit_seconds'] for e in entries)), finished_at_utc=now())
    write(d/'metrics.json', result)
    bd = out/'backup'/scheme/ARM
    for name in ['oof.csv.gz', 'metrics.json']:
        require(not (bd/name).exists(), 'Backup already exists; refusing overwrite')
        shutil.copy2(d/name, bd/name)
    log(out, f'{scheme}/{ARM} OOF assembled mean={result["cv_mean"]:.9f} sd={result["cv_std"]:.9f}')
    return result


def summary(oof, X, fit_seconds=None):
    y, p, fo = oof.y_true.to_numpy(), oof.y_pred.to_numpy(), oof.fold.to_numpy()
    fold_scores = [score(y[fo == f], p[fo == f]) for f in range(5)]
    values = [s['macro_f1'] for s in fold_scores]
    age = X.age.to_numpy() == 995
    geo1 = X.geo_level_1_id.astype(str).to_numpy()
    prof = dict(age995=score(y[age], p[age]))
    for g in config()['profiles']['geo1_regions']:
        m = geo1 == str(g)
        prof[f'geo1_{g}'] = score(y[m], p[m])
    return dict(cv_mean=float(np.mean(values)), cv_std=float(np.std(values, ddof=1)), fold_macro_f1=values,
                fold_grade3_recall=[s['recall'][2] for s in fold_scores],
                fold_confusion=[s['confusion'] for s in fold_scores],
                pooled=score(y, p), profiles=prof, fit_seconds=fit_seconds)


def decomposition_share(p, o, n):
    return None if n == o else float((p - o) / (n - o))


def finalize_all(data, output):
    from experiments.geo_ablation.audit import turnover
    out = Path(output)
    X, y, ids, folds, design = verify(data, out)
    require(not (out/'analysis.json').exists(), 'analysis.json exists; refusing overwrite')
    ledger = read_ledger(out)
    tg = design['study_targets']
    complete = sum(ledger['folds'].get(k, {}).get('status') == 'complete' for k in tg)
    attempts = sum(e['attempts'] for e in ledger['folds'].values())
    require(complete == len(tg) == 15 and len(ledger['folds']) == 15, 'Ledger incomplete; not finalizing')
    c = design['config']
    res = dict(study_id=STUDY_ID, design_sha256=sha256_file(out/'design.json'), code_commit=design['code_commit'],
               ledger=dict(complete=complete, total=15, attempts=attempts, retries=attempts-complete,
                           fit_seconds_sum=float(sum(e['fit_seconds'] for e in ledger['folds'].values()))),
               reference_refits=0, holdout_accessed=False, rules=c['rules'],
               note='P compared with read-only O/N/LR-C OOFs on the same folds; descriptive comparison, '
                    'selection on the same folds is optimistic; no significance or causal claims.',
               schemes={})
    for s in SCHEMES:
        refs = {a: reference(design['reference_oof_path'][a][s], design['reference_oof_sha256'][a][s], ids, y, folds[s])
                for a in REFS}
        mp = assemble(out, y, ids, folds, s)
        p = read_pred(out/s/ARM/'oof.csv.gz')
        for r in refs.values():
            np.testing.assert_array_equal(r.building_id.to_numpy(), p.building_id.to_numpy())
        block = {a: dict(summary(refs[a], X), oof_sha256=design['reference_oof_sha256'][a][s]) for a in REFS}
        block['P'] = dict(summary(p, X, mp['fit_seconds']), oof_sha256=mp['oof_sha256'])
        comps = {}
        for ref_name in REFS:
            deltas = [a - b for a, b in zip(block['P']['fold_macro_f1'], block[ref_name]['fold_macro_f1'])]
            comps[f'P_vs_{ref_name}'] = dict(
                mean_delta=block['P']['cv_mean'] - block[ref_name]['cv_mean'], fold_delta=deltas,
                fold_wins=int(sum(d > 0 for d in deltas)),
                turnover=turnover(y.to_numpy(), refs[ref_name].y_pred.to_numpy(), p.y_pred.to_numpy()))
        block['comparisons'] = comps
        block['decomposition_share_P_minus_O_over_N_minus_O'] = decomposition_share(
            block['P']['cv_mean'], block['O']['cv_mean'], block['N']['cv_mean'])
        res['schemes'][s] = block
    st, g2 = res['schemes']['stratified'], res['schemes']['geo2']
    r2 = c['rules']['combined_candidate']
    conds = dict(
        stratified_mean_P_gt=bool(st['P']['cv_mean'] > r2['stratified_mean_P_gt']),
        stratified_fold_wins_vs_LR_C=bool(st['comparisons']['P_vs_LR_C']['fold_wins']
                                          >= r2['stratified_fold_wins_vs_LR_C_min']),
        geo2_mean_P_ge=bool(g2['P']['cv_mean'] >= r2['geo2_mean_P_ge']),
        geo2_fold_wins_vs_LR_C=bool(g2['comparisons']['P_vs_LR_C']['fold_wins'] >= r2['geo2_fold_wins_vs_LR_C_min']),
        geo2_pooled_grade3_recall_P_ge=bool(g2['P']['pooled']['recall'][2] >= r2['geo2_pooled_grade3_recall_P_ge']))
    res['judgments'] = dict(
        decomposition=dict(kind='descriptive', stratified_share=st['decomposition_share_P_minus_O_over_N_minus_O'],
                           geo2_share=g2['decomposition_share_P_minus_O_over_N_minus_O'],
                           geo1_share=res['schemes']['geo1']['decomposition_share_P_minus_O_over_N_minus_O']),
        combined_candidate=bool(all(conds.values())), combined_conditions=conds,
        combined_inputs=dict(stratified_mean_P=st['P']['cv_mean'],
                             stratified_fold_wins_vs_LR_C=st['comparisons']['P_vs_LR_C']['fold_wins'],
                             geo2_mean_P=g2['P']['cv_mean'],
                             geo2_fold_wins_vs_LR_C=g2['comparisons']['P_vs_LR_C']['fold_wins'],
                             geo2_pooled_grade3_recall_P=g2['P']['pooled']['recall'][2],
                             geo2_pooled_grade3_recall_LR_C=g2['LR_C']['pooled']['recall'][2]))
    res['finished_at_utc'] = now()
    write(out/'analysis.json', res)
    require(not (out/'backup'/'analysis.json').exists(), 'Backup already exists; refusing overwrite')
    shutil.copy2(out/'analysis.json', out/'backup'/'analysis.json')
    log(out, 'analysis written: ' + json.dumps(dict(combined_candidate=res['judgments']['combined_candidate'],
                                                    decomposition=res['judgments']['decomposition'])))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action', choices=['sample-check', 'prepare', 'fit-all', 'finalize', 'status'])
    ap.add_argument('--data', default='data')
    ap.add_argument('--output')
    ap.add_argument('--destination')
    ap.add_argument('--t8-run', help='T8 private run dir (manifests, O/N geo OOFs)')
    ap.add_argument('--code-commit')
    ap.add_argument('--sample')
    ap.add_argument('--t4-oof', help='pinned T4 LGBM_C_v1 OOF = O stratified')
    for s in SCHEMES:
        ap.add_argument(f'--lr-oof-{s}')
    a = ap.parse_args()
    if a.action == 'sample-check':
        sample_check(a.data, a.destination)
    elif a.action == 'prepare':
        t8_run = Path(a.t8_run)
        refs = {'O': {'stratified': a.t4_oof, 'geo2': t8_run/'geo2'/'O'/'oof.csv.gz',
                      'geo1': t8_run/'geo1'/'O'/'oof.csv.gz'},
                'N': {s: t8_run/s/'N'/'oof.csv.gz' for s in SCHEMES},
                'LR_C': {s: getattr(a, f'lr_oof_{s}') for s in SCHEMES}}
        prepare(a.data, t8_run, refs, a.output, a.code_commit, a.sample)
    elif a.action == 'fit-all':
        fit_all(a.data, a.output)
    elif a.action == 'finalize':
        finalize_all(a.data, a.output)
    else:
        ledger = read_ledger(a.output)
        print(json.dumps(dict(complete=sum(e['status'] == 'complete' for e in ledger['folds'].values()), total=15,
                              attempts=sum(e['attempts'] for e in ledger['folds'].values()),
                              running=[k for k, e in ledger['folds'].items() if e['status'] == 'running']), indent=1))


if __name__ == '__main__':
    sys.exit(main())
