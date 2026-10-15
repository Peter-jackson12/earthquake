"""T8: LightGBM categorical representation, one-hot (O) vs native categorical (N).

Single factor = how the 11 categorical inputs reach LightGBM.
  O = T4 LGBM_C_v1 exactly: make_pipeline('C')['preprocess'] (scaled numeric, binary,
      age_is_995, OneHotEncoder CSR float32) + LGBMClassifier(T4 params).
  N = the same C preprocessor with the OneHotEncoder replaced by an OrdinalEncoder
      fitted on the training fold only (unseen -> NaN); dense float32; the 11 code
      columns are declared via categorical_feature. Same LightGBM params.
O stratified is never refit (pinned T4 OOF is reused read-only). The holdout is never read.
Each fold is saved, hashed, reloaded and backed up immediately; completed folds are
never refit; automatic retries = 0. All outputs are private (outside Git).
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
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
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from preprocess import BINARY, CATEGORICAL, NUMERIC, sha256_file  # noqa: E402
from experiments.age995.run import FLAG, make_pipeline as make_C  # noqa: E402
from experiments.geo_group_cv.run import PROBS, load, require, score, write  # noqa: E402
from experiments.model_families import run as t4  # noqa: E402

HERE = Path(__file__).resolve().parent
STUDY_ID = 'T8_lgbm_native_cat_v1'
SCHEMES = ['stratified', 'geo2', 'geo1']
ARMS = ['O', 'N']
SOURCE = ['preprocess.py', 'baseline.py', 'scripts/audit_eda_oof.py', 'configs/data_contract.json',
          'experiments/age995/run.py', 'experiments/geo_group_cv/run.py', 'experiments/geo_ablation/audit.py',
          'experiments/model_families/run.py', 'experiments/model_families/config.json',
          'experiments/lgbm_native_cat/__init__.py', 'experiments/lgbm_native_cat/run.py',
          'experiments/lgbm_native_cat/config.json', 'docs/T8_LGBM_NATIVE_CAT_PLAN.md']
CAT_PARAMS = ['max_cat_to_onehot', 'cat_smooth', 'cat_l2', 'max_cat_threshold', 'min_data_per_group']


def now():
    return datetime.now(timezone.utc).isoformat()


def config(path=HERE/'config.json'):
    c = json.loads(Path(path).read_text(encoding='utf-8'))
    require(c['study_id'] == STUDY_ID and c['schemes'] == SCHEMES and c['n_folds'] == 5, 'Unplanned study/schemes')
    t4c = t4.load_config()
    require(c['lgbm_params'] == t4c['models']['LGBM'], 'LightGBM params differ from T4 LGBM_C_v1')
    require(c['categorical_columns'] == list(CATEGORICAL), 'Categorical column order changed')
    require(c['study_fits'] == {'N': SCHEMES, 'O': ['geo2', 'geo1']} and c['max_new_study_fits'] == 25,
            'Unplanned fit plan')
    require(c['automatic_retries'] == 0 and not c['early_stopping'] and c['class_weight'] is None
            and c['sampling'] is None and not c['tuning'] and not c['threshold_or_decision_rule_change']
            and c['outer_parallel_jobs'] == 1 and c['threadpool_limit'] == 2, 'Unplanned run policy')
    require(c['categorical_feature_indices_N'] == list(range(len(NUMERIC)+len(BINARY)+1, c['n_feature_columns_N'])),
            'Categorical indices inconsistent')
    return c


def environment():
    names = ['numpy', 'pandas', 'scipy', 'scikit-learn', 'lightgbm', 'joblib', 'threadpoolctl']
    return dict(python=platform.python_version(), **{n: importlib.metadata.version(n) for n in names})


def sources():
    return {p: sha256_file(ROOT/p) for p in SOURCE}


def targets():
    c = config()
    return [(s, a, f) for a in ['N', 'O'] for s in c['study_fits'][a] for f in range(5)]


# ---------------------------------------------------------------- arms
def make_preprocessor(arm):
    """Fresh unfitted preprocessor. O = unchanged C; N = C with OHE -> train-fold OrdinalEncoder."""
    require(arm in ARMS, 'Unknown arm')
    pre = make_C('C')['preprocess']
    if arm == 'O':
        return pre
    columns = pre['columns']
    require([n for n, _, _ in columns.transformers] == ['numeric', 'binary', 'categorical'], 'Unexpected C paths')
    require(columns.transformers[0][2] == NUMERIC and columns.transformers[1][2] == BINARY+[FLAG],
            'Noncategorical inputs changed')
    name, ohe, cols = columns.transformers[2]
    require(cols == CATEGORICAL and isinstance(ohe, OneHotEncoder), 'C categorical path changed')
    ordinal = OrdinalEncoder(categories='auto', handle_unknown='use_encoded_value', unknown_value=np.nan,
                             encoded_missing_value=np.nan, dtype=np.float32)
    columns.transformers = columns.transformers[:2] + [(name, ordinal, cols)]
    return pre


def categorical_indices():
    start = len(NUMERIC) + len(BINARY) + 1
    return list(range(start, start + len(CATEGORICAL)))


def check_matrix(arm, matrix):
    if arm == 'O':
        require(sparse.isspmatrix_csr(matrix) and matrix.dtype == np.float32, 'O must be float32 CSR')
        require(np.isfinite(matrix.data).all(), 'Nonfinite O feature')
    else:
        require(isinstance(matrix, np.ndarray) and matrix.dtype == np.float32
                and matrix.shape[1] == len(NUMERIC)+len(BINARY)+1+len(CATEGORICAL), 'N must be dense float32 (39 cols)')
        head = matrix[:, :categorical_indices()[0]]
        require(np.isfinite(head).all(), 'Nonfinite N numeric/binary feature')
        codes = matrix[:, categorical_indices()]
        known = codes[~np.isnan(codes)]
        require((known >= 0).all() and (known == np.floor(known)).all(), 'Codes must be nonnegative integers')


def fit_arm(arm, X_train, y_train, X_valid, params, threads):
    """Fit preprocessor+LightGBM on the training fold only; return (pre, model, p_valid, p_train, info)."""
    from lightgbm import LGBMClassifier
    pre = make_preprocessor(arm)
    with threadpool_limits(limits=threads):
        t0 = time.perf_counter()
        mt = pre.fit_transform(X_train)
        mv = pre.transform(X_valid)
        pre_seconds = time.perf_counter() - t0
    check_matrix(arm, mt)
    check_matrix(arm, mv)
    model = LGBMClassifier(**params)
    kw = {} if arm == 'O' else dict(categorical_feature=categorical_indices())
    with threadpool_limits(limits=threads), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        t0 = time.perf_counter()
        model.fit(mt, np.asarray(y_train).astype(np.int32) - 1, **kw)  # No eval_set / weights / callbacks.
        fit_seconds = time.perf_counter() - t0
        p = checked(model, mv)
        p_train = checked(model, mt)
    require(model.booster_.current_iteration() == params['n_estimators'], 'Full fixed boosting budget not completed')
    info = dict(fit_seconds=fit_seconds, preprocessing_seconds=pre_seconds, encoded_columns=int(mt.shape[1]),
                warnings=[dict(category=w.category.__name__, message=str(w.message)) for w in caught])
    if arm == 'N':
        enc = pre['columns'].named_transformers_['categorical']
        for col, learned in zip(CATEGORICAL, enc.categories_):
            require(set(learned) == set(pre['schema'].transform(X_train)[col]), 'Vocabulary must equal training values')
        codes = mv[:, categorical_indices()]
        info['vocabulary_sizes'] = {c: int(len(v)) for c, v in zip(CATEGORICAL, enc.categories_)}
        info['valid_unseen_nan'] = {c: int(np.isnan(codes[:, i]).sum()) for i, c in enumerate(CATEGORICAL)}
        info['train_nan'] = int(np.isnan(mt[:, categorical_indices()]).sum())
        require(info['train_nan'] == 0, 'Training codes cannot be missing')
    info['age995_median'] = pre['age995'].median_
    return pre, model, p, p_train, mt, mv, info


def checked(model, matrix):
    require(np.array_equal(model.classes_, [0, 1, 2]), 'Internal class order must be 0,1,2')
    p = np.asarray(model.predict_proba(matrix), dtype=np.float64)
    require(p.shape == (matrix.shape[0], 3) and np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all(),
            'Invalid probabilities')
    np.testing.assert_allclose(p.sum(1), 1.0, atol=1e-6, rtol=0)
    return p


def model_text_parameters(path):
    """Effective parameters as written by LightGBM in the saved model text."""
    text = Path(path).read_text(encoding='utf-8')
    params = dict(re.findall(r'^\[([a-z_0-9]+): (.*)\]$', text, flags=re.M))
    infos = re.search(r'^feature_infos=(.*)$', text, flags=re.M).group(1).split(' ')
    return params, infos


def categorical_effective(path):
    params, infos = model_text_parameters(path)
    eff = {k: float(params[k]) for k in CAT_PARAMS}
    retained = {c: (len(infos[i].split(':')) if infos[i] != 'none' else 0)
                for c, i in zip(CATEGORICAL, categorical_indices())} if len(infos) > max(categorical_indices()) else None
    return eff, retained


def same_head(o_matrix, n_matrix):
    """N's numeric/binary/age_is_995 columns must equal O's first 28 columns (no full .toarray())."""
    k = categorical_indices()[0]
    a = o_matrix[:, :k].tocsr()
    b = sparse.csr_matrix(n_matrix[:, :k])
    return a.shape == b.shape and (a != b).nnz == 0


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


def t4_reference(path, ids, y, folds):
    """Pinned T4 LGBM_C_v1 OOF aligned to ids; SHA anchored in config."""
    c = config()['t4_O_stratified_oof']
    require(sha256_file(path) == c['sha256'], 'T4 LGBM_C_v1 OOF SHA differs from verification.json anchor')
    o = read_pred(path).set_index('building_id').loc[np.asarray(ids)].reset_index()
    require(len(o) == len(ids) and o.building_id.is_unique, 'T4 OOF coverage')
    np.testing.assert_array_equal(o.fold.to_numpy(), folds)
    np.testing.assert_array_equal(o.y_true.to_numpy(), np.asarray(y))
    return o


# ---------------------------------------------------------------- control check
def control_check(data, t4_oof, destination):
    """Not a study fit: Arm O refit on stratified fold 0 must equal pinned T4 OOF fold 0."""
    c = config()
    cc = c['control_check']
    dest = Path(destination)
    require(not dest.exists(), 'Do not overwrite an existing control check')
    require(environment() == c['environment'], 'Environment differs from the pinned environment')
    X, y, ids, old, contract = load(data)
    ref = t4_reference(t4_oof, ids, y, old)
    f = cc['fold']
    a, b = np.flatnonzero(old != f), np.flatnonzero(old == f)
    require(len(a) == 166784 and len(b) == 41696, 'Frozen fold size changed')
    pre, model, p, p_train, mt, mv, info = fit_arm('O', X.iloc[a], y.iloc[a], X.iloc[b], c['lgbm_params'],
                                                   c['threadpool_limit'])
    r = ref.iloc[b]
    diff = float(np.max(np.abs(p - r[PROBS].to_numpy())))
    pred = p.argmax(1) + 1
    labels_equal = bool(np.array_equal(pred, r.y_pred.to_numpy()))
    y_equal = bool(np.array_equal(y.iloc[b].to_numpy(), r.y_true.to_numpy()))
    dest.mkdir(parents=True)
    model_path = dest/'O_stratified_fold0.txt'
    model.booster_.save_model(str(model_path))
    t4_model = Path(t4_oof).parent/'fold0.txt'
    out = dict(study_id=STUDY_ID, kind=cc['kind'], counted_as_study_fit=False, classifiers_fitted=1,
               scheme='stratified', fold=f, train_rows=len(a), valid_rows=len(b),
               t4_oof_path=str(Path(t4_oof).resolve()), t4_oof_sha256=sha256_file(t4_oof),
               probability_max_abs_diff=diff, probability_atol=cc['probability_atol'],
               predicted_labels_identical=labels_equal, true_labels_identical=y_equal,
               fold_macro_f1=score(y.iloc[b], pred)['macro_f1'],
               t4_fold_macro_f1=score(r.y_true, r.y_pred)['macro_f1'],
               model_sha256=sha256_file(model_path),
               t4_model_sha256=sha256_file(t4_model) if t4_model.exists() else None,
               model_text_identical_to_t4=(sha256_file(model_path) == sha256_file(t4_model)) if t4_model.exists() else None,
               fit_seconds=info['fit_seconds'], warnings=info['warnings'], environment=environment(),
               holdout_accessed=False, created_at_utc=now())
    out['status'] = 'PASS' if (diff <= cc['probability_atol'] and labels_equal and y_equal) else 'FAIL'
    write(dest/'control_check.json', out)
    print(json.dumps(out, indent=1))
    require(out['status'] == 'PASS', 'Control check failed')
    return out


def probe_categorical_defaults(destination):
    """Synthetic 400-row probe (not a study fit) to read LightGBM 4.6.0 effective categorical defaults."""
    from lightgbm import LGBMClassifier
    rng = np.random.default_rng(0)
    X = np.c_[rng.normal(size=400), rng.integers(0, 6, 400)].astype(np.float32)
    yy = rng.integers(0, 3, 400)
    params = dict(config()['lgbm_params'], n_estimators=2, verbosity=-1)
    m = LGBMClassifier(**params).fit(X, yy, categorical_feature=[1])
    path = Path(destination)/'categorical_defaults_probe.txt'
    m.booster_.save_model(str(path))
    params, _ = model_text_parameters(path)
    eff = {k: float(params[k]) for k in CAT_PARAMS}
    path.unlink()
    return eff


# ---------------------------------------------------------------- prepare/verify
def prepare(data, manifests, t4_oof, lr_oof, output, code_commit, control):
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
        shas[s] = sha256_file(Path(manifests)/s/'manifest.csv')
    require(shas == c['manifest_sha256'], 'Split manifest SHA differs')
    lr_sha = {s: sha256_file(lr_oof[s]) for s in SCHEMES}
    require(lr_sha == c['lr_c_oof_sha256'], 'LR-C OOF SHA differs')
    t4_reference(t4_oof, ids, y, old)
    t4_metrics = Path(t4_oof).parent/'metrics.json'
    require(sha256_file(t4_metrics) == c['t4_O_stratified_oof']['metrics_sha256'], 'T4 metrics SHA differs')
    cc = json.loads(Path(control).read_text(encoding='utf-8'))
    require(cc['status'] == 'PASS' and cc['study_id'] == STUDY_ID
            and cc['t4_oof_sha256'] == c['t4_O_stratified_oof']['sha256'], 'Control check must PASS before prepare')
    output.mkdir(parents=True)
    folds = {'stratified': old}
    for s in ['geo2', 'geo1']:
        (output/s).mkdir()
        shutil.copyfile(Path(manifests)/s/'manifest.csv', output/s/'manifest.csv')
        require(sha256_file(output/s/'manifest.csv') == shas[s], 'Copied manifest SHA differs')
        m = pd.read_csv(output/s/'manifest.csv')
        np.testing.assert_array_equal(m.building_id, ids)
        folds[s] = m.fold.to_numpy()
    (output/'backup').mkdir()
    eff = probe_categorical_defaults(output)
    require(eff == {k: float(v) for k, v in c['lightgbm_categorical_defaults_expected'].items()},
            f'LightGBM categorical defaults differ from expectation: {eff}')
    # LR-C reference means must equal the pre-registered thresholds.
    lr_means = {}
    for s in SCHEMES:
        ref = read_pred(lr_oof[s]).set_index('building_id').loc[ids.to_numpy()].reset_index()
        np.testing.assert_array_equal(ref.fold.to_numpy(), folds[s])
        lr_means[s] = float(np.mean([score(ref.y_true[ref.fold == f], ref.y_pred[ref.fold == f])['macro_f1']
                                     for f in range(5)]))
    for s in ['stratified', 'geo2']:
        require(abs(lr_means[s] - c['lr_c_reference_means'][s]) <= c['lr_c_reference_means']['tolerance'],
                f'LR-C {s} mean differs from the registered threshold')
    design = dict(study_id=STUDY_ID, config=c, code_commit=code_commit, source_sha256=sources(),
                  data_contract=contract, environment=environment(), manifest_sha256=shas,
                  t4_oof_path=str(Path(t4_oof).resolve()), t4_oof_sha256=sha256_file(t4_oof),
                  t4_metrics_sha256=sha256_file(t4_metrics),
                  lr_c_oof_path={s: str(Path(lr_oof[s]).resolve()) for s in SCHEMES}, lr_c_oof_sha256=lr_sha,
                  lr_c_cv_means=lr_means,
                  control_check=dict(path=str(Path(control).resolve()), sha256=sha256_file(control),
                                     status=cc['status'], probability_max_abs_diff=cc['probability_max_abs_diff']),
                  lightgbm_categorical_effective_defaults=eff,
                  categorical_feature_indices_N=categorical_indices(),
                  study_targets=[key(*t) for t in targets()], max_new_study_fits=c['max_new_study_fits'],
                  automatic_retries=0, O_stratified_refits=0, holdout_accessed=False,
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
    pre, model, p, p_train, mt, mv, info = fit_arm(arm, X.iloc[a], y.iloc[a], X.iloc[b], c['lgbm_params'],
                                                   c['threadpool_limit'])
    pred = p.argmax(1) + 1
    head_equal = None
    if arm == 'N':  # N's numeric/binary/flag columns must be byte-equal to O's, built from the same fold
        o_pre = make_preprocessor('O')
        with threadpool_limits(limits=c['threadpool_limit']):
            o_pre.fit(X.iloc[a])
            head_equal = bool(same_head(o_pre.transform(X.iloc[b]), mv))
        require(head_equal, 'N numeric/binary columns differ from O')
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
    reload_diff = float(np.max(np.abs(p2 - p)))
    require(reload_diff <= c['checkpoint']['model_reload_probability_atol']
            and np.array_equal(p2.argmax(1) + 1, pred), f'Model reload mismatch {reload_diff}')
    back = read_pred(preds)
    np.testing.assert_array_equal(back.building_id.to_numpy(), frame.building_id.to_numpy())
    np.testing.assert_array_equal(back.y_pred.to_numpy(), pred)
    np.testing.assert_array_equal(back[PROBS].to_numpy(), p)
    eff, retained = categorical_effective(model_path)
    record = dict(study_id=STUDY_ID, scheme=scheme, arm=arm, fold=f, train_rows=len(a), valid_rows=len(b),
                  train_class_counts=[int((y.iloc[a] == v).sum()) for v in (1, 2, 3)],
                  metrics=score(y.iloc[b], pred), train_macro_f1=score(y.iloc[a], p_train.argmax(1)+1)['macro_f1'],
                  fit_seconds=info['fit_seconds'], preprocessing_seconds=info['preprocessing_seconds'],
                  encoded_columns=info['encoded_columns'], age995_median=info['age995_median'],
                  rounds_completed=int(model.booster_.current_iteration()),
                  vocabulary_sizes=info.get('vocabulary_sizes'), valid_unseen_nan=info.get('valid_unseen_nan'),
                  numeric_binary_columns_equal_O=head_equal,
                  lightgbm_categorical_effective=eff,
                  lightgbm_retained_categories=retained if arm == 'N' else None,
                  categorical_feature=categorical_indices() if arm == 'N' else None,
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


def reference(path, digest, ids, y, folds):
    require(sha256_file(path) == digest, f'Reference OOF changed: {path}')
    r = read_pred(path).set_index('building_id').loc[ids.to_numpy()].reset_index()
    require(len(r) == len(ids) and r.building_id.is_unique, 'Reference OOF coverage')
    np.testing.assert_array_equal(r.fold.to_numpy(), folds)
    np.testing.assert_array_equal(r.y_true.to_numpy(), y.to_numpy())
    return r


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


def finalize_all(data, output):
    from experiments.geo_ablation.audit import turnover
    out = Path(output)
    X, y, ids, folds, design = verify(data, out)
    require(not (out/'analysis.json').exists(), 'analysis.json exists; refusing overwrite')
    ledger = read_ledger(out)
    tg = design['study_targets']
    complete = sum(ledger['folds'].get(k, {}).get('status') == 'complete' for k in tg)
    attempts = sum(e['attempts'] for e in ledger['folds'].values())
    require(complete == len(tg) == 25 and len(ledger['folds']) == 25, 'Ledger incomplete; not finalizing')
    c = design['config']
    res = dict(study_id=STUDY_ID, design_sha256=sha256_file(out/'design.json'), code_commit=design['code_commit'],
               ledger=dict(complete=complete, total=25, attempts=attempts, retries=attempts-complete,
                           fit_seconds_sum=float(sum(e['fit_seconds'] for e in ledger['folds'].values()))),
               O_stratified_refits=0, holdout_accessed=False, rules=c['rules'],
               note='Two arms on the same folds; descriptive comparison, no significance or causal claims.',
               schemes={})
    for s in SCHEMES:
        lr = reference(design['lr_c_oof_path'][s], design['lr_c_oof_sha256'][s], ids, y, folds[s])
        if s == 'stratified':
            o = reference(design['t4_oof_path'], design['t4_oof_sha256'], ids, y, folds[s])
            o_meta = dict(source='pinned T4 LGBM_C_v1 OOF (not refit)', oof_sha256=design['t4_oof_sha256'])
            o_secs = None
        else:
            m = assemble(out, y, ids, folds, s, 'O')
            o = read_pred(out/s/'O'/'oof.csv.gz')
            o_meta = dict(source='T8 study fit', oof_sha256=m['oof_sha256'])
            o_secs = m['fit_seconds']
        mn = assemble(out, y, ids, folds, s, 'N')
        n = read_pred(out/s/'N'/'oof.csv.gz')
        for r in (o, lr):
            np.testing.assert_array_equal(r.building_id.to_numpy(), n.building_id.to_numpy())
        block = {'LR_C': dict(summary(lr, X), oof_sha256=design['lr_c_oof_sha256'][s]),
                 'O': dict(summary(o, X, o_secs), **o_meta),
                 'N': dict(summary(n, X, mn['fit_seconds']), oof_sha256=mn['oof_sha256'])}
        comps = {}
        for ref_name, ref in [('O', o), ('LR_C', lr)]:
            deltas = [a - b for a, b in zip(block['N']['fold_macro_f1'], block[ref_name]['fold_macro_f1'])]
            comps[f'N_vs_{ref_name}'] = dict(
                mean_delta=block['N']['cv_mean'] - block[ref_name]['cv_mean'], fold_delta=deltas,
                fold_wins=int(sum(d > 0 for d in deltas)),
                turnover=turnover(y.to_numpy(), ref.y_pred.to_numpy(), n.y_pred.to_numpy()))
        block['comparisons'] = comps
        res['schemes'][s] = block
    st, g2 = res['schemes']['stratified'], res['schemes']['geo2']
    r1, r2 = c['rules']['primary_representation'], c['rules']['secondary_baseline_challenger']
    res['judgments'] = dict(
        primary_representation_improvement=bool(
            st['comparisons']['N_vs_O']['mean_delta'] >= r1['mean_delta_N_minus_O_min']
            and st['comparisons']['N_vs_O']['fold_wins'] >= r1['fold_wins_min']),
        primary_inputs=dict(mean_delta=st['comparisons']['N_vs_O']['mean_delta'],
                            fold_wins=st['comparisons']['N_vs_O']['fold_wins']),
        secondary_baseline_challenger_candidate=bool(
            st['N']['cv_mean'] > r2['stratified_mean_N_gt']
            and st['comparisons']['N_vs_LR_C']['fold_wins'] >= r2['stratified_fold_wins_vs_LR_C_min']
            and g2['N']['cv_mean'] >= r2['geo2_mean_N_ge']),
        secondary_inputs=dict(stratified_mean_N=st['N']['cv_mean'],
                              stratified_fold_wins_vs_LR_C=st['comparisons']['N_vs_LR_C']['fold_wins'],
                              geo2_mean_N=g2['N']['cv_mean']),
        group_cv_N_vs_O='descriptive only')
    res['finished_at_utc'] = now()
    write(out/'analysis.json', res)
    require(not (out/'backup'/'analysis.json').exists(), 'Backup already exists; refusing overwrite')
    shutil.copy2(out/'analysis.json', out/'backup'/'analysis.json')
    log(out, 'analysis written: ' + json.dumps({k: v for k, v in res['judgments'].items() if not k.endswith('inputs')}))
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['control-check', 'prepare', 'fit-all', 'finalize', 'status'])
    p.add_argument('--data', default='data')
    p.add_argument('--output')
    p.add_argument('--destination')
    p.add_argument('--manifests')
    p.add_argument('--code-commit')
    p.add_argument('--control')
    p.add_argument('--t4-oof')
    for s in SCHEMES:
        p.add_argument(f'--lr-oof-{s}')
    a = p.parse_args()
    if a.action == 'control-check':
        control_check(a.data, a.t4_oof, a.destination)
    elif a.action == 'prepare':
        lr = {s: getattr(a, f'lr_oof_{s}') for s in SCHEMES}
        prepare(a.data, a.manifests, a.t4_oof, lr, a.output, a.code_commit, a.control)
    elif a.action == 'fit-all':
        fit_all(a.data, a.output)
    elif a.action == 'finalize':
        finalize_all(a.data, a.output)
    else:
        ledger = read_ledger(a.output)
        print(json.dumps(dict(complete=sum(e['status'] == 'complete' for e in ledger['folds'].values()), total=25,
                              attempts=sum(e['attempts'] for e in ledger['folds'].values()),
                              running=[k for k, e in ledger['folds'].items() if e['status'] == 'running']), indent=1))


if __name__ == '__main__':
    sys.exit(main())
