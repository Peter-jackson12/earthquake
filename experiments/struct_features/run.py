"""T12: region-free structure-derived features (single factor = appended derived columns).

Bundle S (4 structure summaries), D (3 size/ratio features) or SD (both) are computed per row from raw
inputs (experiments/struct_features/features.py; no region id, no age, no fitted statistic), standardized
by a StandardScaler fit on the training fold, cast to float32 and APPENDED after the existing columns.
  LR_*   = LR-C pipeline (make_pipeline('C')) + derived columns, LogisticRegression unchanged.
  LGBM_* = T4/T8 arm O preprocessor + derived columns, T4 LightGBM params unchanged.
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
import pickle
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
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from preprocess import CATEGORICAL, GEO, _float32, sha256_file  # noqa: E402
from experiments.age995.run import PARAMS as LR_PARAMS, make_pipeline as make_C  # noqa: E402
from experiments.geo_group_cv.run import PROBS, load, require, score, write  # noqa: E402
from experiments.model_families import run as t4  # noqa: E402
from experiments.struct_features import features as F  # noqa: E402

HERE = Path(__file__).resolve().parent
STUDY_ID = 'T12_struct_features_v1'
SCHEMES = ['stratified', 'geo2', 'geo1']
ARMS = ['LR_S', 'LR_D', 'LR_SD', 'LGBM_S', 'LGBM_D', 'LGBM_SD']
BUNDLE = {a: a.split('_')[1] for a in ARMS}
FAMILY = {a: a.split('_')[0] for a in ARMS}
BASELINE_OF = {'LR': 'LR_C', 'LGBM': 'LGBM_O'}
N_TARGETS = 90
SOURCE = ['preprocess.py', 'baseline.py', 'scripts/audit_eda_oof.py', 'configs/data_contract.json',
          'experiments/age995/run.py', 'experiments/geo_group_cv/run.py', 'experiments/geo_ablation/audit.py',
          'experiments/model_families/run.py', 'experiments/model_families/config.json',
          'experiments/struct_features/__init__.py', 'experiments/struct_features/features.py',
          'experiments/struct_features/run.py', 'experiments/struct_features/config.json',
          'docs/T12_STRUCT_FEATURES_PLAN.md']


def now():
    return datetime.now(timezone.utc).isoformat()


def config(path=HERE/'config.json'):
    c = json.loads(Path(path).read_text(encoding='utf-8'))
    require(c['study_id'] == STUDY_ID and c['schemes'] == SCHEMES and c['n_folds'] == 5, 'Unplanned study/schemes')
    require(c['lgbm_params'] == t4.load_config()['models']['LGBM'], 'LightGBM params differ from T4 LGBM_C_v1')
    require(c['lr_params'] == LR_PARAMS, 'LR params differ from LR-C')
    require({k: v['features'] for k, v in c['bundles'].items()} == F.NAMES, 'Bundle features differ from the module')
    require(c['study_fits'] == {a: SCHEMES for a in ARMS} and c['max_new_study_fits'] == N_TARGETS,
            'Unplanned fit plan')
    require(c['automatic_retries'] == 0 and not c['early_stopping'] and c['class_weight'] is None
            and c['sampling'] is None and not c['tuning'] and not c['threshold_or_decision_rule_change']
            and c['outer_parallel_jobs'] == 1 and c['lr_threadpool_limit'] == 1 and c['lgbm_threadpool_limit'] == 2
            and c['stop_on_lr_nonconvergence'] and not c['holdout_access'], 'Unplanned run policy')
    return c


def environment():
    names = ['numpy', 'pandas', 'scipy', 'scikit-learn', 'lightgbm', 'joblib', 'threadpoolctl']
    return dict(python=platform.python_version(), **{n: importlib.metadata.version(n) for n in names})


def sources():
    return {p: sha256_file(ROOT/p) for p in SOURCE}


def targets():
    c = config()
    return [(s, a, f) for a in ARMS for s in c['study_fits'][a] for f in range(5)]


# ---------------------------------------------------------------- pipelines
def add_struct(pre, bundle):
    """Append the derived-feature block AFTER the unchanged C columns."""
    require(bundle in F.BUNDLES, 'Unknown bundle')
    columns = pre['columns']
    require([n for n, _, _ in columns.transformers] == ['numeric', 'binary', 'categorical'], 'Unexpected C paths')
    step = Pipeline([('derive', F.StructFeatures(bundle)), ('scale', StandardScaler()),
                     ('float32', FunctionTransformer(_float32, feature_names_out='one-to-one'))])
    columns.transformers = list(columns.transformers) + [('struct', step, list(F.INPUTS[bundle]))]
    return pre


def make_lr_pipeline(bundle):
    pipe = make_C('C')
    add_struct(pipe['preprocess'], bundle)
    return pipe


def make_lgbm_preprocessor(bundle):
    return add_struct(make_C('C')['preprocess'], bundle)


def check_matrix(pre, matrix, bundle):
    """CSR float32, finite; derived block last; no geo/age-derived column among the appended ones."""
    names = pre['columns'].get_feature_names_out().tolist()
    k = len(F.NAMES[bundle])
    require(sparse.isspmatrix_csr(matrix) and matrix.dtype == np.float32, 'Matrix must be float32 CSR')
    require(np.isfinite(matrix.data).all(), 'Nonfinite feature')
    require(matrix.shape[1] == len(names), 'Names/columns mismatch')
    require(names[-k:] == ['struct__' + n for n in F.NAMES[bundle]], 'Derived block position changed')
    require(not any(n.startswith('struct__') for n in names[:-k]), 'Derived column outside the appended block')
    require(not any(('geo' in n or 'age' in n) for n in names[-k:]), 'Geo/age-derived column present')
    return names


def reference_scaled(X_train, X_valid, bundle):
    """Independent pandas formulation of the derived columns, standardized with training statistics."""
    def raw(X):
        out = {}
        sup = [c for c in X.columns if c.startswith('has_superstructure_')]
        use = [c for c in X.columns if c.startswith('has_secondary_use_')]  # trailing '_' excludes the aggregate
        pre = 'has_superstructure_'
        weak = [pre + s for s in ('adobe_mud', 'mud_mortar_stone', 'stone_flag', 'mud_mortar_brick', 'timber', 'bamboo')]
        strong = [pre + s for s in ('cement_mortar_stone', 'cement_mortar_brick', 'rc_non_engineered', 'rc_engineered')]
        fl = X['count_floors_pre_eq'].astype('float64')
        guard = fl.where(fl >= 1, 1.0)
        if 'S' in bundle:
            out.update(n_super=X[sup].astype('float64').sum(1), n_weak=X[weak].astype('float64').sum(1),
                       n_strong=X[strong].astype('float64').sum(1), n_use=X[use].astype('float64').sum(1))
        if 'D' in bundle:
            out.update(height_per_floor=X['height_percentage'].astype('float64') / guard,
                       area_x_floors=X['area_percentage'].astype('float64') * fl,
                       families_per_floor=X['count_families'].astype('float64') / guard)
        return pd.DataFrame(out)[F.NAMES[bundle]]
    tr, va = raw(X_train), raw(X_valid)
    mean, std = tr.mean(0), tr.std(0, ddof=0)
    return ((tr - mean) / std).to_numpy(), ((va - mean) / std).to_numpy(), mean.to_numpy(), std.to_numpy()


def proba(model, matrix, family):
    expected = [1, 2, 3] if family == 'LR' else [0, 1, 2]
    require(np.array_equal(model.classes_, expected), 'Unexpected class order')
    p = np.asarray(model.predict_proba(matrix), dtype=np.float64)
    require(p.shape == (matrix.shape[0], 3) and np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all(),
            'Invalid probabilities')
    np.testing.assert_allclose(p.sum(1), 1.0, atol=1e-6, rtol=0)
    return p


def fit_arm(arm, X_train, y_train, X_valid, c):
    """Fit preprocessor/model on the training fold only. Validation labels are not an argument."""
    from lightgbm import LGBMClassifier
    bundle, family = BUNDLE[arm], FAMILY[arm]
    k = len(F.NAMES[bundle])
    if family == 'LR':
        pipe = make_lr_pipeline(bundle)
        pre, model, threads = pipe['preprocess'], pipe['model'], c['lr_threadpool_limit']
        require(model.get_params()['C'] == 1.0 and model.get_params()['solver'] == 'lbfgs', 'LR params changed')
    else:
        pre = make_lgbm_preprocessor(bundle)
        model, threads = LGBMClassifier(**c['lgbm_params']), c['lgbm_threadpool_limit']
        pipe = Pipeline([('preprocess', pre), ('model', model)])
    with threadpool_limits(limits=threads):
        t0 = time.perf_counter()
        mt = pre.fit_transform(X_train)
        mv = pre.transform(X_valid)
        pre_seconds = time.perf_counter() - t0
    names = check_matrix(pre, mt, bundle)
    check_matrix(pre, mv, bundle)
    # derived columns equal an independent pandas computation standardized with training statistics
    rt, rv, mean, std = reference_scaled(X_train, X_valid, bundle)
    np.testing.assert_allclose(mt[:, -k:].toarray(), rt, rtol=1e-5, atol=2e-5)
    np.testing.assert_allclose(mv[:, -k:].toarray(), rv, rtol=1e-5, atol=2e-5)
    scaler = pre['columns'].named_transformers_['struct']['scale']
    np.testing.assert_allclose(scaler.mean_, mean, rtol=1e-12, atol=1e-12)
    yt = np.asarray(y_train)
    with threadpool_limits(limits=threads), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        t0 = time.perf_counter()
        if family == 'LR':
            model.fit(mt, yt)
        else:
            model.fit(mt, yt.astype(np.int32) - 1)  # No eval_set / weights / callbacks.
        fit_seconds = time.perf_counter() - t0
        p = proba(model, mv, family)
        p_train = proba(model, mt, family)
    info = dict(fit_seconds=fit_seconds, preprocessing_seconds=pre_seconds, encoded_columns=int(mt.shape[1]),
                feature_names_sha256=hashlib.sha256('\n'.join(names).encode()).hexdigest(),
                warnings=[dict(category=w.category.__name__, message=str(w.message)) for w in caught],
                age995_median=pre['age995'].median_,
                derived=dict(names=F.NAMES[bundle], train_mean=scaler.mean_.tolist(), train_scale=scaler.scale_.tolist(),
                             valid_scaled_mean=rv.mean(0).tolist(), valid_scaled_std=rv.std(0).tolist()))
    if family == 'LR':
        info['converged'] = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
        info['n_iter'] = model.n_iter_.tolist()
        info['derived_coef'] = model.coef_[:, -k:].tolist()
    else:
        require(model.booster_.current_iteration() == c['lgbm_params']['n_estimators'],
                'Full fixed boosting budget not completed')
        info['rounds_completed'] = int(model.booster_.current_iteration())
        gain = model.booster_.feature_importance(importance_type='gain')
        info['derived_gain_share'] = float(gain[-k:].sum() / gain.sum()) if gain.sum() else None
        info['derived_gain'] = dict(zip(F.NAMES[bundle], map(float, gain[-k:])))
    return pipe, pre, model, p, p_train, mt, mv, names, info


def same_prefix_as_C(pre_c, m_c, m_s, names_s, k):
    """Every column of the C matrix is unchanged and the derived block follows it (sparse compare)."""
    names_c = pre_c['columns'].get_feature_names_out().tolist()
    width = len(names_c)
    return bool(names_s[:width] == names_c and m_s.shape[1] == width + k
                and (m_c != m_s[:, :width]).nnz == 0), width


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
def py_reference(row):
    """Plain-python hand computation for one raw row (dict of python scalars), independent of the vectorized code."""
    s = 'has_superstructure_'
    n_super = (row[s+'adobe_mud'] + row[s+'mud_mortar_stone'] + row[s+'stone_flag'] + row[s+'cement_mortar_stone']
               + row[s+'mud_mortar_brick'] + row[s+'cement_mortar_brick'] + row[s+'timber'] + row[s+'bamboo']
               + row[s+'rc_non_engineered'] + row[s+'rc_engineered'] + row[s+'other'])
    n_weak = (row[s+'adobe_mud'] + row[s+'mud_mortar_stone'] + row[s+'stone_flag'] + row[s+'mud_mortar_brick']
              + row[s+'timber'] + row[s+'bamboo'])
    n_strong = (row[s+'cement_mortar_stone'] + row[s+'cement_mortar_brick'] + row[s+'rc_non_engineered']
                + row[s+'rc_engineered'])
    u = 'has_secondary_use_'
    n_use = (row[u+'agriculture'] + row[u+'hotel'] + row[u+'rental'] + row[u+'institution'] + row[u+'school']
             + row[u+'industry'] + row[u+'health_post'] + row[u+'gov_office'] + row[u+'use_police'] + row[u+'other'])
    fl = row['count_floors_pre_eq']
    g = fl if fl >= 1 else 1
    return [float(n_super), float(n_weak), float(n_strong), float(n_use),
            row['height_percentage'] / g, float(row['area_percentage'] * fl), row['count_families'] / g]


def synthetic_edge_frame():
    base = {c: 0 for c in F.INPUTS['SD']}
    base['has_secondary_use'] = 0  # aggregate flag present in the frame but never used
    rows = []
    for floors in (0, 1, 3):
        rows.append(dict(base, count_floors_pre_eq=floors, height_percentage=10, area_percentage=7, count_families=2))
    r = dict(base, count_floors_pre_eq=2, height_percentage=6, area_percentage=5, count_families=1, has_secondary_use=1)
    r.update({c: 1 for c in F.SUPER})
    rows.append(r)  # all 11 superstructure flags on; aggregate use flag on but no specific use
    r = dict(base, count_floors_pre_eq=1, height_percentage=4, area_percentage=3, count_families=0,
             has_superstructure_timber=1, has_superstructure_rc_engineered=1, has_secondary_use_hotel=1,
             has_secondary_use_school=1)
    rows.append(r)
    return pd.DataFrame(rows)


SYNTH_EXPECTED = np.array([
    [0, 0, 0, 0, 10.0, 0.0, 2.0],       # floors 0 -> guard 1
    [0, 0, 0, 0, 10.0, 7.0, 2.0],       # floors 1
    [0, 0, 0, 0, 10.0/3, 21.0, 2.0/3],  # floors 3
    [11, 6, 4, 0, 3.0, 10.0, 0.5],      # all superstructure flags; aggregate use flag excluded
    [2, 1, 1, 2, 4.0, 3.0, 0.0],        # timber (weak) + rc_engineered (strong); hotel + school
])


FRESH_CODE = r'''
import sys, json, joblib, numpy as np, pandas as pd
sys.path.insert(0, sys.argv[1])
ref = json.load(open(sys.argv[2])); X = pd.read_pickle(sys.argv[3])
def modules(obj, seen):
    seen.add(type(obj).__module__)
    for attr in ('steps', 'transformers_'):
        for item in getattr(obj, attr, []) or []:
            part = item[1]
            if hasattr(part, 'get_params'):
                modules(part, seen)
    return seen
out = {}
for name, path in ref['paths'].items():
    obj = joblib.load(path)
    mods = sorted(modules(obj, set()))
    if name == 'lr':
        got = obj.predict_proba(X)
    else:
        import lightgbm
        booster = lightgbm.Booster(model_file=ref['lgbm_model'])
        got = np.asarray(booster.predict(obj.transform(X), num_threads=2), dtype=np.float64)
    out[name] = dict(modules=mods, main_free=not any(m == '__main__' for m in mods),
                     max_abs_diff=float(np.max(np.abs(got - np.asarray(ref['probs'][name])))))
print(json.dumps(out))
'''


def fresh_process_check(ref_file, sample_file):
    """Load fitted artifacts in a new isolated interpreter: no __main__ classes, identical outputs."""
    res = subprocess.run([sys.executable, '-I', '-c', FRESH_CODE, str(ROOT), str(ref_file), str(sample_file)],
                         capture_output=True, text=True, cwd=ROOT)
    require(res.returncode == 0, f'Fresh-process load failed: {res.stderr[-800:]}')
    return json.loads(res.stdout.strip().splitlines()[-1])


def control_check(data, destination):
    """Not a study fit: derived-feature correctness, append invariance, no geo/age columns, fresh-process load."""
    c = config()
    cc = c['control_check']
    dest = Path(destination)
    require(not dest.exists(), 'Do not overwrite an existing control check')
    require(environment() == c['environment'], 'Environment differs from the pinned environment')
    atol, seed = cc['atol'], cc['subsample_seed']
    res = dict(study_id=STUDY_ID, kind=cc['kind'], counted_as_study_fit=False, created_at_utc=now())
    # (a1) synthetic edge cases vs hand-computed constants and vs the plain-python reference
    edge = synthetic_edge_frame()
    got = F.derive(edge, 'SD')
    py = np.array([py_reference(r) for r in edge.to_dict('records')])
    res['a_synthetic_edge_cases'] = dict(
        hand_max_abs_diff=float(np.max(np.abs(got - SYNTH_EXPECTED))), python_max_abs_diff=float(np.max(np.abs(got - py))),
        floors_zero_guarded=bool(got[0, 4] == 10.0 and got[0, 6] == 2.0 and got[0, 5] == 0.0),
        aggregate_use_flag_excluded=bool(got[3, 3] == 0.0),
        pass_=bool(np.max(np.abs(got - SYNTH_EXPECTED)) <= atol and np.max(np.abs(got - py)) <= atol))
    # (a2) real development rows
    X, y, ids, old, contract = load(data)
    n_super = X[F.SUPER].to_numpy(dtype=np.int64).sum(1)
    n_use = X[F.USE].to_numpy(dtype=np.int64).sum(1)
    picks = sorted({0, 1, 2, int(np.argmax(n_super)), int(np.argmax(n_use)),
                    int(np.flatnonzero(X.count_floors_pre_eq.eq(1))[0]), int(np.argmax(X.count_floors_pre_eq.to_numpy())),
                    int(np.flatnonzero(X.count_families.eq(0))[0]), int(np.flatnonzero(X.has_secondary_use.eq(1))[0])})
    real = X.iloc[picks]
    real_got = F.derive(real, 'SD')
    real_py = np.array([py_reference({k2: (v.item() if hasattr(v, 'item') else v) for k2, v in r.items()})
                        for r in real.to_dict('records')])
    res['a_real_rows'] = dict(rows=picks, max_abs_diff=float(np.max(np.abs(real_got - real_py))),
                              pass_=bool(np.max(np.abs(real_got - real_py)) <= atol))
    fl = X.count_floors_pre_eq
    res['observed_count_floors_pre_eq'] = dict(min=int(fl.min()), max=int(fl.max()), rows_floors_0=int((fl == 0).sum()),
                                               rows_floors_1=int((fl == 1).sum()), rows=len(fl))
    # (b) append invariance on a 5k stratified subsample; derived columns equal an independent computation
    sub, _ = train_test_split(np.arange(len(y)), train_size=cc['subsample_rows'], stratify=y, random_state=seed)
    sub = np.sort(sub)
    Xs, ys = X.iloc[sub].reset_index(drop=True), y.iloc[sub].reset_index(drop=True)
    a, b = train_test_split(np.arange(len(ys)), test_size=0.2, stratify=ys, random_state=seed)
    a, b = np.sort(a), np.sort(b)
    pre_c = make_C('C')['preprocess']
    with threadpool_limits(limits=1):
        mc_a, mc_b = pre_c.fit_transform(Xs.iloc[a]), pre_c.transform(Xs.iloc[b])
    names_c = pre_c['columns'].get_feature_names_out().tolist()
    inv = {}
    for fam, build in [('LR', lambda bd: make_lr_pipeline(bd)['preprocess']), ('LGBM', make_lgbm_preprocessor)]:
        for bundle in F.BUNDLES:
            pre = build(bundle)
            with threadpool_limits(limits=1):
                ma, mb = pre.fit_transform(Xs.iloc[a]), pre.transform(Xs.iloc[b])
            names = check_matrix(pre, ma, bundle)
            check_matrix(pre, mb, bundle)
            k = len(F.NAMES[bundle])
            eq_a, width = same_prefix_as_C(pre_c, mc_a, ma, names, k)
            eq_b, _ = same_prefix_as_C(pre_c, mc_b, mb, names, k)
            rt, rv, _, _ = reference_scaled(Xs.iloc[a], Xs.iloc[b], bundle)
            d_a = float(np.max(np.abs(ma[:, -k:].toarray() - rt)))
            d_b = float(np.max(np.abs(mb[:, -k:].toarray() - rv)))
            geo_same = bool(sum('geo_level' in n for n in names) == sum('geo_level' in n for n in names_c))
            inv[f'{fam}_{bundle}'] = dict(c_columns=width, appended=k, c_prefix_equal_train=eq_a, c_prefix_equal_valid=eq_b,
                                         derived_max_abs_diff_train=d_a, derived_max_abs_diff_valid=d_b,
                                         geo_onehot_columns_same_as_C=geo_same, appended_names=names[-k:],
                                         ok=bool(eq_a and eq_b and d_a <= 2e-5 and d_b <= 2e-5 and geo_same))
    res['b_append_invariance'] = dict(arms=inv, pass_=all(v['ok'] for v in inv.values()))
    # (c) no geo/age inputs in any derived block
    used = set(sum(F.INPUTS.values(), []))
    res['c_no_geo_or_age'] = dict(inputs_total=len(used), geo_in_inputs=sorted(used & set(GEO)),
                                  age_in_inputs='age' in used, categorical_in_inputs=sorted(used & set(CATEGORICAL)),
                                  excluded_t3_names_absent=not ({'age_band', 'geo_foundation', 'geo_age', 'foundation_age'}
                                                                & {n for v in F.NAMES.values() for n in v}),
                                  pass_=bool(not (used & set(GEO)) and 'age' not in used and not (used & set(CATEGORICAL))))
    # (d) smoke fits on the subsample (not study fits) + fresh-process artifact load
    dest.mkdir(parents=True)
    smoke, paths, probs = {}, {}, {}
    for arm in ['LR_SD', 'LGBM_SD']:
        pipe, pre, model, p, p_train, mt, mv, names, info = fit_arm(arm, Xs.iloc[a], ys.iloc[a], Xs.iloc[b], c)
        nk = len(F.NAMES['SD'])
        eq_v, _ = same_prefix_as_C(pre_c, mc_b, mv, names, nk)
        eq_t, _ = same_prefix_as_C(pre_c, mc_a, mt, names, nk)
        pred = p.argmax(1) + 1
        raw_ok = all(b'__main__' not in pickle.dumps(o) for o in (pipe, pre))
        smoke[arm] = dict(sample_macro_f1=score(ys.iloc[b], pred)['macro_f1'], columns=int(mt.shape[1]),
                          prefix_equal_C=bool(eq_v and eq_t), converged=info.get('converged'),
                          fit_seconds=info['fit_seconds'], warnings=info['warnings'], pickle_has_no___main__=raw_ok)
        if arm == 'LR_SD':
            joblib.dump(pipe, dest/'lr_sample.joblib', compress=3)
            paths['lr'] = str(dest/'lr_sample.joblib')
            probs['lr'] = p.tolist()
        else:
            model.booster_.save_model(str(dest/'lgbm_sample.txt'))
            joblib.dump(pre, dest/'lgbm_pre_sample.joblib', compress=3)
            paths['lgbm'] = str(dest/'lgbm_pre_sample.joblib')
            probs['lgbm'] = p.tolist()
    Xs.iloc[b].to_pickle(dest/'sample_valid.pkl')
    write(dest/'fresh_ref.json', dict(paths=paths, probs=probs, lgbm_model=str(dest/'lgbm_sample.txt')))
    fresh = fresh_process_check(dest/'fresh_ref.json', dest/'sample_valid.pkl')
    res['d_smoke_fits'] = dict(note='sample fits on 4k subsample rows; not study fits', arms=smoke,
                               pass_=all(v['prefix_equal_C'] and v['converged'] is not False and v['pickle_has_no___main__']
                                         for v in smoke.values()))
    res['d_fresh_process_load'] = dict(arms=fresh, pass_=all(v['main_free'] and v['max_abs_diff'] <= 1e-12
                                                             for v in fresh.values()))
    res['classifiers_fitted_sample_only'] = 2
    res['environment'] = environment()
    res['holdout_accessed'] = False
    res['status'] = 'PASS' if all(v['pass_'] for v in res.values() if isinstance(v, dict) and 'pass_' in v) else 'FAIL'
    write(dest/'control_check.json', res)
    print(json.dumps({k2: (v['pass_'] if isinstance(v, dict) and 'pass_' in v else v) for k2, v in res.items()
                      if k2 not in ('d_smoke_fits', 'b_append_invariance')}, indent=1, default=str))
    require(res['status'] == 'PASS', 'Control check failed')
    return res


# ---------------------------------------------------------------- prepare/verify
def prepare(data, t10_design, manifests, output, code_commit, control):
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
    manifests = Path(manifests)
    shas = {'stratified': sha256_file(Path(data)/'split_manifest.csv')}
    for s in ['geo2', 'geo1']:
        shas[s] = sha256_file(manifests/s/'manifest.csv')
    require(shas == c['manifest_sha256'], 'Split manifest SHA differs')
    cc = json.loads(Path(control).read_text(encoding='utf-8'))
    require(cc['status'] == 'PASS' and cc['study_id'] == STUDY_ID, 'Control check must PASS before prepare')
    t10 = json.loads(Path(t10_design).read_text(encoding='utf-8'))
    paths = {n: {s: t10['baselines'][n][s]['path'] for s in SCHEMES} for n in ['LR_C', 'LGBM_O']}
    for n in paths:
        for s in SCHEMES:
            require(t10['baselines'][n][s]['sha256'] == c['baselines'][n]['oof_sha256'][s], f'T10 anchor differs {n}/{s}')
    output.mkdir(parents=True)
    folds = {'stratified': old}
    for s in ['geo2', 'geo1']:
        (output/s).mkdir()
        shutil.copyfile(manifests/s/'manifest.csv', output/s/'manifest.csv')
        require(sha256_file(output/s/'manifest.csv') == shas[s], 'Copied manifest SHA differs')
        mf = pd.read_csv(output/s/'manifest.csv')
        np.testing.assert_array_equal(mf.building_id, ids)
        folds[s] = mf.fold.to_numpy()
    (output/'backup').mkdir()
    public = json.loads((ROOT/'reports/lgbm_native_cat_20261014.json').read_text(encoding='utf-8'))
    pub = {(m['scheme'], m['arm']): m['fold_macro_f1'] for m in public['models']}
    base = {}
    for name, pub_arm in [('LR_C', 'LR_C'), ('LGBM_O', 'O')]:
        base[name] = {}
        for s in SCHEMES:
            r = reference(paths[name][s], c['baselines'][name]['oof_sha256'][s], ids, y, folds[s])
            vals = fold_means(r)
            sc = score(r.y_true, r.y_pred)
            anchored = (s, pub_arm) in pub
            if anchored:
                np.testing.assert_allclose(vals, pub[(s, pub_arm)], rtol=0, atol=1e-12)
            base[name][s] = dict(path=str(Path(paths[name][s]).resolve()), sha256=c['baselines'][name]['oof_sha256'][s],
                                 cv_mean=float(np.mean(vals)), fold_macro_f1=vals,
                                 pooled_grade3_recall=sc['recall'][2], pooled_error32=sc['error32'],
                                 public_anchor_checked=anchored)
    ref = c['baseline_references']
    for name in ['LR_C', 'LGBM_O']:
        r0 = ref[name]
        require(abs(base[name]['stratified']['cv_mean'] - r0['stratified_mean']) <= ref['tolerance_mean']
                and abs(base[name]['geo2']['cv_mean'] - r0['geo2_mean']) <= ref['tolerance_mean']
                and abs(base[name]['geo2']['pooled_grade3_recall'] - r0['geo2_pooled_grade3_recall']) <= ref['tolerance_recall'],
                f'{name} reference values differ from the registered thresholds')
    design = dict(study_id=STUDY_ID, config=c, code_commit=code_commit, source_sha256=sources(),
                  data_contract=contract, environment=environment(), manifest_sha256=shas,
                  t10_design=dict(path=str(Path(t10_design).resolve()), sha256=sha256_file(t10_design)),
                  baselines=base, observed_count_floors_pre_eq=cc['observed_count_floors_pre_eq'],
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
    family, bundle = FAMILY[arm], BUNDLE[arm]
    nk = len(F.NAMES[bundle])
    a, b = np.flatnonzero(folds[scheme] != f), np.flatnonzero(folds[scheme] == f)
    require(len(a) + len(b) == len(ids) and len(b) > 0, 'Fold membership invalid')
    pipe, pre, model, p, p_train, mt, mv, names, info = fit_arm(arm, X.iloc[a], y.iloc[a], X.iloc[b], c)
    pred = p.argmax(1) + 1
    threads = c['lr_threadpool_limit'] if family == 'LR' else c['lgbm_threadpool_limit']
    # Existing C columns unchanged and the derived block appended (same fold's C preprocessor, sparse compare).
    pre_c = make_C('C')['preprocess']
    with threadpool_limits(limits=threads):
        pre_c.fit(X.iloc[a])
        eq_v, width = same_prefix_as_C(pre_c, pre_c.transform(X.iloc[b]), mv, names, nk)
        eq_t, _ = same_prefix_as_C(pre_c, pre_c.transform(X.iloc[a]), mt, names, nk)
    require(eq_v and eq_t, 'Existing C columns changed')
    preds = d/f'fold_{f}_pred.csv.gz'
    if family == 'LR':
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
        if family == 'LR':
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
    record = dict(study_id=STUDY_ID, scheme=scheme, arm=arm, fold=f, bundle=bundle, family=family,
                  train_rows=len(a), valid_rows=len(b),
                  train_class_counts=[int((y.iloc[a] == v).sum()) for v in (1, 2, 3)],
                  metrics=score(y.iloc[b], pred), train_macro_f1=score(y.iloc[a], p_train.argmax(1)+1)['macro_f1'],
                  fit_seconds=info['fit_seconds'], preprocessing_seconds=info['preprocessing_seconds'],
                  encoded_columns=info['encoded_columns'], c_columns=width, appended_columns=nk,
                  feature_names_sha256=info['feature_names_sha256'], c_columns_unchanged=True,
                  age995_median=info['age995_median'], derived=info['derived'],
                  converged=info.get('converged'), n_iter=info.get('n_iter'), derived_coef=info.get('derived_coef'),
                  rounds_completed=info.get('rounds_completed'), derived_gain_share=info.get('derived_gain_share'),
                  derived_gain=info.get('derived_gain'),
                  effective_parameters={k2: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
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
             f'iter={info.get("n_iter")} reload_diff={reload_diff:.3g} saved+verified+backed-up')
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
    pa, pb = block[a_name]['pooled'], block[b_name]['pooled']
    return dict(mean_delta=block[a_name]['cv_mean'] - block[b_name]['cv_mean'], fold_delta=deltas,
                fold_wins=int(sum(dd > 0 for dd in deltas)),
                pooled_grade3_recall_delta=pa['recall'][2] - pb['recall'][2],
                pooled_error32_ratio=(pa['error32'] / pb['error32']) if pb['error32'] else None,
                turnover=turnover(y, b_oof.y_pred.to_numpy(), a_oof.y_pred.to_numpy()))


def judge(st, g2, arm, r):
    """Pre-registered joint improvement candidate for one arm vs its own baseline."""
    base = BASELINE_OF[FAMILY[arm]]
    cs, cg = st['comparisons'][f'{arm}_vs_{base}'], g2['comparisons'][f'{arm}_vs_{base}']
    ratio = r['pooled_error32_ratio_max']
    conds = {
        'c1_stratified_mean_and_wins': bool(cs['mean_delta'] >= r['stratified_mean_delta_min']
                                            and cs['fold_wins'] >= r['stratified_fold_wins_min']),
        'c2_geo2_mean_and_wins': bool(cg['mean_delta'] >= r['geo2_mean_delta_min'] and cg['fold_wins'] >= r['geo2_fold_wins_min']),
        'c3_grade3_recall': bool(cs['pooled_grade3_recall_delta'] >= r['pooled_grade3_recall_delta_min']
                                 and cg['pooled_grade3_recall_delta'] >= r['pooled_grade3_recall_delta_min']),
        'c4_error32': bool(st[arm]['pooled']['error32'] <= ratio * st[base]['pooled']['error32']
                           and g2[arm]['pooled']['error32'] <= ratio * g2[base]['pooled']['error32'])}
    inputs = dict(stratified_mean_delta=cs['mean_delta'], stratified_fold_wins=cs['fold_wins'],
                  geo2_mean_delta=cg['mean_delta'], geo2_fold_wins=cg['fold_wins'],
                  stratified_grade3_recall_delta=cs['pooled_grade3_recall_delta'],
                  geo2_grade3_recall_delta=cg['pooled_grade3_recall_delta'],
                  stratified_error32_arm_vs_base=[st[arm]['pooled']['error32'], st[base]['pooled']['error32']],
                  geo2_error32_arm_vs_base=[g2[arm]['pooled']['error32'], g2[base]['pooled']['error32']])
    return dict(joint_improvement_candidate=bool(all(conds.values())), conditions=conds, inputs=inputs)


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
                           fit_seconds_sum=float(sum(e['fit_seconds'] for e in ledger['folds'].values())),
                           all_lr_converged=all(e.get('converged', True) for e in ledger['folds'].values())),
               baseline_refits=0, holdout_accessed=False, rules=c['rules'],
               observed_count_floors_pre_eq=design['observed_count_floors_pre_eq'],
               note=c['rules']['multiplicity'] + '; descriptive, no significance or causal claims.',
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
            lr = FAMILY[arm] == 'LR'
            block[arm] = dict(summary(oofs[arm], X, mt['fit_seconds']), oof_sha256=mt['oof_sha256'],
                              derived_names=recs[0]['derived']['names'],
                              converged=[r['converged'] for r in recs] if lr else None,
                              n_iter=[r['n_iter'] for r in recs] if lr else None,
                              derived_coef_by_fold=[r['derived_coef'] for r in recs] if lr else None,
                              derived_gain_share_by_fold=None if lr else [r['derived_gain_share'] for r in recs],
                              derived_gain_by_fold=None if lr else [r['derived_gain'] for r in recs])
        comps = {}
        for arm in ARMS:
            own = BASELINE_OF[FAMILY[arm]]
            comps[f'{arm}_vs_{own}'] = compare(block, arm, own, yv, oofs[arm], oofs[own])
            if own != 'LR_C':
                comps[f'{arm}_vs_LR_C'] = compare(block, arm, 'LR_C', yv, oofs[arm], oofs['LR_C'])
        comps['LGBM_O_vs_LR_C'] = compare(block, 'LGBM_O', 'LR_C', yv, oofs['LGBM_O'], oofs['LR_C'])
        block['comparisons'] = comps
        res['schemes'][s] = block
    st, g2 = res['schemes']['stratified'], res['schemes']['geo2']
    r = c['rules']['joint_improvement_candidate']
    res['judgments'] = {arm: judge(st, g2, arm, r) for arm in ARMS}
    res['judgments']['candidates'] = [a for a in ARMS if res['judgments'][a]['joint_improvement_candidate']]
    res['judgments']['geo1_scheme'] = 'descriptive only'
    res['finished_at_utc'] = now()
    write(out/'analysis.json', res)
    require(not (out/'backup'/'analysis.json').exists(), 'Backup already exists; refusing overwrite')
    shutil.copy2(out/'analysis.json', out/'backup'/'analysis.json')
    log(out, 'analysis written: ' + json.dumps({a: res['judgments'][a]['joint_improvement_candidate'] for a in ARMS}))
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['control-check', 'prepare', 'fit-all', 'finalize', 'status'])
    p.add_argument('--data', default='data')
    p.add_argument('--output')
    p.add_argument('--destination')
    p.add_argument('--t10-design')
    p.add_argument('--manifests')
    p.add_argument('--code-commit')
    p.add_argument('--control')
    a = p.parse_args()
    if a.action == 'control-check':
        control_check(a.data, a.destination)
    elif a.action == 'prepare':
        prepare(a.data, a.t10_design, a.manifests, a.output, a.code_commit, a.control)
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
