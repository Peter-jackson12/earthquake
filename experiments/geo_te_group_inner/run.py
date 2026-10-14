"""T11: group-aware inner-OOF hierarchical geo target encoding (single factor vs T10 = the inner splitter).

T10 encoded TRAINING rows with a random inner StratifiedKFold, so under geo2/geo1 Group CV nearly every training
row saw its own region while every validation row was an unseen region (parent value, cnt = 0). T11 changes only
the inner splitter: StratifiedGroupKFold(5, shuffle=True, random_state=42) with groups = (geo1, geo2) for the
outer geo2 scheme and geo1 for the outer geo1 scheme, so training rows are also encoded as unseen regions.
Everything else (formula, m = 20, composite keys, 9 features, parent fallback, validation encoding from the full
outer training fold, LR/LightGBM parameters, preprocessing) is T10's, imported unchanged.
The stratified scheme is not refit: its inner split equals T10's, so the T10 stratified OOF is reused.
Arms: LR_TE_G, LGBM_TE_G x {geo2, geo1} x 5 folds = 20 fits. Holdout is never read. Private outputs only.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from preprocess import GEO, NUMERIC, TEXT, BINARY, CATEGORICAL, sha256_file  # noqa: E402
from experiments.age995.run import FLAG, PARAMS as LR_PARAMS, make_pipeline as make_C  # noqa: E402
from experiments.geo_group_cv.run import PROBS, load, require, score, write  # noqa: E402
from experiments.model_families import run as t4  # noqa: E402
from experiments.geo_hier_te import run as t10  # noqa: E402
from experiments.geo_te_group_inner.encoder import (GroupInnerTargetEncoder, group_ids,  # noqa: E402
                                                    inner_group_fold_ids)

HERE = Path(__file__).resolve().parent
STUDY_ID = 'T11_geo_te_group_inner_v1'
SCHEMES = ['stratified', 'geo2', 'geo1']
FIT_SCHEMES = ['geo2', 'geo1']
ARMS = ['LR_TE_G', 'LGBM_TE_G']
T10_ARM = {'LR_TE_G': 'LR_TE', 'LGBM_TE_G': 'LGBM_TE'}
BASELINE_OF = {'LR_TE_G': 'LR_C', 'LGBM_TE_G': 'LGBM_O'}
FAMILY = {'LR_TE_G': 'LR', 'LGBM_TE_G': 'LGBM'}
LEVELS, TE_NAMES, HEAD, SEP = t10.LEVELS, t10.TE_NAMES, t10.HEAD, t10.SEP
N_TARGETS = 20
ENCODER_MODULE = 'experiments.geo_te_group_inner.encoder'
SOURCE = [x for x in t10.SOURCE if not x.startswith(('experiments/geo_hier_te/', 'docs/T10'))] + [
    'experiments/geo_hier_te/run.py', 'experiments/geo_hier_te/config.json', 'experiments/geo_hier_te/__init__.py',
    'experiments/geo_te_group_inner/__init__.py', 'experiments/geo_te_group_inner/encoder.py',
    'experiments/geo_te_group_inner/run.py', 'experiments/geo_te_group_inner/config.json',
    'tests/test_geo_te_group_inner.py', 'docs/T11_GEO_TE_GROUP_INNER_PLAN.md']
SOURCE = list(dict.fromkeys(SOURCE))
now, environment, log, event = t10.now, t10.environment, t10.log, t10.event
atomic_write, read_pred, write_pred = t10.atomic_write, t10.read_pred, t10.write_pred
read_ledger, key, check_saved, reference, fold_means = (t10.read_ledger, t10.key, t10.check_saved,
                                                         t10.reference, t10.fold_means)


def config(path=HERE/'config.json'):
    c = json.loads(Path(path).read_text(encoding='utf-8'))
    t = t10.config()
    require(c['study_id'] == STUDY_ID and c['schemes'] == SCHEMES and c['fit_schemes'] == FIT_SCHEMES
            and c['n_folds'] == 5, 'Unplanned study/schemes')
    require(c['lgbm_params'] == t4.load_config()['models']['LGBM'] == t['lgbm_params'], 'LightGBM params differ')
    require(c['lr_params'] == LR_PARAMS == t['lr_params'], 'LR params differ from LR-C')
    require(c['geo_columns'] == list(GEO) and c['text_categorical_columns'] == list(TEXT)
            and list(GEO) + list(TEXT) == list(CATEGORICAL), 'Categorical column order changed')
    te, tt = c['te'], t['te']
    require(all(te[k] == tt[k] for k in ['feature_names', 'smoothing_m', 'formula', 'keys', 'unseen', 'classes_encoded',
                                         'raw_geo_ids_in_matrix', 'dtype'])
            and te['feature_names'] == TE_NAMES and te['smoothing_m'] == 20.0, 'TE design differs from T10')
    inner = te['inner_oof']
    require(inner['splitter'] == 'StratifiedGroupKFold' and inner['n_splits'] == 5 and inner['shuffle']
            and inner['random_state'] == 42, 'Inner splitter changed')
    require(c['group_level_of_scheme'] == {'geo2': 'geo2', 'geo1': 'geo1'}, 'Group level mapping changed')
    require(c['study_fits'] == {a: FIT_SCHEMES for a in ARMS} and c['max_new_study_fits'] == N_TARGETS,
            'Unplanned fit plan')
    require(c['automatic_retries'] == 0 and not c['early_stopping'] and c['class_weight'] is None
            and c['sampling'] is None and not c['tuning'] and not c['threshold_or_decision_rule_change']
            and c['outer_parallel_jobs'] == 1 and c['lr_threadpool_limit'] == 1 and c['lgbm_threadpool_limit'] == 2
            and c['stop_on_lr_nonconvergence'] and not c['holdout_access'], 'Unplanned run policy')
    require(c['manifest_sha256'] == t['manifest_sha256'] and c['baselines'] == t['baselines']
            and c['environment'] == t['environment'], 'Frozen references differ from T10')
    return c


def sources():
    return {p: sha256_file(ROOT/p) for p in SOURCE}


def targets():
    c = config()
    return [(s, a, f) for a in ARMS for s in c['study_fits'][a] for f in range(5)]


# ---------------------------------------------------------------- arms (T10 arms with the group-aware encoder)
def te_step(arm, m, level):
    enc = GroupInnerTargetEncoder(m=m, group_level=level, inner_splits=5, inner_seed=42)
    return Pipeline([('te', enc), ('scale', StandardScaler())]) if arm == 'LR_TE_G' else enc


def patch(pre, arm, m, level):
    """Same replacement as T10.patch: only the C geo one-hot is replaced by the TE step."""
    require(arm in ARMS, 'Unknown arm')
    columns = pre['columns']
    require([n for n, _, _ in columns.transformers] == ['numeric', 'binary', 'categorical'], 'Unexpected C paths')
    require(columns.transformers[0][2] == NUMERIC and columns.transformers[1][2] == BINARY + [FLAG],
            'Noncategorical inputs changed')
    name, ohe, cols = columns.transformers[2]
    require(cols == CATEGORICAL and type(ohe).__name__ == 'OneHotEncoder', 'C categorical path changed')
    columns.transformers = columns.transformers[:2] + [('geo_te', te_step(arm, m, level), list(GEO)), (name, ohe, list(TEXT))]
    return pre


def make_lr_pipeline(m=20.0, level='geo2'):
    pipe = make_C('C')
    patch(pipe['preprocess'], 'LR_TE_G', m, level)
    return pipe


def make_lgbm_preprocessor(m=20.0, level='geo2'):
    return patch(make_C('C')['preprocess'], 'LGBM_TE_G', m, level)


def fit_arm(arm, level, X_train, y_train, X_valid, c):
    """Fit encoder/preprocessor/model on the training fold only. Validation labels are not an argument."""
    from lightgbm import LGBMClassifier
    m = c['te']['smoothing_m']
    if arm == 'LR_TE_G':
        pipe = make_lr_pipeline(m, level)
        pre, model, threads = pipe['preprocess'], pipe['model'], c['lr_threadpool_limit']
        require(model.get_params()['C'] == 1.0 and model.get_params()['solver'] == 'lbfgs', 'LR params changed')
    else:
        pre = make_lgbm_preprocessor(m, level)
        model, threads = LGBMClassifier(**c['lgbm_params']), c['lgbm_threadpool_limit']
        pipe = Pipeline([('preprocess', pre), ('model', model)])
    with threadpool_limits(limits=threads):
        t0 = time.perf_counter()
        mt = pre.fit_transform(X_train, y_train)
        mv = pre.transform(X_valid)
        pre_seconds = time.perf_counter() - t0
    names = t10.check_matrix(pre, mt)
    t10.check_matrix(pre, mv)
    yt = np.asarray(y_train)
    with threadpool_limits(limits=threads), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        t0 = time.perf_counter()
        if arm == 'LR_TE_G':
            model.fit(mt, yt)
        else:
            model.fit(mt, yt.astype(np.int32) - 1)  # No eval_set / weights / callbacks.
        fit_seconds = time.perf_counter() - t0
        p = t10.proba(model, mv, T10_ARM[arm])
        p_train = t10.proba(model, mt, T10_ARM[arm])
    enc = t10.te_encoder(pre)
    require(type(enc).__module__ == ENCODER_MODULE, 'Encoder class must live in the importable encoder module')
    _, valid_unseen = t10.encode(enc.stats_, X_valid[GEO], return_unseen=True)
    valid_te = enc.transform(X_valid[GEO]).astype(np.float64)
    n_tr = len(X_train)
    info = dict(fit_seconds=fit_seconds, preprocessing_seconds=pre_seconds, encoded_columns=int(mt.shape[1]),
                feature_names_sha256=hashlib.sha256('\n'.join(names).encode()).hexdigest(),
                warnings=[dict(category=w.category.__name__, message=str(w.message)) for w in caught],
                age995_median=pre['age995'].median_,
                te=dict(prior=enc.stats_['prior'].tolist(), m=enc.stats_['m'], fit_rows=enc.stats_['n'],
                        group_level=level,
                        levels_seen={lvl: int(len(t)) for lvl, t in zip(LEVELS, enc.stats_['levels'])},
                        train_groups=enc.train_groups_, inner_fold_groups=enc.inner_fold_groups_,
                        inner_fold_sizes=np.bincount(enc.inner_folds_).tolist(),
                        inner_fold_class_counts=enc.inner_fold_class_counts_,
                        train_oof_unseen=enc.train_oof_unseen_,
                        train_oof_unseen_rate={lvl: enc.train_oof_unseen_[lvl]/n_tr for lvl in LEVELS},
                        train_oof_mean=enc.train_oof_mean_, train_oof_std=enc.train_oof_std_,
                        valid_unseen=valid_unseen, valid_mean=valid_te.mean(0).tolist(),
                        valid_std=valid_te.std(0).tolist()))
    require(enc.train_oof_unseen_[level] == n_tr, f'Training rows must be 100% unseen at the {level} level')
    if arm == 'LR_TE_G':
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


def t10_validation_encoding(X_train, y_train, X_valid, m):
    """T10's validation-row encoding for the same fold (float32), computed by T10's own code."""
    return t10.encode(t10.fit_stats(X_train[GEO], np.asarray(y_train), m), X_valid[GEO]).astype(np.float32)


# ---------------------------------------------------------------- control check
FRESH_LOAD = r'''
import sys, joblib, numpy as np
sys.path.insert(0, sys.argv[2])
import experiments.geo_te_group_inner.encoder as enc_mod   # the ONLY project import; no __main__ registration
obj = joblib.load(sys.argv[1])
enc = obj['encoder']
assert type(enc).__module__ == 'experiments.geo_te_group_inner.encoder', type(enc).__module__
assert type(enc) is enc_mod.GroupInnerTargetEncoder
got = enc.transform(obj['query'])
assert got.tobytes() == obj['expected'].tobytes(), 'encoder transform differs after fresh load'
if 'matrix_pre' in obj:
    pre = obj['matrix_pre']
    mv = pre.transform(obj['X_query'])
    assert (mv != obj['expected_matrix']).nnz == 0, 'preprocessor matrix differs after fresh load'
print('FRESH_LOAD_OK', type(enc).__module__)
'''


def fresh_process_load(enc, query, extra=None):
    """Save an encoder (and optionally a fitted preprocessor) and load it in a new python process that imports
    only the encoder module. Returns (ok, output)."""
    obj = dict(encoder=enc, query=query, expected=enc.transform(query))
    if extra:
        obj.update(extra)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp)/'artifact.joblib'
        joblib.dump(obj, path, compress=3)
        proc = subprocess.run([sys.executable, '-c', FRESH_LOAD, str(path), str(ROOT)], cwd=tmp,
                              capture_output=True, text=True)
    return proc.returncode == 0 and 'FRESH_LOAD_OK' in proc.stdout, (proc.stdout + proc.stderr).strip()[-400:]


def control_check(data, destination, t10_run):
    """Not a study fit (no classifier is fitted): encoder checks on the real development folds."""
    c = config()
    dest = Path(destination)
    require(not dest.exists(), 'Do not overwrite an existing control check')
    require(environment() == c['environment'], 'Environment differs from the pinned environment')
    m = c['te']['smoothing_m']
    X, y, ids, old, contract = load(data)
    res = dict(study_id=STUDY_ID, kind=c['control_check']['kind'], counted_as_study_fit=False, created_at_utc=now())
    folds = {}
    for s in FIT_SCHEMES:
        path = Path(t10_run)/s/'manifest.csv'
        require(sha256_file(path) == c['manifest_sha256'][s], 'T10 manifest SHA differs')
        mf = pd.read_csv(path)
        np.testing.assert_array_equal(mf.building_id, ids)
        folds[s] = mf.fold.to_numpy()
    yv = y.to_numpy()
    a_rows, b_rows, c_rows = {}, {}, {}
    all_disjoint = all_unseen100 = all_valid_identical = True
    for s in FIT_SCHEMES:
        level = c['group_level_of_scheme'][s]
        a_rows[s], b_rows[s], c_rows[s] = [], [], []
        for f in range(5):
            tr, va = np.flatnonzero(folds[s] != f), np.flatnonzero(folds[s] == f)
            Gtr, Gva, ytr = X.iloc[tr][GEO].reset_index(drop=True), X.iloc[va][GEO].reset_index(drop=True), yv[tr]
            enc = GroupInnerTargetEncoder(m=m, group_level=level)
            out = enc.fit_transform(Gtr, ytr)
            inner = enc.inner_folds_
            groups = group_ids(Gtr, level)
            sets = [set(groups[inner == i]) for i in range(5)]
            disjoint = all(sets[i].isdisjoint(sets[j]) for i in range(5) for j in range(i + 1, 5))
            nonempty = bool((np.bincount(inner, minlength=5) > 0).all())
            # outer groups of the validation fold must not occur in the training fold (outer-split sanity)
            outer_disjoint = set(group_ids(Gva, level)).isdisjoint(set(groups))
            a_rows[s].append(dict(fold=f, inner_groups_disjoint=bool(disjoint), five_nonempty_inner_folds=nonempty,
                                  outer_groups_disjoint=bool(outer_disjoint),
                                  inner_fold_sizes=np.bincount(inner).tolist(), inner_fold_groups=enc.inner_fold_groups_,
                                  train_groups=enc.train_groups_, inner_fold_class_counts=enc.inner_fold_class_counts_))
            all_disjoint &= disjoint and nonempty and outer_disjoint
            # T10 reference regime for the same fold: random inner folds
            t10_folds = t10.inner_fold_ids(ytr)
            _, t10_unseen = t10.oof_encode(Gtr, ytr, t10_folds, m, return_unseen=True)
            n = len(tr)
            rate = {lvl: enc.train_oof_unseen_[lvl]/n for lvl in LEVELS}
            b_rows[s].append(dict(fold=f, train_rows=n, outer_level=level, t11_unseen_rate=rate,
                                  t10_unseen_rate={lvl: t10_unseen[lvl]/n for lvl in LEVELS},
                                  t11_unseen_rows=enc.train_oof_unseen_, t10_unseen_rows=t10_unseen,
                                  valid_unseen=t10.encode(enc.stats_, Gva, return_unseen=True)[1],
                                  pass_=bool(rate[level] == 1.0)))
            all_unseen100 &= rate[level] == 1.0
            # (c) validation-row encodings byte-identical to T10
            v11 = enc.transform(Gva)
            v10 = t10_validation_encoding(X.iloc[tr], ytr, X.iloc[va], m)
            v10_enc = t10.HierGeoTargetEncoder(m=m).fit(Gtr, ytr).transform(Gva)
            same = v11.tobytes() == v10.tobytes() == v10_enc.tobytes()
            c_rows[s].append(dict(fold=f, valid_rows=len(va), byte_identical=bool(same),
                                  train_encoding_changed_vs_T10=bool(
                                      not np.array_equal(out, t10.HierGeoTargetEncoder(m=m).fit_transform(Gtr, ytr)))))
            all_valid_identical &= same
    res['a_inner_groups_disjoint'] = dict(folds=a_rows, pass_=bool(all_disjoint))
    res['b_training_unseen_at_outer_group_level'] = dict(folds=b_rows, pass_=bool(all_unseen100))
    res['c_validation_encoding_identical_to_T10'] = dict(folds=c_rows, pass_=bool(all_valid_identical))
    # (d) self-label and validation-label independence on a 20k-row subsample of geo2 fold 0 (fixed inner assignment)
    rng = np.random.default_rng(42)
    tr0, va0 = np.flatnonzero(folds['geo2'] != 0), np.flatnonzero(folds['geo2'] == 0)
    sub_tr = np.sort(rng.choice(tr0, 20000, replace=False))
    sub_va = np.sort(rng.choice(va0, 5000, replace=False))
    Gs, ys = X.iloc[sub_tr][GEO].reset_index(drop=True), yv[sub_tr]
    fo = inner_group_fold_ids(ys, group_ids(Gs, 'geo2'))
    base = t10.oof_encode(Gs, ys, fo, m)
    own_same, others_changed, single_same = [], [], []
    for i in rng.choice(len(ys), 12, replace=False):
        mask = fo == fo[i]
        y2 = ys.copy(); y2[mask] = (y2[mask] % 3) + 1
        e2 = t10.oof_encode(Gs, y2, fo, m)
        own_same.append(bool(np.array_equal(e2[mask], base[mask])))
        others_changed.append(bool(not np.array_equal(e2[~mask], base[~mask])))
        y3 = ys.copy(); y3[i] = (y3[i] % 3) + 1
        single_same.append(bool(np.array_equal(t10.oof_encode(Gs, y3, fo, m)[i], base[i])))
    # group-level: every row of the same inner group is encoded from statistics that exclude that group
    same_group_constant = True
    gk = group_ids(Gs, 'geo2')
    for g in rng.choice(np.unique(gk), 10, replace=False):
        idx = np.flatnonzero(gk == g)
        y2 = ys.copy(); y2[idx] = (y2[idx] % 3) + 1  # flip every label of the whole group
        same_group_constant &= bool(np.array_equal(t10.oof_encode(Gs, y2, fo, m)[idx], base[idx]))
    yfull = pd.Series(yv).copy()
    yperm = yfull.copy()
    yperm.iloc[sub_va] = rng.permutation(yfull.iloc[sub_va].to_numpy())
    require(not np.array_equal(yperm.iloc[sub_va].to_numpy(), yfull.iloc[sub_va].to_numpy()), 'Permutation must change labels')
    d_out = {}
    for arm, build in [('LR_TE_G', lambda: make_lr_pipeline(m, 'geo2')['preprocess']),
                       ('LGBM_TE_G', lambda: make_lgbm_preprocessor(m, 'geo2'))]:
        mats = []
        for labels in (yfull, yperm):
            pre = build()
            with threadpool_limits(limits=1):
                mt = pre.fit_transform(X.iloc[sub_tr], labels.iloc[sub_tr])
                mv = pre.transform(X.iloc[sub_va])
            t10.check_matrix(pre, mt)
            names = t10.check_matrix(pre, mv)
            mats.append((mt, mv))
        d_out[arm] = dict(valid_matrix_unchanged=bool((mats[0][1] != mats[1][1]).nnz == 0),
                          train_matrix_unchanged=bool((mats[0][0] != mats[1][0]).nnz == 0),
                          columns=len(names), no_raw_geo_columns=True)
    res['d_self_and_validation_label_independence'] = dict(
        rows_checked=len(own_same), own_inner_fold_unchanged=all(own_same), single_row_flip_own_unchanged=all(single_same),
        whole_group_label_flip_group_rows_unchanged=bool(same_group_constant),
        nonvacuous_other_folds_changed=all(others_changed), arms=d_out,
        pass_=bool(all(own_same) and all(single_same) and same_group_constant and all(others_changed)
                   and all(v['valid_matrix_unchanged'] and v['train_matrix_unchanged'] for v in d_out.values())))
    # (e) fresh-process load of a saved encoder and of a fitted preprocessor containing it
    pre = make_lgbm_preprocessor(m, 'geo2')
    with threadpool_limits(limits=1):
        pre.fit_transform(X.iloc[sub_tr], pd.Series(yv).iloc[sub_tr])
        expected = pre.transform(X.iloc[sub_va])
    enc = t10.te_encoder(pre)
    ok, text = fresh_process_load(enc, X.iloc[sub_va][GEO].reset_index(drop=True),
                                  dict(matrix_pre=pre, X_query=X.iloc[sub_va], expected_matrix=expected))
    res['e_fresh_process_load'] = dict(encoder_module=type(enc).__module__, output=text, pass_=bool(ok))
    res['classifiers_fitted'] = 0
    res['environment'] = environment()
    res['holdout_accessed'] = False
    res['status'] = 'PASS' if all(v['pass_'] for v in res.values() if isinstance(v, dict) and 'pass_' in v) else 'FAIL'
    dest.mkdir(parents=True)
    write(dest/'control_check.json', res)
    brief = {k: v['pass_'] for k, v in res.items() if isinstance(v, dict) and 'pass_' in v}
    print(json.dumps(brief, indent=1))
    for s in FIT_SCHEMES:
        for r in b_rows[s]:
            print(s, r['fold'], 'T11 unseen', {k: round(v, 4) for k, v in r['t11_unseen_rate'].items()},
                  'T10 unseen', {k: round(v, 4) for k, v in r['t10_unseen_rate'].items()})
    require(res['status'] == 'PASS', 'Control check failed')
    return res


# ---------------------------------------------------------------- prepare/verify
def prepare(data, t10_run, output, code_commit, control):
    output, t10_run = Path(output), Path(t10_run)
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
    for s in FIT_SCHEMES:
        shas[s] = sha256_file(t10_run/s/'manifest.csv')
    require(shas == c['manifest_sha256'], 'Split manifest SHA differs')
    cc = json.loads(Path(control).read_text(encoding='utf-8'))
    require(cc['status'] == 'PASS' and cc['study_id'] == STUDY_ID, 'Control check must PASS before prepare')
    require(sha256_file(t10_run/'design.json') == c['t10']['design_sha256'], 'T10 design SHA differs')
    t10_design = json.loads((t10_run/'design.json').read_text(encoding='utf-8'))
    require(t10_design['code_commit'] == c['t10']['code_commit'], 'T10 code commit differs')
    public = json.loads((ROOT/c['t10']['public_json']).read_text(encoding='utf-8'))
    output.mkdir(parents=True)
    folds = {'stratified': old}
    for s in FIT_SCHEMES:
        (output/s).mkdir()
        shutil.copyfile(t10_run/s/'manifest.csv', output/s/'manifest.csv')
        require(sha256_file(output/s/'manifest.csv') == shas[s], 'Copied manifest SHA differs')
        mf = pd.read_csv(output/s/'manifest.csv')
        np.testing.assert_array_equal(mf.building_id, ids)
        folds[s] = mf.fold.to_numpy()
    (output/'backup').mkdir()
    base = {}
    for name in ['LR_C', 'LGBM_O']:
        base[name] = {}
        for s in SCHEMES:
            path = t10_design['baselines'][name][s]['path']
            digest = c['baselines'][name]['oof_sha256'][s]
            r = reference(path, digest, ids, y, folds[s])
            vals = fold_means(r)
            base[name][s] = dict(path=path, sha256=digest, cv_mean=float(np.mean(vals)), fold_macro_f1=vals,
                                 pooled_grade3_recall=score(r.y_true, r.y_pred)['recall'][2])
    ref = c['lr_c_reference']
    require(abs(base['LR_C']['stratified']['cv_mean'] - ref['stratified_mean']) <= ref['tolerance_mean']
            and abs(base['LR_C']['geo2']['cv_mean'] - ref['geo2_mean']) <= ref['tolerance_mean']
            and abs(base['LR_C']['geo2']['pooled_grade3_recall'] - ref['geo2_pooled_grade3_recall']) <= ref['tolerance_recall'],
            'LR-C reference values differ from the registered thresholds')
    t10_arms = {}
    for arm in T10_ARM.values():
        t10_arms[arm] = {}
        for s in SCHEMES:
            path = str((t10_run/s/arm/'oof.csv.gz').resolve())
            digest = c['t10']['arm_oof_sha256'][arm][s]
            meta = json.loads((t10_run/s/arm/'metrics.json').read_text(encoding='utf-8'))
            require(meta['oof_sha256'] == digest, 'T10 arm metrics anchor differs')
            r = reference(path, digest, ids, y, folds[s])
            vals = fold_means(r)
            pub = next(mm for mm in public['models'] if mm['scheme'] == s and mm['arm'] == arm)
            np.testing.assert_allclose(vals, pub['fold_macro_f1'], rtol=0, atol=c['t10']['fold_mean_tolerance_vs_public_json'])
            t10_arms[arm][s] = dict(path=path, sha256=digest, cv_mean=float(np.mean(vals)), fold_macro_f1=vals,
                                    pooled_grade3_recall=score(r.y_true, r.y_pred)['recall'][2])
    unseen = t10.unseen_counts(X, folds)
    design = dict(study_id=STUDY_ID, config=c, code_commit=code_commit, source_sha256=sources(),
                  data_contract=contract, environment=environment(), manifest_sha256=shas,
                  t10_run=str(t10_run.resolve()), baselines=base, t10_arms=t10_arms,
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
    for s in FIT_SCHEMES:
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
    level = c['group_level_of_scheme'][scheme]
    a, b = np.flatnonzero(folds[scheme] != f), np.flatnonzero(folds[scheme] == f)
    require(len(a) + len(b) == len(ids) and len(b) > 0, 'Fold membership invalid')
    pipe, pre, model, p, p_train, mt, mv, names, info = fit_arm(arm, level, X.iloc[a], y.iloc[a], X.iloc[b], c)
    pred = p.argmax(1) + 1
    threads = c['lr_threadpool_limit'] if arm == 'LR_TE_G' else c['lgbm_threadpool_limit']
    # Same numeric/binary/flag and text one-hot values as the C preprocessor fitted on the same fold.
    pre_c = make_C('C')['preprocess']
    with threadpool_limits(limits=threads):
        pre_c.fit(X.iloc[a])
        eq_v, n_geo_ohe = t10.same_blocks_as_C(pre_c, pre_c.transform(X.iloc[b]), mv, names)
        eq_t, _ = t10.same_blocks_as_C(pre_c, pre_c.transform(X.iloc[a]), mt, names)
    require(eq_v and eq_t, 'Non-geo columns differ from the C preprocessor')
    require(info['te']['valid_unseen'] == {lvl: design['valid_unseen_geo_keys'][scheme][f][lvl] for lvl in LEVELS},
            'Unseen counts differ from design')
    # Validation TE block must be byte-identical to T10's validation encoding of the same fold.
    enc_block = np.asarray(t10.te_encoder(pre).transform(X.iloc[b][GEO]))
    require(enc_block.tobytes() == t10_validation_encoding(X.iloc[a], y.iloc[a], X.iloc[b],
                                                           c['te']['smoothing_m']).tobytes(),
            'Validation encoding differs from T10')
    preds = d/f'fold_{f}_pred.csv.gz'
    if arm == 'LR_TE_G':
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
        if arm == 'LR_TE_G':
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
                  pred_share=(np.bincount(pred - 1, minlength=3)/len(pred)).tolist(),
                  fit_seconds=info['fit_seconds'], preprocessing_seconds=info['preprocessing_seconds'],
                  encoded_columns=info['encoded_columns'], feature_names_sha256=info['feature_names_sha256'],
                  geo_onehot_columns_replaced=n_geo_ohe, non_geo_columns_equal_C=True,
                  valid_encoding_identical_to_t10=True, encoder_module=type(t10.te_encoder(pre)).__module__,
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
             f'iter={info.get("n_iter")} reload_diff={reload_diff:.3g} valid_unseen={info["te"]["valid_unseen"]} '
             f'train_unseen_rate={info["te"]["train_oof_unseen_rate"]} saved+verified+backed-up')
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
    s = t10.summary(oof, X, fit_seconds)
    pooled = s['pooled']
    s['accuracy'] = pooled['accuracy']
    s['pred_share'] = (np.bincount(oof.y_pred.to_numpy() - 1, minlength=3)/len(oof)).tolist()
    s['grade3_recall'] = pooled['recall'][2]
    s['rate32'] = pooled['rate32']
    return s


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
               stratified_note=c['stratified_policy'],
               note='Same folds as T10/baselines (optimistic if selecting); descriptive, no significance or causal claims.',
               schemes={})
    yv = y.to_numpy()
    for s in SCHEMES:
        oofs, block = {}, {}
        for name in ['LR_C', 'LGBM_O']:
            oofs[name] = reference(design['baselines'][name][s]['path'], design['baselines'][name][s]['sha256'],
                                   ids, y, folds[s])
            block[name] = dict(summary(oofs[name], X), oof_sha256=design['baselines'][name][s]['sha256'],
                               source='pinned baseline OOF (not refit)')
        for arm in ARMS:
            t10_arm = T10_ARM[arm]
            oofs[t10_arm] = reference(design['t10_arms'][t10_arm][s]['path'], design['t10_arms'][t10_arm][s]['sha256'],
                                      ids, y, folds[s])
            block[t10_arm] = dict(summary(oofs[t10_arm], X), oof_sha256=design['t10_arms'][t10_arm][s]['sha256'],
                                  source='T10 pinned OOF (not refit)')
            if s == 'stratified':  # inner split == T10 by construction: reuse
                oofs[arm] = oofs[t10_arm]
                block[arm] = dict(block[t10_arm], reused_from_T10=True,
                                  source='stratified inner split equals T10 (random StratifiedKFold): T10 OOF reused')
                continue
            mt = assemble(out, y, ids, folds, s, arm)
            oofs[arm] = read_pred(out/s/arm/'oof.csv.gz')
            np.testing.assert_array_equal(oofs[arm].building_id.to_numpy(), oofs['LR_C'].building_id.to_numpy())
            recs = [json.loads((out/s/arm/f'fold_{f}.json').read_text(encoding='utf-8')) for f in range(5)]
            block[arm] = dict(summary(oofs[arm], X, mt['fit_seconds']), oof_sha256=mt['oof_sha256'],
                              reused_from_T10=False, valid_unseen=[r['te']['valid_unseen'] for r in recs],
                              train_oof_unseen_rate=[r['te']['train_oof_unseen_rate'] for r in recs],
                              train_oof_unseen=[r['te']['train_oof_unseen'] for r in recs],
                              train_rows=[r['train_rows'] for r in recs],
                              train_groups=[r['te']['train_groups'] for r in recs],
                              inner_fold_sizes=[r['te']['inner_fold_sizes'] for r in recs],
                              train_macro_f1_oof_encoded=[r['train_macro_f1_oof_encoded'] for r in recs],
                              converged=[r['converged'] for r in recs] if arm == 'LR_TE_G' else None,
                              n_iter=[r['n_iter'] for r in recs] if arm == 'LR_TE_G' else None,
                              te_gain_share=[r['te_gain_share'] for r in recs] if arm == 'LGBM_TE_G' else None,
                              fold_fit_seconds=[r['fit_seconds'] for r in recs])
        comps = {}
        for arm in ARMS:
            base_name = BASELINE_OF[arm]
            comps[f'{arm}_vs_{base_name}'] = t10.compare(block, arm, base_name, yv, oofs[arm], oofs[base_name])
            comps[f'{arm}_vs_{T10_ARM[arm]}'] = t10.compare(block, arm, T10_ARM[arm], yv, oofs[arm], oofs[T10_ARM[arm]])
            comps[f'{T10_ARM[arm]}_vs_{base_name}'] = t10.compare(block, T10_ARM[arm], base_name, yv,
                                                                  oofs[T10_ARM[arm]], oofs[base_name])
        comps['LGBM_TE_G_vs_LR_C'] = t10.compare(block, 'LGBM_TE_G', 'LR_C', yv, oofs['LGBM_TE_G'], oofs['LR_C'])
        comps['LGBM_TE_vs_LR_C'] = t10.compare(block, 'LGBM_TE', 'LR_C', yv, oofs['LGBM_TE'], oofs['LR_C'])
        block['comparisons'] = comps
        block['valid_unseen_geo_keys'] = design['valid_unseen_geo_keys'][s]
        res['schemes'][s] = block
    # ---- pre-registered judgments
    r0, r1, r2 = c['rules']['distribution_repair'], c['rules']['generalization_candidate'], c['rules']['baseline_challenger']
    st, g2, g1 = res['schemes']['stratified'], res['schemes']['geo2'], res['schemes']['geo1']
    judg = {}
    for arm in ARMS:
        B = BASELINE_OF[arm]
        vs = f'{arm}_vs_{B}'
        repair_in = {}
        for s, blk in (('geo2', g2), ('geo1', g1)):
            acc_diff = blk[arm]['accuracy'] - blk[B]['accuracy']
            agree = blk['comparisons'][vs]['turnover']['prediction_agreement']
            repair_in[s] = dict(accuracy_diff=acc_diff, agreement=agree,
                                holds=bool(abs(acc_diff) <= r0['abs_accuracy_diff_max']
                                           and agree >= r0['pooled_prediction_agreement_min']))
        repair = bool(all(v['holds'] for v in repair_in.values()))
        cg = g2['comparisons'][vs]
        cs = st['comparisons'][f'{T10_ARM[arm]}_vs_{B}']  # stratified uses the reused T10 values
        inputs1 = dict(geo2_mean_delta=cg['mean_delta'], geo2_fold_wins=cg['fold_wins'],
                       geo2_pooled_grade3_recall_delta=cg['pooled_grade3_recall_delta'],
                       stratified_mean_delta_reused_T10=cs['mean_delta'])
        gen = bool(repair and cg['mean_delta'] >= r1['geo2_mean_delta_min'] and cg['fold_wins'] >= r1['geo2_fold_wins_min']
                   and cg['pooled_grade3_recall_delta'] >= r1['geo2_pooled_grade3_recall_delta_min']
                   and cs['mean_delta'] >= r1['stratified_mean_delta_min'])
        lrc = f'{arm}_vs_LR_C' if arm == 'LGBM_TE_G' else vs
        inputs2 = dict(stratified_mean_reused_T10=st[arm]['cv_mean'],
                       stratified_fold_wins_vs_LR_C=(st['comparisons']['LGBM_TE_vs_LR_C']['fold_wins'] if arm == 'LGBM_TE_G'
                                                     else st['comparisons'][f'LR_TE_vs_LR_C']['fold_wins']),
                       geo2_mean=g2[arm]['cv_mean'], geo2_fold_wins_vs_LR_C=g2['comparisons'][lrc]['fold_wins'],
                       geo2_pooled_grade3_recall=g2[arm]['pooled']['recall'][2])
        chal = bool(inputs2['stratified_mean_reused_T10'] > r2['stratified_mean_gt']
                    and inputs2['stratified_fold_wins_vs_LR_C'] >= r2['stratified_fold_wins_vs_LR_C_min']
                    and inputs2['geo2_mean'] >= r2['geo2_mean_ge']
                    and inputs2['geo2_fold_wins_vs_LR_C'] >= r2['geo2_fold_wins_vs_LR_C_min']
                    and inputs2['geo2_pooled_grade3_recall'] >= r2['geo2_pooled_grade3_recall_ge'])
        judg[FAMILY[arm]] = dict(distribution_repair=repair, repair_inputs=repair_in,
                                 generalization_candidate=gen, generalization_inputs=inputs1,
                                 baseline_challenger_candidate=chal, challenger_inputs=inputs2)
    res['judgments'] = judg
    res['finished_at_utc'] = now()
    write(out/'analysis.json', res)
    require(not (out/'backup'/'analysis.json').exists(), 'Backup already exists; refusing overwrite')
    shutil.copy2(out/'analysis.json', out/'backup'/'analysis.json')
    log(out, 'analysis written: ' + json.dumps({f: {k: v for k, v in j.items() if isinstance(v, bool)}
                                                for f, j in judg.items()}))
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['control-check', 'prepare', 'fit-all', 'finalize', 'status'])
    p.add_argument('--data', default='data')
    p.add_argument('--output')
    p.add_argument('--destination')
    p.add_argument('--t10-run')
    p.add_argument('--code-commit')
    p.add_argument('--control')
    a = p.parse_args()
    if a.action == 'control-check':
        control_check(a.data, a.destination, a.t10_run)
    elif a.action == 'prepare':
        prepare(a.data, a.t10_run, a.output, a.code_commit, a.control)
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
