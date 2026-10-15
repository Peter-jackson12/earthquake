"""T14: local reproduction + geo2 Group-split check of the teammate's exploratory RF + geo TE + ROS pipeline.

NOT a new method and NOT a factor-separation study.  The pipeline (geo1/2/3 OOF target encoding of class-1/class-3 rates
with m=20, 3 count features, RandomOverSampler on the outer-train only after encoding, raw age 995 and raw numeric geo IDs
kept, 77 columns, RandomForestClassifier(n_jobs=4, random_state=0)) is copied from the sealed exploratory scripts
(`experiments/te_ros_rf_exploratory/`, git blob SHA-256 pinned) and refit locally with per-fold checkpoints.

Stage A: the fixed stratified split (`split_manifest.csv`), 5 folds, compared with the recorded per-fold confusion matrices.
Stage B (only if Stage A is EXACT or WITHIN_TOLERANCE): the geo2 Group splits seed 42 (T5), 101/202/303 (T13), 20 folds,
compared with already audited LR-C / LGBM-O / LGBM_TE_G OOFs (no comparator retraining).  On Group splits the inner OOF
folds of the target encoder are group-aware (T11 choice: StratifiedGroupKFold(5, shuffle, 42) on (geo1, geo2) groups);
on the stratified split the original inner StratifiedKFold(5, shuffle, 0) is used unchanged.

Holdout is never read.  Private outputs only.  25 RF fits maximum, sequential, one level of parallelism (RF n_jobs=4).
"""
from __future__ import annotations
import argparse
import ast
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
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from preprocess import GEO, ID, TEXT, sha256_file  # noqa: E402
from experiments.geo_group_cv import run as t5  # noqa: E402
from experiments.geo_group_cv.run import PROBS, load, require, score, write  # noqa: E402
from experiments.geo_hier_te import run as t10  # noqa: E402
from experiments.geo_te_group_inner import run as t11  # noqa: E402
from experiments.geo_te_group_inner.encoder import group_ids, inner_group_fold_ids  # noqa: E402

HERE = Path(__file__).resolve().parent
STUDY_ID = 'T14_TE_ROS_RF_LOCAL_REPRO'
ARM = 'RF_TE_ROS'
SCHEMES = ['strat', 'geo2_s42', 'geo2_s101', 'geo2_s202', 'geo2_s303']
GEO_SCHEMES = SCHEMES[1:]
STAGE_OF = {'strat': 'A', **{s: 'B' for s in GEO_SCHEMES}}
GROUP_COLUMNS = list(GEO[:2])
N_TARGETS = 25
HOLDOUT_MARKER = 'final_holdout'
PASS_A = ('EXACT', 'WITHIN_TOLERANCE')
ORIGINAL_SCRIPT = 'experiments/te_ros_rf_exploratory/evaluate_te_ros.py'
SOURCE = list(dict.fromkeys(t11.SOURCE + [
    'experiments/geo_ablation/audit.py',
    'experiments/te_ros_rf_exploratory/README.md', ORIGINAL_SCRIPT,
    'experiments/te_ros_rf_exploratory/validate_rf_5fold.py',
    'experiments/te_ros_rf_repro/__init__.py', 'experiments/te_ros_rf_repro/run.py',
    'experiments/te_ros_rf_repro/config.json', 'tests/test_te_ros_rf_repro.py',
    'docs/T14_TE_ROS_RF_REPRO_PLAN.md']))
now, log, event = t10.now, t10.log, t10.event
atomic_write, read_pred, write_pred = t10.atomic_write, t10.read_pred, t10.write_pred
read_ledger, check_saved, reference = t10.read_ledger, t10.check_saved, t10.reference


def key(scheme, arm, fold):
    return f'{scheme}/{arm}/{fold}'


def targets():
    return [(s, ARM, f) for s in SCHEMES for f in range(5)]


def environment():
    names = ['numpy', 'pandas', 'scipy', 'scikit-learn', 'lightgbm', 'joblib', 'threadpoolctl', 'imbalanced-learn']
    return dict(python=platform.python_version(), **{n: importlib.metadata.version(n) for n in names})


def require_not_holdout(path):
    require(HOLDOUT_MARKER not in Path(path).as_posix().lower(), f'Holdout path refused: {path}')


def config(path=HERE/'config.json'):
    c = json.loads(Path(path).read_text(encoding='utf-8'))
    require(c['study_id'] == STUDY_ID and c['arm'] == ARM and c['schemes'] == SCHEMES and c['n_folds'] == 5,
            'Unplanned study/arm/schemes')
    require(c['rf_params'] == dict(n_jobs=4, random_state=0) and c['rf_n_jobs'] == 4
            and c['rf_default_snapshot'] == RandomForestClassifier(**c['rf_params']).get_params(),
            'RF parameters differ from the exploratory pipeline / sklearn defaults changed')
    te = c['te']
    require(te['smoothing_m'] == 20 and te['geo_columns'] == list(GEO) and te['classes_encoded_index'] == [0, 2]
            and te['stratified_inner'] == dict(n_splits=5, shuffle=True, random_state=0)
            and te['group_inner'] == dict(n_splits=5, shuffle=True, random_state=42, group_level='geo2'),
            'TE design differs from the exploratory pipeline / T11')
    require(c['ros'] == dict(random_state=0, after_te=True, outer_train_only=True), 'ROS design changed')
    require(c['expected_model_columns'] == 77 and c['text_columns'] == list(TEXT), 'Feature schema changed')
    require(c['max_new_study_fits'] == N_TARGETS == len(SCHEMES) * 5, 'Unplanned fit plan')
    require(c['automatic_retries'] == 0 and not c['early_stopping'] and c['outer_parallel_jobs'] == 1
            and not c['tuning'] and not c['threshold_or_decision_rule_change'] and not c['holdout_access']
            and c['decision_rule'] == 'argmax of RF predict_proba, labels 1/2/3', 'Unplanned run policy')
    require(set(c['comparators']) == set(SCHEMES), 'Comparator table must cover every scheme')
    require(c['environment'] == environment() or path != HERE/'config.json', 'Environment differs from pinned')
    return c


def sources():
    return {p: sha256_file(ROOT/p) for p in SOURCE}


# ---------------------------------------------------------------- the exploratory pipeline (copied, minimal hooks)
def target_enc(xtr, ytr, xva, cols, m=20, splits=None, return_state=False):
    """Body copied from the exploratory `target_enc` (SHA-pinned script).  Hooks added, nothing else changed:
    `splits` (default = the original StratifiedKFold(5, shuffle, 0) split of the training rows; a Group-aware list of
    (inner_train_pos, inner_oof_pos) pairs is passed on Group splits) and `return_state` (keeps the full-train maps so the
    validation encoding can be rebuilt after reload)."""
    xtr = xtr.copy(); xva = xva.copy()
    Y = pd.get_dummies(ytr).reindex(columns=range(3), fill_value=False).astype(float)
    if splits is None:
        skf = StratifiedKFold(5, shuffle=True, random_state=0)
        splits = list(skf.split(xtr, ytr))
    state = {}
    for c in cols:
        state[c] = {}
        for k in (0, 2):
            name = f'{c}_te{k}'; oof = np.zeros(len(xtr), dtype=float)
            for a, b in splits:
                prior_fold = float(Y.iloc[a][k].mean())
                g = Y.iloc[a].groupby(xtr[c].iloc[a].values)[k].agg(['sum', 'count'])
                smooth = (g['sum'] + m*prior_fold)/(g['count'] + m)
                oof[b] = xtr[c].iloc[b].map(smooth).fillna(prior_fold).to_numpy()
            xtr[name] = oof
            global_prior = float(Y[k].mean())
            g = Y.groupby(xtr[c].values)[k].agg(['sum', 'count'])
            smooth = (g['sum'] + m*global_prior)/(g['count'] + m)
            xva[name] = xva[c].map(smooth).fillna(global_prior).to_numpy()
            state[c][f'te{k}'] = smooth; state[c][f'prior{k}'] = global_prior
        count = xtr[c].value_counts()
        xtr[c + '_cnt'] = xtr[c].map(count).astype(float).to_numpy()
        xva[c + '_cnt'] = xva[c].map(count).fillna(0).astype(float).to_numpy()
        state[c]['cnt'] = count
    return (xtr, xva, state) if return_state else (xtr, xva)


def apply_state(state, F_valid, columns, cats, m=None):
    """Rebuild the validation matrix from stored full-train statistics only (used for the reload check)."""
    xva = F_valid.copy()
    for c in GEO:
        for k in (0, 2):
            xva[f'{c}_te{k}'] = xva[c].map(state[c][f'te{k}']).fillna(state[c][f'prior{k}']).to_numpy()
        xva[c + '_cnt'] = xva[c].map(state[c]['cnt']).fillna(0).astype(float).to_numpy()
    return pd.get_dummies(xva, columns=cats, dtype=int).reindex(columns=columns, fill_value=0).astype('float32')


def original_target_enc():
    """The ORIGINAL function object, extracted from the SHA-pinned exploratory script (for equivalence checks)."""
    path = ROOT/ORIGINAL_SCRIPT
    require(sha256_file(path) == config_pins()['evaluate_te_ros.py'], 'Exploratory script changed')
    tree = ast.parse(path.read_text(encoding='utf-8'))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'target_enc')
    ns = {'pd': pd, 'np': np, 'StratifiedKFold': StratifiedKFold}
    exec(compile(ast.Module([node], []), str(path), 'exec'), ns)  # trusted: committed, hash-pinned file
    return ns['target_enc']


def config_pins():
    return json.loads((HERE/'config.json').read_text(encoding='utf-8'))['original_source_sha256']


def original_frame(X):
    """X (normalized by preprocess: geo as str) -> the exploratory feature frame: geo IDs numeric, text as category."""
    F = X.copy()
    for c in GEO:
        F[c] = F[c].astype('int64')
    for c in TEXT:
        F[c] = pd.Categorical(F[c])  # vocabulary = observed development values (label-free schema, as in the original)
    return F


def inner_splits_for(scheme, X_train, y_train):
    """None = original inner StratifiedKFold; Group splits: T11 group-aware inner folds on (geo1, geo2)."""
    if scheme == 'strat':
        return None, None
    g = group_ids(X_train[GEO].reset_index(drop=True), 'geo2')
    folds = inner_group_fold_ids(np.asarray(y_train), g, 5, 42)
    return [(np.flatnonzero(folds != i), np.flatnonzero(folds == i)) for i in range(5)], folds


def build_fold_matrices(F_train, X_train, y_train, F_valid, scheme, c):
    """Encoding + ROS from TRAINING rows/labels only; the validation frame is encoded with training statistics.
    No validation label is an argument of this function."""
    t0 = time.perf_counter()
    y0 = pd.Series(np.asarray(y_train, dtype=np.int64) - 1, index=F_train.index)
    splits, inner = inner_splits_for(scheme, X_train, y_train)
    A, B, state = target_enc(F_train, y0, F_valid, list(GEO), m=c['te']['smoothing_m'], splits=splits, return_state=True)
    cats = [col for col in F_train.columns if col in TEXT]  # DataFrame column order, as select_dtypes in the original
    Ao = pd.get_dummies(A, columns=cats, dtype=int)
    Bo = pd.get_dummies(B, columns=cats, dtype=int).reindex(columns=Ao.columns, fill_value=0)
    selected = Ao.columns.tolist()
    require(len(selected) == c['expected_model_columns'], f'Feature count drift {len(selected)}')
    Ao = Ao.astype('float32'); Bo = Bo.astype('float32')
    te_seconds = time.perf_counter() - t0
    t1 = time.perf_counter()
    Xos, yos = RandomOverSampler(random_state=c['ros']['random_state']).fit_resample(Ao[selected], y0)
    ros_seconds = time.perf_counter() - t1
    state_out = dict(columns=selected, cats=cats, geo=state, m=c['te']['smoothing_m'])
    return dict(Ao=Ao, Bo=Bo[selected], Xos=Xos, yos=yos, selected=selected, state=state_out, y0=y0,
                inner_folds=inner, te_seconds=te_seconds, ros_seconds=ros_seconds)


def geo_diagnostics(F_train, F_valid, built):
    """Descriptive train/validation mismatch of the regional features (no gating)."""
    diag = {}
    for g in GEO:
        seen = np.isin(F_valid[g].to_numpy(), np.unique(F_train[g].to_numpy()))
        diag[g] = dict(valid_rows_with_id_seen_in_train=float(seen.mean()),
                       train_cnt_mean=float(built['Ao'][g + '_cnt'].mean()),
                       valid_cnt_mean=float(built['Bo'][g + '_cnt'].mean()),
                       valid_cnt_zero_share=float((built['Bo'][g + '_cnt'] == 0).mean()))
    for col in built['selected']:
        if col.endswith(('_te0', '_te2')):
            diag[col] = dict(train_mean=float(built['Ao'][col].mean()), train_std=float(built['Ao'][col].std()),
                             valid_mean=float(built['Bo'][col].mean()), valid_std=float(built['Bo'][col].std()))
    return diag


def importance_shares(model, selected):
    imp = dict(zip(selected, model.feature_importances_))
    total = float(sum(imp.values())) or 1.0
    return dict(raw_geo_ids=sum(imp[g] for g in GEO)/total,
                te=sum(v for k, v in imp.items() if k.endswith(('_te0', '_te2')))/total,
                cnt=sum(v for k, v in imp.items() if k.endswith('_cnt'))/total,
                age=imp['age']/total)


def predict_proba_deterministic(model, X):
    """Single-thread RF predict_proba. With n_jobs>1 the per-tree accumulation order varies between calls (~1e-16),
    which flips argmax on exact top-2 ties (237 such rows in strat fold 1, attempt 1). Training keeps n_jobs=4."""
    keep = model.n_jobs
    model.n_jobs = 1
    try:
        return np.asarray(model.predict_proba(X), dtype=np.float64)
    finally:
        model.n_jobs = keep


def fit_rf(Xos, yos, c):
    model = RandomForestClassifier(**c['rf_params'])
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model.fit(Xos, yos)
    fit_seconds = time.perf_counter() - t0
    require(np.array_equal(model.classes_, [0, 1, 2]), 'Unexpected RF class order')
    return model, fit_seconds, [str(w.message) for w in caught]


# ---------------------------------------------------------------- inputs / design
def load_inputs(data):
    require_not_holdout(data)
    X, y, ids, old_fold, contract = load(data)
    return X, y, ids, np.asarray(old_fold), contract


def comparator_check(c, ids, y, folds):
    """Verify every pre-registered comparator OOF (path, SHA, fold alignment, labels); return their SHAs."""
    seen = {}
    for scheme, comps in c['comparators'].items():
        for name, ref in comps.items():
            r = reference(ref['path'], ref['sha256'], ids, y, folds[scheme])
            seen[f'{scheme}/{name}'] = dict(path=ref['path'], sha256=ref['sha256'], rows=len(r))
    return seen


def geo_manifest_folds(c, ids, X, y):
    """Existing audited geo2 manifests (SHA verified) -> fold arrays; T5 gates re-run (groups/IDs disjoint, supports)."""
    folds = {}
    for scheme in GEO_SCHEMES:
        ref = c['manifests'][scheme]
        require(sha256_file(ref['path']) == ref['sha256'], f'Manifest changed: {scheme}')
        mf = pd.read_csv(ref['path'])
        np.testing.assert_array_equal(mf[ID].to_numpy(), np.asarray(ids))
        folds[scheme] = mf['fold'].to_numpy()
        t5.validate_assignment(X, y, ids, folds[scheme], GROUP_COLUMNS, t5.cfg())
    return folds


def preflight(data, destination):
    """NO classifier is fitted.  Real-data checks of the input contract, split manifests, comparators and (fold 0 of every
    scheme) the encoding + ROS matrices, including bit-equality of the stratified fold-0 encoding with the ORIGINAL
    function and the recorded fold-0 row counts."""
    c = config()
    dest = Path(destination)
    require(not dest.exists(), 'Do not overwrite an existing preflight')
    X, y, ids, old_fold, contract = load_inputs(data)
    F = original_frame(X)
    folds = {'strat': old_fold, **geo_manifest_folds(c, ids, X, y)}
    res = dict(study_id=STUDY_ID, kind='preflight; no classifier fitted', classifiers_fitted=0, holdout_accessed=False,
               created_at_utc=now(), environment=environment(), data_contract=contract,
               original_source_sha256_verified={n: sha256_file(ROOT/'experiments/te_ros_rf_exploratory'/n) == s
                                                for n, s in c['original_source_sha256'].items()
                                                if (ROOT/'experiments/te_ros_rf_exploratory'/n).exists()})
    res['comparators'] = comparator_check(c, ids, y, folds)
    res['geo_manifests'] = {s: c['manifests'][s]['sha256'] for s in GEO_SCHEMES}
    rec = c['recorded_reference']
    per = {}
    for scheme in SCHEMES:
        fo = folds[scheme]
        a, b = np.flatnonzero(fo != 0), np.flatnonzero(fo == 0)
        built = build_fold_matrices(F.iloc[a], X.iloc[a], y.iloc[a], F.iloc[b], scheme, c)
        info = dict(train_rows=len(a), valid_rows=len(b), model_columns=len(built['selected']),
                    ros_rows=len(built['yos']), ros_class_counts=np.bincount(built['yos']).tolist(),
                    te_seconds=built['te_seconds'], ros_seconds=built['ros_seconds'],
                    diagnostics=geo_diagnostics(F.iloc[a], F.iloc[b], built))
        if scheme == 'strat':
            fo0 = rec['fold0']
            info['matches_recorded_fold0_counts'] = bool(len(a) == fo0['train_rows'] and len(b) == fo0['valid_rows']
                                                         and len(built['yos']) == fo0['ros_train_rows']
                                                         and len(built['selected']) == rec['model_columns'])
            y0 = pd.Series(y.iloc[a].astype('int64').to_numpy() - 1, index=F.index[a])
            A0, B0 = original_target_enc()(F.iloc[a], y0, F.iloc[b], list(GEO))
            A0 = pd.get_dummies(A0, columns=built['state']['cats'], dtype=int).astype('float32')
            B0 = (pd.get_dummies(B0, columns=built['state']['cats'], dtype=int)
                  .reindex(columns=A0.columns, fill_value=0).astype('float32'))
            info['encoding_bit_identical_to_original_function'] = bool(
                list(A0.columns) == built['selected'] and A0.equals(built['Ao']) and B0.equals(built['Bo']))
        else:
            fi = built['inner_folds']
            info['inner_group_fold_sizes'] = np.bincount(fi, minlength=5).tolist()
        per[scheme] = info
        print(scheme, {k: v for k, v in info.items() if k != 'diagnostics'}, flush=True)
        del built
    res['fold0_matrices'] = per
    ok = (per['strat']['matches_recorded_fold0_counts'] and per['strat']['encoding_bit_identical_to_original_function']
          and all(v['model_columns'] == c['expected_model_columns'] for v in per.values())
          and all(res['original_source_sha256_verified'].values()) and environment() == c['environment'])
    res['status'] = 'PASS' if ok else 'FAIL'
    write(dest/'preflight.json', res)
    require(ok, 'Preflight failed')
    return res


def prepare(data, output, code_commit, preflight_path):
    output = Path(output)
    require(not output.exists(), 'Do not overwrite an existing study')
    require(len(code_commit) == 40 and all(ch in '0123456789abcdef' for ch in code_commit), 'Full commit SHA required')
    c = config()
    require(environment() == c['environment'], 'Environment differs from the pinned environment')
    require(shutil.disk_usage(output.parent).free/2**30 >= c['min_free_disk_gb_at_prepare'], 'Not enough free disk')
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(['git', 'status', '--porcelain', '--', *SOURCE], cwd=ROOT, capture_output=True, text=True,
                           check=True).stdout.strip()
    require(head == code_commit and not dirty, 'Sealed sources must be committed at code_commit (HEAD) and clean')
    pf = json.loads(Path(preflight_path).read_text(encoding='utf-8'))
    require(pf['status'] == 'PASS' and pf['study_id'] == STUDY_ID and pf['classifiers_fitted'] == 0,
            'Preflight must PASS before prepare')
    X, y, ids, old_fold, contract = load_inputs(data)
    folds = {'strat': old_fold, **geo_manifest_folds(c, ids, X, y)}
    comparators = comparator_check(c, ids, y, folds)
    output.mkdir(parents=True)
    (output/'backup').mkdir()
    for scheme in GEO_SCHEMES:  # private working copies (bytes verified above); never committed to Git
        dst = output/'manifests'/scheme/'manifest.csv'
        dst.parent.mkdir(parents=True)
        shutil.copyfile(c['manifests'][scheme]['path'], dst)
        require(sha256_file(dst) == c['manifests'][scheme]['sha256'], 'Manifest copy mismatch')
    design = dict(study_id=STUDY_ID, config=c, code_commit=code_commit, source_sha256=sources(), data_contract=contract,
                  environment=environment(), manifest_sha256={s: c['manifests'][s]['sha256'] for s in GEO_SCHEMES},
                  comparators=comparators,
                  preflight=dict(path=str(Path(preflight_path).resolve()), sha256=sha256_file(preflight_path)),
                  study_targets=[key(*t) for t in targets()], max_new_study_fits=c['max_new_study_fits'],
                  automatic_retries=0, comparator_refits=0, holdout_accessed=False, created_at_utc=now(),
                  classifiers_fitted=0)
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
    X, y, ids, old_fold, contract = load_inputs(data)
    require(contract == design['data_contract'], 'Data contract changed')
    folds = {'strat': old_fold}
    for scheme in GEO_SCHEMES:
        path = output/'manifests'/scheme/'manifest.csv'
        require(sha256_file(path) == design['manifest_sha256'][scheme], f'Manifest changed: {scheme}')
        mf = pd.read_csv(path)
        np.testing.assert_array_equal(mf[ID].to_numpy(), np.asarray(ids))
        folds[scheme] = mf['fold'].to_numpy()
    return X, y, ids, folds, design


# ---------------------------------------------------------------- stage gate
def stage_a_passed(out):
    path = Path(out)/'stage_a_result.json'
    if not path.exists():
        return False
    return json.loads(path.read_text(encoding='utf-8')).get('status') in PASS_A


# ---------------------------------------------------------------- one fold with checkpointing
def fit_fold(X, F, y, ids, folds, out, scheme, f, design):
    out = Path(out)
    k = key(scheme, ARM, f)
    require(k in design['study_targets'], f'{k} is not a pre-registered study fit')
    ledger = read_ledger(out)
    entry = ledger['folds'].get(k)
    if entry and entry['status'] == 'complete':
        check_saved(out, entry)
        log(out, f'{k} already complete; verified hashes, not refit')
        return entry, False
    require(STAGE_OF[scheme] == 'A' or stage_a_passed(out), 'Stage B requires a passed Stage A result')
    c = design['config']
    require(shutil.disk_usage(out).free/2**30 >= c['min_free_disk_gb_per_fold'], 'Not enough free disk')
    attempts = sum(e['attempts'] for e in ledger['folds'].values())
    require(attempts < design['max_new_study_fits'], 'Fit budget exhausted')
    prior = entry['attempts'] if entry else 0
    entry = dict(dir=f'{scheme}/{ARM}', scheme=scheme, arm=ARM, fold=f, status='running', attempts=prior + 1,
                 started_at_utc=now())
    ledger['folds'][k] = entry
    atomic_write(out/'ledger.json', ledger)
    if prior:
        log(out, f'{k} previous attempt did not finish; manual resume attempt {prior + 1}')
    d = out/entry['dir']
    d.mkdir(parents=True, exist_ok=True)
    for stale in d.glob(f'fold_{f}[._]*'):
        stale.rename(stale.with_name(stale.name + f'.incomplete_attempt{prior}'))
    event(out, fold=k, action='fit_start', attempt=prior + 1)
    fo = folds[scheme]
    a, b = np.flatnonzero(fo != f), np.flatnonzero(fo == f)
    require(len(a) + len(b) == len(ids) and len(b) > 0, 'Fold membership invalid')
    built = build_fold_matrices(F.iloc[a], X.iloc[a], y.iloc[a], F.iloc[b], scheme, c)
    diag = geo_diagnostics(F.iloc[a], F.iloc[b], built)
    model, fit_seconds, fit_warnings = fit_rf(built['Xos'], built['yos'], c)
    t0 = time.perf_counter()
    p = predict_proba_deterministic(model, built['Bo'])
    predict_seconds = time.perf_counter() - t0
    require(p.shape == (len(b), 3) and np.isfinite(p).all(), 'Invalid probabilities')
    pred = p.argmax(1) + 1
    model_path, state_path = d/f'fold_{f}_model.joblib', d/f'fold_{f}_encoder_state.joblib'
    preds = d/f'fold_{f}_pred.csv.gz'
    joblib.dump(model, model_path, compress=c['checkpoint']['model_compress'])
    joblib.dump(built['state'], state_path, compress=3)
    frame = pd.DataFrame({'building_id': ids.iloc[b].to_numpy(), 'fold': f, 'y_true': y.iloc[b].to_numpy(),
                          'y_pred': pred, **{n: p[:, i] for i, n in enumerate(PROBS)}})
    write_pred(preds, frame)
    # reload verification (freshly generated trusted files): model + stored encoder state -> same probabilities
    loaded, state = joblib.load(model_path), joblib.load(state_path)
    Bo2 = apply_state(state['geo'], F.iloc[b], state['columns'], state['cats'])
    require(Bo2.equals(built['Bo']), 'Encoder state does not reproduce the validation matrix')
    p2 = predict_proba_deterministic(loaded, Bo2)
    reload_diff = float(np.max(np.abs(p2 - p)))
    require(reload_diff <= c['checkpoint']['reload_probability_atol'] and np.array_equal(p2.argmax(1) + 1, pred),
            f'Model reload mismatch {reload_diff}')
    back = read_pred(preds)
    np.testing.assert_array_equal(back.building_id.to_numpy(), frame.building_id.to_numpy())
    np.testing.assert_array_equal(back.y_pred.to_numpy(), pred)
    np.testing.assert_array_equal(back[PROBS].to_numpy(), p)
    record = dict(study_id=STUDY_ID, scheme=scheme, arm=ARM, fold=f, stage=STAGE_OF[scheme],
                  train_rows=len(a), valid_rows=len(b), ros_rows=len(built['yos']),
                  train_class_counts=[int((y.iloc[a] == v).sum()) for v in (1, 2, 3)],
                  ros_class_counts=np.bincount(built['yos'], minlength=3).tolist(),
                  model_columns=len(built['selected']),
                  feature_names_sha256=hashlib.sha256('\n'.join(built['selected']).encode()).hexdigest(),
                  inner_te_folds='stratified_random_rs0' if scheme == 'strat' else 'stratified_group_geo2_seed42',
                  metrics=score(y.iloc[b], pred), pred_share=(np.bincount(pred - 1, minlength=3)/len(pred)).tolist(),
                  te_seconds=built['te_seconds'], ros_seconds=built['ros_seconds'], fit_seconds=fit_seconds,
                  predict_seconds=predict_seconds, warnings=fit_warnings, geo_diagnostics=diag,
                  importance_shares=importance_shares(model, built['selected']),
                  effective_parameters={k2: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                                        for k2, v in model.get_params().items()},
                  n_trees=len(model.estimators_), total_nodes=int(sum(t.tree_.node_count for t in model.estimators_)),
                  reload_probability_max_abs_diff=reload_diff, csv_round_trip_exact=True, finished_at_utc=now())
    require(record['effective_parameters'] == c['rf_default_snapshot'], 'RF params drift')
    write(d/f'fold_{f}.json', record)
    shas = {pth.name: sha256_file(pth) for pth in [model_path, state_path, preds, d/f'fold_{f}.json']}
    bd = out/'backup'/entry['dir']
    bd.mkdir(parents=True, exist_ok=True)
    for name in shas:
        require(not (bd/name).exists(), 'Backup already exists; refusing overwrite')
        shutil.copy2(d/name, bd/name)
    entry.update(status='complete', artifact_sha256=shas, macro_f1=record['metrics']['macro_f1'],
                 fit_seconds=fit_seconds, finished_at_utc=now())
    check_saved(out, entry)
    ledger = read_ledger(out)
    ledger['folds'][k] = entry
    atomic_write(out/'ledger.json', ledger)
    event(out, fold=k, action='fit_complete', macro_f1=record['metrics']['macro_f1'], seconds=fit_seconds,
          artifact_sha256=shas)
    log(out, f'{k} F1={record["metrics"]["macro_f1"]:.9f} te={built["te_seconds"]:.0f}s ros={built["ros_seconds"]:.0f}s '
             f'fit={fit_seconds:.0f}s reload_diff={reload_diff:.3g} saved+verified+backed-up')
    return entry, True


def fit_all(data, output, stage):
    require(stage in ('A', 'B'), 'stage must be A or B')
    out = Path(output)
    lock = out/'RUNNING.lock'
    require(not lock.exists(), f'Another run holds {lock}; remove only if that process is dead')
    lock.write_text(f'pid={os.getpid()} started={now()}\n')
    try:
        X, y, ids, folds, design = verify(data, out)
        require(stage == 'A' or stage_a_passed(out), 'Stage B requires stage_a_result.json with EXACT/WITHIN_TOLERANCE '
                                                     '(run finalize-a first)')
        F = original_frame(X)
        before = pd.util.hash_pandas_object(X, index=True).to_numpy()
        todo = [t for t in targets() if STAGE_OF[t[0]] == stage]
        log(out, f'fit-all stage {stage} start: {len(todo)} targets (completed ones are hash-verified, not refit)')
        t0 = time.perf_counter()
        for s, _, f in todo:
            try:
                fit_fold(X, F, y, ids, folds, out, s, f, design)
            except Exception as exc:
                log(out, f'STOPPED {key(s, ARM, f)} {type(exc).__name__}: {exc}')
                event(out, fold=key(s, ARM, f), action='stopped', error=f'{type(exc).__name__}: {exc}',
                      trace=traceback.format_exc())
                raise
        np.testing.assert_array_equal(before, pd.util.hash_pandas_object(X, index=True).to_numpy())
        log(out, f'fit-all stage {stage} finished wall={time.perf_counter() - t0:.1f}s')
    finally:
        lock.unlink(missing_ok=True)


# ---------------------------------------------------------------- pre-registered rules (pure functions)
def judge_stage_a(rules, recorded, observed):
    """`recorded`/`observed`: lists of 5 dicts(confusion 3x3 [true x pred], macro_f1).  Returns the Stage A verdict."""
    ra = rules['stage_a']
    rec_cm = [np.asarray(r['confusion']) for r in recorded]
    obs_cm = [np.asarray(o['confusion']) for o in observed]
    exact = all(np.array_equal(r, o) for r, o in zip(rec_cm, obs_cm))
    f1_diff = [abs(o['macro_f1'] - r['macro_f1']) for r, o in zip(recorded, observed)]
    mean_diff = abs(float(np.mean([o['macro_f1'] for o in observed])) - float(np.mean([r['macro_f1'] for r in recorded])))
    pr, po = sum(rec_cm), sum(obs_cm)
    g3_diff = abs(po[2, 2]/po[2].sum() - pr[2, 2]/pr[2].sum())
    e31_rel = abs(int(po[2, 0]) - int(pr[2, 0]))/max(int(pr[2, 0]), 1)
    cell = [float(np.max(np.abs(r - o)))/float(r.sum()) for r, o in zip(rec_cm, obs_cm)]
    parts = dict(fold_f1_ok=bool(max(f1_diff) <= ra['fold_f1_abs_diff_max']),
                 mean_f1_ok=bool(mean_diff <= ra['mean_f1_abs_diff_max']),
                 pooled_grade3_recall_ok=bool(g3_diff <= ra['pooled_grade3_recall_abs_diff_max']),
                 pooled_error31_ok=bool(e31_rel <= ra['pooled_error31_rel_diff_max']),
                 confusion_cells_ok=bool(max(cell) <= ra['fold_confusion_cell_abs_diff_max_fraction']))
    status = 'EXACT' if exact else ('WITHIN_TOLERANCE' if all(parts.values()) else 'FAIL')
    return dict(status=status, exact_all_five_confusions=bool(exact), parts=parts,
                inputs=dict(fold_f1_abs_diff=f1_diff, mean_f1_abs_diff=mean_diff, pooled_grade3_recall_abs_diff=float(g3_diff),
                            pooled_error31_rel_diff=float(e31_rel), max_cell_diff_fraction_by_fold=cell))


def judge_stage_b(rules, vs_lr, vs_o):
    """Descriptive pre-registered labels; inputs are 20-fold aggregates (mean_delta, fold_wins, n_folds,
    pooled_grade3_recall_delta).  This is NOT a factor attribution and does not select a final model."""
    rb = rules['stage_b']
    r1, r2 = rb['B1_vs_LR_C'], rb['B2_vs_LGBM_O']
    b1 = dict(mean_delta_ok=bool(vs_lr['mean_delta'] > r1['mean_delta_gt']),
              fold_wins_ok=bool(vs_lr['fold_wins'] >= r1['fold_wins_min'] and vs_lr['n_folds'] == r1['n_folds']),
              grade3_recall_ok=bool(vs_lr['pooled_grade3_recall_delta'] >= r1['pooled_grade3_recall_delta_min']),
              split_mean_positive_ok=bool(vs_lr['splits_with_positive_mean'] >= r1['splits_with_positive_mean_min']))
    b2 = dict(mean_delta_ok=bool(vs_o['mean_delta'] >= r2['mean_delta_min']),
              fold_wins_ok=bool(vs_o['fold_wins'] >= r2['fold_wins_min'] and vs_o['n_folds'] == r2['n_folds']),
              grade3_recall_ok=bool(vs_o['pooled_grade3_recall_delta'] >= r2['pooled_grade3_recall_delta_min']),
              split_mean_positive_ok=bool(vs_o['splits_with_positive_mean'] >= r2['splits_with_positive_mean_min']))
    h1, h2 = all(b1.values()), all(b2.values())
    return dict(B1_vs_LR_C=dict(holds=h1, parts=b1, inputs=vs_lr), B2_vs_LGBM_O=dict(holds=h2, parts=b2, inputs=vs_o),
                label=('B1_AND_B2' if h1 and h2 else 'B1_ONLY' if h1 else 'B2_ONLY' if h2 else 'NEITHER'))


# ---------------------------------------------------------------- finalize
def assemble(out, y, ids, folds, scheme):
    out = Path(out)
    ledger = read_ledger(out)
    entries = [ledger['folds'].get(key(scheme, ARM, f)) for f in range(5)]
    require(all(e and e['status'] == 'complete' for e in entries), f'{scheme} incomplete')
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
    bd.mkdir(parents=True, exist_ok=True)
    for name in ['oof.csv.gz', 'metrics.json']:
        require(not (bd/name).exists(), 'Backup already exists; refusing overwrite')
        shutil.copy2(d/name, bd/name)
    log(out, f'{scheme} OOF assembled mean={result["cv_mean"]:.9f} sd={result["cv_std"]:.9f}')
    return result


def finalize_a(data, output):
    """Stage A verdict from the 5 stratified folds (confusion matrices vs the recorded exploratory aggregates)."""
    out = Path(output)
    X, y, ids, folds, design = verify(data, out)
    require(not (out/'stage_a_result.json').exists(), 'stage_a_result.json exists; refusing overwrite')
    ledger = read_ledger(out)
    require(all(ledger['folds'].get(key('strat', ARM, f), {}).get('status') == 'complete' for f in range(5)),
            'Stage A folds incomplete')
    c = design['config']
    observed = []
    for f in range(5):
        e = ledger['folds'][key('strat', ARM, f)]
        check_saved(out, e)
        rec = json.loads((out/'strat'/ARM/f'fold_{f}.json').read_text(encoding='utf-8'))
        observed.append(dict(confusion=rec['metrics']['confusion'], macro_f1=rec['metrics']['macro_f1'],
                             ros_rows=rec['ros_rows'], model_columns=rec['model_columns']))
    recorded = c['recorded_reference']['folds']
    verdict = judge_stage_a(c['rules'], recorded, observed)
    mt = assemble(out, y, ids, folds, 'strat')
    verdict.update(study_id=STUDY_ID, recorded_cv_mean=c['recorded_reference']['cv_macro_f1_mean'],
                   observed_cv_mean=mt['cv_mean'], observed_fold_macro_f1=mt['fold_macro_f1'],
                   recorded_fold_macro_f1=[r['macro_f1'] for r in recorded],
                   observed_ros_rows=[o['ros_rows'] for o in observed], observed_model_columns=[o['model_columns'] for o in observed],
                   created_at_utc=now(), holdout_accessed=False,
                   note='Stage B may start only for EXACT or WITHIN_TOLERANCE; FAIL = stop and report.')
    write(out/'stage_a_result.json', verdict)
    shutil.copy2(out/'stage_a_result.json', out/'backup'/'stage_a_result.json')
    log(out, f'stage A verdict: {verdict["status"]}')
    return verdict


def compare_arms(rf, other, y):
    from experiments.geo_ablation.audit import turnover
    fr = [score(rf.y_true[rf.fold == f], rf.y_pred[rf.fold == f]) for f in range(5)]
    fo = [score(other.y_true[other.fold == f], other.y_pred[other.fold == f]) for f in range(5)]
    deltas = [a['macro_f1'] - b['macro_f1'] for a, b in zip(fr, fo)]
    return dict(fold_delta=deltas, turnover=turnover(np.asarray(y), other.y_pred.to_numpy(), rf.y_pred.to_numpy()),
                other_fold_macro_f1=[b['macro_f1'] for b in fo], rf_fold_macro_f1=[a['macro_f1'] for a in fr])


def pooled_scores(frames):
    y = np.concatenate([f.y_true.to_numpy() for f in frames])
    p = np.concatenate([f.y_pred.to_numpy() for f in frames])
    s = score(y, p)
    return dict(pooled=s, accuracy=s['accuracy'], grade3_recall=s['recall'][2], error31=s['error31'],
                error32=s['error32'], rate31=s['rate31'], rate32=s['rate32'],
                pred_share=(np.bincount(p - 1, minlength=3)/len(p)).tolist())


def finalize_all(data, output):
    out = Path(output)
    X, y, ids, folds, design = verify(data, out)
    require(not (out/'analysis.json').exists(), 'analysis.json exists; refusing overwrite')
    ledger = read_ledger(out)
    tg = design['study_targets']
    complete = sum(ledger['folds'].get(k, {}).get('status') == 'complete' for k in tg)
    attempts = sum(e['attempts'] for e in ledger['folds'].values())
    require(complete == len(tg) == N_TARGETS, 'Ledger incomplete; not finalizing')
    require((out/'stage_a_result.json').exists(), 'stage_a_result.json missing')
    c = design['config']
    stage_a = json.loads((out/'stage_a_result.json').read_text(encoding='utf-8'))
    require(stage_a['status'] in PASS_A, 'Stage A did not pass; Stage B must not exist')
    yv = y.to_numpy()
    res = dict(study_id=STUDY_ID, design_sha256=sha256_file(out/'design.json'), code_commit=design['code_commit'],
               ledger=dict(complete=complete, total=N_TARGETS, attempts=attempts, retries=attempts - complete,
                           fit_seconds_sum=float(sum(e['fit_seconds'] for e in ledger['folds'].values()))),
               holdout_accessed=False, rules=c['rules'], stage_a=stage_a, comparators=design['comparators'],
               note='Reproduction check only. TE vs ROS vs RF vs age995 are NOT separated here; descriptive, no '
                    'significance or causal claims; no holdout; no model selection.', per_scheme={})
    oofs, fold_scores = {}, {}
    for s in SCHEMES:
        mt = assemble(out, y, ids, folds, s)
        oofs[s] = read_pred(out/s/ARM/'oof.csv.gz')
        np.testing.assert_array_equal(oofs[s].building_id.to_numpy(), ids.to_numpy())
        recs = [json.loads((out/s/ARM/f'fold_{f}.json').read_text(encoding='utf-8')) for f in range(5)]
        block = dict(t11.summary(oofs[s], X, mt['fit_seconds']), oof_sha256=mt['oof_sha256'],
                     fold_fit_seconds=[r['fit_seconds'] for r in recs], pred_share_by_fold=[r['pred_share'] for r in recs],
                     importance_shares=[r['importance_shares'] for r in recs],
                     geo_diagnostics=[r['geo_diagnostics'] for r in recs])
        comps = {}
        for name, ref in c['comparators'][s].items():
            other = reference(ref['path'], ref['sha256'], ids, y, folds[s])
            cmp_ = compare_arms(oofs[s], other, yv)
            cmp_['mean_delta'] = float(np.mean(cmp_['fold_delta']))
            cmp_['fold_wins'] = int(sum(d > 0 for d in cmp_['fold_delta']))
            cmp_['other_pooled'] = pooled_scores([other])
            comps[name] = cmp_
            oofs[s, name] = other
        block['comparisons'] = comps
        res['per_scheme'][s] = block
    # ---- Stage B pooled over the 20 geo2 folds (each row appears once per split)
    pooled = {}
    rf_frames = [oofs[s] for s in GEO_SCHEMES]
    for name in ('LR_C', 'LGBM_O', 'LGBM_TE_G'):
        deltas, wins_splits = [], 0
        for s in GEO_SCHEMES:
            d = res['per_scheme'][s]['comparisons'][name]['fold_delta']
            deltas += d
            wins_splits += int(np.mean(d) > 0)
        other = pooled_scores([oofs[s, name] for s in GEO_SCHEMES])
        rf = pooled_scores(rf_frames)
        from experiments.geo_ablation.audit import turnover
        pooled[name] = dict(mean_delta=float(np.mean(deltas)), fold_delta=deltas,
                            fold_wins=int(sum(x > 0 for x in deltas)), n_folds=len(deltas),
                            splits_with_positive_mean=wins_splits,
                            pooled_grade3_recall_delta=rf['grade3_recall'] - other['grade3_recall'],
                            accuracy_diff=rf['accuracy'] - other['accuracy'], rf_pooled=rf, other_pooled=other,
                            turnover=turnover(np.concatenate([yv]*4), np.concatenate([oofs[s, name].y_pred.to_numpy() for s in GEO_SCHEMES]),
                                              np.concatenate([oofs[s].y_pred.to_numpy() for s in GEO_SCHEMES])))
    res['stage_b_pooled_20_folds'] = pooled
    res['judgments'] = dict(stage_a=stage_a['status'],
                            stage_b=judge_stage_b(c['rules'], pooled['LR_C'], pooled['LGBM_O']))
    res['finished_at_utc'] = now()
    write(out/'analysis.json', res)
    require(not (out/'backup'/'analysis.json').exists(), 'Backup already exists; refusing overwrite')
    shutil.copy2(out/'analysis.json', out/'backup'/'analysis.json')
    log(out, f'analysis written: stage A {stage_a["status"]}, stage B label {res["judgments"]["stage_b"]["label"]}')
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['preflight', 'prepare', 'fit-all', 'finalize-a', 'finalize', 'status'])
    p.add_argument('--data', default='data')
    p.add_argument('--output')
    p.add_argument('--destination')
    p.add_argument('--code-commit')
    p.add_argument('--preflight')
    p.add_argument('--stage', choices=['A', 'B'])
    a = p.parse_args()
    if a.action == 'preflight':
        preflight(a.data, a.destination)
    elif a.action == 'prepare':
        prepare(a.data, a.output, a.code_commit, a.preflight)
    elif a.action == 'fit-all':
        fit_all(a.data, a.output, a.stage)
    elif a.action == 'finalize-a':
        finalize_a(a.data, a.output)
    elif a.action == 'finalize':
        finalize_all(a.data, a.output)
    else:
        ledger = read_ledger(a.output)
        print(json.dumps(dict(complete=sum(e['status'] == 'complete' for e in ledger['folds'].values()), total=N_TARGETS,
                              attempts=sum(e['attempts'] for e in ledger['folds'].values()),
                              stage_a_passed=stage_a_passed(a.output),
                              running=[k for k, e in ledger['folds'].items() if e['status'] == 'running']), indent=1))


if __name__ == '__main__':
    sys.exit(main())
