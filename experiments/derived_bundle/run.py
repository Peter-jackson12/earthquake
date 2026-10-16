"""T34 derived 13-column bundle, stratified main split: B (reused OOF) vs S0 / S1 / S2 for LGBM-N and LR-C.

Commands (all private outputs live outside Git under one root; the holdout and the team files are never opened):
  preflight      no classifier is fitted: derived blocks / matrices / reference OOF checks for every fold
  control-check  2 control fits (fold 0, base matrix without derived block) against the reused B OOF (before sealing)
  prepare        seal the design (plan SHA, HEAD blobs, environment, data contract, control result)
  fit-all        30 study fits (+5 LR_B_REFIT only if the LR control failed); one executor, Fold-wise checkpoints
  status         ledger summary
  finalize       aggregate analysis only (judging 5 conditions, descriptive S-S comparisons, diagnostics)

Design: docs/T34_DERIVED_BUNDLE_STRATIFIED_PLAN.md (approved). Existing code (T8 LGBM-N, T1 C policy, geo_group_cv
loader) is imported, never modified. Data come only from `load_development()` through `geo_group_cv.run.load`.

Limits of the safeguards (reviewed 2026-10-16, none changes the planned file set):
  * The runtime guard is a Python-level backstop. `guard_path` resolves links, but the process-wide audit hook only sees
    the path string given to Python-level open / os.listdir / os.scandir: symlinks or junctions into the holdout are not
    resolved by the hook, and opens made inside C++ code (for example LightGBM `Booster(model_file=...)`) are not seen.
    Matching is by substring on every path component (`final_holdout`), so an unrelated component that merely contains
    that text is refused too. An installed audit hook cannot be removed (tests only disarm it).
  * The CPU gate before each fit (mean of 3 samples x 0.5 s below 95 %) is a weak proxy for "no other training is
    running"; it cannot prove the absence of another training job. The control tower must still schedule runs.
  * `psutil` only serves the resource gate; its version is recorded in design.json (`environment_extra`) and re-checked
    in `verify`, but it is not part of the pinned `environment` of the plan.
"""
from __future__ import annotations

import argparse
import errno
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
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from preprocess import GEO, ID, sha256_file  # noqa: E402
from experiments.age995.run import PARAMS as LR_PARAMS, make_pipeline as make_C  # noqa: E402
from experiments.geo_group_cv.run import PROBS, load, require, score, write  # noqa: E402
from experiments.lgbm_native_cat import run as t8  # noqa: E402
from experiments.derived_bundle import features as F  # noqa: E402

HERE = Path(__file__).resolve().parent
STUDY_ID = 'T34_derived_bundle_stratified_v1'
PLAN_SHA256 = '9b9edfed62f70af280b691934da83bfc1d78709627fd429287de1c1b2190255e'
MODELS = ['LR', 'LGBMN']
ARMS = list(F.ARMS)
BASE_ARMS = ('BASE', 'B_REFIT')
SOURCE = ['preprocess.py', 'baseline.py', 'scripts/audit_eda_oof.py', 'configs/data_contract.json',
          'experiments/age995/run.py', 'experiments/age995/interactions/run.py', 'experiments/geo_group_cv/run.py',
          'experiments/model_families/run.py', 'experiments/model_families/config.json',
          'experiments/lgbm_native_cat/__init__.py', 'experiments/lgbm_native_cat/run.py',
          'experiments/lgbm_native_cat/config.json',
          'experiments/derived_bundle/__init__.py', 'experiments/derived_bundle/features.py',
          'experiments/derived_bundle/run.py', 'experiments/derived_bundle/config.json',
          'tests/test_derived_bundle.py', 'docs/T34_DERIVED_BUNDLE_STRATIFIED_PLAN.md']

atomic_write, read_pred, write_pred = t8.atomic_write, t8.read_pred, t8.write_pred
log, event, read_ledger, check_saved, now = t8.log, t8.event, t8.read_ledger, t8.check_saved, t8.now


class ControlFailed(RuntimeError):
    pass


class ResourceBusy(RuntimeError):
    pass


# ======================================================================================== runtime guard (plan section 11)
GUARD_COMPONENT = 'final_holdout'
GUARD_NAMES = ('x_train.csv', 'modeling_ver2_raw.ipynb')
_GUARD = dict(checked=0, refused=0, refused_examples=[], hook_installed=False, armed=False)


class GuardError(PermissionError):
    pass


def guard_violation(path):
    """Reason string if the path (any separator style, any case) has a `final_holdout` component or a forbidden file name."""
    text = os.fsdecode(path) if isinstance(path, bytes) else str(path)
    parts = [p for p in text.replace('\\', '/').lower().split('/') if p]
    if any(GUARD_COMPONENT in p for p in parts):
        return f'path has a {GUARD_COMPONENT} component'
    if parts and parts[-1] in GUARD_NAMES:
        return f'forbidden file name {parts[-1]}'
    return None


def guard_path(path):
    """Refuse the path (and its resolved form) if it touches the holdout or a forbidden file; returns a Path."""
    _GUARD['checked'] += 1
    p = Path(path)
    candidates = [str(path), os.path.normpath(str(path))]
    try:
        candidates.append(str(p.resolve()))
    except OSError:
        pass
    for text in candidates:
        why = guard_violation(text)
        if why:
            _GUARD['refused'] += 1
            if len(_GUARD['refused_examples']) < 20:
                _GUARD['refused_examples'].append(f'{why}: {Path(text).name}')
            raise GuardError(f'Access refused ({why}): {Path(text).name}')
    return p


def _audit_hook(name, args):
    if _GUARD['armed'] and name in ('open', 'os.listdir', 'os.scandir') and args and isinstance(args[0], (str, bytes, os.PathLike)):
        why = guard_violation(args[0])
        if why:
            _GUARD['refused'] += 1
            if len(_GUARD['refused_examples']) < 20:
                _GUARD['refused_examples'].append(f'{name}: {why}')
            raise GuardError(f'Access refused by the audit hook ({name}: {why})')


def install_audit_hook():
    """Process-wide backstop: every Python-level open / listdir / scandir of a guarded path raises (idempotent)."""
    if not _GUARD['hook_installed']:
        sys.addaudithook(_audit_hook)
        _GUARD['hook_installed'] = True
    _GUARD['armed'] = True


def disarm_audit_hook_for_tests():
    """Python cannot remove an audit hook; the test module calls this in tearDownModule so that other test modules
    that run later in the same process are not affected. The command line never disarms."""
    _GUARD['armed'] = False


def guard_summary():
    return dict(checked=_GUARD['checked'], refused=_GUARD['refused'], refused_examples=list(_GUARD['refused_examples']),
                audit_hook_installed=_GUARD['hook_installed'], audit_hook_armed=_GUARD['armed'],
                forbidden_component=GUARD_COMPONENT,
                forbidden_names=list(GUARD_NAMES))


# ======================================================================================== config / seal helpers
def py(obj):
    """Recursively turn numpy scalars / arrays into JSON-safe python values."""
    if isinstance(obj, dict):
        return {str(k): py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [py(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return py(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def jwrite(path, obj):
    write(path, py(obj))


def jatomic(path, obj):
    atomic_write(path, py(obj))


def key(model, arm, fold):
    return f'{model}/{arm}/{fold}'


def study_targets(refit):
    """Fixed execution order (plan 6.1 step 7)."""
    t = [('LR', 'B_REFIT', f) for f in range(5)] if refit else []
    t += [(m, a, f) for m in MODELS for a in ARMS for f in range(5)]
    return t


def config(path=HERE/'config.json'):
    c = json.loads(Path(path).read_text(encoding='utf-8'))
    require(c['study_id'] == STUDY_ID and c['scheme'] == 'stratified' and c['n_folds'] == 5, 'Unplanned study')
    require(c['plan_sha256'] == PLAN_SHA256, 'Plan SHA pin changed')
    require(c['models'] == MODELS and c['arms'] == ARMS, 'Model / arm set changed')
    d = c['derived']
    require(d['numeric'] == F.NUMERIC_COLS and d['age_mean'] == F.AGE_COLS and d['categorical'] == F.COMBO_COLS
            and {a: d['arm_columns'][a] for a in ARMS} == {a: F.derived_names(a) for a in ARMS}
            and d['max_combinations'] == F.MAX_COMBINATIONS, 'Derived column definition changed')
    require([len(d['arm_columns'][a]) for a in ARMS] == [10, 13, 13], 'S0 must have 10 and S1/S2 13 columns')
    require(c['lgbm_params'] == t8.config()['lgbm_params'], 'LightGBM params differ from T8 LGBM-N')
    require(c['lr']['params'] == LR_PARAMS and c['lr']['threadpool_limit'] == 1 and c['lgbm']['threadpool_limit'] == 2
            and c['lgbm_params']['n_jobs'] == 2, 'Model / thread settings differ from the baselines')
    require(c['lgbm']['base_categorical_indices'] == t8.categorical_indices() and c['lgbm']['base_columns'] == 39
            and {a: c['lgbm']['derived_categorical_indices'][a] for a in ARMS}
            == {a: F.categorical_positions(a, 39) for a in ARMS}
            and c['lgbm']['input_columns'] == {'BASE': 39, 'S0': 49, 'S1': 52, 'S2': 52}, 'LGBM column layout changed')
    fp = c['fit_plan']
    require(len(study_targets(False)) == fp['study_fits'] == 30 and len(study_targets(True)) == 35
            and fp['completed_base'] == fp['study_fits'] + fp['control_fits'] == 32
            and fp['completed_with_refit'] == 37 and fp['attempt_cap'] == 36 and fp['attempt_cap_with_refit'] == 41
            and fp['max_infrastructure_retries'] == 4 and fp['max_attempts_per_unit'] == 2, 'Unplanned fit plan')
    require(len({key(*t) for t in study_targets(True)}) == 35, 'Duplicate targets')
    require(c['control']['fold'] == 0 and c['control']['probability_atol'] == 1e-9, 'Control rule changed')
    rule = c['rule']
    require(rule['mean_delta_min'] == 0.003 and rule['wins_min'] == 4 and rule['worst_fold_floor'] == -0.003
            and rule['recall3_delta_min'] == 0.0 and rule['to1_delta_max'] == 0, 'Judging thresholds changed')
    require(not c['holdout_access'] and not c['tuning'] and not c['early_stopping'] and c['class_weight'] is None
            and c['sampling'] is None and not c['threshold_or_decision_rule_change'] and c['outer_parallel_jobs'] == 1
            and c['automatic_retries'] == 0 and not c['automatic_model_replacement'], 'Unplanned run policy')
    return c


def check_plan_sha(c, root=None):
    """The plan document must be exactly the approved one (config pin and module constant)."""
    require(c['plan_sha256'] == PLAN_SHA256, 'Config plan_sha256 differs from the approved pin')
    actual = sha256_file(guard_path(Path(root or ROOT)/c['plan']))
    require(actual == c['plan_sha256'], f'Plan document SHA {actual} differs from the approved {c["plan_sha256"]}')
    return actual


def environment():
    names = ['numpy', 'pandas', 'scipy', 'scikit-learn', 'lightgbm', 'joblib', 'threadpoolctl']
    return dict(python=platform.python_version(), **{n: importlib.metadata.version(n) for n in names})


def environment_extra():
    """Recorded (not pinned by the plan): psutil serves only the resource gate."""
    return dict(psutil=importlib.metadata.version('psutil'))


def require_environment(c):
    require(environment() == c['environment'], 'Environment differs from the pinned environment')


def sources(root=None):
    return {p: sha256_file(Path(root or ROOT)/p) for p in SOURCE}


def git_head(root=None):
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root or ROOT, capture_output=True, text=True,
                          check=True).stdout.strip()
    require(len(head) == 40 and all(ch in '0123456789abcdef' for ch in head), 'Full 40-character HEAD SHA required')
    return head


def source_mismatches(paths, root=None):
    """Git-blob check of ONLY the sealed files: the worktree blob of each must equal the blob at HEAD."""
    root = Path(root or ROOT)
    listing = subprocess.run(['git', 'ls-tree', '-r', 'HEAD', '--', *paths], cwd=root, capture_output=True, text=True,
                             check=True).stdout
    tree = {}
    for line in listing.splitlines():
        meta, path = line.split('\t', 1)
        tree[path] = meta.split()[2]
    bad = [(p, 'missing in worktree') for p in paths if not (root/p).is_file()]
    present = [p for p in paths if (root/p).is_file()]
    if present:
        hashes = subprocess.run(['git', 'hash-object', '--', *present], cwd=root, capture_output=True, text=True,
                                check=True).stdout.split()
        require(len(hashes) == len(present), 'hash-object output mismatch')
        for p, h in zip(present, hashes):
            if p not in tree:
                bad.append((p, 'not committed at HEAD'))
            elif tree[p] != h:
                bad.append((p, 'worktree blob differs from HEAD'))
    return bad


def is_infrastructure_error(exc):
    """Process kill / memory / disk problems only (plan 6.2); everything else is a defect and stops the study."""
    return isinstance(exc, (MemoryError, KeyboardInterrupt)) or (
        isinstance(exc, OSError) and exc.errno in (errno.ENOSPC, getattr(errno, 'EDQUOT', -1)))


def resource_check(root, c, cpu_samples=3, interval=0.5):
    """RAM / disk floors and no sustained CPU saturation, checked before a new attempt is recorded."""
    import psutil
    r = c['resources']
    vm = psutil.virtual_memory()
    free_ram = vm.available/2**30
    free_disk = shutil.disk_usage(root).free/2**30
    cpu = float(np.mean([psutil.cpu_percent(interval=interval) for _ in range(cpu_samples)]))
    rec = dict(free_ram_gb=round(free_ram, 2), free_disk_gb=round(free_disk, 2), cpu_percent_mean=round(cpu, 1))
    if free_ram < r['min_free_ram_gb'] or free_disk < r['min_free_disk_gb_per_fold'] or cpu >= r['max_cpu_percent']:
        raise ResourceBusy(f'Resource floor not met (RAM>={r["min_free_ram_gb"]} GiB, disk>={r["min_free_disk_gb_per_fold"]} '
                           f'GiB, CPU<{r["max_cpu_percent"]}%): {rec}')
    return rec


# ======================================================================================== data and references
def load_data(data, c):
    """Development table + stored stratified folds only (`load_development` via geo_group_cv.load)."""
    X, y, ids, folds, contract = load(guard_path(data))
    require(contract['files']['split_manifest.csv'] == c['manifest_sha256']
            and contract['files']['development.csv'] == c['development_sha256'], 'Data SHA differs from the pin')
    folds = np.asarray(folds)
    require(len(X) == c['expected_rows'], 'Development row count changed')
    require(set(np.unique(folds)) == set(range(5)), 'Expected five folds')
    require(all(int((folds == f).sum()) == c['valid_rows_per_fold'] for f in range(5)), 'Unexpected fold sizes')
    return X, y, ids, folds, contract


def read_reference(local_root, spec, ids, y, folds):
    """Hash-verified, building_id-aligned reference OOF (probabilities read with round_trip precision)."""
    path = guard_path(Path(local_root)/spec['path'])
    require(path.is_file() and sha256_file(path) == spec['sha256'], f'Reference OOF SHA mismatch: {spec["name"]}')
    r = read_pred(path)
    require(r[ID].is_unique and len(r) == len(ids), f'Reference OOF coverage: {spec["name"]}')
    r = r.set_index(ID).loc[np.asarray(ids)].reset_index()
    np.testing.assert_array_equal(r.fold.to_numpy(), folds)
    np.testing.assert_array_equal(r.y_true.to_numpy(), np.asarray(y))
    top = np.sort(r[PROBS].to_numpy(), axis=1)
    mism = r[PROBS].to_numpy().argmax(1) + 1 != r.y_pred.to_numpy()
    require(int((mism & (top[:, 2] != top[:, 1])).sum()) == 0, f'Reference y_pred is not the argmax: {spec["name"]}')
    return r


def fold_f1(frame):
    return [score(frame.y_true[frame.fold == f], frame.y_pred[frame.fold == f])['macro_f1'] for f in range(5)]


def verify_references(c, local_root, ids, y, folds):
    """Plan section 5 items 1-3 (SHA, alignment, recomputed fold means vs metrics.json / 6dp table values)."""
    tol = c['reference_tolerance']
    out = {}
    for name, spec in c['references'].items():
        frame = read_reference(local_root, spec, ids, y, folds)
        vals = fold_f1(frame)
        mean = float(np.mean(vals))
        require(abs(mean - spec['cv_mean']) <= tol['cv_mean_6dp'], f'{name}: recomputed mean {mean} differs from {spec["cv_mean"]}')
        pooled = score(frame.y_true, frame.y_pred)
        rec = dict(sha256=spec['sha256'], rows=int(len(frame)), cv_mean=mean, fold_macro_f1=vals, pooled_confusion=pooled['confusion'],
                   error31=pooled['error31'], error32=pooled['error32'], role=spec['role'])
        ref_folds = None
        if 'metrics_path' in spec:
            mp = guard_path(Path(local_root)/spec['metrics_path'])
            require(sha256_file(mp) == spec['metrics_sha256'], f'{name}: metrics.json SHA mismatch')
            m = json.loads(mp.read_text(encoding='utf-8'))
            if name == 'LGBMN_B':
                ref_folds = m['fold_macro_f1']
                require(abs(m['cv_mean'] - mean) <= tol['fold_macro_f1_vs_metrics_json'], 'LGBMN_B: metrics.json mean differs')
            else:
                require(abs(m['cv_macro_f1_mean'] - mean) <= tol['fold_macro_f1_vs_metrics_json'], f'{name}: metrics.json mean differs')
        if 'fold_metrics_path' in spec:
            fp = guard_path(Path(local_root)/spec['fold_metrics_path'])
            require(sha256_file(fp) == spec['fold_metrics_sha256'], f'{name}: fold_metrics.json SHA mismatch')
            fm = json.loads(fp.read_text(encoding='utf-8'))
            ref_folds = [r['validation']['macro_f1'] for r in fm]
            rec['n_iter'] = [r['n_iter'] for r in fm]
            rec['converged'] = [bool(r['convergence_ok']) for r in fm]
            require(rec['n_iter'] == spec['expected_n_iter'] and all(rec['converged']), f'{name}: n_iter / convergence differ from the record')
        if ref_folds is not None:
            diff = max(abs(a - b) for a, b in zip(vals, ref_folds))
            require(diff <= tol['fold_macro_f1_vs_metrics_json'], f'{name}: fold Macro F1 differs from metrics ({diff})')
            rec['fold_macro_f1_max_diff_vs_metrics'] = float(diff)
        out[name] = rec
    return py(out)


# ======================================================================================== matrices
def threads_for(model, c):
    return c['lr']['threadpool_limit'] if model == 'LR' else c['lgbm']['threadpool_limit']


def base_fit(model, X_tr, X_va, c):
    """Baseline preprocessing exactly as the pinned baselines: LR = T1 C policy CSR; LGBM-N = T8 N dense (39 columns)."""
    with threadpool_limits(limits=threads_for(model, c)):
        pre = make_C('C')['preprocess'] if model == 'LR' else t8.make_preprocessor('N')
        t0 = time.perf_counter()
        mt = pre.fit_transform(X_tr)
        mv = pre.transform(X_va)
        seconds = time.perf_counter() - t0
    if model == 'LR':
        for m in (mt, mv):
            require(sparse.isspmatrix_csr(m) and m.dtype == np.float32 and np.isfinite(m.data).all(), 'LR base must be finite float32 CSR')
    else:
        t8.check_matrix('N', mt)
        t8.check_matrix('N', mv)
    return dict(pre=pre, mt=mt, mv=mv, seconds=seconds)


def join(model, base, block):
    if model == 'LR':
        out = sparse.hstack([base, block], format='csr', dtype=np.float32)
        require(sparse.isspmatrix_csr(out) and out.dtype == np.float32, 'LR input must be float32 CSR')
    else:
        out = np.hstack([base, block]).astype(np.float32)
    return out


def prefix_equal(model, full, base):
    w = base.shape[1]
    if full.shape[0] != base.shape[0] or full.shape[1] < w:
        return False
    if model == 'LR':
        return bool((full[:, :w] != base).nnz == 0)
    return bool(np.array_equal(full[:, :w], base, equal_nan=True))


def check_input_matrix(model, arm, m, c):
    if model == 'LR':
        require(sparse.isspmatrix_csr(m) and m.dtype == np.float32 and np.isfinite(m.data).all(), 'LR input invalid')
        return
    k = c['lgbm']['input_columns']['BASE' if arm in BASE_ARMS else arm]
    require(isinstance(m, np.ndarray) and m.dtype == np.float32 and m.shape[1] == k, f'LGBM input must be dense float32 with {k} columns')
    cat = list(c['lgbm']['base_categorical_indices']) + ([] if arm in BASE_ARMS else list(c['lgbm']['derived_categorical_indices'][arm]))
    num = np.delete(m, cat, axis=1)
    require(np.isfinite(num).all(), 'Nonfinite numeric / binary LGBM input')
    codes = m[:, cat]
    known = codes[~np.isnan(codes)]
    require((known >= 0).all() and (known == np.floor(known)).all(), 'Categorical codes must be nonnegative integers')


def build_inputs(model, arm, X_tr, X_va, c, base=None, base_check=None, verify=True):
    """Base matrices + (S arms) the derived block fitted on the training rows only. Returns a dict with every
    object that is saved, the matrices and the per-fold check results (plan 7: items i-iv)."""
    base = base or base_fit(model, X_tr, X_va, c)
    bundle, stats_t, stats_v = None, None, None
    if arm in BASE_ARMS:
        mt, mv = base['mt'], base['mv']
    else:
        bundle = F.DerivedBundle(arm, 'lr' if model == 'LR' else 'lgbm').fit(X_tr)      # label-free, training rows only
        bt, stats_t = bundle.transform(X_tr, return_stats=True)
        bv, stats_v = bundle.transform(X_va, return_stats=True)
        mt, mv = join(model, base['mt'], bt), join(model, base['mv'], bv)
    check_input_matrix(model, arm, mt, c)
    check_input_matrix(model, arm, mv, c)
    checks = {}
    if verify and arm not in BASE_ARMS:
        base2 = base_check or base_fit(model, X_tr, X_va, c)                           # independent second base fit
        checks['prefix_equals_independent_base_train'] = prefix_equal(model, mt, base2['mt'])
        checks['prefix_equals_independent_base_valid'] = prefix_equal(model, mv, base2['mv'])
        require(all(checks.values()), 'Base columns differ from the independent base matrix')
        checks['independent_reference'] = F.check_against_reference(bundle, X_tr, X_va)
        checks['arm_relations'] = F.check_arm_relations(X_tr, X_va)
        checks['stats_train_only'] = F.verify_fitted_on(bundle, X_tr)
    return dict(base=base, bundle=bundle, mt=mt, mv=mv, region_stats_train=stats_t, region_stats_valid=stats_v, checks=checks)


# ======================================================================================== classifiers
def check_proba(p, n):
    require(p.shape == (n, 3) and np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all(), 'Invalid probabilities')
    np.testing.assert_allclose(p.sum(1), 1.0, atol=1e-6, rtol=0)


def fit_classifier(model, arm, mt, mv, y_tr, c):
    """No eval_set, weights, resampling or early stopping. Returns (clf, p_valid, p_train, info)."""
    info = {}
    if model == 'LGBMN':
        from lightgbm import LGBMClassifier
        cat = list(c['lgbm']['base_categorical_indices']) + ([] if arm in BASE_ARMS else list(c['lgbm']['derived_categorical_indices'][arm]))
        clf = LGBMClassifier(**c['lgbm_params'])
        with threadpool_limits(limits=c['lgbm']['threadpool_limit']), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            t0 = time.perf_counter()
            clf.fit(mt, np.asarray(y_tr).astype(np.int32) - 1, categorical_feature=cat)
            info['fit_seconds'] = time.perf_counter() - t0
            p = t8.checked(clf, mv)
            p_train = t8.checked(clf, mt)
        require(clf.booster_.current_iteration() == c['lgbm_params']['n_estimators'], 'Full fixed boosting budget not completed')
        info.update(rounds_completed=int(clf.booster_.current_iteration()), trees=int(clf.booster_.num_trees()),
                    categorical_feature=cat, warnings=[dict(category=w.category.__name__, message=str(w.message)) for w in caught])
    else:
        clf = LogisticRegression(**c['lr']['params'])
        with threadpool_limits(limits=c['lr']['threadpool_limit']), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            t0 = time.perf_counter()
            clf.fit(mt, np.asarray(y_tr))
            info['fit_seconds'] = time.perf_counter() - t0
            require(list(clf.classes_) == [1, 2, 3], 'Class order must be 1, 2, 3')
            p = np.asarray(clf.predict_proba(mv), dtype=np.float64)
            p_train = np.asarray(clf.predict_proba(mt), dtype=np.float64)
        n_iter = np.asarray(clf.n_iter_).tolist()
        warned = any(issubclass(w.category, ConvergenceWarning) for w in caught)
        info.update(n_iter=n_iter, max_iter=c['lr']['params']['max_iter'], convergence_warning=bool(warned),
                    converged=bool(not warned and max(n_iter) < c['lr']['params']['max_iter']),
                    warnings=[dict(category=w.category.__name__, message=str(w.message)) for w in caught])
    check_proba(p, mv.shape[0])
    check_proba(p_train, mt.shape[0])
    return clf, p, p_train, info


def lgbm_model_internals(path):
    """Effective categorical parameters as written into the saved LightGBM model text (diagnostic only)."""
    params, _ = t8.model_text_parameters(path)
    return {k: float(params[k]) for k in t8.CAT_PARAMS}


# ======================================================================================== one fit
def unit_files(model, d, f, arm):
    files = dict(pred=d/f'fold_{f}_pred.csv.gz', base=d/f'fold_{f}_base_pre.joblib', json=d/f'fold_{f}.json')
    files['model'] = d/(f'fold_{f}_model.txt' if model == 'LGBMN' else f'fold_{f}_model.joblib')
    if arm not in BASE_ARMS:
        files['bundle'] = d/f'fold_{f}_bundle.joblib'
    return files


def reload_predict(model, arm, files, X_va, c):
    """Reload the saved base preprocessor, bundle and classifier (trusted files written seconds ago) and predict."""
    pre = joblib.load(files['base'])
    with threadpool_limits(limits=threads_for(model, c)):
        mv = pre.transform(X_va)
        if arm not in BASE_ARMS:
            bundle = joblib.load(files['bundle'])
            mv = join(model, mv, bundle.transform(X_va))
        if model == 'LGBMN':
            import lightgbm
            booster = lightgbm.Booster(model_file=str(files['model']))
            p = np.asarray(booster.predict(mv, num_threads=c['lgbm_params']['n_jobs']), dtype=np.float64)
        else:
            p = np.asarray(joblib.load(files['model']).predict_proba(mv), dtype=np.float64)
    return p


def execute_fit(X, y, ids, folds, d, unit, c):
    """Fit one (model, arm, fold) on the training rows only, write every artifact into d and verify by reloading."""
    model, arm, f = unit['model'], unit['arm'], unit['fold']
    a, b = np.flatnonzero(folds != f), np.flatnonzero(folds == f)
    require(len(a) == c['train_rows_per_fold'] and len(b) == c['valid_rows_per_fold'] and not np.intersect1d(a, b).size,
            'Fold membership differs from the frozen split')
    X_tr, X_va = X.iloc[a], X.iloc[b]
    y_tr, y_va = y.iloc[a].to_numpy(), y.iloc[b].to_numpy()
    built = build_inputs(model, arm, X_tr, X_va, c)
    clf, p, p_train, info = fit_classifier(model, arm, built['mt'], built['mv'], y_tr, c)
    pred = p.argmax(1) + 1
    files = unit_files(model, d, f, arm)
    d.mkdir(parents=True, exist_ok=True)
    joblib.dump(built['base']['pre'], files['base'], compress=3)
    if built['bundle'] is not None:
        joblib.dump(built['bundle'], files['bundle'], compress=3)
    if model == 'LGBMN':
        clf.booster_.save_model(str(files['model']))
    else:
        joblib.dump(clf, files['model'], compress=3)
    frame = pd.DataFrame({ID: ids.iloc[b].to_numpy(), 'fold': f, 'y_true': y_va, 'y_pred': pred,
                          **{n: p[:, i] for i, n in enumerate(PROBS)}})
    write_pred(files['pred'], frame)
    # (a) reloaded models / preprocessors reproduce the probabilities, (b) CSV round trip is exact
    p2 = reload_predict(model, arm, files, X_va, c)
    reload_diff = float(np.max(np.abs(p2 - p)))
    require(reload_diff <= c['checkpoint']['model_reload_probability_atol'] and np.array_equal(p2.argmax(1) + 1, pred),
            f'Model reload mismatch {reload_diff}')
    back = read_pred(files['pred'])
    np.testing.assert_array_equal(back[ID].to_numpy(), frame[ID].to_numpy())
    np.testing.assert_array_equal(back.y_pred.to_numpy(), pred)
    np.testing.assert_array_equal(back[PROBS].to_numpy(), p)
    metrics = score(y_va, pred)
    record = dict(study_id=STUDY_ID, model=model, arm=arm, fold=f, kind=unit['kind'], train_rows=int(len(a)), valid_rows=int(len(b)),
                  valid_class_counts=[int((y_va == v).sum()) for v in (1, 2, 3)],
                  train_class_counts=[int((y_tr == v).sum()) for v in (1, 2, 3)],
                  metrics=metrics, train_macro_f1=score(y_tr, p_train.argmax(1) + 1)['macro_f1'],
                  input_columns=int(built['mt'].shape[1]), base_columns=int(built['base']['mt'].shape[1]),
                  base_age_median=built['base']['pre']['age995'].median_,
                  base_preprocessing_seconds=built['base']['seconds'], info=info,
                  reload_probability_max_abs_diff=reload_diff, csv_round_trip_exact=True,
                  checks=built['checks'], effective_parameters=clf.get_params(), finished_at_utc=now())
    if built['bundle'] is not None:
        record['derived'] = built['bundle'].summary()
        record['derived_feature_names_sha256'] = hashlib.sha256('\n'.join(built['bundle'].feature_names()).encode()).hexdigest()
        record['region_stats_train'] = built['region_stats_train']
        record['region_stats_valid'] = built['region_stats_valid']
        codes = built['bundle'].combo_codes(X_va)
        record['valid_unseen_combinations'] = {n: int(np.isnan(codes[:, i]).sum()) for i, n in enumerate(F.COMBO_COLS)}
    if model == 'LGBMN':
        gain = clf.booster_.feature_importance(importance_type='gain')
        split = clf.booster_.feature_importance(importance_type='split')
        n0 = built['base']['mt'].shape[1]
        record['info']['categorical_effective_parameters'] = lgbm_model_internals(files['model'])
        record['gain_share_derived'] = float(gain[n0:].sum()/gain.sum()) if gain.sum() > 0 and gain.shape[0] > n0 else 0.0
        names = F.derived_names(arm) if arm not in BASE_ARMS else []
        record['derived_gain'] = {n: float(v) for n, v in zip(names, gain[n0:])}
        record['derived_split_count'] = {n: int(v) for n, v in zip(names, split[n0:])}
    else:
        n0 = built['base']['mt'].shape[1]
        if built['bundle'] is not None:
            record['derived_coef_abs_mean'] = float(np.abs(clf.coef_[:, n0:]).mean())
    jwrite(files['json'], record)
    return record, [files[k] for k in ('model', 'base', 'bundle', 'pred', 'json') if k in files], p


def run_unit(X, y, ids, folds, root, unit, ctx, retry_interrupted=False, retry_reason=None, execute_fn=None, resource_fn=None):
    """Ledgered, hash-verified, resumable fit. `ctx` carries config, code commit, allowed keys and the attempt budget."""
    root = Path(root)
    c = ctx['config']
    k = unit['key']
    require(k in ctx['allowed_keys'], f'{k} is not a registered fit')
    ledger = read_ledger(root)
    entry = ledger['folds'].get(k)
    if entry and entry['status'] == 'complete':
        check_saved(root, entry)
        log(root, f'{k} already complete; hashes verified, not refit')
        return entry, False
    failed = [kk for kk, e in ledger['folds'].items() if e['status'] == 'failed']
    require(not failed, f'Failed unit(s) {failed[:3]}: stop and report (no retry, no bypass; a design revision needs a new ID)')
    prior = entry['attempts'] if entry else 0
    reasons = list(entry.get('retry_reasons', [])) if entry else []
    if entry:                                    # status running / interrupted: a process vanished or infra error
        require(retry_interrupted and retry_reason and retry_reason.strip(),
                f'{k} did not finish: explicit --retry-interrupted and an infrastructure --retry-reason are required')
        require(prior < c['fit_plan']['max_attempts_per_unit'], f'{k}: only one infrastructure retry per unit is allowed')
        retries = ctx['retries_before'] + sum(max(e['attempts'] - 1, 0) for e in ledger['folds'].values())
        require(retries < ctx['max_retries'], 'Infrastructure retry reserve exhausted')
        reasons.append(retry_reason)
    used = ctx['attempts_before'] + sum(e['attempts'] for e in ledger['folds'].values())
    require(used < ctx['attempt_cap'], f'Attempt cap {ctx["attempt_cap"]} exhausted')
    resources = (resource_fn or resource_check)(root, c)
    entry = dict(dir=f'{unit["model"]}/{unit["arm"]}', model=unit['model'], arm=unit['arm'], fold=unit['fold'], kind=unit['kind'],
                 status='running', attempts=prior + 1, retry_reasons=reasons, started_at_utc=now(), code_commit=ctx['code_commit'])
    ledger['folds'][k] = entry
    jatomic(root/'ledger.json', ledger)
    event(root, event='attempt', status='running', key=k, attempt=prior + 1, code_commit=ctx['code_commit'],
          resources=resources, retry_reason=retry_reason if prior else None)
    d = root/entry['dir']
    d.mkdir(parents=True, exist_ok=True)
    f = unit['fold']
    for location in (d, root/'backup'/entry['dir']):
        stale = [p for p in (location.glob(f'fold_{f}[._]*') if location.exists() else []) if '.incomplete_attempt' not in p.name]
        if not prior:
            require(not stale, 'Untracked fold files exist for a unit without an earlier attempt')
        for path in stale:                       # a partial artifact of an unfinished attempt is preserved, never deleted
            target = path.with_name(path.name + f'.incomplete_attempt{prior}')
            require(not target.exists(), 'Preserved incomplete artifact already exists')
            path.rename(target)
    try:
        record, files, proba = (execute_fn or execute_fit)(X, y, ids, folds, d, unit, c)
        shas = {p.name: sha256_file(p) for p in files}
        bd = root/'backup'/entry['dir']
        bd.mkdir(parents=True, exist_ok=True)
        for p in files:
            require(not (bd/p.name).exists(), 'Backup already exists; refusing to overwrite')
            shutil.copy2(p, bd/p.name)
        entry.update(status='complete', artifact_sha256=shas, macro_f1=record['metrics']['macro_f1'],
                     fit_seconds=record['info']['fit_seconds'], finished_at_utc=now())
        check_saved(root, entry)
    except BaseException as exc:
        status = 'interrupted' if is_infrastructure_error(exc) else 'failed'
        try:
            ledger = read_ledger(root)
            ledger['folds'][k].update(status=status, error=f'{type(exc).__name__}: {exc}', ended_at_utc=now())
            jatomic(root/'ledger.json', ledger)
            event(root, event='attempt', status=status, key=k, attempt=prior + 1, code_commit=ctx['code_commit'],
                  error=f'{type(exc).__name__}: {exc}', trace=traceback.format_exc())
            log(root, f'STOPPED {k} ({status}) {type(exc).__name__}: {exc}')
        except OSError:
            pass
        raise
    ledger = read_ledger(root)
    ledger['folds'][k] = entry
    jatomic(root/'ledger.json', ledger)
    event(root, event='attempt', status='complete', key=k, attempt=prior + 1, code_commit=ctx['code_commit'],
          macro_f1=entry['macro_f1'], artifact_sha256=shas)
    log(root, f'{k} F1={entry["macro_f1"]:.9f} fit={entry["fit_seconds"]:.1f}s; saved+verified+backed-up')
    return entry, True


def make_unit(model, arm, fold, kind):
    return dict(key=key(model, arm, fold), model=model, arm=arm, fold=fold, kind=kind)


# ======================================================================================== control check
def tie_aware_compare(new, ref0, atol):
    """Aligned comparison of a control refit with the reused B OOF rows of the same fold."""
    new = new.sort_values(ID).reset_index(drop=True)
    ref0 = ref0.sort_values(ID).reset_index(drop=True)
    aligned = bool(len(new) == len(ref0) and np.array_equal(new[ID].to_numpy(), ref0[ID].to_numpy())
                   and np.array_equal(new.y_true.to_numpy(), ref0.y_true.to_numpy()))
    if not aligned:
        return dict(status='FAIL', aligned=False)
    diff = np.abs(new[PROBS].to_numpy() - ref0[PROBS].to_numpy())
    mismatch = new.y_pred.to_numpy() != ref0.y_pred.to_numpy()
    top = np.sort(ref0[PROBS].to_numpy(), axis=1)
    tie = top[:, 2] == top[:, 1]
    unexplained = int((mismatch & ~tie).sum())
    max_diff = float(diff.max())
    ok = bool(max_diff <= atol and unexplained == 0)
    return dict(status='PASS' if ok else 'FAIL', aligned=True, rows=int(len(new)), probability_max_abs_diff=max_diff,
                probability_atol=atol, bit_identical=bool(max_diff == 0.0), labels_identical=bool(not mismatch.any()),
                label_mismatch_rows=int(mismatch.sum()), label_mismatch_on_exact_top2_ties=int((mismatch & tie).sum()),
                unexplained_label_mismatch_rows=unexplained, top2_tie_rows=int(tie.sum()),
                new_macro_f1=score(new.y_true, new.y_pred)['macro_f1'],
                reference_macro_f1=score(ref0.y_true, ref0.y_pred)['macro_f1'])


def control_ctx(c, head):
    return dict(config=c, code_commit=head, allowed_keys={key('LGBMN', 'BASE', 0), key('LR', 'BASE', 0)},
                attempt_cap=c['fit_plan']['attempt_cap'], attempts_before=0, retries_before=0,
                max_retries=c['fit_plan']['max_infrastructure_retries'])


def control_dir_check(dest, result):
    ledger = read_ledger(dest)
    for e in ledger['folds'].values():
        require(e['status'] == 'complete', 'Control ledger contains an unfinished unit')
        check_saved(dest, e)
    return ledger


def read_preflight(root, c, head):
    p = Path(root)/'preflight'/'preflight.json'
    require(p.is_file(), 'Run preflight first')
    pf = json.loads(p.read_text(encoding='utf-8'))
    require(pf['status'] == 'PASS' and pf['classifiers_fitted'] == 0 and pf['code_commit'] == head
            and pf['plan_sha256'] == c['plan_sha256'] and pf['config_sha256'] == sha256_file(HERE/'config.json')
            and pf['source_sha256'] == sources(), 'Preflight does not match the current code / config / plan (rerun it)')
    return pf


def control_check(data, out_root, local_root=None, retry_interrupted=False, retry_reason=None, ctx_override=None,
                  execute_fn=None, resource_fn=None):
    """Plan section 5: refit the base matrices of Fold 0 (LGBM-N, LR-C) and compare with the reused B OOF."""
    install_audit_hook()
    root = guard_path(out_root)
    dest = root/'control_check'
    c = config()
    check_plan_sha(c)
    bad = source_mismatches(SOURCE)
    require(not bad, f'Sealed sources must equal their HEAD blobs: {bad}')
    require_environment(c)
    head = git_head()
    read_preflight(root, c, head)
    local_root = Path(local_root) if local_root else Path(data).resolve().parents[1]
    result_path = dest/'control_check.json'
    if result_path.exists():                     # completed earlier: verify, do not refit
        res = json.loads(result_path.read_text(encoding='utf-8'))
        control_dir_check(dest, res)
        require(res['code_commit'] == head, 'Control check was made at another commit')
        require(res['lgbm']['status'] == 'PASS', 'CONTROL_FAILED_LGBM is recorded; do not seal')
        return res
    lock = dest/'RUNNING.lock'
    dest.mkdir(parents=True, exist_ok=True)
    (dest/'backup').mkdir(exist_ok=True)
    if not (dest/'ledger.json').exists():
        jatomic(dest/'ledger.json', dict(study_id=STUDY_ID, scope='control', folds={}))
    try:
        with lock.open('x', encoding='utf-8') as handle:
            handle.write(f'pid={os.getpid()} started={now()}\n')
    except FileExistsError:
        raise ValueError(f'Another run holds {lock}; remove only after confirming that process is dead')
    try:
        X, y, ids, folds, contract = load_data(data, c)
        refs = verify_references(c, local_root, ids, y, folds)
        ctx = ctx_override or control_ctx(c, head)
        result = dict(study_id=STUDY_ID, code_commit=head, plan_sha256=c['plan_sha256'], config_sha256=sha256_file(HERE/'config.json'),
                      source_sha256=sources(), environment=environment(), environment_extra=environment_extra(),
                      data_contract=contract, references=refs,
                      holdout_accessed=False, counted_as_study_fit=False, created_at_utc=now())
        statuses = {}
        for model, ref_name, bad_status in (('LGBMN', 'LGBMN_B', 'CONTROL_FAILED_LGBM'), ('LR', 'LR_B', 'CONTROL_FAILED_LR')):
            unit = make_unit(model, 'BASE', c['control']['fold'], 'control')
            run_unit(X, y, ids, folds, dest, unit, ctx, retry_interrupted, retry_reason, execute_fn, resource_fn)
            new = read_pred(dest/model/'BASE'/f'fold_{c["control"]["fold"]}_pred.csv.gz')
            ref = read_reference(local_root, c['references'][ref_name], ids, y, folds)
            cmp = tie_aware_compare(new, ref[ref.fold == c['control']['fold']], c['control']['probability_atol'])
            require(cmp['aligned'], f'{model} control: ID / y_true alignment with the reference differs (a data defect, not a '
                                    f'numeric control failure): stop and report')
            cmp['status'] = 'PASS' if cmp['status'] == 'PASS' else bad_status
            led = read_ledger(dest)['folds'][unit['key']]
            cmp.update(reference=ref_name, reference_sha256=c['references'][ref_name]['sha256'], fold=c['control']['fold'],
                       artifact_sha256=led['artifact_sha256'], attempts=led['attempts'])
            result['lgbm' if model == 'LGBMN' else 'lr'] = cmp
            statuses[model] = cmp['status']
            if model == 'LGBMN' and cmp['status'] != 'PASS':
                result['lr'] = dict(status='NOT_RUN')
                break
        ledger = read_ledger(dest)
        result.update(attempts=sum(e['attempts'] for e in ledger['folds'].values()),
                      retries=sum(max(e['attempts'] - 1, 0) for e in ledger['folds'].values()),
                      lr_b_refit_required=bool(result.get('lr', {}).get('status') == 'CONTROL_FAILED_LR'),
                      control_fits_completed=sum(e['status'] == 'complete' for e in ledger['folds'].values()),
                      guard=guard_summary())
        jatomic(result_path, result)
        shutil.copy2(result_path, dest/'backup'/'control_check.json')
        log(dest, f'control check: LGBM-N {statuses.get("LGBMN")}, LR-C {statuses.get("LR", "NOT_RUN")}')
        require(statuses.get('LGBMN') == 'PASS', 'CONTROL_FAILED_LGBM: do not seal (artifacts kept in control_check/)')
        return result
    finally:
        lock.unlink(missing_ok=True)


# ======================================================================================== preflight / prepare / verify
def value_ranges(arr, names):
    return {n: dict(min=float(arr[:, i].min()), max=float(arr[:, i].max()), mean=float(arr[:, i].mean())) for i, n in enumerate(names)}


def preflight_checks(data, c, local_root):
    """No classifier is fitted. Derived blocks of every fold; matrices and checks of Fold 0 for both models."""
    X, y, ids, folds, contract = load_data(data, c)
    refs = verify_references(c, local_root, ids, y, folds)
    area_zero = int((X['area_percentage'] == 0).sum())
    require(area_zero == 0, f'area_percentage == 0 in {area_zero} rows: no pre-registered rule, revise the design before sealing')
    geo_info = {}
    for g in GEO:
        v = X[g]
        geo_info[g] = dict(unique=int(v.nunique()), all_digit_strings=bool(v.str.fullmatch(r'\d+').all()),
                           max_length=int(v.str.len().max()), distinct_after_single_column_key=int(v.nunique()))
    sep_clash = {col: bool(X[col].astype(str).str.contains(F.SEP, regex=False).any())
                 for pair in F.COMBO_SOURCES.values() for col in pair}
    require(not any(sep_clash.values()), 'Combination separator occurs in a source column')
    raw = F.raw_numeric(X)
    fold_info, matrices = {}, {}
    for f in range(5):
        a, b = np.flatnonzero(folds != f), np.flatnonzero(folds == f)
        X_tr, X_va = X.iloc[a], X.iloc[b]
        rec = dict(train_rows=int(len(a)), valid_rows=int(len(b)), mixed_995=F.mixed_995_summary(X_tr, X_va),
                   arm_relations=F.check_arm_relations(X_tr, X_va), arms={})
        for arm in ARMS:
            arm_rec = {}
            for kind in F.KINDS:
                bundle = F.DerivedBundle(arm, kind).fit(X_tr)
                arm_rec[kind] = dict(reference=F.check_against_reference(bundle, X_tr, X_va), summary=bundle.summary())
            bundle = F.DerivedBundle(arm, 'lgbm').fit(X_tr)
            bt, st = bundle.transform(X_tr, return_stats=True)
            bv, sv = bundle.transform(X_va, return_stats=True)
            k = F.n_numeric(arm)
            codes = bundle.combo_codes(X_va)
            arm_rec.update(train_ranges=value_ranges(bt[:, :k], F.numeric_names(arm)), valid_ranges=value_ranges(bv[:, :k], F.numeric_names(arm)),
                           region_stats_train=st, region_stats_valid=sv,
                           valid_unseen_combination_rows={n: int(np.isnan(codes[:, i]).sum()) for i, n in enumerate(F.COMBO_COLS)})
            rec['arms'][arm] = arm_rec
        fold_info[str(f)] = rec
        if f == c['control']['fold']:
            for model in MODELS:
                base = base_fit(model, X_tr, X_va, c)
                base2 = base_fit(model, X_tr, X_va, c)
                require(prefix_equal(model, base['mt'], base2['mt']) and prefix_equal(model, base['mv'], base2['mv']),
                        'Base preprocessing is not reproducible')
                for arm in ARMS:
                    built = build_inputs(model, arm, X_tr, X_va, c, base=base, base_check=base2)
                    m = built['mt']
                    matrices[f'{model}/{arm}'] = dict(
                        train_shape=list(m.shape), valid_shape=list(built['mv'].shape), dtype=str(m.dtype),
                        container='csr' if sparse.issparse(m) else 'dense', nnz=int(m.nnz) if sparse.issparse(m) else None,
                        base_columns=int(base['mt'].shape[1]), derived_columns=int(m.shape[1] - base['mt'].shape[1]),
                        checks=built['checks'])
                matrices[f'{model}/BASE'] = dict(train_shape=list(base['mt'].shape), valid_shape=list(base['mv'].shape),
                                                 dtype=str(base['mt'].dtype),
                                                 container='csr' if sparse.issparse(base['mt']) else 'dense')
    require(matrices['LGBMN/S0']['train_shape'][1] == 49 and matrices['LGBMN/S1']['train_shape'][1] == 52
            and matrices['LGBMN/S2']['train_shape'][1] == 52, 'LGBM-N input widths differ from 49 / 52 / 52')
    return dict(study_id=STUDY_ID, status='PASS', classifiers_fitted=0, holdout_accessed=False, data_contract=contract,
                references=refs, raw_derived_numeric_ranges_all_rows=value_ranges(raw, F.NUMERIC_COLS),
                area_percentage_zero_rows=area_zero, geo_keys=geo_info, separator_clash=sep_clash,
                folds=fold_info, fold0_matrices=matrices, guard=guard_summary())


def preflight(data, out_root, local_root=None):
    install_audit_hook()
    root = guard_path(out_root)
    dest = root/'preflight'
    require(not dest.exists(), 'Preflight destination exists')
    c = config()
    check_plan_sha(c)
    bad = source_mismatches(SOURCE)
    require(not bad, f'Sealed sources must equal their HEAD blobs (commit first): {bad}')
    require_environment(c)
    head = git_head()
    local_root = Path(local_root) if local_root else Path(data).resolve().parents[1]
    result = preflight_checks(data, c, local_root)
    result.update(code_commit=head, plan_sha256=c['plan_sha256'], config_sha256=sha256_file(HERE/'config.json'),
                  source_sha256=sources(), environment=environment(), environment_extra=environment_extra(),
                  created_at_utc=now())
    jwrite(dest/'preflight.json', result)
    return result


def prepare(data, out_root, local_root=None, code_commit=None):
    install_audit_hook()
    root = guard_path(out_root)
    run = root/'run'
    c = config()
    check_plan_sha(c)                                        # first: nothing else is trusted if the plan differs
    require(not run.exists(), 'Study output exists')
    head = git_head()
    require(code_commit in (None, head), 'code_commit differs from HEAD')
    bad = source_mismatches(SOURCE)
    require(not bad, f'Sealed sources must equal their HEAD blobs: {bad}')
    require_environment(c)
    pf = read_preflight(root, c, head)
    cc_path = root/'control_check'/'control_check.json'
    require(cc_path.is_file(), 'Run control-check first')
    cc = json.loads(cc_path.read_text(encoding='utf-8'))
    control_dir_check(root/'control_check', cc)
    require(cc['code_commit'] == head and cc['plan_sha256'] == c['plan_sha256'] and cc['source_sha256'] == sources()
            and cc['config_sha256'] == sha256_file(HERE/'config.json'), 'Control check is bound to another commit / config / plan')
    require(cc['lgbm']['status'] == 'PASS', 'CONTROL_FAILED_LGBM: do not seal')
    require(cc['lr']['status'] in ('PASS', 'CONTROL_FAILED_LR'), 'LR control incomplete')
    refit = cc['lr']['status'] == 'CONTROL_FAILED_LR'
    local_root = Path(local_root) if local_root else Path(data).resolve().parents[1]
    X, y, ids, folds, contract = load_data(data, c)
    refs = verify_references(c, local_root, ids, y, folds)
    require(refs == pf['references'] and contract == pf['data_contract'], 'References / data differ from the preflight record')
    anchor = root
    while not anchor.exists():
        anchor = anchor.parent
    require(shutil.disk_usage(anchor).free/2**30 >= c['resources']['min_free_disk_gb_at_prepare'], 'Insufficient disk')
    cap = c['fit_plan']['attempt_cap_with_refit' if refit else 'attempt_cap']
    targets = [key(*t) for t in study_targets(refit)]
    require(cc['attempts'] < cap and cc['retries'] <= c['fit_plan']['max_infrastructure_retries'], 'Control attempts leave no budget')
    (run/'backup').mkdir(parents=True)
    design = dict(study_id=STUDY_ID, config=c, config_sha256=sha256_file(HERE/'config.json'), plan_sha256=c['plan_sha256'],
                  code_commit=head, source_sha256=sources(), environment=environment(), environment_extra=environment_extra(),
                  data_contract=contract,
                  data_dir=str(Path(data).resolve()), local_root=str(local_root.resolve()), references=refs,
                  preflight_sha256=sha256_file(root/'preflight'/'preflight.json'),
                  control=dict(path='control_check/control_check.json', sha256=sha256_file(cc_path), code_commit=cc['code_commit'],
                               lgbm_status=cc['lgbm']['status'], lr_status=cc['lr']['status'], attempts=cc['attempts'],
                               retries=cc['retries'],
                               ledger_sha256=sha256_file(root/'control_check'/'ledger.json')),
                  lr_b_refit=refit, study_targets=targets, planned_completed_fits=len(targets),
                  completed_fits_including_control=len(targets) + cc['control_fits_completed'],
                  attempt_cap=cap, classifiers_fitted=0, holdout_accessed=False, created_at_utc=now())
    jwrite(run/'design.json', design)
    jwrite(run/'design_seal.json', dict(design_sha256=sha256_file(run/'design.json'), code_commit=head, classifiers_fitted=0,
                                        created_at_utc=now()))
    jatomic(run/'ledger.json', dict(study_id=STUDY_ID, scope='study', folds={}))
    for name in ('design.json', 'design_seal.json'):
        shutil.copy2(run/name, run/'backup'/name)
    log(run, f'prepared and sealed design {sha256_file(run/"design.json")} commit={head} lr_b_refit={refit}')
    return design


def verify(out_root, data=None):
    """Sealed design + frozen sources / environment / data / control bindings. Returns (X, y, ids, folds, design)."""
    install_audit_hook()
    root = guard_path(out_root)
    run = root/'run'
    seal = json.loads((run/'design_seal.json').read_text(encoding='utf-8'))
    require(sha256_file(run/'design.json') == seal['design_sha256'], 'Design seal changed')
    design = json.loads((run/'design.json').read_text(encoding='utf-8'))
    c = config()
    check_plan_sha(c)
    require(design['config'] == c and design['plan_sha256'] == c['plan_sha256'], 'Frozen config / plan changed')
    require(design['source_sha256'] == sources() and design['environment'] == environment(), 'Frozen source / environment changed')
    require(design['environment_extra'] == environment_extra(), 'Recorded psutil version changed')
    require(sha256_file(root/'preflight'/'preflight.json') == design['preflight_sha256'], 'Preflight record changed after sealing')
    ctl = design['control']
    require(sha256_file(root/ctl['path']) == ctl['sha256'] and sha256_file(root/'control_check'/'ledger.json') == ctl['ledger_sha256'],
            'Control check changed after sealing')
    control_dir_check(root/'control_check', None)
    require(json.loads((root/ctl['path']).read_text(encoding='utf-8'))['code_commit'] == design['code_commit'] == ctl['code_commit'],
            'Control check / design code commit differ')
    X, y, ids, folds, contract = load_data(data or design['data_dir'], c)
    require(contract == design['data_contract'], 'Data contract changed')
    require(verify_references(c, design['local_root'], ids, y, folds) == design['references'], 'Reference OOF changed')
    return X, y, ids, folds, design


def study_ctx(design):
    c = design['config']
    return dict(config=c, code_commit=design['code_commit'], allowed_keys=set(design['study_targets']),
                attempt_cap=design['attempt_cap'], attempts_before=design['control']['attempts'],
                retries_before=design['control']['retries'], max_retries=c['fit_plan']['max_infrastructure_retries'])


def study_units(design):
    units = []
    for k in design['study_targets']:
        model, arm, fold = k.split('/')
        units.append(make_unit(model, arm, int(fold), 'refit' if arm == 'B_REFIT' else 'study'))
    return units


def fit_all(out_root, data=None, retry_interrupted=False, retry_reason=None, execute_fn=None, resource_fn=None, verify_fn=None):
    """Single executor: exclusive RUNNING.lock, fixed order, completed folds are hash-checked and skipped."""
    install_audit_hook()
    root = guard_path(out_root)
    run = root/'run'
    require((run/'design_seal.json').is_file(), 'Run prepare first')
    lock = run/'RUNNING.lock'
    try:
        with lock.open('x', encoding='utf-8') as handle:
            handle.write(f'pid={os.getpid()} started={now()}\n')
    except FileExistsError:
        raise ValueError(f'Another run holds {lock}; remove only after the control tower confirms no executor is alive')
    try:
        X, y, ids, folds, design = (verify_fn or verify)(root, data)
        ledger = read_ledger(run)
        bad = [k for k, e in ledger['folds'].items() if e['status'] == 'failed']
        require(not bad, f'Failed unit(s) present: {bad[:3]} (stop and report)')
        before = pd.util.hash_pandas_object(X, index=True).to_numpy()
        ctx = study_ctx(design)
        t0 = time.perf_counter()
        log(run, f'fit-all start; targets={len(design["study_targets"])} attempt_cap={design["attempt_cap"]} lr_b_refit={design["lr_b_refit"]}')
        for unit in study_units(design):
            run_unit(X, y, ids, folds, run, unit, ctx, retry_interrupted, retry_reason, execute_fn, resource_fn)
        np.testing.assert_array_equal(before, pd.util.hash_pandas_object(X, index=True).to_numpy())
        log(run, f'fit-all finished wall={time.perf_counter() - t0:.1f}s')
    finally:
        try:
            jwrite(run/f'guard_log_{time.strftime("%Y%m%dT%H%M%S")}_{os.getpid()}.json', guard_summary())   # never overwritten
        finally:
            lock.unlink(missing_ok=True)


def status(out_root):
    root = Path(out_root)
    run = root/'run'
    if not (run/'ledger.json').exists():
        return dict(prepared=False, preflight=(root/'preflight'/'preflight.json').exists(),
                    control_check=(root/'control_check'/'control_check.json').exists())
    design = json.loads((run/'design.json').read_text(encoding='utf-8'))
    folds = read_ledger(run)['folds']
    per_arm = {}
    for k in design['study_targets']:
        model, arm, _ = k.split('/')
        slot = per_arm.setdefault(f'{model}/{arm}', dict(complete=0, of=0))
        slot['of'] += 1
        slot['complete'] += int(folds.get(k, {}).get('status') == 'complete')
    attempts = sum(e['attempts'] for e in folds.values())
    return dict(prepared=True, complete=sum(e['status'] == 'complete' for e in folds.values()), total=len(design['study_targets']),
                attempts_study=attempts, attempts_including_control=attempts + design['control']['attempts'],
                attempt_cap=design['attempt_cap'], lr_b_refit=design['lr_b_refit'],
                running=[k for k, e in folds.items() if e['status'] == 'running'],
                interrupted=[k for k, e in folds.items() if e['status'] == 'interrupted'],
                failed=[k for k, e in folds.items() if e['status'] == 'failed'],
                lock_present=(run/'RUNNING.lock').exists(), per_arm=per_arm, analysis_written=(run/'analysis.json').exists())


# ======================================================================================== analysis (aggregates only)
def pooled_counts(cm):
    return dict(error31=int(cm[2][0]), error32=int(cm[2][1]), error23=int(cm[1][2]), error12=int(cm[0][1]),
                error21=int(cm[1][0]), error13=int(cm[0][2]), correct3=int(cm[2][2]))


def summarize_oof(oof, support3):
    y, p, fo = oof.y_true.to_numpy(), oof.y_pred.to_numpy(), oof.fold.to_numpy()
    scores = [score(y[fo == f], p[fo == f]) for f in range(5)]
    vals = [s['macro_f1'] for s in scores]
    pooled = score(y, p)
    cm = pooled['confusion']
    require(int(sum(cm[2])) == support3, 'Grade-3 support differs from the pinned value')
    out = dict(fold_macro_f1=vals, cv_mean=float(np.mean(vals)), cv_sd=float(np.std(vals, ddof=1)),
               fold_accuracy=[s['accuracy'] for s in scores],
               fold_per_class=[dict(precision=s['precision'], recall=s['recall'], f1=s['f1'], support=s['support']) for s in scores],
               pooled=pooled, recall3=pooled['recall'][2],
               rate31=pooled['error31']/support3, rate32=pooled['error32']/support3, support3=support3)
    out.update(pooled_counts(cm))
    return out


def turnover(y, c, r):
    """Same-row transitions between prediction sets c (reference) and r (candidate); geo_ablation/audit.py definition."""
    y, c, r = np.asarray(y), np.asarray(c), np.asarray(r)
    out = dict(corrected_predictions=int(((c != y) & (r == y)).sum()), new_errors=int(((c == y) & (r != y)).sum()),
               prediction_agreement=float(np.mean(c == r)))
    for label, mc, mr in (('error31', (y == 3) & (c == 1), (y == 3) & (r == 1)), ('error32', (y == 3) & (c == 2), (y == 3) & (r == 2))):
        out[label + '_removed'] = int((mc & ~mr).sum())
        out[label + '_new'] = int((~mc & mr).sum())
        out[label + '_common'] = int((mc & mr).sum())
    return out


def difference(a_sum, b_sum, a_oof, b_oof):
    """a - b between two OOF sets over the same folds: Fold deltas, mean +/- SD, wins, pooled count deltas."""
    d = [x - z for x, z in zip(a_sum['fold_macro_f1'], b_sum['fold_macro_f1'])]
    row = dict(fold_delta=d, mean_delta=float(np.mean(d)), delta_sd=float(np.std(d, ddof=1)), wins=int(sum(x > 0 for x in d)),
               worst_fold_delta=float(min(d)), delta_recall3=a_sum['recall3'] - b_sum['recall3'],
               delta_correct3=a_sum['correct3'] - b_sum['correct3'], delta_error31=a_sum['error31'] - b_sum['error31'],
               delta_rate31=a_sum['rate31'] - b_sum['rate31'], delta_error32=a_sum['error32'] - b_sum['error32'],
               delta_error23=a_sum['error23'] - b_sum['error23'], delta_error12=a_sum['error12'] - b_sum['error12'],
               delta_error21=a_sum['error21'] - b_sum['error21'], delta_error13=a_sum['error13'] - b_sum['error13'],
               delta_pooled_macro_f1=a_sum['pooled']['macro_f1'] - b_sum['pooled']['macro_f1'],
               error31_a=a_sum['error31'], error31_b=b_sum['error31'])
    row['turnover'] = turnover(a_oof.y_true.to_numpy(), b_oof.y_pred.to_numpy(), a_oof.y_pred.to_numpy())
    return row


def safety_flag(n_treat, n_base, rule):
    """T30 / T31 double flag: relative +10% AND absolute +30 more 3->1 errors than the reference."""
    return bool(n_treat > n_base*(1 + rule['safety_3to1_relative']) and (n_treat - n_base) >= rule['safety_3to1_absolute'])


def judge(row, rule, caveat):
    """Plan 8.2: five conditions, direction tag, CONVERGENCE_CAVEAT; ALL_MET is a candidate flag only."""
    eps, thr = rule['boundary_epsilon'], rule['mean_delta_min']
    conds = {1: bool(row['mean_delta'] >= thr - eps), 2: bool(row['wins'] >= rule['wins_min']),
             3: bool(row['delta_correct3'] >= 0), 4: bool(row['delta_error31'] <= rule['to1_delta_max']),
             5: bool(row['worst_fold_delta'] > rule['worst_fold_floor'])}
    unmet = [k for k, v in conds.items() if not v]
    tag = ('HIGHER' if row['mean_delta'] >= thr - eps else
           'LOWER' if row['mean_delta'] <= -thr + eps and row['wins'] <= rule['lower_tag_max_wins'] else 'NEUTRAL')
    all_met = not unmet
    return dict(conditions=conds, unmet_conditions=unmet, verdict='ALL_MET' if all_met else 'NOT_ALL_MET', direction_tag=tag,
                convergence_caveat=bool(caveat), candidate=bool(all_met and not caveat))


def family_summary(judged):
    """Plan 8.2 multiple-comparison notes. The model-dependence and partial-arm notes use CANDIDATES (ALL_MET without
    CONVERGENCE_CAVEAT); the raw ALL_MET list is kept beside them."""
    all_met = {m: [a for a in ARMS if judged[f'{m}/{a}']['verdict'] == 'ALL_MET'] for m in MODELS}
    cand = {m: [a for a in ARMS if judged[f'{m}/{a}']['candidate']] for m in MODELS}
    return dict(all_met_by_model=all_met, candidates_by_model=cand,
                model_dependent=bool(sum(bool(v) for v in cand.values()) == 1),
                partial_arms_note=('같은 모델에서 일부 팔만 후보(ALL_MET, 수렴 주의 없음)이다: 세 팔 표를 나란히 보고 충족한 팔만 부각하지 않는다.'
                                   if any(0 < len(v) < len(ARMS) for v in cand.values()) else None),
                basis='model_dependent / partial_arms_note use candidates; ALL_MET with CONVERGENCE_CAVEAT is not a candidate')


def subgroup_summary(oof, mask, rule):
    sub = oof[mask]
    n = int(len(sub))
    s = dict(rows=n, correct=int((sub.y_true == sub.y_pred).sum()) if n else 0,
             class_support=[int((sub.y_true == v).sum()) for v in (1, 2, 3)] if n else [0, 0, 0])
    if n >= rule['subgroup_min_rows']:
        sc = score(sub.y_true, sub.y_pred)
        s.update(macro_f1=sc['macro_f1'], accuracy=sc['accuracy'], recall3=sc['recall'][2], error31=sc['error31'], error32=sc['error32'])
    else:
        s.update(macro_f1=None, note='fewer than 200 rows: counts only')
    return s


def assemble(root, model, arm, ids, y, folds):
    """Fold predictions -> OOF (exactly one row per development row, ID / fold / label alignment checked)."""
    root = Path(root)
    ledger = read_ledger(root)
    for f in range(5):
        e = ledger['folds'].get(key(model, arm, f))
        require(e and e['status'] == 'complete', f'{key(model, arm, f)} incomplete')
        check_saved(root, e)
    d = root/model/arm
    frame = pd.concat([read_pred(d/f'fold_{f}_pred.csv.gz') for f in range(5)])
    require(frame[ID].is_unique and len(frame) == len(ids), 'OOF coverage mismatch')
    frame = frame.set_index(ID).loc[ids.to_numpy()].reset_index()
    np.testing.assert_array_equal(frame.fold.to_numpy(), folds)
    np.testing.assert_array_equal(frame.y_true.to_numpy(), np.asarray(y))
    require(np.array_equal(frame[PROBS].to_numpy().argmax(1) + 1, frame.y_pred.to_numpy()), 'argmax != y_pred')
    path = d/'oof.csv.gz'
    if path.exists():
        np.testing.assert_array_equal(read_pred(path)[PROBS].to_numpy(), frame[PROBS].to_numpy())
    else:
        write_pred(path, frame)
        np.testing.assert_array_equal(read_pred(path)[PROBS].to_numpy(), frame[PROBS].to_numpy())
    back = root/'backup'/model/arm/'oof.csv.gz'
    back.parent.mkdir(parents=True, exist_ok=True)
    if not back.exists():
        shutil.copy2(path, back)
    require(sha256_file(back) == sha256_file(path), 'OOF backup mismatch')
    return frame


def fold_records(root, model, arm):
    return [json.loads((Path(root)/model/arm/f'fold_{f}.json').read_text(encoding='utf-8')) for f in range(5)]


def lr_convergence(records):
    return dict(n_iter=[r['info']['n_iter'] for r in records], converged=[bool(r['info']['converged']) for r in records],
                convergence_warning=[bool(r['info']['convergence_warning']) for r in records])


def analyze(root, X, y, ids, folds, design):
    root = Path(root)
    run = root/'run'
    c = design['config']
    rule = c['rule']
    ledger = read_ledger(run)
    targets = design['study_targets']
    complete = sum(ledger['folds'].get(k, {}).get('status') == 'complete' for k in targets)
    attempts = sum(e['attempts'] for e in ledger['folds'].values())
    total_attempts = attempts + design['control']['attempts']
    require(complete == len(targets) == design['planned_completed_fits'], 'Ledger incomplete; not finalizing')
    require(total_attempts <= design['attempt_cap'], 'Attempt cap exceeded')
    support3 = c['grade3_support']
    local_root = design['local_root']
    refs = c['references']
    oofs = {('LGBMN', 'B'): read_reference(local_root, refs['LGBMN_B'], ids, y, folds)}
    if design['lr_b_refit']:
        oofs[('LR', 'B')] = assemble(run, 'LR', 'B_REFIT', ids, y, folds)
    else:
        oofs[('LR', 'B')] = read_reference(local_root, refs['LR_B'], ids, y, folds)
    cited = read_reference(local_root, refs['LR_ORIGINAL'], ids, y, folds)
    for m in MODELS:
        for a in ARMS:
            oofs[(m, a)] = assemble(run, m, a, ids, y, folds)
    summ = {k: summarize_oof(v, support3) for k, v in oofs.items()}
    cited_summary = summarize_oof(cited, support3)
    # --- convergence (LR): treatment folds and B
    conv = {a: lr_convergence(fold_records(run, 'LR', a)) for a in ARMS}
    if design['lr_b_refit']:
        conv['B'] = lr_convergence(fold_records(run, 'LR', 'B_REFIT'))
        conv['B']['source'] = 'LR_B_REFIT (new fits)'
    else:
        conv['B'] = dict(n_iter=design['references']['LR_B']['n_iter'], converged=design['references']['LR_B']['converged'],
                         source='cited from the pinned fold_metrics.json of the local refit')
    b_conv_ok = all(conv['B']['converged'])
    judged, descriptive, family_tags = {}, {}, {}
    for m in MODELS:
        for a in ARMS:
            row = difference(summ[(m, a)], summ[(m, 'B')], oofs[(m, a)], oofs[(m, 'B')])
            caveat = m == 'LR' and not (b_conv_ok and all(conv[a]['converged']))
            row.update(judge(row, rule, caveat))
            row['safety_3to1_double_flag'] = safety_flag(row['error31_a'], row['error31_b'], rule)
            row['n_iter'] = conv[a]['n_iter'] if m == 'LR' else None
            row['b_n_iter'] = conv['B']['n_iter'] if m == 'LR' else None
            judged[f'{m}/{a}'] = row
        descriptive_rows = {}
        for left, right in (('S1', 'S0'), ('S2', 'S0'), ('S2', 'S1')):
            d = difference(summ[(m, left)], summ[(m, right)], oofs[(m, left)], oofs[(m, right)])
            d['safety_3to1_double_flag'] = safety_flag(d['error31_a'], d['error31_b'], rule)
            descriptive_rows[f'{left}-{right}'] = d
        descriptive[m] = descriptive_rows
    family_tags.update(family_summary(judged))
    # --- diagnostics from the fold records and the data (no row-level output)
    diag = dict(folds={}, convergence=conv)
    for m in MODELS:
        for a in ARMS:
            recs = fold_records(run, m, a)
            diag['folds'][f'{m}/{a}'] = dict(
                fit_seconds=[r['info']['fit_seconds'] for r in recs], train_macro_f1=[r['train_macro_f1'] for r in recs],
                input_columns=[r['input_columns'] for r in recs], valid_unseen_combinations=[r['valid_unseen_combinations'] for r in recs],
                region_stats_valid=[r['region_stats_valid'] for r in recs],
                vocabulary_sizes=[r['derived']['vocabulary_sizes'] for r in recs],
                trees=[r['info'].get('trees') for r in recs] if m == 'LGBMN' else None,
                gain_share_derived=[r.get('gain_share_derived') for r in recs] if m == 'LGBMN' else None,
                derived_gain=[r.get('derived_gain') for r in recs] if m == 'LGBMN' else None,
                categorical_effective_parameters=recs[0]['info'].get('categorical_effective_parameters') if m == 'LGBMN' else None,
                importance_note='diagnostic only; not a selection basis' if m == 'LGBMN' else None)
    age = X['age'].to_numpy()
    geo1 = X[GEO[0]].to_numpy()
    profiles, subgroups = {}, {}
    masks = {'age995': age == F.AGE_SENTINEL, 'age_not995': age != F.AGE_SENTINEL}
    masks.update({f'geo1_{v}': geo1 == str(v) for v in (26, 20, 13)})
    mixed_by_fold, self_inclusion = {}, {}
    has995 = {g: np.zeros(len(X), dtype=bool) for g in GEO}
    for f in range(5):
        a_, b_ = np.flatnonzero(folds != f), np.flatnonzero(folds == f)
        X_tr, X_va = X.iloc[a_], X.iloc[b_]
        mixed_by_fold[str(f)] = F.mixed_995_summary(X_tr, X_va)
        for g, m_ in F.has_995_region_mask(X_tr, X_va).items():
            has995[g][b_] = m_
        for arm in ('S1', 'S2'):
            self_inclusion.setdefault(arm, {})[str(f)] = F.self_inclusion_diagnostics(X_tr, X_va, arm)
    for g in GEO:
        masks[f'{g}_region_has_995_in_training'] = has995[g]
        masks[f'{g}_region_without_995_in_training'] = ~has995[g]
    for k, oof in oofs.items():
        for name, mask in masks.items():
            profiles.setdefault(f'{k[0]}/{k[1]}', {})[name] = subgroup_summary(oof, mask, rule)
    diag.update(subgroups=profiles, mixed_995_by_fold=mixed_by_fold, self_inclusion_by_fold=self_inclusion)
    result = dict(
        study_id=STUDY_ID, design_sha256=sha256_file(run/'design.json'), code_commit=design['code_commit'], holdout_accessed=False,
        scheme='stratified', guide_line=c['guide_line'], fits=complete, attempts_study=attempts, attempts_including_control=total_attempts,
        retries=total_attempts - design['completed_fits_including_control'],
        lr_b_refit=design['lr_b_refit'], rule=rule, control=json.loads((root/design['control']['path']).read_text(encoding='utf-8')),
        baseline={f'{m}': {k2: v for k2, v in summ[(m, 'B')].items()} for m in MODELS},
        cited_lr_original=cited_summary, arms={f'{m}/{a}': summ[(m, a)] for m in MODELS for a in ARMS},
        judged=judged, descriptive=descriptive, family=family_tags, diagnostics=diag, guard=guard_summary(),
        notes=['층화 seed 42 5-Fold 하나의 기술 판정이며 p-value·신뢰구간·인과·일반화 주장이 아니다.',
               'ALL_MET은 후보 표시일 뿐이다. 별도 확인 평가(T29) 전에는 채택·승격·"개선 확인" 문구를 쓰지 않는다.',
               '변수별 기여는 분해하지 않았다(묶음 평가). S0 대비 S1/S2 차이는 지역 평균 연령 3열 전체의 효과이며 지역 지문·자기 포함 평균이 섞일 수 있다.',
               'LGBM-N과 LR-C의 범주 표현이 달라 모델 간 순위를 비교하지 않는다.',
               'CONVERGENCE_CAVEAT가 붙은 LR 비교의 ALL_MET은 후보로도 쓰지 않는다.',
               'B는 같은 Fold를 이미 많이 쓴 선택 통계의 기준선이며 독립 성능이 아니다.'],
        finished_at_utc=now())
    return py(result)


def no_row_level_data(obj, limit=64):
    """Aggregate files must not contain long arrays (a row-level leak guard for analysis.json)."""
    if isinstance(obj, dict):
        return all(no_row_level_data(v, limit) for v in obj.values())
    if isinstance(obj, list):
        return len(obj) <= limit and all(no_row_level_data(v, limit) for v in obj)
    return True


def finalize(out_root, data=None, verify_fn=None):
    install_audit_hook()
    root = guard_path(out_root)
    run = root/'run'
    require(not (run/'analysis.json').exists(), 'Analysis exists')
    require(not (run/'RUNNING.lock').exists(), 'An executor lock is present')
    X, y, ids, folds, design = (verify_fn or verify)(root, data)
    result = analyze(root, X, y, ids, folds, design)
    require(no_row_level_data(result), 'analysis.json would contain row-level arrays')
    jatomic(run/'analysis.json', result)
    back = run/'backup'/'analysis.json'
    require(not back.exists(), 'Backup already exists')
    shutil.copy2(run/'analysis.json', back)
    log(run, 'analysis written')
    return result


# ======================================================================================== CLI
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('action', choices=['preflight', 'control-check', 'prepare', 'fit-all', 'status', 'finalize'])
    p.add_argument('--data', help='earthquake/data directory (preflight / control-check / prepare; later steps default to design.json)')
    p.add_argument('--out', '--output', dest='out', required=True, help='private root, e.g. local_reproduction/T34_20261016_derived_bundle')
    p.add_argument('--local-root', help='folder that contains local_reproduction/ (default: two levels above --data)')
    p.add_argument('--code-commit')
    p.add_argument('--retry-interrupted', action='store_true')
    p.add_argument('--retry-reason')
    a = p.parse_args(argv)
    install_audit_hook()
    require(not a.retry_interrupted or a.retry_reason, '--retry-interrupted requires --retry-reason')
    retry = dict(retry_interrupted=a.retry_interrupted, retry_reason=a.retry_reason if a.retry_interrupted else None)
    if a.action == 'preflight':
        require(a.data, '--data required')
        preflight(a.data, a.out, a.local_root)
    elif a.action == 'control-check':
        require(a.data, '--data required')
        control_check(a.data, a.out, a.local_root, **retry)
    elif a.action == 'prepare':
        require(a.data, '--data required')
        prepare(a.data, a.out, a.local_root, a.code_commit)
    elif a.action == 'fit-all':
        fit_all(a.out, a.data, **retry)
    elif a.action == 'finalize':
        finalize(a.out, a.data)
    else:
        print(json.dumps(status(a.out), indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
