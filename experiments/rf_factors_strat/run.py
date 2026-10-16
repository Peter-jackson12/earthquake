"""T30 RF 요인 분리 (층화 주 평가). 기준은 감사된 T14 Stage A 층화 OOF를 SHA 고정 재사용한다.

신규 RF fit 41개(상한 45 시도): C0 통제 1 + N1/N2 잡음 통제 10 + NO_ROS/DROP_GEO23_TE/AGE_C/DROP_TE_ALL/CW_NO_ROS/ONEHOT
6팔 x 5 Fold = 30. 홀드아웃 접근 없음. 학습은 fit-all에서만 수행하며 preflight/prepare는 분류기를 학습하지 않는다.
T14/T16의 봉인 소스는 import만 하고 수정하지 않는다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import joblib
import numpy as np
import pandas as pd
from scipy import sparse

from preprocess import GEO, ID, sha256_file
from experiments.rf_factors import run as t16
from experiments.te_ros_rf_repro import run as t14

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
STUDY_ID = 'T30_rf_factors_stratified_v1'
SCHEME = 'strat'
BASE = t16.BASE
C0 = 'C0_BASE_REFIT'
ARM_ORDER = [C0, 'N1_RS1', 'N2_RS2', 'NO_ROS', 'DROP_GEO23_TE', 'AGE_C', 'DROP_TE_ALL', 'CW_NO_ROS', 'ONEHOT']
CONTROLS = [C0, 'N1_RS1', 'N2_RS2']
TIER1 = ['ONEHOT', 'NO_ROS', 'DROP_GEO23_TE', 'AGE_C']
TIER3 = ['DROP_TE_ALL', 'CW_NO_ROS']
JUDGED = ['NO_ROS', 'DROP_GEO23_TE', 'AGE_C', 'DROP_TE_ALL', 'ONEHOT']  # contrasted against the baseline
ARM_FOLDS = {a: ([0] if a == C0 else list(range(5))) for a in ARM_ORDER}
N_TARGETS = 41
MAX_ATTEMPTS = 45
TE_ALL = [f'{g}_te{k}' for g in GEO for k in (0, 2)]
ARM_SPEC = {
    C0: dict(matrix_arm=BASE, drop_all_te=False, contrast='control: identical pipeline refit on fold 0'),
    'N1_RS1': dict(matrix_arm=BASE, drop_all_te=False, contrast='RF random_state 0 -> 1'),
    'N2_RS2': dict(matrix_arm=BASE, drop_all_te=False, contrast='RF random_state 0 -> 2'),
    'NO_ROS': dict(matrix_arm='NO_ROS', drop_all_te=False, contrast='ROS off'),
    'DROP_GEO23_TE': dict(matrix_arm='DROP_GEO23_TE', drop_all_te=False, contrast='geo2/geo3 TE 4 columns removed'),
    'AGE_C': dict(matrix_arm='AGE_C', drop_all_te=False, contrast='age 995 -> train median + age_is_995'),
    'DROP_TE_ALL': dict(matrix_arm=BASE, drop_all_te=True,
                        contrast='all 6 geo TE columns removed (raw geo + counts kept)'),
    'CW_NO_ROS': dict(matrix_arm='NO_ROS', drop_all_te=False, contrast='vs NO_ROS: class_weight balanced only'),
    'ONEHOT': dict(matrix_arm='ONEHOT', drop_all_te=False, contrast='geo TE 6 columns replaced by geo one-hot')}
ARM_RF = {C0: {}, 'N1_RS1': dict(random_state=1), 'N2_RS2': dict(random_state=2), 'NO_ROS': {}, 'DROP_GEO23_TE': {},
          'AGE_C': {}, 'DROP_TE_ALL': {}, 'CW_NO_ROS': dict(class_weight='balanced'), 'ONEHOT': {}}
# Only the sealed sources that the T14/T16 chain does not already list; T14/T16 sources are blob-checked as well.
SOURCE = list(dict.fromkeys(t16.SOURCE + [
    'experiments/rf_factors_strat/__init__.py', 'experiments/rf_factors_strat/run.py',
    'experiments/rf_factors_strat/config.json', 'tests/test_rf_factors_strat.py',
    'docs/T30_RF_FACTORS_STRATIFIED_PLAN.md']))
require, score, write = t14.require, t14.score, t14.write
now, log, event = t14.now, t14.log, t14.event
atomic_write, read_pred, write_pred = t14.atomic_write, t14.read_pred, t14.write_pred
read_ledger, check_saved = t14.read_ledger, t14.check_saved
PROBS = t14.PROBS


def key(scheme, arm, fold):
    return f'{scheme}/{arm}/{fold}'


def targets():
    """Fixed execution order: C0 -> N1/N2 -> cheap arms -> ONEHOT (arm-major)."""
    return [(SCHEME, a, f) for a in ARM_ORDER for f in ARM_FOLDS[a]]


def sources():
    return {p: sha256_file(ROOT/p) for p in SOURCE}


def merged_rf(c, arm):
    return {**c['rf_params'], **c['arm_rf_params'][arm]}


# ---------------------------------------------------------------------------------------------------- config / seal
def config():
    c = json.loads((HERE/'config.json').read_text(encoding='utf-8'))
    require(c['study_id'] == STUDY_ID and c['scheme'] == SCHEME and c['n_folds'] == 5 and c['arm_order'] == ARM_ORDER
            and c['arm_folds'] == {a: ARM_FOLDS[a] for a in ARM_ORDER}, 'Unplanned design')
    require(len(targets()) == N_TARGETS == c['planned_completed_fits'] and c['max_new_study_fits'] == MAX_ATTEMPTS
            and c['max_infrastructure_retries'] == 4, 'Unplanned budget')
    require(len({key(*t) for t in targets()}) == N_TARGETS, 'Duplicate targets')
    base14, base16 = t14.config(), t16.config()
    require(c['baseline_design_sha256'] == base16['baseline_design_sha256']
            == '965f8741dd48ea0b277a9a512b551fd05ac1f365bd9a4d11fd9769688db09a74', 'Baseline design pin changed')
    require(c['baseline_oof']['sha256'] == '69a74d410da08bf06c9da35211e3fe6f1e424bd4afbc72c4de2d2d3c21c09969',
            'Baseline OOF pin changed')
    require(c['split_manifest_sha256'] == '7673dbeb136e9a916c67a3b19928b3fbc621ae627a5667a677a1ca6eae838721',
            'Split manifest pin changed')
    require(c['rf_params'] == base14['rf_params'] and c['te'] == base14['te'], 'T14 settings changed')
    require(c['arm_changes'] == base16['arm_changes'], 'T16 factor contrasts changed')
    require(c['arm_spec'] == ARM_SPEC and c['arm_rf_params'] == ARM_RF, 'Arm definitions changed')
    require(set(c['expected_model_columns']) == set(ARM_ORDER) - {'ONEHOT'}, 'Expected column table changed')
    require(not c['holdout_access'] and not c['tuning'] and not c['early_stopping']
            and not c['threshold_or_decision_rule_change'] and c['outer_parallel_jobs'] == 1
            and c['automatic_retries'] == 0 and c['baseline_refits_control_only'] == 1, 'Unplanned execution')
    require(c['environment'] == t14.environment(), 'Pinned environment changed')
    return c


def source_mismatches(paths, root=ROOT):
    """Git-blob check of ONLY the sealed files: the working-tree blob of each must equal the blob at HEAD.
    Unrelated untracked/modified files elsewhere in the worktree are ignored on purpose."""
    root = Path(root)
    listing = subprocess.run(['git', 'ls-tree', '-r', 'HEAD', '--', *paths], cwd=root, capture_output=True,
                             text=True, check=True).stdout
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
                bad.append((p, 'working-tree blob differs from HEAD'))
    return bad


# ---------------------------------------------------------------------------------------------------- baseline reuse
def baseline_design(c):
    return t16.baseline_design(c)


def baseline_reference(c, ids, y, folds):
    ref = c['baseline_oof']
    t14.require_not_holdout(ref['path'])
    return t14.reference(ref['path'], ref['sha256'], ids, y, folds)


def baseline_check(c, ids, y, folds):
    """Plan section 2, items 1-3 plus the recorded T14 numbers (pooled confusion and fold F1)."""
    baseline_design(c)
    ref = baseline_reference(c, ids, y, folds)
    exp, tol = c['baseline_expected'], c['baseline_expected']['tolerance_6dp']
    f1 = [score(ref.y_true[ref.fold == f], ref.y_pred[ref.fold == f])['macro_f1'] for f in range(5)]
    require(all(abs(a-b) <= tol for a, b in zip(f1, exp['fold_macro_f1_6dp'])), 'Baseline fold F1 differs from record')
    require(abs(float(np.mean(f1))-exp['mean_macro_f1_6dp']) <= tol, 'Baseline mean F1 differs from record')
    pooled = score(ref.y_true, ref.y_pred)
    require(pooled['confusion'] == exp['pooled_confusion'] and pooled['error31'] == exp['error31']
            and pooled['error32'] == exp['error32'], 'Baseline pooled confusion differs from record')
    return dict(sha256=c['baseline_oof']['sha256'], rows=len(ref), fold_macro_f1=f1, pooled_confusion=pooled['confusion'])


def input_bundle(data, c):
    """Development table + stored stratified folds only (`load_development`); no holdout path is touched."""
    X, y, ids, folds, contract = t14.load_inputs(data)
    require(contract['files']['split_manifest.csv'] == c['split_manifest_sha256'], 'Split manifest differs from pin')
    folds = np.asarray(folds)
    require(set(np.unique(folds)) == set(range(5)), 'Expected five folds')
    require(all(int((folds == f).sum()) == c['valid_rows_per_fold'] for f in range(5)), 'Unexpected fold sizes')
    return X, y, ids, folds, contract


# ---------------------------------------------------------------------------------------------------- matrices
def drop_te_all(built):
    """DROP_TE_ALL: remove the 6 geo TE columns from Ao/Bo/Xos and keep the stored state in sync, so reload
    (`apply_state` rebuilds from `state['columns']`) reproduces exactly the filtered validation matrix."""
    require(all(n in built['selected'] for n in TE_ALL), 'TE columns missing')
    keep = [n for n in built['selected'] if n not in TE_ALL]
    for name in ('Ao', 'Bo', 'Xos'):
        require(isinstance(built[name], pd.DataFrame), 'DROP_TE_ALL needs dense frames')
    state = dict(built['state'])
    state['columns'] = [n for n in state['columns'] if n not in TE_ALL]
    state['selected'] = [n for n in state['selected'] if n not in TE_ALL]
    state['drop'] = list(state['drop']) + TE_ALL
    require(state['columns'] == keep == state['selected'], 'State/column filter out of sync')
    return dict(built, Ao=built['Ao'].drop(columns=TE_ALL), Bo=built['Bo'].drop(columns=TE_ALL),
                Xos=built['Xos'].drop(columns=TE_ALL), selected=keep, state=state)


def build_fold_matrices(F_train, X_train, y_train, F_valid, scheme, c, arm):
    """Validation labels are not an argument. Delegates to the sealed T16 builder (arm 'BASE' or a T16 arm)."""
    require(arm in ARM_ORDER and scheme == SCHEME, 'Unknown arm/scheme')
    spec = c['arm_spec'][arm]
    built = t16.build_fold_matrices(F_train, X_train, y_train, F_valid, scheme, c, spec['matrix_arm'])
    return drop_te_all(built) if spec['drop_all_te'] else built


apply_state, matrix_equal = t16.apply_state, t16.matrix_equal


# ---------------------------------------------------------------------------------------------------- preflight/prepare
def preflight_checks(data, c):
    """No classifier is fitted. Fold 0 matrices of every arm; baseline bit-identity with T14; expected columns."""
    X, y, ids, folds, contract = input_bundle(data, c)
    refs = baseline_check(c, ids, y, folds)
    F = t14.original_frame(X)
    a, b = np.flatnonzero(folds != 0), np.flatnonzero(folds == 0)
    base14 = t14.build_fold_matrices(F.iloc[a], X.iloc[a], y.iloc[a], F.iloc[b], SCHEME, t14.config())
    rows, names = {}, {}
    for arm in ARM_ORDER:
        built = build_fold_matrices(F.iloc[a], X.iloc[a], y.iloc[a], F.iloc[b], SCHEME, c, arm)
        require(matrix_equal(built['Bo'], apply_state(built['state'], F.iloc[b])), f'Encoder reload differs: {arm}')
        names[arm] = list(built['selected'])
        if c['arm_spec'][arm]['matrix_arm'] == BASE and not c['arm_spec'][arm]['drop_all_te']:
            require(matrix_equal(base14['Ao'], built['Ao']) and matrix_equal(base14['Bo'], built['Bo'])
                    and matrix_equal(base14['Xos'], built['Xos']) and np.array_equal(base14['yos'], built['yos']),
                    f'Baseline reconstruction differs from T14: {arm}')
        want = c['expected_model_columns'].get(arm)
        require(want is None or len(built['selected']) == want, f'Column count differs: {arm} {len(built["selected"])}')
        rows[arm] = dict(columns=len(built['selected']), train_rows=len(a), valid_rows=len(b),
                         fit_rows=len(built['yos']), fit_class_counts=np.bincount(built['yos'], minlength=3).tolist(),
                         sparse=bool(sparse.issparse(built['Xos'])), rf_params=merged_rf(c, arm))
        del built
    base = set(names[C0])
    require(set(names['NO_ROS']) == base and set(names['CW_NO_ROS']) == base and rows['NO_ROS']['fit_rows'] == len(a)
            and rows['CW_NO_ROS']['fit_rows'] == len(a), 'NO_ROS contrasts changed more than ROS')
    require(base-set(names['DROP_GEO23_TE']) == {f'{g}_te{k}' for g in GEO[1:] for k in (0, 2)}
            and set(names['DROP_GEO23_TE']) <= base, 'DROP_GEO23_TE differs by other than 4 TE columns')
    require(base-set(names['DROP_TE_ALL']) == set(TE_ALL) and set(names['DROP_TE_ALL']) <= base,
            'DROP_TE_ALL differs by other than 6 TE columns')
    require(set(names['AGE_C'])-base == {'age_is_995'} and base-set(names['AGE_C']) == set(),
            'AGE_C differs by other than age_is_995')
    require(base-set(names['ONEHOT']) == set(TE_ALL), 'ONEHOT must remove exactly the 6 TE columns')
    require(all(n.startswith('onehot__') for n in set(names['ONEHOT'])-base), 'ONEHOT adds non-onehot columns')
    return dict(study_id=STUDY_ID, status='PASS', classifiers_fitted=0, holdout_accessed=False,
                data_contract=contract, baseline=refs, fold0_matrices=rows, source_sha256=sources(),
                config_sha256=sha256_file(HERE/'config.json'), environment=t14.environment(), created_at_utc=now())


def preflight(data, destination):
    dest = Path(destination)
    t14.require_not_holdout(dest)
    require(not dest.exists(), 'Preflight destination exists')
    result = preflight_checks(data, config())
    write(dest/'preflight.json', result)
    return result


def prepare(data, output, code_commit=None):
    out = Path(output)
    t14.require_not_holdout(out)
    require(not out.exists(), 'Study output exists')
    c = config()
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    require(len(head) == 40 and all(ch in '0123456789abcdef' for ch in head), 'Full HEAD SHA required')
    require(code_commit is None or code_commit == head, 'code_commit differs from HEAD')
    bad = source_mismatches(SOURCE)
    require(not bad, f'Sealed sources must equal their HEAD blobs: {bad}')
    require(sha256_file(ROOT/c['plan']) == c['plan_sha256'], 'Plan document changed')
    anchor = out.parent
    while not anchor.exists():
        anchor = anchor.parent
    require(shutil.disk_usage(anchor).free/2**30 >= c['min_free_disk_gb_at_prepare'], 'Insufficient disk')
    pf = preflight_checks(data, c)
    out.mkdir(parents=True)
    (out/'backup').mkdir()
    write(out/'preflight.json', pf)
    design = dict(study_id=STUDY_ID, config=c, code_commit=head, source_sha256=sources(),
                  environment=t14.environment(), data_contract=pf['data_contract'], baseline=pf['baseline'],
                  data_dir=str(Path(data).resolve()), preflight_sha256=sha256_file(out/'preflight.json'),
                  study_targets=[key(*t) for t in targets()], max_new_study_fits=c['max_new_study_fits'],
                  classifiers_fitted=0, holdout_accessed=False, created_at_utc=now())
    write(out/'design.json', design)
    write(out/'design_seal.json', dict(design_sha256=sha256_file(out/'design.json'), code_commit=head))
    atomic_write(out/'ledger.json', dict(study_id=STUDY_ID, folds={}))
    for name in ['design.json', 'design_seal.json', 'preflight.json']:
        shutil.copy2(out/name, out/'backup'/name)
    log(out, f'prepared and sealed design {sha256_file(out/"design.json")} commit={head}')
    return design


def verify(data, output):
    out = Path(output)
    seal = json.loads((out/'design_seal.json').read_text(encoding='utf-8'))
    require(sha256_file(out/'design.json') == seal['design_sha256'], 'Design seal changed')
    design = json.loads((out/'design.json').read_text(encoding='utf-8'))
    require(design['config'] == config() and design['source_sha256'] == sources()
            and design['environment'] == t14.environment(), 'Sealed configuration/source/environment changed')
    X, y, ids, folds, contract = input_bundle(data, design['config'])
    require(contract == design['data_contract'], 'Data contract changed')
    require(baseline_check(design['config'], ids, y, folds) == design['baseline'], 'Baseline changed')
    return X, y, ids, folds, design


# ---------------------------------------------------------------------------------------------------- C0 gate
def gate_path(out):
    return Path(out)/'c0_gate.json'


def gate_passed(out, design):
    p = gate_path(out)
    if not p.exists():
        return False
    g = json.loads(p.read_text(encoding='utf-8'))
    entry = read_ledger(out)['folds'].get(key(SCHEME, C0, 0))
    return bool(g.get('status') == 'PASS' and entry and entry['status'] == 'complete'
                and g['c0_pred_sha256'] == entry['artifact_sha256']['fold_0_pred.csv.gz']
                and g['baseline_oof_sha256'] == design['config']['baseline_oof']['sha256'])


def ensure_gate(out, ids, y, folds, design):
    """Compare the C0 refit with the T14 baseline OOF of fold 0. FAIL stops the whole study (no retry)."""
    out = Path(out)
    p = gate_path(out)
    if p.exists():
        g = json.loads(p.read_text(encoding='utf-8'))
        require(g['status'] == 'PASS', 'C0 gate FAILED earlier; study must stay stopped (design revision needed)')
        require(gate_passed(out, design), 'Recorded C0 gate does not match the ledger')
        return g
    c = design['config']
    gc = c['c0_gate']
    entry = read_ledger(out)['folds'][key(SCHEME, C0, 0)]
    require(entry['status'] == 'complete', 'C0 not complete')
    check_saved(out, entry)
    pred_path = out/SCHEME/C0/'fold_0_pred.csv.gz'
    new = read_pred(pred_path).sort_values(ID).reset_index(drop=True)  # float_precision='round_trip'
    ref = baseline_reference(c, ids, y, folds)
    ref0 = ref[ref.fold == gc['fold']].sort_values(ID).reset_index(drop=True)
    require(len(new) == len(ref0) and np.array_equal(new[ID].to_numpy(), ref0[ID].to_numpy())
            and np.array_equal(new.y_true.to_numpy(), ref0.y_true.to_numpy()), 'C0/baseline alignment differs')
    diff = np.abs(new[PROBS].to_numpy() - ref0[PROBS].to_numpy())
    row_diff = diff.max(axis=1)
    mismatch = new.y_pred.to_numpy() != ref0.y_pred.to_numpy()
    top = np.sort(new[PROBS].to_numpy(), axis=1)
    tie = top[:, 2] == top[:, 1]
    allowed = mismatch & (row_diff == 0) & tie
    max_diff = float(diff.max())
    ok = bool(max_diff <= gc['max_abs_prob_diff'] and int(mismatch.sum()) == int(allowed.sum()))
    result = dict(status='PASS' if ok else 'FAIL', rows=len(new), max_abs_prob_diff=max_diff,
                  bit_identical=bool(max_diff == 0.0), label_mismatches=int(mismatch.sum()),
                  label_mismatches_allowed_exact_ties=int(allowed.sum()), top2_tie_rows=int(tie.sum()),
                  c0_macro_f1=score(new.y_true, new.y_pred)['macro_f1'],
                  baseline_fold0_macro_f1=score(ref0.y_true, ref0.y_pred)['macro_f1'],
                  c0_pred_sha256=entry['artifact_sha256']['fold_0_pred.csv.gz'],
                  baseline_oof_sha256=c['baseline_oof']['sha256'], rule=gc, created_at_utc=now())
    atomic_write(p, result)
    shutil.copy2(p, out/'backup'/'c0_gate.json')
    event(out, fold=key(SCHEME, C0, 0), action='c0_gate', status=result['status'], max_abs_prob_diff=max_diff)
    log(out, f'C0 gate {result["status"]}: max|dp|={max_diff:.3g}, label mismatches={result["label_mismatches"]}')
    require(ok, f'C0 gate FAILED (max|dp|={max_diff}, label mismatches={int(mismatch.sum())}); all later fits are blocked')
    return result


# ---------------------------------------------------------------------------------------------------- one fold
def fit_fold(X, F, y, ids, folds, out, scheme, arm, f, design, retry_interrupted=False, retry_reason=None):
    out = Path(out)
    k = key(scheme, arm, f)
    require(scheme == SCHEME and k in design['study_targets'], 'Unregistered fit')
    ledger = read_ledger(out)
    entry = ledger['folds'].get(k)
    if entry and entry['status'] == 'complete':
        check_saved(out, entry)
        return entry, False
    c = design['config']
    require(arm == C0 or gate_passed(out, design), 'C0 gate has not passed; no other fit may run')
    require(shutil.disk_usage(out).free/2**30 >= c['min_free_disk_gb_per_fold'], 'Insufficient disk')
    prior = entry['attempts'] if entry else 0
    if entry:
        require(retry_interrupted and retry_reason and retry_reason.strip(),
                'Interrupted fit: explicit --retry-interrupted and infrastructure --retry-reason required')
        require(prior == 1, 'Only one infrastructure retry per fold is allowed')
        require(sum(max(e['attempts']-1, 0) for e in ledger['folds'].values()) < c['max_infrastructure_retries'],
                'Infrastructure retry reserve exhausted')
    require(sum(e['attempts'] for e in ledger['folds'].values()) < design['max_new_study_fits'], 'Budget exhausted')
    entry = dict(dir=f'{scheme}/{arm}', scheme=scheme, arm=arm, fold=f, status='running', attempts=prior+1,
                 started_at_utc=now())
    ledger['folds'][k] = entry
    atomic_write(out/'ledger.json', ledger)
    d = out/entry['dir']
    d.mkdir(parents=True, exist_ok=True)
    for location in [d, out/'backup'/entry['dir']]:
        stale = list(location.glob(f'fold_{f}[._]*')) if location.exists() else []
        if not prior:
            require(not stale, 'Untracked fold files exist')
        for path in stale:
            target = path.with_name(path.name+f'.incomplete_attempt{prior}')
            require(not target.exists(), 'Preserved incomplete artifact exists')
            path.rename(target)
    event(out, fold=k, action='fit_start', attempt=prior+1, infrastructure_retry_reason=retry_reason if prior else None)
    a, b = np.flatnonzero(folds != f), np.flatnonzero(folds == f)
    built = build_fold_matrices(F.iloc[a], X.iloc[a], y.iloc[a], F.iloc[b], scheme, c, arm)
    want = c['expected_model_columns'].get(arm)
    require(want is None or len(built['selected']) == want, f'Column count drift: {arm} {len(built["selected"])}')
    rf = merged_rf(c, arm)
    model, seconds, fit_warnings = t14.fit_rf(built['Xos'], built['yos'], dict(c, rf_params=rf))
    require(model.get_params() == t14.RandomForestClassifier(**rf).get_params(), 'Effective RF parameters differ from seal')
    p = t14.predict_proba_deterministic(model, built['Bo'])
    require(p.shape == (len(b), 3) and np.isfinite(p).all(), 'Invalid probabilities')
    pred = p.argmax(1)+1
    model_path, state_path = d/f'fold_{f}_model.joblib', d/f'fold_{f}_encoder_state.joblib'
    joblib.dump(model, model_path, compress=3)
    joblib.dump(built['state'], state_path, compress=3)
    preds = d/f'fold_{f}_pred.csv.gz'
    frame = pd.DataFrame({ID: ids.iloc[b].to_numpy(), 'fold': f, 'y_true': y.iloc[b].to_numpy(), 'y_pred': pred,
                          **{n: p[:, i] for i, n in enumerate(PROBS)}})
    write_pred(preds, frame)
    Bo2 = apply_state(joblib.load(state_path), F.iloc[b])
    require(matrix_equal(built['Bo'], Bo2), 'Reload matrix mismatch')
    p2 = t14.predict_proba_deterministic(joblib.load(model_path), Bo2)
    require(np.array_equal(p, p2), 'Reload probability mismatch')
    np.testing.assert_array_equal(read_pred(preds)[PROBS].to_numpy(), p)
    record = dict(study_id=STUDY_ID, scheme=scheme, arm=arm, fold=f, train_rows=len(a), valid_rows=len(b),
                  fit_rows=len(built['yos']), fit_class_counts=np.bincount(built['yos'], minlength=3).tolist(),
                  feature_names=built['selected'], model_columns=len(built['selected']),
                  feature_names_sha256=hashlib.sha256('\n'.join(built['selected']).encode()).hexdigest(),
                  rf_params=rf, effective_parameters=model.get_params(), metrics=score(y.iloc[b], pred),
                  fit_seconds=seconds, te_seconds=built['te_seconds'], ros_seconds=built['ros_seconds'],
                  warnings=fit_warnings, age_median=built['state']['age'].median_ if built['state']['age'] else None,
                  reload_probability_max_abs_diff=float(np.max(np.abs(p-p2))), csv_round_trip_exact=True,
                  feature_importances=model.feature_importances_.tolist(), finished_at_utc=now())
    write(d/f'fold_{f}.json', record)
    files = [model_path, state_path, preds, d/f'fold_{f}.json']
    shas = {pth.name: sha256_file(pth) for pth in files}
    backup = out/'backup'/entry['dir']
    backup.mkdir(parents=True, exist_ok=True)
    for path in files:
        require(not (backup/path.name).exists(), 'Backup exists')
        shutil.copy2(path, backup/path.name)
    entry.update(status='complete', artifact_sha256=shas, macro_f1=record['metrics']['macro_f1'],
                 fit_seconds=seconds, finished_at_utc=now())
    check_saved(out, entry)
    ledger['folds'][k] = entry
    atomic_write(out/'ledger.json', ledger)
    event(out, fold=k, action='fit_complete', artifact_sha256=shas)
    log(out, f'{k} F1={entry["macro_f1"]:.9f}, fit={seconds:.1f}s; 저장·SHA·재로딩·백업 검증 완료')
    return entry, True


def run_targets(X, y, ids, folds, out, design, retry_interrupted=False, retry_reason=None):
    F = t14.original_frame(X)
    for s, arm, f in targets():
        try:
            fit_fold(X, F, y, ids, folds, out, s, arm, f, design, retry_interrupted, retry_reason)
            if arm == C0:
                ensure_gate(out, ids, y, folds, design)
        except Exception:
            event(out, fold=key(s, arm, f), action='stopped', trace=traceback.format_exc())
            raise


def fit_all(data, output, retry_interrupted=False, retry_reason=None):
    out = Path(output)
    t14.require_not_holdout(out)
    # Exclusive creation prevents two executors entering the same study.
    with (out/'RUNNING.lock').open('x', encoding='utf-8') as handle:
        handle.write(f'pid={os.getpid()} started={now()}\n')
    try:
        X, y, ids, folds, design = verify(data, out)
        before = pd.util.hash_pandas_object(X, index=True).to_numpy()
        t0 = time.perf_counter()
        run_targets(X, y, ids, folds, out, design, retry_interrupted, retry_reason)
        np.testing.assert_array_equal(before, pd.util.hash_pandas_object(X, index=True).to_numpy())
        log(out, f'fit-all finished wall={time.perf_counter()-t0:.1f}s')
    finally:
        (out/'RUNNING.lock').unlink(missing_ok=True)


# ---------------------------------------------------------------------------------------------------- assembly / analysis
def assemble(out, ids, y, folds, scheme, arm):
    ledger = read_ledger(out)
    for f in ARM_FOLDS[arm]:
        entry = ledger['folds'].get(key(scheme, arm, f))
        require(entry and entry['status'] == 'complete', 'Incomplete fold')
        check_saved(out, entry)
    require(len(ARM_FOLDS[arm]) == 5, 'Only five-fold arms can be assembled')
    d = Path(out)/scheme/arm
    frame = pd.concat([read_pred(d/f'fold_{f}_pred.csv.gz') for f in range(5)])
    require(frame[ID].is_unique and len(frame) == len(ids), 'OOF coverage mismatch')
    frame = frame.set_index(ID).loc[ids.to_numpy()].reset_index()
    np.testing.assert_array_equal(frame.fold, folds)
    np.testing.assert_array_equal(frame.y_true, y)
    path = d/'oof.csv.gz'
    if path.exists():
        pd.testing.assert_frame_equal(read_pred(path), frame, check_dtype=False)
    else:
        write_pred(path, frame)
    back = Path(out)/'backup'/scheme/arm/'oof.csv.gz'
    if not back.exists():
        shutil.copy2(path, back)
    require(sha256_file(back) == sha256_file(path), 'OOF backup mismatch')
    return frame


def label_for(deltas, rule):
    """Plan 7.1 label from paired fold deltas (candidate - reference)."""
    eps, thr, k = rule['boundary_epsilon'], rule['mean_delta_min'], rule['same_sign_folds_min']
    mean = float(np.mean(deltas))
    pos, neg = sum(d > 0 for d in deltas), sum(d < 0 for d in deltas)
    if mean >= thr-eps and pos >= k:
        return 'CHANGE_IMPROVES'
    if mean <= -(thr-eps) and neg >= k:
        return 'CHANGE_WORSENS'
    if abs(mean) < thr-eps:
        return 'NEGLIGIBLE'
    return 'MIXED'


def flags_for(recall3_delta, ref_error31, new_error31, rule):
    """Plan 7.2/7.4 flags. The 3->1 reference count is passed in (744 for the baseline, NO_ROS's own for CW_NO_ROS)."""
    eps = rule['boundary_epsilon']
    flags = []
    if recall3_delta <= -(rule['recall3_delta']-eps):
        flags.append('RECALL3_DOWN')
    if recall3_delta >= rule['recall3_delta']-eps:
        flags.append('RECALL3_UP')
    threshold = max(rule['to1_relative']*ref_error31, rule['to1_min_abs_count'])
    diff = new_error31-ref_error31
    if diff >= threshold-eps:
        flags.append('TO1_UP')
    elif diff <= -(threshold-eps):
        flags.append('TO1_DOWN')
    return flags


def importance_blocks(names, importances):
    total = float(sum(importances)) or 1.0
    blocks = dict(raw_geo=0.0, te=0.0, cnt=0.0, onehot_geo=0.0, other=0.0)
    for n, v in zip(names, importances):
        part = ('raw_geo' if n in GEO else 'te' if n.endswith(('_te0', '_te2')) else 'cnt' if n.endswith('_cnt')
                else 'onehot_geo' if n.startswith('onehot__') else 'other')
        blocks[part] += float(v)/total
    return blocks


def contrast(new, old, y, rule, label=True):
    """Paired contrast `new - old` over the same five folds (plan section 6)."""
    comp = t14.compare_arms(new, old, y)
    d = comp['fold_delta']
    new_pool, old_pool = t14.pooled_scores([new]), t14.pooled_scores([old])
    recall_delta = new_pool['grade3_recall']-old_pool['grade3_recall']
    result = dict(fold_macro_f1=comp['rf_fold_macro_f1'], reference_fold_macro_f1=comp['other_fold_macro_f1'],
                  fold_macro_f1_mean=float(np.mean(comp['rf_fold_macro_f1'])),
                  fold_macro_f1_sd=float(np.std(comp['rf_fold_macro_f1'], ddof=1)),
                  fold_delta=d, mean_delta=float(np.mean(d)), delta_sd=float(np.std(d, ddof=1)),
                  positive_folds=int(sum(x > 0 for x in d)), negative_folds=int(sum(x < 0 for x in d)),
                  pooled=new_pool, reference_pooled=old_pool,
                  pooled_macro_f1=new_pool['pooled']['macro_f1'], reference_pooled_macro_f1=old_pool['pooled']['macro_f1'],
                  grade3_recall_delta=recall_delta, error31_delta=new_pool['error31']-old_pool['error31'],
                  error32_delta=new_pool['error32']-old_pool['error32'], turnover=comp['turnover'],
                  reference_error31=old_pool['error31'])
    if label:
        result['label'] = label_for(d, rule)
        result['flags'] = flags_for(recall_delta, old_pool['error31'], new_pool['error31'], rule)
        result['tradeoff'] = bool(result['label'] == 'CHANGE_IMPROVES'
                                  and ({'RECALL3_DOWN', 'TO1_UP'} & set(result['flags'])))
    return result


def arm_stats(out, arm, c):
    recs = [json.loads((Path(out)/SCHEME/arm/f'fold_{f}.json').read_text(encoding='utf-8')) for f in ARM_FOLDS[arm]]
    return dict(model_columns=[r['model_columns'] for r in recs], fit_rows=[r['fit_rows'] for r in recs],
                fit_class_counts=[r['fit_class_counts'] for r in recs], fit_seconds=[r['fit_seconds'] for r in recs],
                te_seconds=[r['te_seconds'] for r in recs], ros_seconds=[r['ros_seconds'] for r in recs],
                rf_params=recs[0]['rf_params'], feature_names_sha256=[r['feature_names_sha256'] for r in recs],
                importance_blocks=[importance_blocks(r['feature_names'], r['feature_importances']) for r in recs],
                importance_note='RF impurity importance; regional block sums; no causal interpretation')


def sign(x):
    return 0 if x == 0 else (1 if x > 0 else -1)


def analyze(out, ids, y, folds, design):
    out = Path(out)
    c = design['config']
    rule = c['rule']
    ledger = read_ledger(out)
    complete = sum(ledger['folds'].get(k, {}).get('status') == 'complete' for k in design['study_targets'])
    attempts = sum(e['attempts'] for e in ledger['folds'].values())
    require(complete == len(design['study_targets']) == N_TARGETS == c['planned_completed_fits'],
            'Ledger incomplete; not finalizing')
    require(attempts <= design['max_new_study_fits'] == MAX_ATTEMPTS, 'Attempt cap exceeded')
    require(gate_passed(out, design), 'C0 gate has not passed')
    yv = np.asarray(y)
    ref = baseline_reference(c, ids, y, folds)
    frames = {arm: assemble(out, ids, y, folds, SCHEME, arm) for arm in ARM_ORDER if arm not in (C0,)}
    result = dict(study_id=STUDY_ID, design_sha256=sha256_file(out/'design.json'), code_commit=design['code_commit'],
                  holdout_accessed=False, scheme=SCHEME, guide_line=c['guide_line'],
                  fits=complete, attempts=attempts, retries=attempts-complete, rule=rule,
                  baseline=dict(arm=BASE, oof_sha256=c['baseline_oof']['sha256'], recheck=design['baseline'],
                                pooled=t14.pooled_scores([ref]), error31=t14.pooled_scores([ref])['error31']),
                  c0_gate=json.loads(gate_path(out).read_text(encoding='utf-8')), per_arm={}, notes=[
                      '층화 seed 42 5-Fold 하나의 기술 판정이다. 최고 팔 선정·팔 간 우열·효과 합산·승격은 사전등록 밖.',
                      '기준 0.700427은 56개 설정 중 최고값(선택 통계)이다. 승자 편향 보정을 하지 않는다.',
                      'Fold는 독립 표본이 아니며 p-value·신뢰구간·인과 효과를 주장하지 않는다.',
                      'N1/N2는 RF random_state만 바꾸므로 잡음의 하한이다(ROS·내부 TE Fold 시드는 고정).'])
    for arm in JUDGED:
        con = contrast(frames[arm], ref, yv, rule)
        con.update(contrast_with=BASE, spec=c['arm_spec'][arm], stats=arm_stats(out, arm, c),
                   oof_sha256=sha256_file(out/SCHEME/arm/'oof.csv.gz'))
        result['per_arm'][arm] = con
    # noise controls: signed deltas, no label, no correction of the arm deltas
    noise = {}
    for arm in ('N1_RS1', 'N2_RS2'):
        con = contrast(frames[arm], ref, yv, rule, label=False)
        con.update(contrast_with=BASE, spec=c['arm_spec'][arm], stats=arm_stats(out, arm, c),
                   oof_sha256=sha256_file(out/SCHEME/arm/'oof.csv.gz'))
        noise[arm] = con
    max_mean = max(abs(v['mean_delta']) for v in noise.values())
    max_fold = max(abs(d) for v in noise.values() for d in v['fold_delta'])
    triggered = bool(max_mean >= rule['noise_annotation']['max_abs_mean_delta']-rule['boundary_epsilon']
                     or max_fold >= rule['noise_annotation']['max_abs_fold_delta']-rule['boundary_epsilon'])
    result['noise'] = dict(arms=noise, signed_mean_delta={a: v['mean_delta'] for a, v in noise.items()},
                           signed_fold_delta={a: v['fold_delta'] for a, v in noise.items()},
                           max_abs_mean_delta=max_mean, max_abs_fold_delta=max_fold, annotation_triggered=triggered,
                           annotation=('주석: RF 난수 재추첨 잡음이 임계와 같은 규모' if triggered else None),
                           lower_bound_note='RF random_state만 변경한 잡음의 하한. 승자 편향 보정에 쓰지 않는다.')
    if triggered:
        for v in result['per_arm'].values():
            v['annotation'] = result['noise']['annotation']
    # CW_NO_ROS: judged against NO_ROS (one factor); baseline contrast is reference only
    cw = contrast(frames['CW_NO_ROS'], frames['NO_ROS'], yv, rule)
    cw.update(contrast_with='NO_ROS', spec=c['arm_spec']['CW_NO_ROS'], stats=arm_stats(out, 'CW_NO_ROS', c),
              oof_sha256=sha256_file(out/SCHEME/'CW_NO_ROS'/'oof.csv.gz'),
              to1_reference_note='3->1 reference count is the NO_ROS count, not the baseline 744')
    cw_ref = contrast(frames['CW_NO_ROS'], ref, yv, rule)
    cw_ref['reference_only'] = True
    cw['vs_baseline_reference_only'] = cw_ref
    if triggered:
        cw['annotation'] = result['noise']['annotation']
    result['cw_no_ros'] = cw
    # descriptive geo2 comparison (citation only)
    geo2 = c['reference_tables']['T16_geo2_group_20_folds']
    result['geo2_direction'] = {a: dict(strat_mean_delta=result['per_arm'][a]['mean_delta'], geo2_mean_delta=geo2[a]['mean_delta'],
                                         same_sign=bool(sign(result['per_arm'][a]['mean_delta']) == sign(geo2[a]['mean_delta'])))
                                for a in TIER1}
    result['reference_tables'] = c['reference_tables']
    result['to1_reference_counts'] = dict(baseline=result['baseline']['error31'],
                                          NO_ROS=t14.pooled_scores([frames['NO_ROS']])['error31'])
    result['finished_at_utc'] = now()
    return result


def finalize(data, output):
    out = Path(output)
    require(not (out/'analysis.json').exists(), 'Analysis exists')
    X, y, ids, folds, design = verify(data, out)
    result = analyze(out, ids, y, folds, design)
    write(out/'analysis.json', result)
    back = out/'backup'/'analysis.json'
    require(not back.exists(), 'Backup already exists')
    shutil.copy2(out/'analysis.json', back)
    log(out, 'analysis written')
    return result


def status(output):
    out = Path(output)
    ledger = read_ledger(out)
    folds = ledger['folds']
    per_arm = {a: dict(complete=sum(folds.get(key(SCHEME, a, f), {}).get('status') == 'complete' for f in ARM_FOLDS[a]),
                       of=len(ARM_FOLDS[a])) for a in ARM_ORDER}
    gate = gate_path(out)
    return dict(complete=sum(e['status'] == 'complete' for e in folds.values()), total=N_TARGETS,
                attempts=sum(e['attempts'] for e in folds.values()), attempt_cap=MAX_ATTEMPTS,
                running=[k for k, e in folds.items() if e['status'] == 'running'],
                c0_gate=json.loads(gate.read_text(encoding='utf-8'))['status'] if gate.exists() else None,
                lock_present=(out/'RUNNING.lock').exists(), per_arm=per_arm,
                analysis_written=(out/'analysis.json').exists())


def resolve_data(args):
    if args.data:
        return args.data
    design = json.loads((Path(args.out)/'design.json').read_text(encoding='utf-8'))
    return design['data_dir']


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['preflight', 'prepare', 'fit-all', 'status', 'finalize'])
    p.add_argument('--data', help='earthquake/data directory (prepare/preflight; later steps default to design.json)')
    p.add_argument('--out', '--output', dest='out', required=True)
    p.add_argument('--code-commit')
    p.add_argument('--retry-interrupted', action='store_true')
    p.add_argument('--retry-reason')
    a = p.parse_args(argv)
    if a.action == 'preflight':
        require(a.data, '--data required')
        preflight(a.data, a.out)
    elif a.action == 'prepare':
        require(a.data, '--data required')
        prepare(a.data, a.out, a.code_commit)
    elif a.action == 'fit-all':
        fit_all(resolve_data(a), a.out, a.retry_interrupted, a.retry_reason)
    elif a.action == 'finalize':
        finalize(resolve_data(a), a.out)
    else:
        print(json.dumps(status(a.out), indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
