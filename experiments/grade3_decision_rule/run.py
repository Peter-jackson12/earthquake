"""T15: grade-3 decision rule (one scalar logit offset b on grade 3) with leak-free nested tuning.

Factor under test: ONLY the decision rule  pred = argmax_k(p_k * w_k), w = (1, 1, exp(b)),  b from a fixed grid,
applied identically to LR_C, LGBM_O and LGBM_TE_G and compared with each arm's OWN unadjusted argmax.

Leak-free protocol. For every outer fold f of a saved split, b is fitted only from inner-CV out-of-fold probabilities
computed INSIDE the outer-training rows (group-aware inner folds on Group schemes). The outer-validation rows, their
labels, and the OOF models of the other outer folds are never used for fitting b. The outer-validation probabilities
come from the already saved (audited) outer models/OOFs, reused read-only after SHA-256 verification; they are only
re-weighted by exp(b) at prediction time. No holdout (earthquake/data/final_holdout) is ever read.
"""
from __future__ import annotations
import argparse
import hashlib
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
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from preprocess import GEO, ID, sha256_file  # noqa: E402
from experiments.geo_group_cv import run as t5  # noqa: E402
from experiments.geo_group_cv.run import PROBS, groups, load, require, score, write  # noqa: E402
from experiments.geo_hier_te import run as t10  # noqa: E402
from experiments.geo_te_group_inner import run as t11  # noqa: E402
from experiments.geo2_te_confirm import run as t13  # noqa: E402

HERE = Path(__file__).resolve().parent
STUDY_ID = 'T15_grade3_decision_rule_v1'
ARMS = ['LR_C', 'LGBM_O', 'LGBM_TE_G']
N_TARGETS = 450
FORBIDDEN = 'final_holdout'
TE_LEVEL = {'group2': 'geo2', 'group1': 'geo1'}
SOURCE = list(dict.fromkeys(t13.SOURCE + [
    'experiments/geo2_te_confirm/run.py', 'experiments/geo2_te_confirm/config.json',
    'experiments/grade3_decision_rule/__init__.py', 'experiments/grade3_decision_rule/run.py',
    'experiments/grade3_decision_rule/config.json', 'tests/test_grade3_decision_rule.py',
    'docs/T15_GRADE3_DECISION_RULE_PLAN.md']))
now, environment, log, event = t10.now, t10.environment, t10.log, t10.event
atomic_write, read_pred, write_pred = t10.atomic_write, t10.read_pred, t10.write_pred
read_ledger, check_saved = t10.read_ledger, t10.check_saved


# ---------------------------------------------------------------- guard / config
def guard(path):
    """Refuse any path that points into the final holdout."""
    require(FORBIDDEN not in Path(path).as_posix().lower(), f'Holdout access refused: {path}')
    return Path(path)


def config(path=HERE/'config.json'):
    c = json.loads(Path(path).read_text(encoding='utf-8'))
    t = t13.config()
    require(c['study_id'] == STUDY_ID and c['arms'] == ARMS and c['n_folds'] == 5, 'Unplanned study/arms')
    require(c['lgbm_params'] == t['lgbm_params'] and c['lr_params'] == t['lr_params'], 'Model params differ from T13/T11')
    require(c['geo_columns'] == t['geo_columns'] and c['text_categorical_columns'] == t['text_categorical_columns'],
            'Categorical columns changed')
    te = c['te']
    require(te['smoothing_m'] == t['te']['smoothing_m'] == 20.0 and te['inner_splits'] == 5 and te['inner_seed'] == 42,
            'TE design differs from T11/T13')
    r = c['rule']
    require(r['grid'][0] == 0.0 and r['grid'] == sorted(r['grid']) and len(r['grid']) == 21 and r['parameters'] == 1
            and not r['negative_b'] and all(g >= 0 for g in r['grid']), 'Rule grid changed')
    i = c['inner']
    require(i['n_splits'] == 5 and i['seed'] == 42 and i['shuffle'] and i['min_class_support'] == 30, 'Inner CV changed')
    require(list(c['schemes']) == c['primary_schemes'] + c['secondary_schemes'], 'Scheme list changed')
    require(c['max_new_study_fits'] == N_TARGETS == len(c['schemes']) * len(ARMS) * 5 * i['n_splits']
            and c['outer_model_refits'] == 0, 'Unplanned fit plan')
    require(c['automatic_retries'] == 0 and not c['early_stopping'] and c['class_weight'] is None
            and c['sampling'] is None and not c['tuning_of_model_hyperparameters'] and c['outer_parallel_jobs'] == 1
            and c['lr_threadpool_limit'] == 1 and c['lgbm_threadpool_limit'] == 2 and c['stop_on_lr_nonconvergence']
            and not c['holdout_access'] and c['holdout_access_guard'] == FORBIDDEN, 'Unplanned run policy')
    require(c['environment'] == t['environment'], 'Frozen environment differs from T13')
    for name, s in c['schemes'].items():
        require(s['kind'] in ('group2', 'group1', 'strat') and set(s['refs']) == set(ARMS), f'Scheme malformed: {name}')
        require(all(FORBIDDEN not in v['path'].lower() for v in s['refs'].values()) and
                FORBIDDEN not in s['manifest']['path'].lower(), 'Holdout path in config')
    return c


def sources():
    return {p: sha256_file(ROOT/p) for p in SOURCE}


def targets(c=None):
    c = c or config()
    return [(s, a, f, j) for s in c['schemes'] for a in ARMS for f in range(5) for j in range(c['inner']['n_splits'])]


def inner_key(scheme, arm, f, j):
    return f'{scheme}/{arm}/f{f}/i{j}'


def unit_key(scheme, arm, f):
    return f'{scheme}/{arm}/f{f}'


# ---------------------------------------------------------------- the decision rule (pure functions)
def adjust_predict(p, b):
    """argmax_k(p_k * w_k) + 1 with w = (1, 1, exp(b)); b = 0 is the plain argmax."""
    p = np.asarray(p, dtype=np.float64)
    require(p.ndim == 2 and p.shape[1] == 3 and b >= 0, 'Invalid probabilities or negative offset')
    return (p * np.array([1.0, 1.0, np.exp(float(b))])).argmax(1) + 1


def offset_curve(y, p, grid):
    return [score(y, adjust_predict(p, b))['macro_f1'] for b in grid]


def select_offset(y_inner, p_inner, grid):
    """Fit b from INNER out-of-fold rows only. Ties (exact) go to the smallest b."""
    curve = offset_curve(y_inner, p_inner, grid)
    best = max(curve)
    idx = next(i for i, v in enumerate(curve) if v >= best)
    return float(grid[idx]), curve


def decide(y_inner, p_inner, p_outer, grid):
    """Fit b on inner OOF, apply to outer probabilities. Outer labels are NOT an argument (cannot be used)."""
    b, curve = select_offset(y_inner, p_inner, grid)
    return b, adjust_predict(p_outer, b), curve


# ---------------------------------------------------------------- inner folds (a deterministic function of outer-train only)
def inner_folds(X_train, y_train, kind, c):
    """Inner fold id per outer-training row. Group schemes: StratifiedGroupKFold over the outer scheme's own groups."""
    i = c['inner']
    n = len(X_train)
    folds = np.full(n, -1, dtype=int)
    if kind == 'strat':
        it = StratifiedKFold(i['n_splits'], shuffle=True, random_state=i['seed']).split(np.zeros(n), np.asarray(y_train))
    else:
        g, _ = groups(X_train, c['group_columns'][kind])
        it = StratifiedGroupKFold(i['n_splits'], shuffle=True, random_state=i['seed']).split(np.zeros(n), np.asarray(y_train), g)
    for k, (_, b) in enumerate(it):
        require((folds[b] == -1).all(), 'Inner folds overlap')
        folds[b] = k
    require((folds >= 0).all(), 'Inner coverage must equal one')
    return folds


def inner_gate(X_train, y_train, folds, kind, c):
    """Group disjointness and class support in every inner train/validation split. Failure stops the study."""
    y = np.asarray(y_train).astype(int)
    g = groups(X_train, c['group_columns'][kind])[0] if kind != 'strat' else None
    for k in range(c['inner']['n_splits']):
        a, b = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
        require(len(a) > 0 and len(b) > 0, 'Empty inner split')
        if g is not None:
            require(not set(g[a]) & set(g[b]), 'Inner group leakage')
        for rows in (a, b):
            require((np.bincount(y[rows], minlength=4)[1:4] >= c['inner']['min_class_support']).all(),
                    'Inner class support gate failed')


# ---------------------------------------------------------------- scheme data
def scheme_folds(c, name, ids, old):
    s = c['schemes'][name]
    if s['kind'] == 'strat':
        require(sha256_file(guard(s['manifest']['path'])) == s['manifest']['sha256'], f'Manifest changed: {name}')
        return np.asarray(old, dtype=int)
    require(sha256_file(guard(s['manifest']['path'])) == s['manifest']['sha256'], f'Manifest changed: {name}')
    mf = pd.read_csv(s['manifest']['path'])
    np.testing.assert_array_equal(mf[ID].to_numpy(), np.asarray(ids))
    return mf['fold'].to_numpy(dtype=int)


def read_reference(c, name, arm, ids, y, folds):
    """Saved outer OOF (read-only). SHA, alignment, labels and argmax==saved prediction are all verified."""
    ref = c['schemes'][name]['refs'][arm]
    r = t10.reference(guard(ref['path']), ref['sha256'], ids, y, folds)
    require(np.array_equal(adjust_predict(r[PROBS].to_numpy(), 0.0), r.y_pred.to_numpy()),
            f'Saved prediction is not argmax of saved probabilities: {name}/{arm}')
    return r


# ---------------------------------------------------------------- real inner fit (audited arms, unchanged)
def fit_inner_arm(arm, kind, Xa, ya, Xb, c):
    """Fit one arm on inner-train rows and predict inner-validation rows. Returns the checkpoint interface."""
    require(arm in ARMS, 'Unknown arm')
    if arm == 'LR_C' or arm == 'LGBM_O':
        out = t13.fit_one(arm, Xa, ya, Xb, c)
        pre, model, p = out['pre'], out['model'], out['p']
    elif kind == 'strat':
        _, pre, model, p, _, _, _, _, info = t10.fit_arm('LGBM_TE', Xa, ya, Xb, c)
        out = dict(p=p, converged=True, fit_seconds=info['fit_seconds'], pipe=None, info=info)
    else:
        _, pre, model, p, _, _, _, _, info = t11.fit_arm('LGBM_TE_G', TE_LEVEL[kind], Xa, ya, Xb, c)
        out = dict(p=p, converged=True, fit_seconds=info['fit_seconds'], pipe=None, info=info)

    def save(prefix):
        prefix = Path(prefix)
        if arm == 'LR_C':
            joblib.dump(out['pipe'], f'{prefix}_model.joblib', compress=3)
            return [Path(f'{prefix}_model.joblib')]
        model.booster_.save_model(f'{prefix}_model.txt')
        joblib.dump(pre, f'{prefix}_preprocessor.joblib', compress=3)
        return [Path(f'{prefix}_model.txt'), Path(f'{prefix}_preprocessor.joblib')]

    def verify(paths, Xv, p_saved):
        if arm == 'LR_C':
            with threadpool_limits(limits=c['lr_threadpool_limit']):
                p2 = np.asarray(joblib.load(paths[0]).predict_proba(Xv), dtype=np.float64)
            require(np.array_equal(p2, p_saved), 'LR reload mismatch')
            return 0.0
        import lightgbm
        with threadpool_limits(limits=c['lgbm_threadpool_limit']):
            booster = lightgbm.Booster(model_file=str(paths[0]))
            p2 = np.asarray(booster.predict(joblib.load(paths[1]).transform(Xv), num_threads=c['lgbm_params']['n_jobs']),
                            dtype=np.float64)
        return float(np.max(np.abs(p2 - p_saved)))

    info = out.get('info') or {}
    return dict(p=np.asarray(p, dtype=np.float64), converged=bool(out.get('converged', True)),
                fit_seconds=float(out['fit_seconds']), warnings=info.get('warnings', []), save=save, verify=verify)


# ---------------------------------------------------------------- prepare / verify
def prepare(data, output, code_commit):
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
    inner_sha, scheme_info, ref_info = {}, {}, {}
    for name, s in c['schemes'].items():
        folds = scheme_folds(c, name, ids, old)
        if s['kind'] != 'strat':
            scheme_info[name] = t5.validate_assignment(X, y, ids, folds, c['group_columns'][s['kind']], t5.cfg())
        for arm in ARMS:
            r = read_reference(c, name, arm, ids, y, folds)
            ref_info[f'{name}/{arm}'] = dict(sha256=s['refs'][arm]['sha256'], rows=len(r),
                                             grade3_pred_share=float((r.y_pred == 3).mean()))
        for f in range(5):
            a = np.flatnonzero(folds != f)
            inner = inner_folds(X.iloc[a], y.iloc[a], s['kind'], c)
            inner_gate(X.iloc[a], y.iloc[a], inner, s['kind'], c)
            inner_sha[unit_scheme(name, f)] = hashlib.sha256(inner.astype(np.int8).tobytes()).hexdigest()
    output.mkdir(parents=True)
    (output/'backup').mkdir()
    design = dict(study_id=STUDY_ID, config=c, code_commit=code_commit, source_sha256=sources(), data_contract=contract,
                  environment=environment(), scheme_validation=scheme_info, references=ref_info,
                  inner_fold_sha256=inner_sha, study_targets=[inner_key(*t) for t in targets(c)],
                  max_new_study_fits=c['max_new_study_fits'], outer_model_refits=0, automatic_retries=0,
                  holdout_accessed=False, created_at_utc=now(), classifiers_fitted=0)
    write(output/'design.json', design)
    write(output/'design_seal.json', dict(design_sha256=sha256_file(output/'design.json'), code_commit=code_commit,
                                          created_at_utc=now(), classifiers_fitted=0))
    atomic_write(output/'ledger.json', dict(study_id=STUDY_ID, folds={}, decisions={}))
    log(output, f'prepared and sealed design {sha256_file(output/"design.json")} commit={code_commit}')
    return design


def unit_scheme(name, f):
    return f'{name}/f{f}'


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
    folds = {name: scheme_folds(design['config'], name, ids, old) for name in design['config']['schemes']}
    return X, y, ids, folds, design


# ---------------------------------------------------------------- one inner fit with checkpointing
def fit_inner_fold(X, y, ids, folds, out, scheme, arm, f, j, design, fit_fn=fit_inner_arm, inner_cache=None):
    out = Path(out)
    c = design['config']
    k = inner_key(scheme, arm, f, j)
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
    entry = dict(dir=f'{scheme}/{arm}', scheme=scheme, arm=arm, outer_fold=f, inner_fold=j, status='running',
                 attempts=prior + 1, started_at_utc=now())
    ledger['folds'][k] = entry
    atomic_write(out/'ledger.json', ledger)
    if prior:
        log(out, f'{k} previous attempt did not finish; manual resume attempt {prior+1}')
    d = out/entry['dir']
    d.mkdir(parents=True, exist_ok=True)
    stem = f'f{f}_i{j}'
    for stale in d.glob(f'{stem}[._]*'):
        stale.rename(stale.with_name(stale.name + f'.incomplete_attempt{prior}'))
    event(out, fold=k, action='fit_start', attempt=prior + 1)
    kind = c['schemes'][scheme]['kind']
    fo = folds[scheme]
    outer_train = np.flatnonzero(fo != f)
    outer_valid = np.flatnonzero(fo == f)
    cache = inner_cache if inner_cache is not None else {}
    ck = unit_scheme(scheme, f)
    if ck not in cache:
        cache[ck] = inner_folds(X.iloc[outer_train], y.iloc[outer_train], kind, c)
        require(hashlib.sha256(cache[ck].astype(np.int8).tobytes()).hexdigest() == design['inner_fold_sha256'][ck],
                'Inner folds differ from the sealed design')
    inner = cache[ck]
    a, b = outer_train[inner != j], outer_train[inner == j]
    require(len(a) > 0 and len(b) > 0 and not set(a) & set(b), 'Inner split invalid')
    require(not (set(a) | set(b)) & set(outer_valid.tolist()), 'Outer-validation rows leaked into the inner fit')
    res = fit_fn(arm, kind, X.iloc[a], y.iloc[a], X.iloc[b], c)
    p = res['p']
    pred = p.argmax(1) + 1
    paths = res['save'](d/stem)
    preds = d/f'{stem}_pred.csv.gz'
    frame = pd.DataFrame({'building_id': ids.iloc[b].to_numpy(), 'outer_fold': f, 'inner_fold': j,
                          'y_true': y.iloc[b].to_numpy(), 'y_pred': pred, **{n: p[:, i] for i, n in enumerate(PROBS)}})
    write_pred(preds, frame)
    reload_diff = float(res['verify'](paths, X.iloc[b], p))  # reload verification of freshly written files
    require(reload_diff <= c['checkpoint']['model_reload_probability_atol_lgbm'], f'Model reload mismatch {reload_diff}')
    back = read_pred(preds)
    np.testing.assert_array_equal(back[PROBS].to_numpy(), p)
    np.testing.assert_array_equal(back.building_id.to_numpy(), frame.building_id.to_numpy())
    record = dict(study_id=STUDY_ID, scheme=scheme, arm=arm, outer_fold=f, inner_fold=j, train_rows=len(a),
                  valid_rows=len(b), metrics=score(y.iloc[b], pred), fit_seconds=res['fit_seconds'],
                  converged=res['converged'], warnings=res.get('warnings', []), reload_probability_max_abs_diff=reload_diff,
                  csv_round_trip_exact=True, finished_at_utc=now())
    write(d/f'{stem}.json', record)
    shas = {pth.name: sha256_file(pth) for pth in [*paths, preds, d/f'{stem}.json']}
    bd = out/'backup'/entry['dir']
    bd.mkdir(parents=True, exist_ok=True)
    for name in shas:
        require(not (bd/name).exists(), 'Backup already exists; refusing overwrite')
        shutil.copy2(d/name, bd/name)
    converged = res['converged']
    entry.update(status='complete' if converged else 'complete_nonconverged', artifact_sha256=shas,
                 macro_f1=record['metrics']['macro_f1'], fit_seconds=res['fit_seconds'], converged=converged,
                 finished_at_utc=now())
    check_saved(out, entry)
    ledger = read_ledger(out)
    ledger['folds'][k] = entry
    atomic_write(out/'ledger.json', ledger)
    event(out, fold=k, action='fit_complete', seconds=res['fit_seconds'], converged=converged, artifact_sha256=shas)
    log(out, f'{k} innerF1={record["metrics"]["macro_f1"]:.6f} fit={res["fit_seconds"]:.1f}s saved+verified+backed-up')
    require(converged, 'LR nonconvergence: stop without changing parameters or retrying')
    return entry, True


# ---------------------------------------------------------------- decision stage for one (scheme, arm, outer fold)
def decide_unit(X, y, ids, folds, out, scheme, arm, f, design):
    out = Path(out)
    c = design['config']
    k = unit_key(scheme, arm, f)
    ledger = read_ledger(out)
    done = ledger.setdefault('decisions', {}).get(k)
    d = out/scheme/arm
    if done and done['status'] == 'complete':
        check_saved(out, done)
        return done, False
    inner_entries = [ledger['folds'].get(inner_key(scheme, arm, f, j)) for j in range(c['inner']['n_splits'])]
    require(all(e and e['status'] == 'complete' for e in inner_entries), f'{k}: inner fits incomplete')
    for e in inner_entries:
        check_saved(out, e)
    inner = pd.concat([read_pred(d/f'f{f}_i{j}_pred.csv.gz') for j in range(c['inner']['n_splits'])], ignore_index=True)
    fo = folds[scheme]
    tr_ids, va_ids = set(np.asarray(ids)[fo != f].tolist()), set(np.asarray(ids)[fo == f].tolist())
    got = set(inner.building_id.tolist())
    require(got == tr_ids and not got & va_ids and inner.building_id.is_unique,
            'Inner OOF must cover exactly the outer-training rows and none of the outer-validation rows')
    ref = read_reference(c, scheme, arm, ids, y, fo)
    rf = ref[ref.fold == f]
    # --- tuning step: only inner OOF labels/probabilities and the outer probabilities (no outer labels) enter
    b, pred_adj, curve = decide(inner.y_true.to_numpy(), inner[PROBS].to_numpy(), rf[PROBS].to_numpy(), c['rule']['grid'])
    # --- evaluation step (outer labels used only now, after b is fixed)
    y_out = rf.y_true.to_numpy()
    s0, s1 = score(y_out, rf.y_pred.to_numpy()), score(y_out, pred_adj)
    sweep = offset_curve(y_out, rf[PROBS].to_numpy(), c['rule']['grid'])
    record = dict(study_id=STUDY_ID, scheme=scheme, arm=arm, outer_fold=f, selected_b=b,
                  selected_at_grid_max=bool(b == c['rule']['grid'][-1]), selected_at_zero=bool(b == 0.0),
                  inner_rows=len(inner), inner_macro_f1_at_zero=curve[0], inner_macro_f1_at_selected=max(curve),
                  inner_curve=curve, outer_rows=len(rf), outer_unadjusted=s0, outer_adjusted=s1,
                  outer_delta_macro_f1=s1['macro_f1'] - s0['macro_f1'],
                  outer_sweep_descriptive_post_hoc_not_used_for_selection=sweep, finished_at_utc=now())
    write(d/f'decision_f{f}.json', record)
    frame = pd.DataFrame({'building_id': rf.building_id.to_numpy(), 'fold': f, 'y_true': y_out,
                          'y_pred_unadjusted': rf.y_pred.to_numpy(), 'y_pred_adjusted': pred_adj,
                          **{n: rf[n].to_numpy() for n in PROBS}})
    write_pred(d/f'adjusted_f{f}_pred.csv.gz', frame)
    shas = {n: sha256_file(d/n) for n in [f'decision_f{f}.json', f'adjusted_f{f}_pred.csv.gz']}
    bd = out/'backup'/scheme/arm
    bd.mkdir(parents=True, exist_ok=True)
    for n in shas:
        require(not (bd/n).exists(), 'Backup already exists; refusing overwrite')
        shutil.copy2(d/n, bd/n)
    entry = dict(dir=f'{scheme}/{arm}', status='complete', artifact_sha256=shas, selected_b=b, finished_at_utc=now())
    check_saved(out, entry)
    ledger = read_ledger(out)
    ledger.setdefault('decisions', {})[k] = entry
    atomic_write(out/'ledger.json', ledger)
    log(out, f'{k} b={b} outerF1 {s0["macro_f1"]:.6f}->{s1["macro_f1"]:.6f} saved+backed-up')
    return entry, True


def fit_all(data, output, only_schemes=None, fit_fn=fit_inner_arm):
    out = Path(output)
    lock = out/'RUNNING.lock'
    require(not lock.exists(), f'Another run holds {lock}; remove only if that process is dead')
    lock.write_text(f'pid={os.getpid()} started={now()}\n')
    try:
        X, y, ids, folds, design = verify(data, out)
        before = pd.util.hash_pandas_object(X, index=True).to_numpy()
        c = design['config']
        schemes = [s for s in c['schemes'] if only_schemes is None or s in only_schemes]
        log(out, f'fit-all start: schemes={schemes} (completed fits are hash-verified, not refit)')
        t0, cache = time.perf_counter(), {}
        for s in schemes:
            for f in range(5):
                for a in ARMS:
                    try:
                        for j in range(c['inner']['n_splits']):
                            fit_inner_fold(X, y, ids, folds, out, s, a, f, j, design, fit_fn, cache)
                        decide_unit(X, y, ids, folds, out, s, a, f, design)
                    except Exception as exc:
                        log(out, f'STOPPED {unit_key(s, a, f)} {type(exc).__name__}: {exc}')
                        event(out, fold=unit_key(s, a, f), action='stopped', error=f'{type(exc).__name__}: {exc}',
                              trace=traceback.format_exc())
                        raise
        np.testing.assert_array_equal(before, pd.util.hash_pandas_object(X, index=True).to_numpy())
        log(out, f'fit-all finished wall={time.perf_counter()-t0:.1f}s')
    finally:
        lock.unlink(missing_ok=True)


# ---------------------------------------------------------------- rules (pure functions)
def judge_arm(rules, geo2, strat):
    """Pre-registered per-arm rule.

    geo2: fold_deltas (list, 20 outer folds, seed-major), seed_mean_deltas {seed: mean}, grade3_recall = (unadj, adj),
          grade1_recall = (unadj, adj), error31 = (unadj, adj)
    strat: mean_delta over its 5 outer folds
    """
    d = geo2['fold_deltas']
    parts = dict(
        P1_mean_delta_ok=bool(np.mean(d) >= rules['P1_mean_delta_min']),
        P2_fold_wins_ok=bool(sum(x > 0 for x in d) >= rules['P2_fold_wins_min'] and len(d) == rules['P2_n_folds']),
        P3_positive_seeds_ok=bool(sum(v > 0 for v in geo2['seed_mean_deltas'].values()) >= rules['P3_positive_seed_means_min']
                                  and len(geo2['seed_mean_deltas']) == rules['P3_n_seeds']),
        G1_grade3_recall_gain_ok=bool(geo2['grade3_recall'][1] - geo2['grade3_recall'][0]
                                      >= rules['G1_pooled_grade3_recall_gain_min']),
        G2_error31_ratio_ok=bool(geo2['error31'][1] <= rules['G2_error31_ratio_max'] * geo2['error31'][0]),
        G3_grade1_recall_drop_ok=bool(geo2['grade1_recall'][0] - geo2['grade1_recall'][1]
                                      <= rules['G3_pooled_grade1_recall_drop_max']),
        G4_stratified_not_worse_ok=bool(strat['mean_delta'] >= rules['G4_stratified_mean_delta_min']))
    return dict(candidate=bool(all(parts.values())), parts=parts,
                inputs=dict(mean_delta=float(np.mean(d)), fold_wins=int(sum(x > 0 for x in d)), n_folds=len(d),
                            seed_mean_deltas=geo2['seed_mean_deltas'], stratified_mean_delta=strat['mean_delta'],
                            grade3_recall=geo2['grade3_recall'], grade1_recall=geo2['grade1_recall'],
                            error31=geo2['error31']))


# ---------------------------------------------------------------- finalize
def arm_scheme_block(out, scheme, arm):
    d = Path(out)/scheme/arm
    recs = [json.loads((d/f'decision_f{f}.json').read_text(encoding='utf-8')) for f in range(5)]
    oof = pd.concat([read_pred(d/f'adjusted_f{f}_pred.csv.gz') for f in range(5)], ignore_index=True)
    return recs, oof


def finalize_all(data, output):
    out = Path(output)
    X, y, ids, folds, design = verify(data, out)
    require(not (out/'analysis.json').exists(), 'analysis.json exists; refusing overwrite')
    ledger = read_ledger(out)
    c = design['config']
    tg = design['study_targets']
    complete = sum(ledger['folds'].get(k, {}).get('status') == 'complete' for k in tg)
    require(complete == len(tg) == N_TARGETS and len(ledger['folds']) == N_TARGETS
            and len(ledger['decisions']) == len(c['schemes']) * len(ARMS) * 5, 'Ledger incomplete; not finalizing')
    attempts = sum(e['attempts'] for e in ledger['folds'].values())
    res = dict(study_id=STUDY_ID, design_sha256=sha256_file(out/'design.json'), code_commit=design['code_commit'],
               ledger=dict(inner_fits=complete, total=N_TARGETS, attempts=attempts, retries=attempts - complete,
                           fit_seconds_sum=float(sum(e['fit_seconds'] for e in ledger['folds'].values())),
                           all_lr_converged=all(e['converged'] for e in ledger['folds'].values())),
               holdout_accessed=False, rules=c['rules'], rule=c['rule'], schemes={}, judgments={},
               note='Descriptive; stratified seed 42 is optimistic (folds reused by many studies); geo1 diagnostic only; '
                    'outer models were not refit (saved OOFs reused read-only); no holdout.')
    per = {}
    for s in c['schemes']:
        res['schemes'][s] = {}
        for a in ARMS:
            recs, oof = arm_scheme_block(out, s, a)
            np.testing.assert_array_equal(oof.fold.to_numpy(), np.repeat(np.arange(5), [r['outer_rows'] for r in recs]))
            yy = oof.y_true.to_numpy()
            p0, p1 = oof.y_pred_unadjusted.to_numpy(), oof.y_pred_adjusted.to_numpy()
            blk = dict(selected_b=[r['selected_b'] for r in recs],
                       fold_f1_unadjusted=[r['outer_unadjusted']['macro_f1'] for r in recs],
                       fold_f1_adjusted=[r['outer_adjusted']['macro_f1'] for r in recs],
                       fold_delta=[r['outer_delta_macro_f1'] for r in recs],
                       pooled_unadjusted=score(yy, p0), pooled_adjusted=score(yy, p1),
                       pred_share_unadjusted=(np.bincount(p0 - 1, minlength=3)/len(p0)).tolist(),
                       pred_share_adjusted=(np.bincount(p1 - 1, minlength=3)/len(p1)).tolist(),
                       selected_at_grid_max_fraction=float(np.mean([r['selected_at_grid_max'] for r in recs])),
                       selected_at_zero_fraction=float(np.mean([r['selected_at_zero'] for r in recs])))
            blk['mean_delta'] = float(np.mean(blk['fold_delta']))
            res['schemes'][s][a] = per[s, a] = blk
    seeds = [c['schemes'][s]['seed'] for s in c['primary_schemes']]
    for a in ARMS:
        deltas = [x for s in c['primary_schemes'] for x in per[s, a]['fold_delta']]
        pooled0, pooled1 = [], []
        for s in c['primary_schemes']:
            pooled0.append(per[s, a]['pooled_unadjusted'])
            pooled1.append(per[s, a]['pooled_adjusted'])
        cm0, cm1 = (np.sum([p['confusion'] for p in pooled0], axis=0), np.sum([p['confusion'] for p in pooled1], axis=0))
        rec = lambda cm, k: float(cm[k, k]/cm[k].sum())  # noqa: E731
        geo2 = dict(fold_deltas=deltas, seed_mean_deltas={str(sd): per[s, a]['mean_delta']
                                                           for sd, s in zip(seeds, c['primary_schemes'])},
                    grade3_recall=(rec(cm0, 2), rec(cm1, 2)), grade1_recall=(rec(cm0, 0), rec(cm1, 0)),
                    error31=(int(cm0[2, 0]), int(cm1[2, 0])))
        res['judgments'][a] = dict(judge_arm(c['rules'], geo2, dict(mean_delta=per['strat_s42', a]['mean_delta'])),
                                   geo2_pooled_confusion_unadjusted=cm0.tolist(),
                                   geo2_pooled_confusion_adjusted=cm1.tolist())
    res['common_rule_holds'] = bool(all(j['candidate'] for j in res['judgments'].values()))
    res['finished_at_utc'] = now()
    write(out/'analysis.json', res)
    require(not (out/'backup'/'analysis.json').exists(), 'Backup already exists; refusing overwrite')
    shutil.copy2(out/'analysis.json', out/'backup'/'analysis.json')
    log(out, 'analysis written: ' + ', '.join(f'{a}={j["candidate"]}' for a, j in res['judgments'].items()))
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'fit-all', 'finalize', 'status'])
    p.add_argument('--data', default='data')
    p.add_argument('--output')
    p.add_argument('--code-commit')
    p.add_argument('--schemes', nargs='*', help='subset of schemes (default: all, primary first)')
    a = p.parse_args()
    if a.action == 'prepare':
        prepare(a.data, a.output, a.code_commit)
    elif a.action == 'fit-all':
        fit_all(a.data, a.output, a.schemes)
    elif a.action == 'finalize':
        finalize_all(a.data, a.output)
    else:
        ledger = read_ledger(a.output)
        print(json.dumps(dict(inner_complete=sum(e['status'] == 'complete' for e in ledger['folds'].values()),
                              total=N_TARGETS, decisions=len(ledger.get('decisions', {})),
                              attempts=sum(e['attempts'] for e in ledger['folds'].values()),
                              running=[k for k, e in ledger['folds'].items() if e['status'] == 'running']), indent=1))


if __name__ == '__main__':
    sys.exit(main())
