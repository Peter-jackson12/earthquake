"""T38: wider nested-CV tuning of DT, RF and sklearn GBC (follow-up of T37 whose selections hit the grid edges).

Same outer/inner splits, preprocessing (v1.0.1) and selection rule as T37 (sealed module imported read-only).  No class
weighting or sampling.  GBC inner fits are grouped by depth: one 600-stage fit per (outer, inner, depth) gives the
300-stage candidate through `staged_predict_proba` (identity with a separate 300-stage fit is checked in preflight).
Outer folds are always refit with the selected candidate.  J1 compares with the T37 tuned OOF of the same model.

Design: docs/T38_WIDE_TUNING_PLAN.md.  Actions: preflight (3 control fits + checks, no study fit), prepare (seal),
fit-all (resumable), status, finalize.  Holdout is never read.  All outputs are private (outside Git).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from preprocess import sha256_file  # noqa: E402
from experiments.geo_group_cv.run import PROBS, require, score, write  # noqa: E402
from experiments.ey_pipeline_benchmark import run as T33  # noqa: E402  (read-only reuse of sealed helpers)
from experiments.required_tuning import run as T37  # noqa: E402  (read-only reuse of sealed helpers)

HERE = Path(__file__).resolve().parent
STUDY_ID = 'T38_wide_tuning_v1'
N_FOLDS = 5
N_INNER = 3
MODELS = ['DT', 'RF', 'GBC']
SOURCE = list(dict.fromkeys(T37.SOURCE + [
    'experiments/wide_tuning/__init__.py', 'experiments/wide_tuning/run.py', 'experiments/wide_tuning/config.json',
    'tests/test_wide_tuning.py', 'docs/T38_WIDE_TUNING_PLAN.md']))

now, log, event, atomic_write = T33.now, T33.log, T33.event, T33.atomic_write
read_pred, write_pred, read_ledger, check_saved, backup_files = (T33.read_pred, T33.write_pred, T33.read_ledger,
                                                                 T33.check_saved, T33.backup_files)
git, require_not_holdout, load_inputs, reference, run_lock = (T33.git, T33.require_not_holdout, T33.load_inputs,
                                                              T33.reference, T33.run_lock)
environment, inner_split, select, ids_sha256 = T37.environment, T37.inner_split, T37.select, T37.ids_sha256
fits_started, recovered_units, checked_proba, move_aside = (T37.fits_started, T37.recovered_units, T37.checked_proba,
                                                            T37.move_aside)
Context, begin_fit, input_checks, split_checks = T37.Context, T37.begin_fit, T37.input_checks, T37.split_checks

REGISTERED_FIXED = {
    'DT': dict(random_state=42),
    'RF': dict(n_estimators=100, min_samples_leaf=2, random_state=42, n_jobs=4),
    'GBC': dict(learning_rate=0.1, max_features=None, subsample=1.0, random_state=42),
}
_dt = [(16, 50)] + [(d, l) for d in (16, 20, 24, 32, None) for l in (5, 20, 50, 100) if (d, l) != (16, 50)]
_rf = [(0.3, 20)] + [(mf, d) for mf in (0.3, 0.45, 0.6, 0.8) for d in (20, 30) if (mf, d) != (0.3, 20)]
REGISTERED_CANDIDATES = {
    'DT': [dict(max_depth=d, min_samples_leaf=l) for d, l in _dt],
    'RF': [dict(max_features=mf, max_depth=d) for mf, d in _rf],
    'GBC': [dict(n_estimators=300, max_depth=5), dict(n_estimators=600, max_depth=5), dict(n_estimators=300, max_depth=7),
            dict(n_estimators=600, max_depth=7)],
}
GBC_GROUPS = {'d5': ('GBC0', 'GBC1'), 'd7': ('GBC2', 'GBC3')}   # (300-stage candidate, 600-stage candidate)


def sources():
    return {p: sha256_file(ROOT/p) for p in SOURCE}


def config(path=HERE/'config.json'):
    c = json.loads(Path(path).read_text(encoding='utf-8'))
    require(c['study_id'] == STUDY_ID and c['n_folds'] == N_FOLDS and c['models'] == MODELS, 'Unplanned study/models')
    require(c['inner'] == dict(n_splits=3, shuffle=True, random_state=42, splitter='StratifiedKFold',
                               rows='outer training fold in building_id order'), 'Inner split changed')
    require(c['fixed_params'] == REGISTERED_FIXED, 'Fixed model settings changed')
    require({m: list(c['candidates'][m].values()) for m in MODELS} == REGISTERED_CANDIDATES
            and all(list(c['candidates'][m]) == [f'{m}{i}' for i in range(len(REGISTERED_CANDIDATES[m]))] for m in MODELS),
            'Candidates changed')
    require(c['gbc_groups'] == {k: list(v) for k, v in GBC_GROUPS.items()}, 'GBC staged grouping changed')
    require(c['class_weighting'] is None and c['sampling'] is None, 'Weighting/sampling must be off')
    require(c['budget'] == dict(inner_fits=450, outer_refits=15, planned_fits=465, max_fits=490, max_recovered_units=10,
                                max_attempts_per_unit=2, automatic_retries=0, control_fits=3), 'Unplanned fit budget')
    require(len(inner_targets(c)) == 450, 'Budget arithmetic')
    require(c['threads'] == dict(rf_n_jobs=4, other_models_threads=1, outer_parallel_jobs=1), 'Thread plan changed')
    require(c['selection'] == dict(metric='pooled inner Macro F1 over the 3 inner validation folds, labels [1,2,3]',
                                   tie_break='registered candidate order (T37 anchor first)'), 'Selection rule changed')
    j1 = dict(mean_delta=0.003, fold_wins_min=4, advantage='IMPROVED_DESCRIPTIVE', disadvantage='WORSENED_DESCRIPTIVE',
              otherwise='NOT_MET')
    require(c['rules']['J1'] == dict(j1, delta='T38 outer fold Macro F1 minus T37 tuned fold Macro F1 on the same fold')
            and c['rules']['J1b'] == dict(j1, delta='T38 outer fold Macro F1 minus A2 baseline fold Macro F1 on the same fold'),
            'Rules changed')
    require(set(c['references']['T37']) == set(MODELS) and set(c['references']['A2']) == set(MODELS), 'References changed')
    require(c['control'] == dict(probability_atol=1e-9, fits={'DT': dict(fold=0, candidate='DT0'),
                                                              'GBC': dict(fold=0, candidate='GBC0'),
                                                              'RF': dict(fold=1, candidate='RF0')}), 'Control changed')
    require(c['checkpoint']['reload_probability_atol'] == 1e-12, 'Reload tolerance changed')
    require(not c['holdout_access'] and not c['threshold_or_decision_rule_change'] and not c['automatic_model_replacement'],
            'Unplanned run policy')
    require(c['environment'] == environment() or Path(path) != HERE/'config.json', 'Environment differs from pinned')
    return c


# ---------------------------------------------------------------- keys
def candidates(c, m):
    return list(c['candidates'][m])


def fit_groups(c, m):
    """Inner fit units of model m: one per candidate, except GBC (one per depth group)."""
    return list(c['gbc_groups']) if m == 'GBC' else candidates(c, m)


def group_candidates(c, m, g):
    return list(c['gbc_groups'][g]) if m == 'GBC' else [g]


def inner_key(m, f, k, g):
    return f'{m}/I/{f}/{k}/{g}'


def outer_key(m, f):
    return f'{m}/O/{f}'


def inner_targets(c):
    return [(m, f, k, g) for m in MODELS for f in range(N_FOLDS) for k in range(N_INNER) for g in fit_groups(c, m)]


def model_params(c, m, cand):
    return dict(c['fixed_params'][m], **c['candidates'][m][cand])


def pred_name(f, k, cand):
    return f'f{f}_k{k}_{cand}_pred.csv.gz'


def fit(c, m, cand, X, y):
    """T37 fit (fresh v1.0.1 pipeline, warnings recorded, RF predicts sequentially) with this study's parameters."""
    return T37.fit_pipeline(dict(c, candidates={m: {cand: c['candidates'][m][cand]}}), m, cand, X, y)


def staged(pipe, X, stages):
    """Probabilities after each requested number of boosting stages, from one pass over staged_predict_proba."""
    Xt = pipe['preprocess'].transform(X)
    out, want = {}, set(stages)
    for i, p in enumerate(pipe['model'].staged_predict_proba(Xt), start=1):
        if i in want:
            out[i] = np.asarray(p, dtype=np.float64)
    require(set(out) == want, 'Requested stages not reached')
    return out


def probabilities(c, m, g, pipe, X):
    """{candidate: probabilities} produced by one fitted inner unit."""
    if m != 'GBC':
        return {g: checked_proba(pipe, X)}
    cands = group_candidates(c, m, g)
    ns = [c['candidates'][m][x]['n_estimators'] for x in cands]
    require(np.array_equal(np.asarray(pipe.classes_), [1, 2, 3]), 'Class order must be 1,2,3')
    st = staged(pipe, X, ns)
    full = checked_proba(pipe, X)
    require(np.array_equal(st[max(ns)], full), 'Last stage differs from predict_proba')
    return {x: st[n] for x, n in zip(cands, ns)}


def save_unit(ctx, key, m, fit_cand, prob_map, pipe, seconds, conv, other, a, b, d, stem, extra, pred_files):
    out, c = ctx.out, ctx.c
    names = [f'{stem}_model.joblib', f'{stem}.json'] + list(pred_files.values())
    y_va = ctx.y.to_numpy()[b]
    joblib.dump(pipe, d/names[0], compress=3)
    metrics = {}
    for cand, p in prob_map.items():
        pred = p.argmax(1) + 1
        frame = pd.DataFrame({'building_id': ctx.ids.to_numpy()[b], 'y_true': y_va, 'y_pred': pred,
                              **{n: p[:, i] for i, n in enumerate(PROBS)}})
        for col, v in extra['pred_columns'].items():
            frame.insert(1, col, v)
        write_pred(d/pred_files[cand], frame)
        back = read_pred(d/pred_files[cand])
        np.testing.assert_array_equal(back[PROBS].to_numpy(), p)
        metrics[cand] = score(y_va, pred)
    back_pipe = joblib.load(d/names[0])
    p2 = probabilities(c, m, extra['group'], back_pipe, ctx.X.iloc[b]) if extra.get('group') else \
        {fit_cand: checked_proba(back_pipe, ctx.X.iloc[b])}
    diff = max(float(np.max(np.abs(p2[x] - prob_map[x]))) for x in prob_map)
    require(diff <= c['checkpoint']['reload_probability_atol']
            and all(np.array_equal(p2[x].argmax(1), prob_map[x].argmax(1)) for x in prob_map), f'{key}: reload mismatch {diff}')
    record = dict(study_id=STUDY_ID, unit=key, model=m, fitted_params=model_params(c, m, fit_cand),
                  candidates=list(prob_map), metrics=metrics, train_rows=len(a), valid_rows=len(b),
                  train_ids_sha256=ids_sha256(ctx.ids.to_numpy()[a]), valid_ids_sha256=ids_sha256(ctx.ids.to_numpy()[b]),
                  train_class_counts=[int((ctx.y.to_numpy()[a] == v).sum()) for v in (1, 2, 3)], fit_seconds=seconds,
                  convergence_warnings=conv, other_warnings=other, reload_probability_max_abs_diff=diff,
                  csv_round_trip_exact=True, finished_at_utc=now(),
                  **{k2: v for k2, v in extra.items() if k2 not in ('pred_columns', 'group')})
    write(d/names[1], record)
    shas = {n: sha256_file(d/n) for n in names}
    rel = d.relative_to(out).as_posix()
    backup_files(out, rel, shas)
    ledger = read_ledger(out)
    entry = ledger['units'][key]
    entry.update(dir=rel, artifact_sha256=shas, status='complete', model=m, fitted_candidate=fit_cand,
                 candidates=list(prob_map), macro_f1={x: metrics[x]['macro_f1'] for x in metrics}, fit_seconds=seconds,
                 finished_at_utc=now())
    atomic_write(out/'ledger.json', ledger)
    check_saved(out, entry)
    event(out, unit=key, action='unit_complete', macro_f1=entry['macro_f1'], seconds=seconds, artifact_sha256=shas)
    log(out, f'{key} complete: F1=' + json.dumps({x: round(v, 6) for x, v in entry['macro_f1'].items()})
        + f' fit={seconds:.0f}s reload_diff={diff:.3g}')
    return entry


def inner_unit(ctx, m, f, k, g):
    key = inner_key(m, f, k, g)
    entry = read_ledger(ctx.out)['units'].get(key)
    if entry and entry['status'] == 'complete':
        check_saved(ctx.out, entry)
        return entry
    c = ctx.c
    a, b = ctx.positions(f, k)
    require(not set(b) & set(np.flatnonzero(ctx.folds == f)), f'{key}: outer validation row in the inner step')
    d = ctx.out/m/'inner'/f'f{f}'
    cands = group_candidates(c, m, g)
    fit_cand = cands[-1]                                      # GBC: the 600-stage candidate of the depth group
    pred_files = {x: pred_name(f, k, x) for x in cands}
    stem = f'f{f}_k{k}_{g}'
    d.mkdir(parents=True, exist_ok=True)
    prior = begin_fit(ctx, key)
    move_aside(d, [f'{stem}_model.joblib', f'{stem}.json'] + list(pred_files.values()), prior)
    require(not set(a) & set(b), f'{key}: train/valid overlap')
    pipe, seconds, conv, other = fit(c, m, fit_cand, ctx.X.iloc[a], ctx.y.iloc[a])
    prob_map = probabilities(c, m, g, pipe, ctx.X.iloc[b])
    return save_unit(ctx, key, m, fit_cand, prob_map, pipe, seconds, conv, other, a, b, d, stem,
                     dict(outer_fold=f, inner_fold=k, group=g if m == 'GBC' else None,
                          pred_columns={'inner_fold': k, 'outer_fold': f}), pred_files)


def pooled_inner_scores(out, c, m, f):
    ledger = read_ledger(out)
    scores, n = {}, None
    for g in fit_groups(c, m):
        for cand in group_candidates(c, m, g):
            parts = []
            for k in range(N_INNER):
                e = ledger['units'].get(inner_key(m, f, k, g))
                require(e and e['status'] == 'complete', f'{inner_key(m, f, k, g)} incomplete')
                check_saved(out, e)
                parts.append(read_pred(Path(out)/e['dir']/pred_name(f, k, cand)))
            pooled = pd.concat(parts)
            require(pooled.building_id.is_unique, 'Inner validation folds overlap')
            n = len(pooled) if n is None else n
            require(len(pooled) == n, 'Inner coverage differs between candidates')
            scores[cand] = score(pooled.y_true.to_numpy(), pooled.y_pred.to_numpy())['macro_f1']
    return {cand: scores[cand] for cand in candidates(c, m)}, n


def selection(ctx, m, f):
    out, c = ctx.out, ctx.c
    scores, n = pooled_inner_scores(out, c, m, f)
    require(n == len(ctx.positions(f)[0]), 'Pooled inner rows must equal the outer training fold')
    chosen = select(scores, candidates(c, m))
    ledger = read_ledger(out)
    sk = f'{m}/{f}'
    prev = ledger['selections'].get(sk)
    if prev:
        require(prev['selected'] == chosen and prev['pooled_inner_macro_f1'] == scores, f'Selection {sk} changed')
        return prev
    rec = dict(model=m, outer_fold=f, pooled_inner_macro_f1=scores, selected=chosen, pooled_rows=n, decided_at_utc=now())
    ledger['selections'][sk] = rec
    atomic_write(out/'ledger.json', ledger)
    event(out, action='selection', model=m, outer_fold=f, selected=chosen, pooled_inner_macro_f1=scores)
    log(out, f'{m} outer fold {f}: selected {chosen} ' + json.dumps({k2: round(v, 6) for k2, v in scores.items()}))
    return rec


def outer_unit(ctx, m, f):
    key = outer_key(m, f)
    sel = selection(ctx, m, f)
    entry = read_ledger(ctx.out)['units'].get(key)
    if entry and entry['status'] == 'complete':
        check_saved(ctx.out, entry)
        require(entry['fitted_candidate'] == sel['selected'], 'Outer unit does not match the selection')
        return entry
    cand = sel['selected']
    a, b = ctx.positions(f)
    d = ctx.out/m/'outer'
    stem = f'fold_{f}'
    pred_files = {cand: f'fold_{f}_pred.csv.gz'}
    d.mkdir(parents=True, exist_ok=True)
    prior = begin_fit(ctx, key)
    move_aside(d, [f'{stem}_model.joblib', f'{stem}.json'] + list(pred_files.values()), prior)
    pipe, seconds, conv, other = fit(ctx.c, m, cand, ctx.X.iloc[a], ctx.y.iloc[a])
    prob_map = {cand: checked_proba(pipe, ctx.X.iloc[b])}
    return save_unit(ctx, key, m, cand, prob_map, pipe, seconds, conv, other, a, b, d, stem,
                     dict(outer_fold=f, group=None, pred_columns={'fold': f}), pred_files)


# ---------------------------------------------------------------- preflight
def reference_check(c, local_root, ids, y, folds):
    res = {}
    for kind in ('T37', 'A2'):
        res[kind] = {}
        for m in MODELS:
            spec = c['references'][kind][m]
            r = reference(Path(local_root)/spec['path'], spec['sha256'], ids, y, folds)
            fo = r.fold.to_numpy()
            values = [score(r.y_true[fo == f], r.y_pred[fo == f])['macro_f1'] for f in range(N_FOLDS)]
            require(abs(float(np.mean(values)) - spec['cv_mean']) <= 1e-6, f'{kind} {m} CV mean differs from the registry')
            res[kind][m] = dict(sha256=spec['sha256'], fold_macro_f1=values, cv_mean=float(np.mean(values)))
    return res


def control_fits(c, X, y, ids, folds, local_root):
    res = {}
    tol = c['control']['probability_atol']
    for m, spec in c['control']['fits'].items():
        f, cand = spec['fold'], spec['candidate']
        ref_spec = c['references']['T37'][m]
        r = reference(Path(local_root)/ref_spec['path'], ref_spec['sha256'], ids, y, folds)
        rows = r[r.fold == f]
        a, b = np.flatnonzero(folds != f), np.flatnonzero(folds == f)
        pipe, seconds, _, _ = fit(c, m, cand, X.iloc[a], y.iloc[a])
        p = checked_proba(pipe, X.iloc[b])
        ref = rows[PROBS].to_numpy()
        diff = float(np.max(np.abs(p - ref)))
        mism = p.argmax(1) + 1 != rows.y_pred.to_numpy()
        top2 = np.sort(ref, axis=1)[:, -2:]
        tie_only = bool(np.all(top2[mism, 1] - top2[mism, 0] <= tol))
        res[m] = dict(fold=f, candidate=cand, max_abs_probability_diff=diff, mismatched_rows=int(mism.sum()),
                      mismatches_only_on_ties=tie_only, fit_seconds=seconds, passes=bool(diff <= tol and tie_only))
    return dict(models=res, control_fits=len(res), counted_as_study_fit=False)


def staged_identity(c, X_small, y_small):
    res = {}
    for depth in (5, 7):
        base = dict(c['fixed_params']['GBC'], max_depth=depth)
        sc = dict(c, candidates={'GBC': {'A': dict(n_estimators=20, max_depth=depth), 'B': dict(n_estimators=10, max_depth=depth)}})
        p20, _, _, _ = T37.fit_pipeline(dict(sc, fixed_params=dict(c['fixed_params'], GBC=base)), 'GBC', 'A', X_small, y_small)
        p10, _, _, _ = T37.fit_pipeline(dict(sc, fixed_params=dict(c['fixed_params'], GBC=base)), 'GBC', 'B', X_small, y_small)
        st = staged(p20, X_small, [10])[10]
        res[f'd{depth}'] = dict(bit_identical=bool(np.array_equal(st, checked_proba(p10, X_small))),
                                max_abs_diff=float(np.max(np.abs(st - checked_proba(p10, X_small)))))
    return dict(depths=res, rows=int(len(X_small)), passes=all(v['bit_identical'] for v in res.values()),
                note='20-stage model, first 10 stages vs separate 10-stage fit; outside the budget')


def probe(c, X_small, y_small):
    res = {}
    for m in MODELS:
        for cand in candidates(c, m):
            sc = json.loads(json.dumps(c))
            if m == 'GBC':
                sc['candidates']['GBC'][cand] = dict(sc['candidates']['GBC'][cand], n_estimators=10)
            if m == 'RF':
                sc['fixed_params']['RF'] = dict(sc['fixed_params']['RF'], n_estimators=10)
            p1 = checked_proba(fit(sc, m, cand, X_small, y_small)[0], X_small)
            pipe = fit(sc, m, cand, X_small, y_small)[0]
            p2 = checked_proba(pipe, X_small)
            with tempfile.TemporaryDirectory() as td:
                joblib.dump(pipe, Path(td)/'m.joblib', compress=3)
                pr = checked_proba(joblib.load(Path(td)/'m.joblib'), X_small)
            tol = c['checkpoint']['reload_probability_atol']
            rep, rel = float(np.max(np.abs(p1 - p2))), float(np.max(np.abs(pr - p2)))
            res[f'{m}/{cand}'] = dict(repeat_max_abs_diff=rep, reload_max_abs_diff=rel,
                                      passes=bool(rep <= tol and rel <= tol))
    return dict(candidates=res, probe_fits=2 * len(res), counted_as_study_fit=False,
                note='first 2,000 outer-fold-0 training rows, small trees; outside the budget')


def preflight(data, destination, local_root=None):
    c = config()
    dest = Path(destination)
    require(not dest.exists(), 'Do not overwrite an existing preflight (use a new folder name)')
    require_not_holdout(data)
    local_root = Path(local_root) if local_root else dest.resolve().parents[1]
    require(environment() == c['environment'], 'Environment differs from the pinned environment')
    plan_sha = sha256_file(ROOT/c['plan']['path'])
    require(plan_sha == c['plan']['sha256'], 'Plan document SHA differs from the registered SHA')
    X, y, ids, folds, contract = load_inputs(data)
    res = dict(study_id=STUDY_ID, kind='preflight; no study fit (3 control fits + small checks)', classifiers_fitted_study=0,
               holdout_accessed=False, created_at_utc=now(), environment=environment(), data_contract=contract,
               plan_sha256=plan_sha, source_sha256=sources())
    res['inputs'] = input_checks(X, y, ids, folds, contract, c)
    res['splits'] = split_checks(y, ids, folds, c)
    res['references'] = reference_check(c, local_root, ids, y, folds)
    a0 = np.flatnonzero(folds != 0)
    res['staged_identity'] = staged_identity(c, X.iloc[a0[:5000]], y.iloc[a0[:5000]])
    res['model_probe'] = probe(c, X.iloc[a0[:2000]], y.iloc[a0[:2000]])
    res['control'] = control_fits(c, X, y, ids, folds, local_root)
    ok = (res['staged_identity']['passes'] and all(v['passes'] for v in res['model_probe']['candidates'].values())
          and all(v['passes'] for v in res['control']['models'].values()))
    res['status'] = 'PASS' if ok else 'FAIL'
    write(dest/'preflight.json', res)
    print(json.dumps({k: res[k] for k in ('status', 'staged_identity', 'control')}, indent=1, default=str))
    require(ok, 'Preflight failed')
    return res


# ---------------------------------------------------------------- prepare / verify / fit-all / status
def prepare(data, out, preflight_path, code_commit=None, local_root=None):
    out = Path(out)
    require(not out.exists(), 'Do not overwrite an existing study')
    c = config()
    require_not_holdout(data)
    local_root = Path(local_root) if local_root else out.resolve().parents[1]
    require(environment() == c['environment'], 'Environment differs from the pinned environment')
    require(shutil.disk_usage(out.resolve().parent).free / 2**30 >= c['min_free_disk_gb_at_prepare'], 'Not enough free disk')
    head = git('rev-parse', 'HEAD')
    require(code_commit is None or code_commit == head, 'code_commit must equal HEAD')
    for p in SOURCE:
        require(git('hash-object', '--', p) == git('rev-parse', f'HEAD:{p}'), f'Sealed source differs from HEAD: {p}')
    plan_sha = sha256_file(ROOT/c['plan']['path'])
    require(plan_sha == c['plan']['sha256'], 'Plan document SHA differs from the registered SHA')
    pf = json.loads(Path(preflight_path).read_text(encoding='utf-8'))
    require(pf['status'] == 'PASS' and pf['study_id'] == STUDY_ID and pf['classifiers_fitted_study'] == 0,
            'Preflight must PASS first')
    require(pf['source_sha256'] == sources() and pf['plan_sha256'] == plan_sha and pf['environment'] == environment(),
            'Sources changed since the preflight; rerun it')
    X, y, ids, folds, contract = load_inputs(data)
    require(contract == pf['data_contract'], 'Data contract differs from the preflight')
    refs = reference_check(c, local_root, ids, y, folds)
    out.mkdir(parents=True)
    (out/'backup').mkdir()
    design = dict(study_id=STUDY_ID, config=c, code_commit=head, source_sha256=sources(), plan_sha256=plan_sha,
                  local_root=str(local_root.resolve()), data_contract=contract, inputs=pf['inputs'],
                  environment=environment(), references=refs, splits=pf['splits'],
                  preflight=dict(path=str(Path(preflight_path).resolve()), sha256=sha256_file(preflight_path),
                                 status=pf['status'], control=pf['control'], staged_identity=pf['staged_identity']),
                  inner_targets=[inner_key(*t) for t in inner_targets(c)],
                  outer_targets=[outer_key(m, f) for m in MODELS for f in range(N_FOLDS)],
                  max_fits=c['budget']['max_fits'], planned_fits=c['budget']['planned_fits'],
                  automatic_retries=0, holdout_accessed=False, created_at_utc=now(), classifiers_fitted=0)
    write(out/'design.json', design)
    write(out/'design_seal.json', dict(design_sha256=sha256_file(out/'design.json'), code_commit=head, created_at_utc=now(),
                                       classifiers_fitted=0))
    atomic_write(out/'ledger.json', dict(study_id=STUDY_ID, units={}, selections={}))
    log(out, f'prepared and sealed design {sha256_file(out/"design.json")} commit={head}')
    return design


def verify(data, out):
    out = Path(out)
    seal = json.loads((out/'design_seal.json').read_text(encoding='utf-8'))
    require(sha256_file(out/'design.json') == seal['design_sha256'], 'Design seal changed')
    design = json.loads((out/'design.json').read_text(encoding='utf-8'))
    require(design['config'] == config(), 'Frozen config changed')
    require(design['source_sha256'] == sources(), 'Frozen source changed')
    require(design['environment'] == environment(), 'Frozen environment changed')
    X, y, ids, folds, contract = load_inputs(data)
    require(contract == design['data_contract'], 'Data contract changed')
    return X, y, ids, folds, design


def _guard(out, key, fn, *args):
    try:
        return fn(*args)
    except Exception as exc:
        log(out, f'STOPPED {key} {type(exc).__name__}: {exc}')
        event(out, unit=key, action='stopped', error=f'{type(exc).__name__}: {exc}', trace=traceback.format_exc())
        raise


def fit_all(data, out, only_models=None):
    out = Path(out)
    with run_lock(out):
        X, y, ids, folds, design = verify(data, out)
        ctx = Context(X, y, ids, folds, out, design)
        c = ctx.c
        before = pd.util.hash_pandas_object(X, index=True).to_numpy()
        todo = [m for m in MODELS if only_models is None or m in only_models]
        log(out, f'fit-all start: models {todo} (complete units are hash-verified, not refit)')
        t0 = time.perf_counter()
        for m in todo:
            tm = time.perf_counter()
            for f in range(N_FOLDS):
                for k in range(N_INNER):
                    for g in fit_groups(c, m):
                        _guard(out, inner_key(m, f, k, g), inner_unit, ctx, m, f, k, g)
                _guard(out, outer_key(m, f), outer_unit, ctx, m, f)
            log(out, f'model {m} done in {time.perf_counter() - tm:.0f}s (wall so far {time.perf_counter() - t0:.0f}s)')
        np.testing.assert_array_equal(before, pd.util.hash_pandas_object(X, index=True).to_numpy())
        log(out, f'fit-all finished wall={time.perf_counter() - t0:.1f}s')


def status(out):
    ledger = read_ledger(out)
    units = ledger['units']
    design = json.loads((Path(out)/'design.json').read_text(encoding='utf-8'))
    per = {m: dict(inner=sum(e['status'] == 'complete' for k, e in units.items() if k.startswith(f'{m}/I/')),
                   outer=sum(e['status'] == 'complete' for k, e in units.items() if k.startswith(f'{m}/O/'))) for m in MODELS}
    return dict(per_model=per, inner_total=len(design['inner_targets']), fits_started=fits_started(ledger),
                planned_fits=design['planned_fits'], max_fits=design['max_fits'], recovered_units=recovered_units(ledger),
                running=[k for k, e in units.items() if e['status'] == 'running'],
                selections={k: s['selected'] for k, s in ledger['selections'].items()})


# ---------------------------------------------------------------- finalize
def finalize_all(data, out):
    from experiments.geo_ablation.audit import turnover
    out = Path(out)
    X, y, ids, folds, design = verify(data, out)
    require(not (out/'analysis.json').exists(), 'analysis.json exists; refusing overwrite')
    c = design['config']
    ledger = read_ledger(out)
    tg = design['inner_targets'] + design['outer_targets']
    complete = sum(ledger['units'].get(k, {}).get('status') == 'complete' for k in tg)
    require(complete == len(tg) and len(ledger['units']) == len(tg), 'Ledger incomplete; not finalizing')
    for e in ledger['units'].values():
        check_saved(out, e)
    yv = y.to_numpy()
    cfg = dict(c, profiles=dict(geo1_regions=[26, 20, 13]))
    res = dict(study_id=STUDY_ID, design_sha256=sha256_file(out/'design.json'), code_commit=design['code_commit'],
               holdout_accessed=False, rules=c['rules'], candidates=c['candidates'], fixed_params=c['fixed_params'],
               note='Nested-CV evaluation of wider limited tuning for DT, RF, GBC (no weighting or sampling), compared '
                    'with the T37 tuned procedure. Descriptive labels only; no significance or causal claims; not '
                    'independent performance; no adoption.', models={})
    for m in MODELS:
        order = candidates(c, m)
        sel = {}
        for f in range(N_FOLDS):
            scores, _ = pooled_inner_scores(out, c, m, f)
            s = ledger['selections'][f'{m}/{f}']
            require(select(scores, order) == s['selected'] and scores == s['pooled_inner_macro_f1'],
                    f'Selection {m}/{f} not reproducible')
            sel[f] = s
        d = out/m/'outer'
        oof = pd.concat([read_pred(d/f'fold_{f}_pred.csv.gz') for f in range(N_FOLDS)])
        oof = oof.set_index('building_id').loc[ids.to_numpy()].reset_index()
        require(len(oof) == len(ids) and oof.building_id.is_unique, f'{m} OOF coverage changed')
        np.testing.assert_array_equal(oof.fold.to_numpy(), folds)
        np.testing.assert_array_equal(oof.y_true.to_numpy(), yv)
        oof = oof[['building_id', 'fold', 'y_true', 'y_pred'] + PROBS]
        write_pred(d/'oof.csv.gz', oof)
        backup_files(out, f'{m}/outer', ['oof.csv.gz'])
        new_s = T33.summarize(oof, X, cfg)
        block = dict(tuned=dict(new_s, oof_sha256=sha256_file(d/'oof.csv.gz')))
        for kind, rule in (('T37', 'J1'), ('A2', 'J1b')):
            spec = c['references'][kind][m]
            r = reference(Path(design['local_root'])/spec['path'], spec['sha256'], ids, y, folds)
            rs = T33.summarize(r, X, cfg)
            block[f'reference_{kind}'] = dict(rs, oof_sha256=spec['sha256'], path=spec['path'])
            block[rule] = T33.d1_label(new_s['fold_macro_f1'], rs['fold_macro_f1'], c['rules'][rule])
            block[f'safety_vs_{kind}_no_verdict'] = dict(
                recall3=dict(new=new_s['recall3'], reference=rs['recall3'], delta=new_s['recall3'] - rs['recall3']),
                error31=dict(new=new_s['error31'], reference=rs['error31'], delta=new_s['error31'] - rs['error31']),
                error32=dict(new=new_s['error32'], reference=rs['error32'], delta=new_s['error32'] - rs['error32']))
            block[f'turnover_{kind}_to_new'] = turnover(yv, r.y_pred.to_numpy(), oof.y_pred.to_numpy())
        block['J3_selection'] = [dict(
            outer_fold=f, selected=sel[f]['selected'], params=c['candidates'][m][sel[f]['selected']],
            pooled_inner_macro_f1=sel[f]['pooled_inner_macro_f1'],
            inner_rank=sorted(order, key=lambda k2: (-sel[f]['pooled_inner_macro_f1'][k2], order.index(k2))),
            selected_inner_minus_outer=sel[f]['pooled_inner_macro_f1'][sel[f]['selected']] - new_s['fold_macro_f1'][f],
            selected_inner_minus_anchor_inner=(sel[f]['pooled_inner_macro_f1'][sel[f]['selected']]
                                               - sel[f]['pooled_inner_macro_f1'][order[0]]))
            for f in range(N_FOLDS)]
        block['selection_frequency'] = {cand: sum(1 for r in block['J3_selection'] if r['selected'] == cand) for cand in order}
        units = {k: e for k, e in ledger['units'].items() if k.startswith(f'{m}/')}
        block['fits'] = dict(inner=sum(1 for k in units if '/I/' in k), outer_refits=sum(1 for k in units if '/O/' in k),
                             fit_seconds_sum=float(sum(e['fit_seconds'] for e in units.values())),
                             inner_fit_seconds_by_group={g: float(sum(ledger['units'][inner_key(m, f, k, g)]['fit_seconds']
                                                                      for f in range(N_FOLDS) for k in range(N_INNER)))
                                                         for g in fit_groups(c, m)})
        res['models'][m] = block
    started = fits_started(ledger)
    res['ledger'] = dict(units_complete=complete, units_total=len(tg), fits_started=started,
                         planned_fits=c['budget']['planned_fits'], max_fits=c['budget']['max_fits'],
                         recovered_units=recovered_units(ledger),
                         fit_seconds_sum=float(sum(e['fit_seconds'] for e in ledger['units'].values())))
    res['judgments'] = {m: dict(J1=res['models'][m]['J1']['label'], J1b=res['models'][m]['J1b']['label'],
                                J2='no verdict', J3='record only') for m in MODELS}
    res['finished_at_utc'] = now()
    write(out/'analysis.json', res)
    backup_files(out, '.', ['analysis.json'])
    log(out, 'analysis written: ' + json.dumps({m: dict(J1=res['models'][m]['J1']['label'],
                                                         d=round(res['models'][m]['J1']['mean_delta'], 6),
                                                         freq=res['models'][m]['selection_frequency']) for m in MODELS}))
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('action', choices=['preflight', 'prepare', 'fit-all', 'status', 'finalize'])
    p.add_argument('--data')
    p.add_argument('--out')
    p.add_argument('--preflight')
    p.add_argument('--code-commit')
    p.add_argument('--local-root')
    p.add_argument('--models', nargs='*')
    a = p.parse_args()
    if a.action == 'preflight':
        preflight(a.data, a.out, a.local_root)
    elif a.action == 'prepare':
        prepare(a.data, a.out, a.preflight, a.code_commit, a.local_root)
    elif a.action == 'fit-all':
        fit_all(a.data, a.out, a.models)
    elif a.action == 'status':
        print(json.dumps(status(a.out), indent=1))
    else:
        r = finalize_all(a.data, a.out)
        print(json.dumps(r['judgments'], indent=1))


if __name__ == '__main__':
    main()
