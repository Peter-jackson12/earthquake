"""T37: limited nested-CV tuning of the four required models (LR, DT, sklearn GBC, RF), no class weighting/sampling.

Outer: the saved stratified 5-fold `split_manifest.csv` (evaluation only).  Inner: stratified 3-fold inside each outer
training fold.  Per model, pre-registered candidates (the first one, '*' in the plan, is the A2 baseline setting) are
scored on the pooled inner validation predictions; the best (ties -> registered order) is refit on the whole outer
training fold and scored once on the outer validation fold.  When the baseline candidate is selected, the rows of that
outer fold in the audited local baseline OOF are reused (no fit).  Preprocessing is `preprocess.make_model_pipeline`
(v1.0.1), fitted on the training rows of each fit only.

Design: docs/T37_REQUIRED_MODELS_TUNING_PLAN.md.  Actions: preflight (4 control fits, no study fit), prepare (seal),
fit-all (resumable), status, finalize.  Holdout is never read.  All outputs are private (outside Git).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
import sys
import tempfile
import time
import traceback
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from preprocess import make_model_pipeline, sha256_file  # noqa: E402
from experiments.geo_group_cv import run as t5  # noqa: E402
from experiments.geo_group_cv.run import PROBS, require, score, write  # noqa: E402
from experiments.ey_pipeline_benchmark import run as T33  # noqa: E402  (read-only reuse of sealed helpers)

HERE = Path(__file__).resolve().parent
STUDY_ID = 'T37_required_models_tuning_v1'
N_FOLDS = 5
N_INNER = 3
MODELS = ['LR', 'DT', 'GBC', 'RF']
SOURCE = list(dict.fromkeys(t5.SOURCE + [
    'configs/data_contract.json', 'experiments/geo_ablation/audit.py',
    'experiments/ey_pipeline_benchmark/__init__.py', 'experiments/ey_pipeline_benchmark/features.py',
    'experiments/ey_pipeline_benchmark/run.py',
    'experiments/required_tuning/__init__.py', 'experiments/required_tuning/run.py',
    'experiments/required_tuning/config.json', 'tests/test_required_tuning.py',
    'docs/T37_REQUIRED_MODELS_TUNING_PLAN.md']))

now, log, event, atomic_write = T33.now, T33.log, T33.event, T33.atomic_write
read_pred, write_pred, read_ledger, check_saved, backup_files = (T33.read_pred, T33.write_pred, T33.read_ledger,
                                                                 T33.check_saved, T33.backup_files)
git, require_not_holdout, load_inputs, reference, run_lock = (T33.git, T33.require_not_holdout, T33.load_inputs,
                                                              T33.reference, T33.run_lock)
CONSTRUCTORS = dict(LR=LogisticRegression, DT=DecisionTreeClassifier, GBC=GradientBoostingClassifier,
                    RF=RandomForestClassifier)
REGISTERED_CANDIDATES = {
    'LR': [dict(C=c) for c in (1.0, 0.03, 0.1, 0.3, 3.0, 10.0)],
    'DT': [dict(max_depth=12, min_samples_leaf=5)] + [dict(max_depth=d, min_samples_leaf=l) for d in (8, 12, 16)
                                                      for l in (5, 20, 50) if (d, l) != (12, 5)],
    'GBC': [dict(n_estimators=100, max_depth=3), dict(n_estimators=300, max_depth=3), dict(n_estimators=100, max_depth=5),
            dict(n_estimators=300, max_depth=5)],
    'RF': [dict(max_features=0.1, min_samples_leaf=2)] + [dict(max_features=m, min_samples_leaf=l) for m in (0.1, 0.2, 0.3)
                                                          for l in (1, 2, 5) if (m, l) != (0.1, 2)],
}
REGISTERED_FIXED = {
    'LR': dict(solver='lbfgs', max_iter=1500, tol=0.0001, random_state=42),
    'DT': dict(random_state=42),
    'GBC': dict(learning_rate=0.1, max_features=None, random_state=42),
    'RF': dict(n_estimators=100, max_depth=20, random_state=42, n_jobs=4),
}


def environment():
    names = ['numpy', 'pandas', 'scipy', 'scikit-learn', 'joblib', 'threadpoolctl']
    return dict(python=platform.python_version(), **{n: importlib.metadata.version(n) for n in names})


def sources():
    return {p: sha256_file(ROOT/p) for p in SOURCE}


def config(path=HERE/'config.json'):
    c = json.loads(Path(path).read_text(encoding='utf-8'))
    require(c['study_id'] == STUDY_ID and c['n_folds'] == N_FOLDS and c['models'] == MODELS, 'Unplanned study/models')
    require(c['inner'] == dict(n_splits=3, shuffle=True, random_state=42, splitter='StratifiedKFold',
                               rows='outer training fold in building_id order'), 'Inner split changed')
    require(c['fixed_params'] == REGISTERED_FIXED, 'Fixed model settings changed')
    require({m: [v for v in c['candidates'][m].values()] for m in MODELS} == REGISTERED_CANDIDATES
            and all(list(c['candidates'][m]) == [f'{m}{i}' for i in range(len(REGISTERED_CANDIDATES[m]))] for m in MODELS),
            'Candidates changed')
    require(c['class_weighting'] is None and c['sampling'] is None, 'Weighting/sampling must be off')
    n_inner = sum(N_FOLDS * N_INNER * len(REGISTERED_CANDIDATES[m]) for m in MODELS)
    require(c['budget'] == dict(inner_fits=420, max_outer_refits=20, planned_fits_max=440, max_fits=460,
                                max_recovered_units=10, max_attempts_per_unit=2, automatic_retries=0, control_fits=4)
            and n_inner == 420, 'Unplanned fit budget')
    require(c['threads'] == dict(rf_n_jobs=4, other_models_threads=1, outer_parallel_jobs=1), 'Thread plan changed')
    require(c['selection'] == dict(metric='pooled inner Macro F1 over the 3 inner validation folds, labels [1,2,3]',
                                   tie_break='registered candidate order (baseline first)'), 'Selection rule changed')
    require(c['rules']['J1'] == dict(mean_delta=0.003, fold_wins_min=4, advantage='IMPROVED_DESCRIPTIVE',
                                     disadvantage='WORSENED_DESCRIPTIVE', otherwise='NOT_MET',
                                     delta='tuned outer fold Macro F1 minus baseline fold Macro F1 on the same fold'),
            'J1 rule changed')
    require(set(c['baselines']) == set(MODELS), 'Baseline registry changed')
    require(c['control'] == dict(tree_probability_atol=1e-9, lr_probability_atol=1e-6, lr_min_prediction_agreement=0.9999,
                                 fold=0), 'Control tolerance changed')
    require(c['checkpoint']['reload_probability_atol'] == 1e-12, 'Reload tolerance changed')
    require(not c['holdout_access'] and not c['threshold_or_decision_rule_change'] and not c['automatic_model_replacement'],
            'Unplanned run policy')
    require(c['environment'] == environment() or Path(path) != HERE/'config.json', 'Environment differs from pinned')
    return c


# ---------------------------------------------------------------- keys and pure rules
def candidates(c, m):
    return list(c['candidates'][m])


def baseline_candidate(c, m):
    return candidates(c, m)[0]


def inner_key(m, f, k, cand):
    return f'{m}/I/{f}/{k}/{cand}'


def outer_key(m, f):
    return f'{m}/O/{f}'


def inner_targets(c):
    return [(m, f, k, cand) for m in MODELS for f in range(N_FOLDS) for k in range(N_INNER) for cand in candidates(c, m)]


def model_params(c, m, cand):
    return dict(c['fixed_params'][m], **c['candidates'][m][cand])


def make_pipeline(c, m, cand):
    return make_model_pipeline(CONSTRUCTORS[m](**model_params(c, m, cand)), scale_numeric=(m == 'LR'))


def inner_split(y_outer_train, c):
    i = c['inner']
    skf = StratifiedKFold(n_splits=i['n_splits'], shuffle=i['shuffle'], random_state=i['random_state'])
    y = np.asarray(y_outer_train)
    return [(np.asarray(tr), np.asarray(va)) for tr, va in skf.split(np.zeros(len(y)), y)]


def select(pooled_scores, order):
    require(set(pooled_scores) == set(order), 'Selection needs a score for every candidate')
    best = max(pooled_scores[cand] for cand in order)
    return next(cand for cand in order if pooled_scores[cand] == best)


def ids_sha256(ids):
    return hashlib.sha256(json.dumps(sorted(int(x) for x in ids)).encode()).hexdigest()


def fits_started(ledger):
    return sum(e.get('attempts', 0) for e in ledger['units'].values())


def recovered_units(ledger):
    return sum(1 for e in ledger['units'].values() if e.get('attempts', 0) > 1)


# ---------------------------------------------------------------- fitting
def fit_pipeline(c, m, cand, X, y):
    pipe = make_pipeline(c, m, cand)
    threads = c['threads']['rf_n_jobs'] if m == 'RF' else c['threads']['other_models_threads']
    with threadpool_limits(limits=threads), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        t0 = time.perf_counter()
        pipe.fit(X, y)
        seconds = time.perf_counter() - t0
    if m == 'RF':
        # trained with n_jobs=4; predict sequentially so the tree-average summation order (and argmax on ties) is fixed
        pipe['model'].set_params(n_jobs=1)
    conv = sum(1 for w in caught if issubclass(w.category, ConvergenceWarning))
    other = [dict(category=w.category.__name__, message=str(w.message)[:200]) for w in caught
             if not issubclass(w.category, ConvergenceWarning)][:10]
    return pipe, seconds, conv, other


def checked_proba(pipe, X):
    require(np.array_equal(np.asarray(pipe.classes_), [1, 2, 3]), 'Class order must be 1,2,3')
    p = np.asarray(pipe.predict_proba(X), dtype=np.float64)
    require(p.shape == (len(X), 3) and np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all(), 'Invalid probabilities')
    np.testing.assert_allclose(p.sum(1), 1.0, atol=1e-6, rtol=0)
    return p


class Context:
    def __init__(self, X, y, ids, folds, out, design):
        self.X, self.y, self.ids, self.folds = X, y, ids, np.asarray(folds)
        self.out, self.design, self.c = Path(out), design, design['config']

    def positions(self, f, k=None):
        a, b = np.flatnonzero(self.folds != f), np.flatnonzero(self.folds == f)
        if k is None:
            return a, b
        tr, va = inner_split(self.y.to_numpy()[a], self.c)[k]
        return a[tr], a[va]


def begin_fit(ctx, key):
    out, cap = ctx.out, ctx.c['budget']
    ledger = read_ledger(out)
    entry = ledger['units'].get(key)
    require(fits_started(ledger) < cap['max_fits'], f'Fit budget exhausted ({cap["max_fits"]} fits)')
    prior = entry.get('attempts', 0) if entry else 0
    require(prior < cap['max_attempts_per_unit'], f'{key}: at most {cap["max_attempts_per_unit"]} attempts')
    if prior >= 1:
        require(recovered_units(ledger) < cap['max_recovered_units'], f'At most {cap["max_recovered_units"]} recovered units')
    entry = entry or dict(artifact_sha256={}, attempts=0)
    entry.update(attempts=prior + 1, status='running', started_at_utc=now())
    ledger['units'][key] = entry
    atomic_write(out/'ledger.json', ledger)
    event(out, unit=key, action='fit_start', attempt=prior + 1)
    return prior


def move_aside(d, names, prior):
    for name in names:
        if (d/name).exists():
            n = 0
            while (d/f'{name}.incomplete_attempt{prior}_{n}').exists():
                n += 1
            (d/name).rename(d/f'{name}.incomplete_attempt{prior}_{n}')


def fit_and_save(ctx, key, m, cand, a, b, d, stem, extra):
    """One study fit on rows a, prediction on rows b; save pipeline/pred/json, reload-verify, back up, ledger."""
    out, c = ctx.out, ctx.c
    names = [f'{stem}_model.joblib', f'{stem}_pred.csv.gz', f'{stem}.json']
    d.mkdir(parents=True, exist_ok=True)
    prior = begin_fit(ctx, key)
    move_aside(d, names, prior)
    require(not set(a) & set(b), f'{key}: train/valid overlap')
    X_tr, y_tr = ctx.X.iloc[a], ctx.y.iloc[a]
    X_va, y_va = ctx.X.iloc[b], ctx.y.to_numpy()[b]
    pipe, seconds, conv, other = fit_pipeline(c, m, cand, X_tr, y_tr)
    p = checked_proba(pipe, X_va)
    pred = p.argmax(1) + 1
    joblib.dump(pipe, d/names[0], compress=3)
    frame = pd.DataFrame({'building_id': ctx.ids.to_numpy()[b], 'y_true': y_va, 'y_pred': pred,
                          **{n: p[:, i] for i, n in enumerate(PROBS)}})
    for col, v in extra['pred_columns'].items():
        frame.insert(1, col, v)
    write_pred(d/names[1], frame)
    p2 = checked_proba(joblib.load(d/names[0]), X_va)
    diff = float(np.max(np.abs(p2 - p)))
    require(diff <= c['checkpoint']['reload_probability_atol'] and np.array_equal(p2.argmax(1) + 1, pred),
            f'{key}: model reload mismatch {diff}')
    back = read_pred(d/names[1])
    np.testing.assert_array_equal(back.building_id.to_numpy(), frame.building_id.to_numpy())
    np.testing.assert_array_equal(back[PROBS].to_numpy(), p)
    metrics = score(y_va, pred)
    record = dict(study_id=STUDY_ID, unit=key, model=m, candidate=cand, params=model_params(c, m, cand),
                  train_rows=len(a), valid_rows=len(b), train_ids_sha256=ids_sha256(ctx.ids.to_numpy()[a]),
                  valid_ids_sha256=ids_sha256(ctx.ids.to_numpy()[b]),
                  train_class_counts=[int((y_tr.to_numpy() == v).sum()) for v in (1, 2, 3)], metrics=metrics,
                  pred_share=(np.bincount(pred - 1, minlength=3) / len(pred)).tolist(), fit_seconds=seconds,
                  convergence_warnings=conv, other_warnings=other,
                  encoded_columns=int(pipe['preprocess'].transform(X_va.iloc[:2]).shape[1]),
                  reload_probability_max_abs_diff=diff, csv_round_trip_exact=True, finished_at_utc=now(),
                  **{k2: v for k2, v in extra.items() if k2 != 'pred_columns'})
    write(d/names[2], record)
    shas = {n: sha256_file(d/n) for n in names}
    rel = d.relative_to(out).as_posix()
    backup_files(out, rel, shas)
    ledger = read_ledger(out)
    entry = ledger['units'][key]
    entry.update(dir=rel, artifact_sha256=shas, status='complete', model=m, candidate=cand, macro_f1=metrics['macro_f1'],
                 fit_seconds=seconds, convergence_warnings=conv, finished_at_utc=now())
    atomic_write(out/'ledger.json', ledger)
    check_saved(out, entry)
    event(out, unit=key, action='unit_complete', macro_f1=metrics['macro_f1'], seconds=seconds, artifact_sha256=shas)
    log(out, f'{key} complete: F1={metrics["macro_f1"]:.6f} fit={seconds:.0f}s conv_warn={conv} reload_diff={diff:.3g}')
    return entry


def inner_unit(ctx, m, f, k, cand):
    key = inner_key(m, f, k, cand)
    entry = read_ledger(ctx.out)['units'].get(key)
    if entry and entry['status'] == 'complete':
        check_saved(ctx.out, entry)
        return entry
    a, b = ctx.positions(f, k)
    require(not set(b) & set(np.flatnonzero(ctx.folds == f)), f'{key}: outer validation row in the inner step')
    return fit_and_save(ctx, key, m, cand, a, b, ctx.out/m/'inner'/f'f{f}', f'f{f}_k{k}_{cand}',
                        dict(outer_fold=f, inner_fold=k, pred_columns={'inner_fold': k, 'outer_fold': f}))


def pooled_inner_scores(out, c, m, f):
    ledger = read_ledger(out)
    scores, n = {}, None
    for cand in candidates(c, m):
        parts = []
        for k in range(N_INNER):
            e = ledger['units'].get(inner_key(m, f, k, cand))
            require(e and e['status'] == 'complete', f'{inner_key(m, f, k, cand)} incomplete')
            check_saved(out, e)
            parts.append(read_pred(Path(out)/e['dir']/f'f{f}_k{k}_{cand}_pred.csv.gz'))
        pooled = pd.concat(parts)
        require(pooled.building_id.is_unique, 'Inner validation folds overlap')
        n = len(pooled) if n is None else n
        require(len(pooled) == n, 'Inner coverage differs between candidates')
        scores[cand] = score(pooled.y_true.to_numpy(), pooled.y_pred.to_numpy())['macro_f1']
    return scores, n


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


def baseline_rows(ctx, m, f):
    spec = ctx.c['baselines'][m]
    src = Path(ctx.design['local_root'])/spec['path']
    require_not_holdout(src)
    r = reference(src, spec['sha256'], ctx.ids, ctx.y, ctx.folds)
    return r[r.fold == f][['building_id', 'fold', 'y_true', 'y_pred'] + PROBS].reset_index(drop=True), src


def outer_unit(ctx, m, f):
    out, c = ctx.out, ctx.c
    key = outer_key(m, f)
    sel = selection(ctx, m, f)
    entry = read_ledger(out)['units'].get(key)
    if entry and entry['status'] == 'complete':
        check_saved(out, entry)
        require(entry['candidate'] == sel['selected'], 'Outer unit does not match the selection')
        return entry
    a, b = ctx.positions(f)
    d = out/m/'outer'
    cand = sel['selected']
    if cand != baseline_candidate(c, m):
        return fit_and_save(ctx, key, m, cand, a, b, d, f'fold_{f}',
                            dict(outer_fold=f, reused_baseline=False, pred_columns={'fold': f}))
    rows, src = baseline_rows(ctx, m, f)
    np.testing.assert_array_equal(rows.building_id.to_numpy(), ctx.ids.to_numpy()[b])
    np.testing.assert_array_equal(rows.y_true.to_numpy(), ctx.y.to_numpy()[b])
    d.mkdir(parents=True, exist_ok=True)
    names = [f'fold_{f}_pred.csv.gz', f'fold_{f}.json']
    move_aside(d, names, 0)
    write_pred(d/names[0], rows)
    back = read_pred(d/names[0])
    np.testing.assert_array_equal(back[PROBS].to_numpy(), rows[PROBS].to_numpy())
    metrics = score(rows.y_true.to_numpy(), rows.y_pred.to_numpy())
    write(d/names[1], dict(study_id=STUDY_ID, unit=key, model=m, candidate=cand, reused_baseline=True,
                           source=str(src), source_sha256=c['baselines'][m]['sha256'], metrics=metrics,
                           valid_ids_sha256=ids_sha256(ctx.ids.to_numpy()[b]), fit_seconds=0.0, finished_at_utc=now()))
    shas = {n: sha256_file(d/n) for n in names}
    rel = d.relative_to(out).as_posix()
    backup_files(out, rel, shas)
    ledger = read_ledger(out)
    ledger['units'][key] = dict(dir=rel, artifact_sha256=shas, attempts=0, status='complete', model=m, candidate=cand,
                                reused_baseline=True, macro_f1=metrics['macro_f1'], fit_seconds=0.0, finished_at_utc=now())
    atomic_write(out/'ledger.json', ledger)
    event(out, unit=key, action='baseline_reused', macro_f1=metrics['macro_f1'], artifact_sha256=shas)
    log(out, f'{key}: baseline {cand} selected, baseline OOF rows reused (no fit) F1={metrics["macro_f1"]:.6f}')
    return ledger['units'][key]


# ---------------------------------------------------------------- preflight
def baseline_check(c, local_root, ids, y, folds):
    res = {}
    for m in MODELS:
        spec = c['baselines'][m]
        r = reference(Path(local_root)/spec['path'], spec['sha256'], ids, y, folds)
        fo = r.fold.to_numpy()
        values = [score(r.y_true[fo == f], r.y_pred[fo == f])['macro_f1'] for f in range(N_FOLDS)]
        require(abs(float(np.mean(values)) - spec['cv_mean']) <= 1e-6, f'{m} baseline CV mean differs from the registry')
        res[m] = dict(sha256=spec['sha256'], fold_macro_f1=values, cv_mean=float(np.mean(values)))
    return res


def split_checks(y, ids, folds, c):
    out = []
    for f in range(N_FOLDS):
        a, b = np.flatnonzero(folds != f), np.flatnonzero(folds == f)
        parts = inner_split(y.to_numpy()[a], c)
        allv = np.concatenate([va for _, va in parts])
        require(len(allv) == len(a) and len(set(allv)) == len(a), f'Inner folds of {f} are not a partition')
        for tr, va in parts:
            require(not set(tr) & set(va) and len(tr) + len(va) == len(a), f'Inner overlap in fold {f}')
            require(not set(a[va]) & set(b) and not set(a[tr]) & set(b), 'Outer validation row inside the inner step')
        out.append(dict(outer_fold=f, inner_valid_rows=[len(va) for _, va in parts],
                        inner_valid_ids_sha256=[ids_sha256(ids.to_numpy()[a[va]]) for _, va in parts]))
    return out


def control_fits(c, X, y, ids, folds, local_root):
    """Baseline candidate of every model refit on outer fold 0 and compared to the baseline OOF (4 control fits)."""
    ctl = c['control']
    f = ctl['fold']
    a, b = np.flatnonzero(folds != f), np.flatnonzero(folds == f)
    res = {}
    for m in MODELS:
        spec = c['baselines'][m]
        r = reference(Path(local_root)/spec['path'], spec['sha256'], ids, y, folds)
        ref = r[r.fold == f][PROBS].to_numpy()
        pipe, seconds, conv, _ = fit_pipeline(c, m, baseline_candidate(c, m), X.iloc[a], y.iloc[a])
        p = checked_proba(pipe, X.iloc[b])
        diff = float(np.max(np.abs(p - ref)))
        mism = p.argmax(1) + 1 != r[r.fold == f].y_pred.to_numpy()
        agree = float(1 - mism.mean())
        top2 = np.sort(ref, axis=1)[:, -2:]
        tie_only = bool(np.all(top2[mism, 1] - top2[mism, 0] <= ctl['tree_probability_atol']))
        if m == 'LR':
            ok = diff <= ctl['lr_probability_atol'] and agree >= ctl['lr_min_prediction_agreement']
        else:
            ok = diff <= ctl['tree_probability_atol'] and tie_only   # label differences allowed only on exact ties
        res[m] = dict(max_abs_probability_diff=diff, prediction_agreement=agree, mismatched_rows=int(mism.sum()),
                      mismatches_only_on_ties=tie_only, fit_seconds=seconds, convergence_warnings=conv, passes=bool(ok))
    return dict(models=res, control_fits=len(MODELS), counted_as_study_fit=False)


def probe(c, X_small, y_small):
    """Small fits for every candidate (NOT study fits): joblib save/reload exactness and repeat determinism."""
    res = {}
    small = json.loads(json.dumps(c))
    small['fixed_params']['GBC'] = dict(small['fixed_params']['GBC'])
    small['fixed_params']['RF'] = dict(small['fixed_params']['RF'], n_estimators=10)
    for m in MODELS:
        for cand in candidates(c, m):
            sc = json.loads(json.dumps(small))
            if m == 'GBC':
                sc['candidates']['GBC'][cand] = dict(sc['candidates']['GBC'][cand], n_estimators=10)
            p1 = checked_proba(fit_pipeline(sc, m, cand, X_small, y_small)[0], X_small)
            pipe, _, _, _ = fit_pipeline(sc, m, cand, X_small, y_small)
            p2 = checked_proba(pipe, X_small)
            with tempfile.TemporaryDirectory() as td:
                joblib.dump(pipe, Path(td)/'m.joblib', compress=3)
                pr = checked_proba(joblib.load(Path(td)/'m.joblib'), X_small)
            diff = float(np.max(np.abs(pr - p2)))
            rep = float(np.max(np.abs(p1 - p2)))
            tol = c['checkpoint']['reload_probability_atol']
            res[f'{m}/{cand}'] = dict(repeat_max_abs_diff=rep, repeat_within_tolerance=bool(rep <= tol),
                                      reload_max_abs_diff=diff, reload_within_tolerance=bool(diff <= tol))
    return dict(candidates=res, counted_as_study_fit=False, probe_fits=2 * len(res),
                note='first 2,000 development rows of outer-fold-0 training rows, small trees; outside the budget')


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
    res = dict(study_id=STUDY_ID, kind='preflight; no study fit (4 control fits)', classifiers_fitted_study=0,
               holdout_accessed=False, created_at_utc=now(), environment=environment(), data_contract=contract,
               plan_sha256=plan_sha, source_sha256=sources())
    res['inputs'] = input_checks(X, y, ids, folds, contract, c)
    res['splits'] = split_checks(y, ids, folds, c)
    res['baselines'] = baseline_check(c, local_root, ids, y, folds)
    a0 = np.flatnonzero(folds != 0)[:2000]
    res['model_probe'] = probe(c, X.iloc[a0], y.iloc[a0])
    res['control'] = control_fits(c, X, y, ids, folds, local_root)
    ok = (all(v['passes'] for v in res['control']['models'].values())
          and all(v['repeat_within_tolerance'] and v['reload_within_tolerance'] for v in res['model_probe']['candidates'].values()))
    res['status'] = 'PASS' if ok else 'FAIL'
    write(dest/'preflight.json', res)
    print(json.dumps({k: res[k] for k in ('status', 'control')}, indent=1, default=str))
    require(ok, 'Preflight failed')
    return res


def input_checks(X, y, ids, folds, contract, c):
    sp = c['split']
    require(sp['manifest_sha256'] == contract['files']['split_manifest.csv']
            and sp['development_sha256'] == contract['files']['development.csv'], 'Registered split/development SHA differs')
    tr = [int((folds != f).sum()) for f in range(N_FOLDS)]
    va = [int((folds == f).sum()) for f in range(N_FOLDS)]
    require(all(s == sp['train_rows_per_fold'] for s in tr) and all(s == sp['valid_rows_per_fold'] for s in va),
            f'Fold sizes differ: {tr} {va}')
    require(ids.is_unique and len(X) == len(y) == len(ids) == 208480, 'Development rows changed')
    require(list(X.columns) == list(X.columns.unique()) and X.shape[1] == 38, 'Expected the 38 raw inputs')
    return dict(fold_train_rows=tr, fold_valid_rows=va, rows=int(len(X)), raw_inputs=int(X.shape[1]))


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
    baselines = baseline_check(c, local_root, ids, y, folds)
    out.mkdir(parents=True)
    (out/'backup').mkdir()
    design = dict(study_id=STUDY_ID, config=c, code_commit=head, source_sha256=sources(), plan_sha256=plan_sha,
                  local_root=str(local_root.resolve()), data_contract=contract, inputs=pf['inputs'],
                  environment=environment(), baselines=baselines, splits=pf['splits'],
                  preflight=dict(path=str(Path(preflight_path).resolve()), sha256=sha256_file(preflight_path),
                                 status=pf['status'], control=pf['control']),
                  inner_targets=[inner_key(*t) for t in inner_targets(c)],
                  outer_targets=[outer_key(m, f) for m in MODELS for f in range(N_FOLDS)],
                  max_fits=c['budget']['max_fits'], planned_fits_max=c['budget']['planned_fits_max'],
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
                    for cand in candidates(c, m):
                        _guard(out, inner_key(m, f, k, cand), inner_unit, ctx, m, f, k, cand)
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
                planned_fits_max=design['planned_fits_max'], max_fits=design['max_fits'],
                recovered_units=recovered_units(ledger), running=[k for k, e in units.items() if e['status'] == 'running'],
                selections={k: s['selected'] for k, s in ledger['selections'].items()})


# ---------------------------------------------------------------- finalize (design 5)
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
    res = dict(study_id=STUDY_ID, design_sha256=sha256_file(out/'design.json'), code_commit=design['code_commit'],
               holdout_accessed=False, rules=c['rules'], candidates=c['candidates'], fixed_params=c['fixed_params'],
               note='Nested-CV evaluation of limited tuning procedures for the four required models, no weighting or '
                    'sampling. Descriptive labels only; no significance or causal claims; not independent performance; '
                    'no adoption (design 5, 6).', models={})
    for m in MODELS:
        sel = {}
        for f in range(N_FOLDS):
            scores, _ = pooled_inner_scores(out, c, m, f)
            s = ledger['selections'][f'{m}/{f}']
            require(select(scores, candidates(c, m)) == s['selected'] and scores == s['pooled_inner_macro_f1'],
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
        spec = c['baselines'][m]
        base = reference(Path(design['local_root'])/spec['path'], spec['sha256'], ids, y, folds)
        cfg = dict(c, profiles=dict(geo1_regions=[26, 20, 13]))
        tuned_s, base_s = T33.summarize(oof, X, cfg), T33.summarize(base, X, cfg)
        j1 = T33.d1_label(tuned_s['fold_macro_f1'], base_s['fold_macro_f1'], c['rules']['J1'])
        j3 = []
        for f in range(N_FOLDS):
            s = sel[f]
            order = candidates(c, m)
            j3.append(dict(outer_fold=f, selected=s['selected'], params=c['candidates'][m][s['selected']],
                           pooled_inner_macro_f1=s['pooled_inner_macro_f1'],
                           inner_rank=sorted(order, key=lambda k2: (-s['pooled_inner_macro_f1'][k2], order.index(k2))),
                           selected_inner_minus_outer=s['pooled_inner_macro_f1'][s['selected']] - tuned_s['fold_macro_f1'][f],
                           selected_inner_minus_baseline_inner=(s['pooled_inner_macro_f1'][s['selected']]
                                                                - s['pooled_inner_macro_f1'][order[0]]),
                           reused_baseline=ledger['units'][outer_key(m, f)].get('reused_baseline', False)))
        units = {k: e for k, e in ledger['units'].items() if k.startswith(f'{m}/')}
        res['models'][m] = dict(
            tuned=dict(tuned_s, oof_sha256=sha256_file(d/'oof.csv.gz')),
            baseline=dict(base_s, oof_sha256=spec['sha256'], path=spec['path']),
            J1=j1,
            J2_safety_no_verdict=dict(
                recall3=dict(tuned=tuned_s['recall3'], baseline=base_s['recall3'], delta=tuned_s['recall3'] - base_s['recall3']),
                error31=dict(tuned=tuned_s['error31'], baseline=base_s['error31'], delta=tuned_s['error31'] - base_s['error31']),
                error32=dict(tuned=tuned_s['error32'], baseline=base_s['error32'], delta=tuned_s['error32'] - base_s['error32'])),
            turnover_baseline_to_tuned=turnover(yv, base.y_pred.to_numpy(), oof.y_pred.to_numpy()),
            J3_selection=j3,
            selection_frequency={cand: sum(1 for r in j3 if r['selected'] == cand) for cand in candidates(c, m)},
            fits=dict(inner=sum(1 for k in units if '/I/' in k),
                      outer_refits=sum(1 for f in range(N_FOLDS) if not ledger['units'][outer_key(m, f)].get('reused_baseline')),
                      fit_seconds_sum=float(sum(e['fit_seconds'] for e in units.values())),
                      inner_fit_seconds_by_candidate={cand: float(sum(ledger['units'][inner_key(m, f, k, cand)]['fit_seconds']
                                                                      for f in range(N_FOLDS) for k in range(N_INNER)))
                                                      for cand in candidates(c, m)},
                      convergence_warnings=int(sum(e.get('convergence_warnings', 0) for e in units.values()))))
    started = fits_started(ledger)
    res['ledger'] = dict(units_complete=complete, units_total=len(tg), fits_started=started,
                         planned_fits_max=c['budget']['planned_fits_max'], max_fits=c['budget']['max_fits'],
                         recovered_units=recovered_units(ledger),
                         fit_seconds_sum=float(sum(e['fit_seconds'] for e in ledger['units'].values())))
    res['judgments'] = {m: dict(J1=res['models'][m]['J1']['label'], J2='no verdict', J3='record only') for m in MODELS}
    res['finished_at_utc'] = now()
    write(out/'analysis.json', res)
    backup_files(out, '.', ['analysis.json'])
    log(out, 'analysis written: ' + json.dumps({m: dict(J1=res['models'][m]['J1']['label'],
                                                         mean_delta=round(res['models'][m]['J1']['mean_delta'], 6),
                                                         freq=res['models'][m]['selection_frequency']) for m in MODELS}))
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('action', choices=['preflight', 'prepare', 'fit-all', 'status', 'finalize'])
    p.add_argument('--data', help='absolute path of earthquake/data (development.csv + split_manifest.csv)')
    p.add_argument('--out', help='preflight: new destination folder; others: the run folder')
    p.add_argument('--preflight', help='prepare: path of a PASS preflight.json')
    p.add_argument('--code-commit', help='optional; must equal HEAD')
    p.add_argument('--local-root', help='local_reproduction root (default: <out>/../..)')
    p.add_argument('--models', nargs='*', help='fit-all: restrict to models')
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
