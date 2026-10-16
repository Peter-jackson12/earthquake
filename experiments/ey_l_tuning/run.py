"""T36: limited nested-CV tuning of the T35 L pipeline (EY revised notebook, final LightGBM setting).

Outer: the saved stratified 5-fold `split_manifest.csv` (evaluation only).  Inner: stratified 3-fold inside each outer
training fold; six pre-registered candidates (num_leaves x n_estimators, C0 = the T35 L setting) are scored on the pooled
inner validation predictions, the best one (ties -> registered order, C0 first) is refit on the whole outer training fold
and scored once on the outer validation fold.  When C0 is selected the audited T35 L fold prediction is reused (no fit).
Features come from the sealed T35 module `experiments/ey_pipeline_v2/features.py` (import only), rebuilt inside every
inner fold so no inner-validation or outer-validation label reaches the statistics.

Design: docs/T36_EY_L_NESTED_TUNING_PLAN.md.  Actions: preflight (no study fit), prepare (seal), fit-all (resumable),
status, finalize.  Holdout is never read.  All outputs are private (outside Git).  Budget: <= 95 fits, cap 105, no retry.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from preprocess import sha256_file  # noqa: E402
from experiments.geo_group_cv.run import PROBS, require, score, write  # noqa: E402
from experiments.ey_pipeline_benchmark import run as T33  # noqa: E402  (read-only reuse of sealed helpers)
from experiments.ey_pipeline_benchmark.features import notebook_frame  # noqa: E402
from experiments.ey_pipeline_v2 import features as F  # noqa: E402  (sealed T35 features, import only)
from experiments.ey_pipeline_v2 import run as T35  # noqa: E402  (sealed T35 helpers, import only)

HERE = Path(__file__).resolve().parent
STUDY_ID = 'T36_ey_l_nested_tuning_v1'
N_FOLDS = 5
N_INNER = 3
CANDIDATES = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']
SOURCE = list(dict.fromkeys(T35.SOURCE + [
    'experiments/ey_l_tuning/__init__.py', 'experiments/ey_l_tuning/run.py', 'experiments/ey_l_tuning/config.json',
    'tests/test_ey_l_tuning.py', 'docs/T36_EY_L_NESTED_TUNING_PLAN.md']))

now, log, event, atomic_write = T33.now, T33.log, T33.event, T33.atomic_write
read_pred, write_pred, read_ledger, check_saved, backup_files = (T33.read_pred, T33.write_pred, T33.read_ledger,
                                                                 T33.check_saved, T33.backup_files)
git, require_not_holdout, load_inputs, reference, run_lock = (T33.git, T33.require_not_holdout, T33.load_inputs,
                                                              T33.reference, T33.run_lock)
environment, checked_proba, jsonable_params = T35.environment, T35.checked_proba, T35.jsonable_params


def sources():
    return {p: sha256_file(ROOT/p) for p in SOURCE}


def config(path=HERE/'config.json'):
    c = json.loads(Path(path).read_text(encoding='utf-8'))
    require(c['study_id'] == STUDY_ID and c['n_folds'] == N_FOLDS and c['inner']['n_splits'] == N_INNER,
            'Unplanned study/folds')
    require(c['inner'] == dict(n_splits=3, shuffle=True, random_state=42, splitter='StratifiedKFold',
                               rows='outer training fold in building_id order'), 'Inner split changed')
    require(c['fixed_lgbm_params'] == dict(random_state=42, n_jobs=4, deterministic=True, verbose=-1, learning_rate=0.05,
                                           min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.5),
            'Fixed LightGBM setting changed')
    require(c['candidates'] == {'C0': dict(num_leaves=127, n_estimators=800), 'C1': dict(num_leaves=63, n_estimators=800),
                                'C2': dict(num_leaves=255, n_estimators=800), 'C3': dict(num_leaves=63, n_estimators=1600),
                                'C4': dict(num_leaves=127, n_estimators=1600), 'C5': dict(num_leaves=255, n_estimators=1600)}
            and c['candidate_order'] == CANDIDATES and c['baseline_candidate'] == 'C0', 'Candidates changed')
    require(c['weights'] == dict(strength=0.5), 'Weights changed')
    require(c['budget'] == dict(inner_fits=90, max_outer_refits=5, planned_fits_max=95, max_fits=105,
                                max_recovered_units=5, max_attempts_per_unit=2, automatic_retries=0), 'Unplanned fit budget')
    require(N_FOLDS * N_INNER * len(CANDIDATES) == c['budget']['inner_fits']
            and c['budget']['inner_fits'] + N_FOLDS == c['budget']['planned_fits_max'], 'Budget arithmetic')
    require(c['threads'] == dict(model_n_jobs=4, outer_parallel_jobs=1, feature_building_threads=1), 'Thread plan changed')
    require(c['selection'] == dict(metric='pooled inner Macro F1 over the 3 inner validation folds, labels [1,2,3]',
                                   tie_break='candidate_order (C0 first)'), 'Selection rule changed')
    require(c['rules']['J1'] == dict(mean_delta=0.003, fold_wins_min=4, advantage='IMPROVED_DESCRIPTIVE',
                                     disadvantage='WORSENED_DESCRIPTIVE', otherwise='NOT_MET',
                                     delta='tuned outer fold Macro F1 minus T35 L fold Macro F1 on the same fold'),
            'J1 rule changed')
    b = c['baseline']
    require(b['oof_sha256'] == '0ea83e83edf697e9440dbf3230675ff180afda8774f7410ee496ab362604fabc'
            and len(b['fold_pred_sha256']) == N_FOLDS and len(b['fold_matrix_sha256']) == N_FOLDS, 'Baseline registry changed')
    require(c['checkpoint']['reload_probability_atol'] == 1e-12, 'Reload tolerance changed')
    require(not c['holdout_access'] and not c['threshold_or_decision_rule_change'] and not c['automatic_model_replacement'],
            'Unplanned run policy')
    require(c['environment'] == environment() or Path(path) != HERE/'config.json', 'Environment differs from pinned')
    return c


# ---------------------------------------------------------------- keys and pure rules
def inner_key(f, k, cand):
    return f'I/{f}/{k}/{cand}'


def outer_key(f):
    return f'O/{f}'


def inner_targets():
    return [(f, k, cand) for f in range(N_FOLDS) for k in range(N_INNER) for cand in CANDIDATES]


def model_params(c, cand):
    return dict(c['fixed_lgbm_params'], **c['candidates'][cand])


def inner_split(y_outer_train, c):
    """Positions (within the outer training fold, building_id order) of the three inner folds."""
    i = c['inner']
    skf = StratifiedKFold(n_splits=i['n_splits'], shuffle=i['shuffle'], random_state=i['random_state'])
    y = np.asarray(y_outer_train)
    return [(np.asarray(tr), np.asarray(va)) for tr, va in skf.split(np.zeros(len(y)), y)]


def select(pooled_scores, order=CANDIDATES):
    """Highest pooled inner Macro F1; exact ties go to the earliest candidate in the registered order (C0 first)."""
    require(set(pooled_scores) == set(order), 'Selection needs a score for every candidate')
    best = max(pooled_scores[cand] for cand in order)
    return next(cand for cand in order if pooled_scores[cand] == best)


def ids_sha256(ids):
    return hashlib.sha256(json.dumps(sorted(int(x) for x in ids)).encode()).hexdigest()


def fits_started(ledger):
    return sum(e.get('attempts', 0) for e in ledger['units'].values())


def recovered_units(ledger):
    return sum(1 for e in ledger['units'].values() if e.get('attempts', 0) > 1)


# ---------------------------------------------------------------- model helpers
def fit_model(c, cand, X, y0, w):
    from lightgbm import LGBMClassifier
    model = LGBMClassifier(**model_params(c, cand))
    with threadpool_limits(limits=c['threads']['model_n_jobs']):
        t0 = time.perf_counter()
        model.fit(X, y0, sample_weight=w)
        seconds = time.perf_counter() - t0
    return model, seconds


def reload_proba(path, X, n_estimators, threads):
    """Re-inference from the saved LightGBM text model only (no pickle)."""
    import lightgbm
    b = lightgbm.Booster(model_file=str(path))
    require(b.current_iteration() == n_estimators and b.num_trees() == 3 * n_estimators,
            f'Saved model is not {n_estimators} iterations')
    return np.asarray(b.predict(X.to_numpy(dtype=np.float64), num_threads=threads), dtype=np.float64)


# ---------------------------------------------------------------- matrices
def write_matrix(d, stem, X_va, valid_ids, train_sha, diagnostics, extra):
    np.savez_compressed(d/f'{stem}_valid_matrix.npz', values=X_va.to_numpy(dtype=np.float64),
                        columns=np.array([str(x) for x in X_va.columns], dtype=str),
                        building_id=np.asarray(valid_ids, dtype=np.int64))
    write(d/f'{stem}_features.json', dict(columns=[str(x) for x in X_va.columns], rows=int(len(X_va)),
                                          valid_matrix_sha256=F.matrix_sha256(X_va), train_matrix_sha256=train_sha,
                                          sha_convention='ey_pipeline_v2.features.matrix_sha256', diagnostics=diagnostics,
                                          **extra))


def read_matrix(d, stem):
    z = np.load(Path(d)/f'{stem}_valid_matrix.npz', allow_pickle=False)
    return pd.DataFrame(z['values'], columns=z['columns'].tolist()), z['building_id']


class Context:
    def __init__(self, X, y, ids, folds, out, design):
        self.X, self.y, self.ids, self.folds = X, y, ids, np.asarray(folds)
        self.out, self.design, self.c = Path(out), design, design['config']
        self.frame = notebook_frame(X, y)
        self._m = {}

    def positions(self, f, k=None):
        """Outer (k None): (train, valid) positions in the full frame.  Inner: (inner train, inner valid) positions."""
        a, b = np.flatnonzero(self.folds != f), np.flatnonzero(self.folds == f)
        if k is None:
            return a, b
        tr, va = inner_split(self.y.to_numpy()[a], self.c)[k]
        return a[tr], a[va]

    def build(self, f, k=None):
        """Matrices for outer fold f (k None) or inner fold (f, k); persisted and reload-verified on first build."""
        name = f'outer/{f}' if k is None else f'inner/{f}/{k}'
        if name in self._m:
            return self._m[name]
        self._m = {}
        a, b = self.positions(f, k)
        require(not set(a) & set(b), f'{name}: train/valid overlap')
        if k is not None:
            require(not set(b) & set(np.flatnonzero(self.folds == f)), f'{name}: outer validation row in the inner step')
        train = self.frame.iloc[a].reset_index(drop=True)
        valid = self.frame.iloc[b].drop(columns=[F.GRADE]).reset_index(drop=True)
        t0 = time.perf_counter()
        with threadpool_limits(limits=self.c['threads']['feature_building_threads']):
            X_tr, X_va, _, _, diag = F.fold_matrices(train, valid)
        diag['build_seconds'] = time.perf_counter() - t0
        m = dict(a=a, b=b, train=X_tr, valid=X_va, train_sha=F.matrix_sha256(X_tr), valid_sha=F.matrix_sha256(X_va),
                 y0=train[F.GRADE].to_numpy() - 1, diagnostics=diag)
        if k is None:
            reg = self.c['baseline']['fold_matrix_sha256'][str(f)]
            require(m['train_sha'] == reg['train'] and m['valid_sha'] == reg['valid'],
                    f'Outer fold {f} matrices differ from the T35 ledger')
        stem = f'outer_{f}' if k is None else f'inner_{f}_{k}'
        files = [f'{stem}_valid_matrix.npz', f'{stem}_features.json']
        ledger = read_ledger(self.out)
        entry = ledger['features'].get(name)
        if entry:
            require(entry['train_matrix_sha256'] == m['train_sha'] and entry['valid_matrix_sha256'] == m['valid_sha'],
                    f'{name} matrices are not reproducible')
            check_saved(self.out, entry)
        else:
            d = self.out/'shared'
            d.mkdir(parents=True, exist_ok=True)
            for fn in files:
                if (d/fn).exists():
                    n = 0
                    while (d/f'{fn}.incomplete_attempt{n}').exists():
                        n += 1
                    (d/fn).rename(d/f'{fn}.incomplete_attempt{n}')
            vids = self.ids.to_numpy()[b]
            extra = dict(outer_fold=f, inner_fold=k, train_ids_sha256=ids_sha256(self.ids.to_numpy()[a]),
                         valid_ids_sha256=ids_sha256(vids))
            write_matrix(d, stem, X_va, vids, m['train_sha'], diag, extra)
            back, back_ids = read_matrix(d, stem)
            require(F.matrix_sha256(back) == m['valid_sha'] and np.array_equal(back_ids, vids), 'Stored matrix differs on reload')
            shas = {fn: sha256_file(d/fn) for fn in files}
            backup_files(self.out, 'shared', shas)
            ledger = read_ledger(self.out)
            ledger['features'][name] = dict(dir='shared', artifact_sha256=shas, train_matrix_sha256=m['train_sha'],
                                            valid_matrix_sha256=m['valid_sha'], train_rows=len(a), valid_rows=len(b),
                                            valid_ids_sha256=extra['valid_ids_sha256'], diagnostics=diag,
                                            created_at_utc=now())
            atomic_write(self.out/'ledger.json', ledger)
            event(self.out, action='matrices_saved', name=name, artifact_sha256=shas,
                  train_matrix_sha256=m['train_sha'], valid_matrix_sha256=m['valid_sha'])
            log(self.out, f'{name} matrices built in {diag["build_seconds"]:.0f}s ({X_tr.shape[1]} columns), saved+verified')
        m['stem'] = stem
        self._m[name] = m
        return m


# ---------------------------------------------------------------- fits
def begin_fit(ctx, k):
    out, cap = ctx.out, ctx.c['budget']
    ledger = read_ledger(out)
    entry = ledger['units'].get(k)
    require(fits_started(ledger) < cap['max_fits'], f'Fit budget exhausted ({cap["max_fits"]} fits)')
    prior = entry.get('attempts', 0) if entry else 0
    require(prior < cap['max_attempts_per_unit'], f'{k}: at most {cap["max_attempts_per_unit"]} attempts')
    if prior >= 1:
        require(recovered_units(ledger) < cap['max_recovered_units'], f'At most {cap["max_recovered_units"]} recovered units')
    entry = entry or dict(artifact_sha256={}, attempts=0)
    entry.update(attempts=prior + 1, status='running', started_at_utc=now())
    ledger['units'][k] = entry
    atomic_write(out/'ledger.json', ledger)
    event(out, unit=k, action='fit_start', attempt=prior + 1)
    return prior


def move_aside(d, names, prior):
    for name in names:
        if (d/name).exists():
            n = 0
            while (d/f'{name}.incomplete_attempt{prior}_{n}').exists():
                n += 1
            (d/name).rename(d/f'{name}.incomplete_attempt{prior}_{n}')


def fit_and_save(ctx, k, m, cand, d, stem, record_extra):
    """One study fit: train on m['train'], predict m['valid'], save model/pred/json, reload-verify, back up, ledger."""
    out, c = ctx.out, ctx.c
    names = [f'{stem}_model.txt', f'{stem}_pred.csv.gz', f'{stem}.json']
    d.mkdir(parents=True, exist_ok=True)
    prior = begin_fit(ctx, k)
    move_aside(d, names, prior)
    w = F.class_weights(pd.Series(m['y0']), c['weights']['strength'])
    model, seconds = fit_model(c, cand, m['train'], m['y0'], w)
    p = checked_proba(model, m['valid'])
    pred = p.argmax(1) + 1
    y_true = ctx.y.to_numpy()[m['b']]
    model.booster_.save_model(str(d/names[0]))
    frame = pd.DataFrame({'building_id': ctx.ids.to_numpy()[m['b']], 'y_true': y_true, 'y_pred': pred,
                          **{n: p[:, i] for i, n in enumerate(PROBS)}})
    for col, v in record_extra['pred_columns'].items():
        frame.insert(1, col, v)
    write_pred(d/names[1], frame)
    stored, stored_ids = read_matrix(out/'shared', m['stem'])
    require(F.matrix_sha256(stored) == m['valid_sha'] and np.array_equal(stored_ids, frame.building_id.to_numpy()),
            'Stored validation matrix differs from the one used')
    n_est = c['candidates'][cand]['n_estimators']
    p2 = reload_proba(d/names[0], stored, n_est, c['threads']['model_n_jobs'])
    diff = float(np.max(np.abs(p2 - p)))
    require(diff <= c['checkpoint']['reload_probability_atol'] and np.array_equal(p2.argmax(1) + 1, pred),
            f'{k}: model reload mismatch {diff}')
    back = read_pred(d/names[1])
    np.testing.assert_array_equal(back.building_id.to_numpy(), frame.building_id.to_numpy())
    np.testing.assert_array_equal(back[PROBS].to_numpy(), p)
    metrics = score(y_true, pred)
    record = dict(study_id=STUDY_ID, unit=k, candidate=cand, params=model_params(c, cand), train_rows=len(m['a']),
                  valid_rows=len(m['b']), train_class_counts=[int((m['y0'] == v).sum()) for v in (0, 1, 2)],
                  metrics=metrics, pred_share=(np.bincount(pred - 1, minlength=3) / len(pred)).tolist(),
                  matrices=dict(stem=m['stem'], train_matrix_sha256=m['train_sha'], valid_matrix_sha256=m['valid_sha'],
                                columns=int(m['train'].shape[1])),
                  fit_seconds=seconds, effective_parameters=jsonable_params(model.get_params()),
                  reload_probability_max_abs_diff=diff, csv_round_trip_exact=True, finished_at_utc=now(),
                  **{a: b for a, b in record_extra.items() if a != 'pred_columns'})
    write(d/names[2], record)
    shas = {n: sha256_file(d/n) for n in names}
    rel = d.relative_to(out).as_posix()
    backup_files(out, rel, shas)
    ledger = read_ledger(out)
    entry = ledger['units'][k]
    entry.update(dir=rel, artifact_sha256=shas, status='complete', candidate=cand, macro_f1=metrics['macro_f1'],
                 fit_seconds=seconds, finished_at_utc=now())
    atomic_write(out/'ledger.json', ledger)
    check_saved(out, entry)
    event(out, unit=k, action='unit_complete', macro_f1=metrics['macro_f1'], seconds=seconds, artifact_sha256=shas)
    log(out, f'{k} ({cand}) complete: F1={metrics["macro_f1"]:.6f} fit={seconds:.0f}s reload_diff={diff:.3g}')
    return entry


def inner_unit(ctx, f, k, cand):
    key = inner_key(f, k, cand)
    entry = read_ledger(ctx.out)['units'].get(key)
    if entry and entry['status'] == 'complete':
        check_saved(ctx.out, entry)
        return entry
    m = ctx.build(f, k)
    return fit_and_save(ctx, key, m, cand, ctx.out/'inner'/f'f{f}', f'f{f}_k{k}_{cand}',
                        dict(outer_fold=f, inner_fold=k, pred_columns={'inner_fold': k, 'outer_fold': f}))


def pooled_inner_scores(out, f):
    """Pooled inner Macro F1 per candidate for outer fold f, from the saved inner predictions only."""
    ledger = read_ledger(out)
    scores, n = {}, None
    for cand in CANDIDATES:
        parts = []
        for k in range(N_INNER):
            e = ledger['units'].get(inner_key(f, k, cand))
            require(e and e['status'] == 'complete', f'{inner_key(f, k, cand)} incomplete')
            check_saved(out, e)
            parts.append(read_pred(Path(out)/e['dir']/f'f{f}_k{k}_{cand}_pred.csv.gz'))
        pooled = pd.concat(parts)
        require(pooled.building_id.is_unique, 'Inner validation folds overlap')
        n = len(pooled) if n is None else n
        require(len(pooled) == n, 'Inner coverage differs between candidates')
        scores[cand] = score(pooled.y_true.to_numpy(), pooled.y_pred.to_numpy())['macro_f1']
    return scores, n


def selection(ctx, f):
    out = ctx.out
    scores, n = pooled_inner_scores(out, f)
    require(n == len(ctx.positions(f)[0]), 'Pooled inner rows must equal the outer training fold')
    chosen = select(scores)
    ledger = read_ledger(out)
    prev = ledger['selections'].get(str(f))
    if prev:
        require(prev['selected'] == chosen and prev['pooled_inner_macro_f1'] == scores, f'Selection of fold {f} changed')
        return prev
    rec = dict(outer_fold=f, pooled_inner_macro_f1=scores, selected=chosen, pooled_rows=n, decided_at_utc=now())
    ledger['selections'][str(f)] = rec
    atomic_write(out/'ledger.json', ledger)
    event(out, action='selection', outer_fold=f, selected=chosen, pooled_inner_macro_f1=scores)
    log(out, f'outer fold {f}: selected {chosen} ' + json.dumps({k2: round(v, 6) for k2, v in scores.items()}))
    return rec


def outer_unit(ctx, f):
    out, c = ctx.out, ctx.c
    key = outer_key(f)
    sel = selection(ctx, f)
    entry = read_ledger(out)['units'].get(key)
    if entry and entry['status'] == 'complete':
        check_saved(out, entry)
        require(entry['candidate'] == sel['selected'], 'Outer unit does not match the selection')
        return entry
    m = ctx.build(f)   # also checks equality with the T35 ledger matrices
    d = out/'outer'
    cand = sel['selected']
    if cand != c['baseline_candidate']:
        return fit_and_save(ctx, key, m, cand, d, f'fold_{f}',
                            dict(outer_fold=f, reused_baseline=False, pred_columns={'fold': f}))
    src = Path(ctx.design['local_root'])/c['baseline']['run_dir']/'L'/f'fold_{f}_pred.csv.gz'
    require_not_holdout(src)
    require(sha256_file(src) == c['baseline']['fold_pred_sha256'][str(f)], f'T35 L fold {f} prediction changed')
    d.mkdir(parents=True, exist_ok=True)
    name = f'fold_{f}_pred.csv.gz'
    move_aside(d, [name, f'fold_{f}.json'], 0)
    shutil.copyfile(src, d/name)
    back = read_pred(d/name)
    np.testing.assert_array_equal(back.building_id.to_numpy(), ctx.ids.to_numpy()[m['b']])
    np.testing.assert_array_equal(back.y_true.to_numpy(), ctx.y.to_numpy()[m['b']])
    metrics = score(back.y_true.to_numpy(), back.y_pred.to_numpy())
    write(d/f'fold_{f}.json', dict(study_id=STUDY_ID, unit=key, candidate=cand, reused_baseline=True,
                                   source=str(src), source_sha256=c['baseline']['fold_pred_sha256'][str(f)],
                                   metrics=metrics, matrices=dict(train_matrix_sha256=m['train_sha'],
                                                                  valid_matrix_sha256=m['valid_sha']),
                                   fit_seconds=0.0, finished_at_utc=now()))
    shas = {n: sha256_file(d/n) for n in [name, f'fold_{f}.json']}
    backup_files(out, 'outer', shas)
    ledger = read_ledger(out)
    ledger['units'][key] = dict(dir='outer', artifact_sha256=shas, attempts=0, status='complete', candidate=cand,
                                reused_baseline=True, macro_f1=metrics['macro_f1'], fit_seconds=0.0, finished_at_utc=now())
    atomic_write(out/'ledger.json', ledger)
    event(out, unit=key, action='baseline_reused', macro_f1=metrics['macro_f1'], artifact_sha256=shas)
    log(out, f'{key}: C0 selected, T35 L prediction reused (no fit) F1={metrics["macro_f1"]:.6f}')
    return ledger['units'][key]


# ---------------------------------------------------------------- preflight (no study fit)
def baseline_check(c, local_root, ids, y, folds):
    b = c['baseline']
    run_dir = Path(local_root)/b['run_dir']
    r = reference(run_dir/'L'/'oof.csv.gz', b['oof_sha256'], ids, y, folds)
    fo = r.fold.to_numpy()
    values = [score(r.y_true[fo == f], r.y_pred[fo == f])['macro_f1'] for f in range(N_FOLDS)]
    require(abs(float(np.mean(values)) - b['cv_mean']) <= 1e-9, 'Baseline CV mean differs from the registry')
    for f in range(N_FOLDS):
        require(sha256_file(run_dir/'L'/f'fold_{f}_pred.csv.gz') == b['fold_pred_sha256'][str(f)], f'Baseline fold {f} changed')
    return dict(oof_sha256=b['oof_sha256'], fold_macro_f1=values, cv_mean=float(np.mean(values)))


def split_checks(y, ids, folds, c):
    out = []
    for f in range(N_FOLDS):
        a, b = np.flatnonzero(folds != f), np.flatnonzero(folds == f)
        parts = inner_split(y.to_numpy()[a], c)
        vals = [va for _, va in parts]
        allv = np.concatenate(vals)
        require(len(allv) == len(a) and len(set(allv)) == len(a), f'Inner folds of {f} are not a partition')
        for tr, va in parts:
            require(not set(tr) & set(va) and len(tr) + len(va) == len(a), f'Inner overlap in fold {f}')
            require(not set(a[va]) & set(b) and not set(a[tr]) & set(b), 'Outer validation row inside the inner step')
        yy = y.to_numpy()[a]
        out.append(dict(outer_fold=f, inner_valid_rows=[len(v) for v in vals],
                        inner_valid_class_share=[(np.bincount(yy[v] - 1, minlength=3) / len(v)).tolist() for v in vals],
                        inner_valid_ids_sha256=[ids_sha256(ids.to_numpy()[a[v]]) for v in vals]))
    return out


def probe_models(c):
    """Synthetic 400-row fits for every candidate (NOT study fits): save/reload exactness and repeat determinism."""
    rng = np.random.default_rng(0)
    n = 400
    X = pd.DataFrame({'a': rng.normal(size=n), 'b': rng.integers(0, 5, n).astype(np.int64), 'c': rng.integers(0, 2, n)})
    y0 = ((X.a + X.c * 0.7 + rng.normal(0, .5, n) > 0.5).astype(int) + (X.a > 1).astype(int)).to_numpy()
    w = F.class_weights(pd.Series(y0), 0.5)
    res = {}
    for cand in CANDIDATES:
        small = dict(c, candidates={cand: dict(c['candidates'][cand], n_estimators=20)},
                     fixed_lgbm_params=dict(c['fixed_lgbm_params'], min_child_samples=5))
        m1, _ = fit_model(small, cand, X, y0, w)
        m2, _ = fit_model(small, cand, X, y0, w)
        p1, p2 = checked_proba(m1, X), checked_proba(m2, X)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/'probe.txt'
            m1.booster_.save_model(str(path))
            pr = reload_proba(path, X, 20, 1)
        diff = float(np.max(np.abs(pr - p1)))
        res[cand] = dict(repeat_identical=bool(np.array_equal(p1, p2)), reload_max_abs_diff=diff,
                         reload_within_tolerance=bool(diff <= c['checkpoint']['reload_probability_atol']))
    return dict(candidates=res, counted_as_study_fit=False, probe_fits=2 * len(CANDIDATES),
                note='synthetic 400-row probe with 20 trees, outside the study budget; not used for any decision')


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
    res = dict(study_id=STUDY_ID, kind='preflight; no study classifier fitted', classifiers_fitted=0, holdout_accessed=False,
               created_at_utc=now(), environment=environment(), data_contract=contract, plan_sha256=plan_sha,
               source_sha256=sources())
    res['inputs'] = T33.input_checks(X, y, ids, folds, contract, c)
    res['splits'] = split_checks(y, ids, folds, c)
    res['baseline'] = baseline_check(c, local_root, ids, y, folds)
    frame = notebook_frame(X, y)
    a, b = np.flatnonzero(folds != 0), np.flatnonzero(folds == 0)
    with threadpool_limits(limits=1):
        X_tr, X_va, _, _, _ = F.fold_matrices(frame.iloc[a].reset_index(drop=True),
                                               frame.iloc[b].drop(columns=[F.GRADE]).reset_index(drop=True))
        reg = c['baseline']['fold_matrix_sha256']['0']
        res['outer_fold0_matches_t35'] = bool(F.matrix_sha256(X_tr) == reg['train'] and F.matrix_sha256(X_va) == reg['valid'])
        tr, va = inner_split(y.to_numpy()[a], c)[0]
        res['inner_label_probe_f0_k0'] = F.label_probe(frame.iloc[a[tr]].reset_index(drop=True),
                                                       frame.iloc[a[va]].reset_index(drop=True))
    res['model_probe'] = probe_models(c)
    ok = (res['outer_fold0_matches_t35'] and res['inner_label_probe_f0_k0']['passes']
          and all(v['repeat_identical'] and v['reload_within_tolerance'] for v in res['model_probe']['candidates'].values()))
    res['status'] = 'PASS' if ok else 'FAIL'
    write(dest/'preflight.json', res)
    print(json.dumps({k: res[k] for k in ('status', 'outer_fold0_matches_t35', 'inner_label_probe_f0_k0', 'baseline')},
                     indent=1, default=str))
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
    require(pf['status'] == 'PASS' and pf['study_id'] == STUDY_ID and pf['classifiers_fitted'] == 0, 'Preflight must PASS first')
    require(pf['source_sha256'] == sources() and pf['plan_sha256'] == plan_sha and pf['environment'] == environment(),
            'Sources changed since the preflight; rerun it')
    X, y, ids, folds, contract = load_inputs(data)
    require(contract == pf['data_contract'], 'Data contract differs from the preflight')
    inputs = T33.input_checks(X, y, ids, folds, contract, c)
    baseline = baseline_check(c, local_root, ids, y, folds)
    out.mkdir(parents=True)
    (out/'backup').mkdir()
    design = dict(study_id=STUDY_ID, config=c, code_commit=head, source_sha256=sources(), plan_sha256=plan_sha,
                  local_root=str(local_root.resolve()), data_contract=contract, inputs=inputs, environment=environment(),
                  baseline=baseline, splits=pf['splits'],
                  preflight=dict(path=str(Path(preflight_path).resolve()), sha256=sha256_file(preflight_path),
                                 status=pf['status'], inner_label_probe_f0_k0=pf['inner_label_probe_f0_k0']),
                  inner_targets=[inner_key(*t) for t in inner_targets()], outer_targets=[outer_key(f) for f in range(N_FOLDS)],
                  max_fits=c['budget']['max_fits'], planned_fits_max=c['budget']['planned_fits_max'], automatic_retries=0,
                  holdout_accessed=False, created_at_utc=now(), classifiers_fitted=0)
    write(out/'design.json', design)
    write(out/'design_seal.json', dict(design_sha256=sha256_file(out/'design.json'), code_commit=head, created_at_utc=now(),
                                       classifiers_fitted=0))
    atomic_write(out/'ledger.json', dict(study_id=STUDY_ID, units={}, features={}, selections={}))
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


def fit_all(data, out, only_folds=None):
    out = Path(out)
    with run_lock(out):
        X, y, ids, folds, design = verify(data, out)
        ctx = Context(X, y, ids, folds, out, design)
        before = pd.util.hash_pandas_object(X, index=True).to_numpy()
        todo = [f for f in range(N_FOLDS) if only_folds is None or f in only_folds]
        log(out, f'fit-all start: outer folds {todo} (complete units are hash-verified, not refit)')
        t0 = time.perf_counter()
        for f in todo:
            for k in range(N_INNER):
                for cand in CANDIDATES:
                    _guard(out, inner_key(f, k, cand), inner_unit, ctx, f, k, cand)
            _guard(out, outer_key(f), outer_unit, ctx, f)
            log(out, f'outer fold {f} done (wall so far {time.perf_counter() - t0:.0f}s)')
        np.testing.assert_array_equal(before, pd.util.hash_pandas_object(X, index=True).to_numpy())
        log(out, f'fit-all finished wall={time.perf_counter() - t0:.1f}s')


def _guard(out, key, fn, *args):
    try:
        return fn(*args)
    except Exception as exc:
        log(out, f'STOPPED {key} {type(exc).__name__}: {exc}')
        event(out, unit=key, action='stopped', error=f'{type(exc).__name__}: {exc}', trace=traceback.format_exc())
        raise


def status(out):
    ledger = read_ledger(out)
    units = ledger['units']
    design = json.loads((Path(out)/'design.json').read_text(encoding='utf-8'))
    return dict(inner_complete=sum(e['status'] == 'complete' for k, e in units.items() if k.startswith('I/')),
                inner_total=len(design['inner_targets']),
                outer_complete=sum(e['status'] == 'complete' for k, e in units.items() if k.startswith('O/')),
                fits_started=fits_started(ledger), planned_fits_max=design['planned_fits_max'], max_fits=design['max_fits'],
                recovered_units=recovered_units(ledger), running=[k for k, e in units.items() if e['status'] == 'running'],
                selections={f: s['selected'] for f, s in ledger['selections'].items()})


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
    sel = {}
    for f in range(N_FOLDS):
        scores, _ = pooled_inner_scores(out, f)
        s = ledger['selections'][str(f)]
        require(select(scores) == s['selected'] and scores == s['pooled_inner_macro_f1'], f'Selection {f} not reproducible')
        sel[f] = s
    oof = pd.concat([read_pred(out/'outer'/f'fold_{f}_pred.csv.gz') for f in range(N_FOLDS)])
    oof = oof.set_index('building_id').loc[ids.to_numpy()].reset_index()
    require(len(oof) == len(ids) and oof.building_id.is_unique, 'OOF coverage changed')
    np.testing.assert_array_equal(oof.fold.to_numpy(), folds)
    np.testing.assert_array_equal(oof.y_true.to_numpy(), y.to_numpy())
    oof = oof[['building_id', 'fold', 'y_true', 'y_pred'] + PROBS]
    write_pred(out/'outer'/'oof.csv.gz', oof)
    oof_sha = sha256_file(out/'outer'/'oof.csv.gz')
    backup_files(out, 'outer', ['oof.csv.gz'])
    base = reference(Path(design['local_root'])/c['baseline']['run_dir']/'L'/'oof.csv.gz', c['baseline']['oof_sha256'],
                     ids, y, folds)
    tuned_s, base_s = T33.summarize(oof, X, c), T33.summarize(base, X, c)
    j1 = T33.d1_label(tuned_s['fold_macro_f1'], base_s['fold_macro_f1'], c['rules']['J1'])
    yv = y.to_numpy()
    j3 = []
    for f in range(N_FOLDS):
        s = sel[f]
        outer_f1 = tuned_s['fold_macro_f1'][f]
        j3.append(dict(outer_fold=f, selected=s['selected'], params=c['candidates'][s['selected']],
                       pooled_inner_macro_f1=s['pooled_inner_macro_f1'],
                       inner_rank=sorted(CANDIDATES, key=lambda k2: (-s['pooled_inner_macro_f1'][k2], CANDIDATES.index(k2))),
                       selected_inner_minus_outer=s['pooled_inner_macro_f1'][s['selected']] - outer_f1,
                       selected_inner_minus_C0_inner=s['pooled_inner_macro_f1'][s['selected']] - s['pooled_inner_macro_f1']['C0'],
                       reused_baseline=ledger['units'][outer_key(f)].get('reused_baseline', False)))
    started = fits_started(ledger)
    inner_fit_seconds = {cand: float(sum(ledger['units'][inner_key(f, k, cand)]['fit_seconds']
                                         for f in range(N_FOLDS) for k in range(N_INNER))) for cand in CANDIDATES}
    res = dict(study_id=STUDY_ID, design_sha256=sha256_file(out/'design.json'), code_commit=design['code_commit'],
               ledger=dict(units_complete=complete, units_total=len(tg), fits_started=started,
                           inner_fits=sum(1 for k in ledger['units'] if k.startswith('I/')),
                           outer_refits=sum(1 for f in range(N_FOLDS) if not ledger['units'][outer_key(f)].get('reused_baseline')),
                           planned_fits_max=c['budget']['planned_fits_max'], max_fits=c['budget']['max_fits'],
                           recovered_units=recovered_units(ledger),
                           fit_seconds_sum=float(sum(e['fit_seconds'] for e in ledger['units'].values())),
                           inner_fit_seconds_by_candidate=inner_fit_seconds),
               holdout_accessed=False, rules=c['rules'], candidates=c['candidates'],
               note='Nested-CV evaluation of a limited tuning procedure on the T35 L pipeline. Descriptive labels only; '
                    'no significance or causal claims; not independent performance; no adoption (design 5, 6).',
               tuned=dict(tuned_s, oof_sha256=oof_sha), baseline_T35_L=dict(base_s, oof_sha256=c['baseline']['oof_sha256']),
               J1=j1,
               J2_safety_no_verdict=dict(
                   recall3=dict(tuned=tuned_s['recall3'], baseline=base_s['recall3'], delta=tuned_s['recall3'] - base_s['recall3']),
                   error31=dict(tuned=tuned_s['error31'], baseline=base_s['error31'], delta=tuned_s['error31'] - base_s['error31']),
                   error32=dict(tuned=tuned_s['error32'], baseline=base_s['error32'], delta=tuned_s['error32'] - base_s['error32']),
                   rate31=dict(tuned=tuned_s['rate31'], baseline=base_s['rate31']),
                   rate32=dict(tuned=tuned_s['rate32'], baseline=base_s['rate32'])),
               turnover_baseline_to_tuned=turnover(yv, base.y_pred.to_numpy(), oof.y_pred.to_numpy()),
               J3_selection=j3,
               selection_frequency={cand: sum(1 for r in j3 if r['selected'] == cand) for cand in CANDIDATES},
               judgments=dict(J1=j1['label'], J2='no verdict', J3='record only'),
               finished_at_utc=now())
    write(out/'analysis.json', res)
    backup_files(out, '.', ['analysis.json'])
    log(out, 'analysis written: ' + json.dumps(dict(J1=j1['label'], mean_delta=j1['mean_delta'],
                                                    selections={f: sel[f]['selected'] for f in sel})))
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('action', choices=['preflight', 'prepare', 'fit-all', 'status', 'finalize'])
    p.add_argument('--data', help='absolute path of earthquake/data (development.csv + split_manifest.csv)')
    p.add_argument('--out', help='preflight: new destination folder; others: the run folder')
    p.add_argument('--preflight', help='prepare: path of a PASS preflight.json')
    p.add_argument('--code-commit', help='optional; must equal HEAD')
    p.add_argument('--local-root', help='local_reproduction root (default: <out>/../..)')
    p.add_argument('--folds', nargs='*', type=int, help='fit-all: restrict to outer folds')
    a = p.parse_args()
    if a.action == 'preflight':
        preflight(a.data, a.out, a.local_root)
    elif a.action == 'prepare':
        prepare(a.data, a.out, a.preflight, a.code_commit, a.local_root)
    elif a.action == 'fit-all':
        fit_all(a.data, a.out, a.folds)
    elif a.action == 'status':
        print(json.dumps(status(a.out), indent=1))
    else:
        r = finalize_all(a.data, a.out)
        print(json.dumps(dict(r['judgments'], selections=r['selection_frequency']), indent=1))


if __name__ == '__main__':
    main()
