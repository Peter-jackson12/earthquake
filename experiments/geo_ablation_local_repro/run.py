"""T6R: independent local reproduction of T6 R12/R1/R0 with per-fold checkpoints.

Separate study from the original T6 (whose private evidence was not recovered).
Model/feature policy, hyperparameters and splits come unchanged from
experiments/geo_ablation (make_pipeline) and the T5 manifests (SHA-checked).
C is never refit. Holdout is never read.

Each fold is saved, hashed, reloaded and backed up immediately after it finishes;
completed folds are never refit. All outputs are private (outside Git).
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
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
from threadpoolctl import threadpool_limits
from preprocess import sha256_file
from experiments.geo_group_cv.run import load, score, require, write, PROBS
from experiments.geo_ablation import run as t6

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
STUDY_ID = 'T6R_geo_ablation_local_repro_v1'
SCHEMES = ['stratified', 'geo2', 'geo1']
VARIANTS = ['R12', 'R1', 'R0']
SOURCE = t6.SOURCE + ['experiments/geo_ablation_local_repro/run.py']


def now():
    return datetime.now(timezone.utc).isoformat()


def sources():
    return {p: sha256_file(ROOT/p) for p in SOURCE}


def atomic_write(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name+'.tmp')
    write(tmp, obj)
    os.replace(tmp, path)


def log(out, message):
    line = f'{now()} {message}'
    print(line, flush=True)
    with open(Path(out)/'run.log', 'a', encoding='utf-8') as fh:
        fh.write(line+'\n')


def event(out, **kw):
    with open(Path(out)/'attempts.jsonl', 'a', encoding='utf-8') as fh:
        fh.write(json.dumps(dict(at_utc=now(), **kw), ensure_ascii=False)+'\n')


def prepare(data, manifests, c_stratified_oof, output, code_commit):
    output = Path(output)
    require(not output.exists(), 'Do not overwrite an existing study')
    require(len(code_commit) == 40, 'Full source commit required')
    cfg = t6.config()
    require(t6.environment() == cfg['environment'], 'Reproduction environment differs')
    X, y, ids, old, contract = load(data)
    shas = {'stratified': sha256_file(Path(data)/'split_manifest.csv')}
    for s in ['geo2', 'geo1']:
        shas[s] = sha256_file(Path(manifests)/s/'manifest.csv')
    require(shas == cfg['manifest_sha256'], 'Split manifest SHA differs from T6/T5 contract')
    c_sha = sha256_file(c_stratified_oof)
    require(c_sha == cfg['C_oof_sha256']['stratified'], 'C stratified OOF SHA differs')
    output.mkdir(parents=True)
    for s in ['geo2', 'geo1']:
        (output/s).mkdir()
        shutil.copyfile(Path(manifests)/s/'manifest.csv', output/s/'manifest.csv')
        m = pd.read_csv(output/s/'manifest.csv')
        np.testing.assert_array_equal(m.building_id, ids)
    (output/'backup').mkdir()
    design = dict(study_id=STUDY_ID, relation='independent reproduction; not the original T6 run',
        original_t6_config=cfg, code_commit=code_commit, source_sha256=sources(),
        data_contract=contract, environment=t6.environment(), manifest_sha256=shas,
        c_reused_oof_sha256={'stratified': c_sha, 'geo2': None, 'geo1': None},
        c_refits=0, max_new_fits=45, threads_per_fit=1, automatic_retries=0,
        stop_on_nonconvergence=True, holdout_accessed=False, created_at_utc=now(), classifiers_fitted=0)
    write(output/'design.json', design)
    write(output/'design_seal.json', dict(design_sha256=sha256_file(output/'design.json'),
        created_at_utc=now(), classifiers_fitted=0))
    atomic_write(output/'ledger.json', dict(study_id=STUDY_ID, folds={}))
    return design


def verify(data, output):
    output = Path(output)
    seal = json.loads((output/'design_seal.json').read_text())
    require(sha256_file(output/'design.json') == seal['design_sha256'], 'Design seal changed')
    design = json.loads((output/'design.json').read_text())
    require(design['source_sha256'] == sources(), 'Frozen source changed')
    require(design['environment'] == t6.environment(), 'Frozen environment changed')
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


def key(scheme, variant, fold):
    return f'{scheme}/{variant}/{fold}'


def check_saved(output, entry):
    """Hash-check a completed fold's artifacts in the run and backup copies."""
    d = Path(output)/entry['dir']
    b = Path(output)/'backup'/entry['dir']
    for name, digest in entry['artifact_sha256'].items():
        require(sha256_file(d/name) == digest, f'Artifact changed: {entry["dir"]}/{name}')
        require(sha256_file(b/name) == digest, f'Backup changed: {entry["dir"]}/{name}')


def fit_fold(X, y, ids, folds, output, scheme, variant, f, design):
    out = Path(output)
    ledger = json.loads((out/'ledger.json').read_text())
    k = key(scheme, variant, f)
    entry = ledger['folds'].get(k)
    if entry and entry['status'] in ('complete', 'complete_nonconverged'):
        check_saved(out, entry)
        log(out, f'{k} already {entry["status"]}; verified hashes, not refit')
        return entry
    n_fits = sum(e['attempts'] for e in ledger['folds'].values())
    require(n_fits < design['max_new_fits'], 'Fit budget exhausted')
    prior = entry['attempts'] if entry else 0
    if prior:
        log(out, f'{k} previous attempt did not finish (status={entry["status"]}); manual resume attempt {prior+1}')
    rel = f'{scheme}/{variant}'
    d = out/rel
    d.mkdir(parents=True, exist_ok=True)
    for stale in d.glob(f'fold_{f}*'):  # partial files of an unfinished attempt; never of a complete fold
        stale.rename(stale.with_name(stale.name+f'.incomplete_attempt{prior}'))
    entry = dict(dir=rel, scheme=scheme, variant=variant, fold=f, status='running',
                 attempts=prior+1, started_at_utc=now())
    ledger['folds'][k] = entry
    atomic_write(out/'ledger.json', ledger)
    event(out, fold=k, action='fit_start', attempt=prior+1)
    a, b = np.flatnonzero(folds[scheme] != f), np.flatnonzero(folds[scheme] == f)
    pipe = t6.make_pipeline(variant)
    start = time.perf_counter()
    with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        pipe.fit(X.iloc[a], y.iloc[a])
    seconds = time.perf_counter()-start
    converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    np.testing.assert_array_equal(pipe['model'].classes_, [1, 2, 3])
    with threadpool_limits(limits=1):
        pred = pipe.predict(X.iloc[b])
        prob = pipe.predict_proba(X.iloc[b])
        matrix = pipe['preprocess'].transform(X.iloc[b])
        require(sparse.isspmatrix_csr(matrix) and matrix.dtype == np.float32, 'Matrix type changed')
        names = pipe['preprocess']['columns'].get_feature_names_out().tolist()
        for excluded in t6.config()['variants'][variant]:
            require(not any(excluded in n for n in names), 'Excluded feature reintroduced')
        model = d/f'fold_{f}.joblib'
        joblib.dump(pipe, model, compress=3)
        frame = pd.DataFrame({'building_id': ids.iloc[b].to_numpy(), 'fold': f, 'y_true': y.iloc[b].to_numpy(),
                              'y_pred': pred, **{c: prob[:, i] for i, c in enumerate(PROBS)}})
        preds = d/f'fold_{f}_pred.csv.gz'
        frame.to_csv(preds, index=False, compression={'method': 'gzip', 'mtime': 0}, float_format='%.17g')
        loaded = joblib.load(model)  # freshly generated trusted model only
        np.testing.assert_array_equal(loaded.predict(X.iloc[b]), pred)
        np.testing.assert_array_equal(loaded.predict_proba(X.iloc[b]), prob)
        back = pd.read_csv(preds)
        np.testing.assert_array_equal(back.building_id, frame.building_id)
        np.testing.assert_array_equal(back.y_pred, pred)
        np.testing.assert_allclose(back[PROBS].to_numpy(), prob, rtol=0, atol=1e-15)
    record = dict(study_id=STUDY_ID, scheme=scheme, variant=variant, fold=f, train_rows=len(a),
                  valid_rows=len(b), metrics=score(y.iloc[b], pred), fit_seconds=seconds,
                  n_iter=pipe['model'].n_iter_.tolist(), converged=converged,
                  warnings=[dict(category=w.category.__name__, message=str(w.message)) for w in caught],
                  median=pipe['preprocess']['age995'].median_, encoded_columns=matrix.shape[1],
                  feature_names_sha256=__import__('hashlib').sha256('\n'.join(names).encode()).hexdigest(),
                  effective_parameters=pipe['model'].get_params(), reload_exact=True, finished_at_utc=now())
    write(d/f'fold_{f}.json', record)
    shas = {p.name: sha256_file(p) for p in [model, preds, d/f'fold_{f}.json']}
    bd = out/'backup'/rel
    bd.mkdir(parents=True, exist_ok=True)
    for name in shas:
        require(not (bd/name).exists(), 'Backup already exists; refusing overwrite')
        shutil.copy2(d/name, bd/name)
    entry.update(status='complete' if converged else 'complete_nonconverged', artifact_sha256=shas,
                 macro_f1=record['metrics']['macro_f1'], fit_seconds=seconds, converged=converged,
                 finished_at_utc=now())
    check_saved(out, entry)
    ledger['folds'][k] = entry
    atomic_write(out/'ledger.json', ledger)
    event(out, fold=k, action='fit_complete', converged=converged, macro_f1=record['metrics']['macro_f1'],
          seconds=seconds, artifact_sha256=shas)
    log(out, f'{k} F1={record["metrics"]["macro_f1"]:.9f} fit={seconds:.1f}s iter={record["n_iter"]} '
             f'converged={converged} saved+verified+backed-up')
    require(converged, 'Nonconvergence: stop without changing parameters or retrying')
    return entry


def finalize(output, y, ids, folds, scheme, variant):
    """Assemble OOF from saved fold predictions; no fitting."""
    out = Path(output)
    ledger = json.loads((out/'ledger.json').read_text())
    entries = [ledger['folds'].get(key(scheme, variant, f)) for f in range(5)]
    if not all(e and e['status'] == 'complete' for e in entries):
        return None
    d = out/scheme/variant
    if (d/'metrics.json').exists():
        return json.loads((d/'metrics.json').read_text())
    for e in entries:
        check_saved(out, e)
    parts = pd.concat([pd.read_csv(d/f'fold_{f}_pred.csv.gz') for f in range(5)])
    oof = parts.set_index('building_id').loc[ids.to_numpy()].reset_index()
    require(len(oof) == len(ids) and oof.building_id.is_unique, 'OOF coverage changed')
    np.testing.assert_array_equal(oof.fold.to_numpy(), folds[scheme])
    np.testing.assert_array_equal(oof.y_true.to_numpy(), y.to_numpy())
    oof.to_csv(d/'oof.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0}, float_format='%.17g')
    values = [e['macro_f1'] for e in entries]
    result = dict(study_id=STUDY_ID, scheme=scheme, variant=variant, cv_mean=float(np.mean(values)),
                  cv_std=float(np.std(values, ddof=1)), fold_macro_f1=values,
                  pooled=score(oof.y_true, oof.y_pred), oof_sha256=sha256_file(d/'oof.csv.gz'),
                  fit_seconds=sum(e['fit_seconds'] for e in entries), finished_at_utc=now())
    write(d/'metrics.json', result)
    bd = out/'backup'/scheme/variant
    for name in ['oof.csv.gz', 'metrics.json']:
        require(not (bd/name).exists(), 'Backup already exists; refusing overwrite')
        shutil.copy2(d/name, bd/name)
    log(out, f'{scheme}/{variant} finalized mean={result["cv_mean"]:.9f} sd={result["cv_std"]:.9f}')
    return result


def run(data, output, targets):
    out = Path(output)
    lock = out/'RUNNING.lock'
    require(not lock.exists(), f'Another run holds {lock}; remove only if that process is dead')
    lock.write_text(f'pid={os.getpid()} started={now()}\n')
    try:
        X, y, ids, folds, design = verify(data, out)
        before = pd.util.hash_pandas_object(X, index=True).to_numpy()
        for scheme, variant, f in targets:
            fit_fold(X, y, ids, folds, out, scheme, variant, f, design)
            if f == 4 or all(json.loads((out/'ledger.json').read_text())['folds'].get(key(scheme, variant, g), {})
                             .get('status') == 'complete' for g in range(5)):
                finalize(out, y, ids, folds, scheme, variant)
        np.testing.assert_array_equal(before, pd.util.hash_pandas_object(X, index=True).to_numpy())
        log(out, 'run finished')
    except Exception as exc:
        log(out, f'STOPPED {type(exc).__name__}: {exc}')
        event(out, action='stopped', error=f'{type(exc).__name__}: {exc}', trace=traceback.format_exc())
        raise
    finally:
        lock.unlink(missing_ok=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'fit', 'fit-all', 'status'])
    p.add_argument('--data', default='data')
    p.add_argument('--output', required=True)
    p.add_argument('--manifests'); p.add_argument('--c-stratified-oof'); p.add_argument('--code-commit')
    p.add_argument('--scheme', choices=SCHEMES); p.add_argument('--variant', choices=VARIANTS)
    p.add_argument('--fold', type=int, choices=range(5))
    a = p.parse_args()
    if a.action == 'prepare':
        prepare(a.data, a.manifests, a.c_stratified_oof, a.output, a.code_commit)
    elif a.action == 'fit':
        run(a.data, a.output, [(a.scheme, a.variant, a.fold)])
    elif a.action == 'fit-all':
        run(a.data, a.output, [(s, v, f) for s in SCHEMES for v in VARIANTS for f in range(5)])
    else:
        ledger = json.loads((Path(a.output)/'ledger.json').read_text())
        done = sum(e['status'] == 'complete' for e in ledger['folds'].values())
        print(json.dumps(dict(complete=done, total=45, folds=ledger['folds']), indent=1))


if __name__ == '__main__':
    sys.exit(main())
