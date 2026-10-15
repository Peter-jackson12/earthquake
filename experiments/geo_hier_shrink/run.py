"""T7: hierarchical geo shrinkage via per-block OHE weights (pre-registered).

Model = C (experiments/age995 make_pipeline('C')) with the single categorical
OneHotEncoder split into geo1/geo2/geo3/text OneHotEncoders (same options and
dtype, same column order) and ColumnTransformer transformer_weights scaling the
geo2/geo3 blocks. Scaling a one-hot block by s is equivalent to multiplying its
L2 penalty by 1/s^2. LR parameters, numeric/binary paths and splits unchanged.

C is never refit (its OOF is reused read-only). The holdout is never read.
Each fold is saved, hashed, reloaded and backed up immediately; completed folds
are never refit. All outputs are private (outside Git).
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
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from threadpoolctl import threadpool_limits
from preprocess import CATEGORICAL, GEO, TEXT, NUMERIC, BINARY, sha256_file
from experiments.age995.run import make_pipeline as original, PARAMS, FLAG
from experiments.geo_group_cv.run import load, score, require, write, PROBS
from experiments.geo_ablation import run as t6

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
STUDY_ID = 'T7_geo_hier_shrink_v1'
SCHEMES = ['stratified', 'geo2', 'geo1']
BLOCKS = {'geo1': ['geo_level_1_id'], 'geo2': ['geo_level_2_id'], 'geo3': ['geo_level_3_id'], 'text': list(TEXT)}
SOURCE = t6.SOURCE + ['experiments/geo_ablation_local_repro/run.py',
                      'experiments/geo_hier_shrink/config.json', 'experiments/geo_hier_shrink/run.py',
                      'docs/T7_GEO_HIER_SHRINK_PLAN.md']
THREAD_ENV = {k: '1' for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                                'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS']}


def now():
    return datetime.now(timezone.utc).isoformat()


def config():
    c = json.loads((HERE/'config.json').read_text(encoding='utf-8'))
    require(c['study_id'] == STUDY_ID and c['lr'] == PARAMS, 'C hyperparameters changed')
    require(c['schemes'] == SCHEMES, 'Unplanned validation schemes')
    expected = {}
    for s, tag in [(0.5, '050'), (0.25, '025'), (0.1, '010')]:
        expected['H3_'+tag] = dict(geo1=1.0, geo2=1.0, geo3=s, text=1.0)
    for s, tag in [(0.5, '050'), (0.25, '025'), (0.1, '010')]:
        expected['H23_'+tag] = dict(geo1=1.0, geo2=s, geo3=s, text=1.0)
    require(c['variants'] == expected, 'Unplanned variant weights')
    require(c['max_new_classifier_fits'] == 90 and c['automatic_retries'] == 0
            and c['stop_on_nonconvergence'] and c['threads_per_fit'] == 1
            and c['max_concurrent_fold_processes'] == 4, 'Unplanned run policy')
    require(GEO + TEXT == CATEGORICAL, 'Categorical order changed')
    return c


VARIANTS = list(config()['variants'])


def make_pipeline(weights):
    """C with the categorical OHE split per block. weights: dict over geo1/geo2/geo3/text."""
    require(set(weights) == set(BLOCKS), 'Weights must cover geo1/geo2/geo3/text')
    pipe = original('C')
    columns = pipe['preprocess']['columns']
    require(columns.remainder == 'drop', 'Remainder must stay drop')
    require([n for n, _, _ in columns.transformers] == ['numeric', 'binary', 'categorical'], 'Unexpected C paths')
    require(columns.transformers[0][2] == NUMERIC and columns.transformers[1][2] == BINARY+[FLAG],
            'Nongeographic inputs changed')
    name, ohe, cols = columns.transformers[2]
    require(cols == CATEGORICAL and isinstance(ohe, OneHotEncoder), 'C categorical path changed')
    params = ohe.get_params()
    blocks = [(b, OneHotEncoder(**params), list(c)) for b, c in BLOCKS.items()]
    columns.transformers = columns.transformers[:2] + blocks
    columns.transformer_weights = {b: float(weights[b]) for b in BLOCKS}
    return pipe


def variant_pipeline(variant):
    require(variant in config()['variants'], 'Only the six pre-registered variants may be trained')
    return make_pipeline(config()['variants'][variant])


def block_index(names):
    return {b: np.array([i for i, n in enumerate(names) if n.startswith(b+'__')]) for b in BLOCKS}


def check_blocks(matrix, names, weights):
    """Nonzero entries in each OHE block must equal float32(weight); no densification."""
    idx = block_index(names)
    require(sum(len(v) for v in idx.values()) + len(NUMERIC) + len(BINARY) + 1 == len(names), 'Block layout changed')
    out = {}
    for b, cols in idx.items():
        data = matrix[:, cols].tocsr().data
        require(data.size == 0 or np.all(data == np.float32(weights[b])), f'Block {b} weight not applied')
        out[b] = dict(columns=int(len(cols)), nonzeros=int(data.size))
    return out


def coef_blocks(pipe, names):
    coef = pipe['model'].coef_
    w = pipe['preprocess']['columns'].transformer_weights
    out = {}
    for b, cols in block_index(names).items():
        part = coef[:, cols]
        out[b] = dict(l2_encoded=float(np.linalg.norm(part)), l2_effective=float(np.linalg.norm(part*w[b])),
                      mean_abs_effective=float(np.mean(np.abs(part*w[b]))))
    return out


def sources():
    return {p: sha256_file(ROOT/p) for p in SOURCE}


def atomic_write(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name+f'.{os.getpid()}.tmp')
    write(tmp, obj)
    for i in range(50):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:  # Windows: reader holds the file briefly
            time.sleep(0.1)
    os.replace(tmp, path)


@contextmanager
def locked(out, name='ledger.lock', timeout=300):
    path = Path(out)/name
    start = time.time()
    while True:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f'pid={os.getpid()}'.encode())
            os.close(fd)
            break
        except FileExistsError:
            require(time.time()-start < timeout, f'Lock {path} held too long; check for a dead process')
            time.sleep(0.05)
    try:
        yield
    finally:
        os.remove(path)


def read_ledger(out):
    return json.loads((Path(out)/'ledger.json').read_text(encoding='utf-8'))


def log(out, message):
    line = f'{now()} [pid {os.getpid()}] {message}'
    print(line, flush=True)
    with locked(out, 'log.lock'):
        with open(Path(out)/'run.log', 'a', encoding='utf-8') as fh:
            fh.write(line+'\n')


def event(out, **kw):
    with locked(out, 'log.lock'):
        with open(Path(out)/'attempts.jsonl', 'a', encoding='utf-8') as fh:
            fh.write(json.dumps(dict(at_utc=now(), pid=os.getpid(), **kw), ensure_ascii=False)+'\n')


def read_pred(path):
    return pd.read_csv(path, float_precision='round_trip')


# ---------------------------------------------------------------- control check
def control_check(data, destination):
    """Sample check (NOT a study run): all-ones split-OHE pipeline == C on a 5,000-row subsample."""
    c = config()['control_check']
    dest = Path(destination)
    require(not dest.exists(), 'Do not overwrite an existing control check')
    X, y, ids, old, contract = load(data)
    pick, _ = train_test_split(np.arange(len(y)), train_size=c['rows'], stratify=y, random_state=c['seed'])
    pick = np.sort(pick)
    a, b = train_test_split(pick, test_size=c['valid_fraction'], stratify=y.iloc[pick], random_state=c['seed'])
    ones = dict(geo1=1.0, geo2=1.0, geo3=1.0, text=1.0)
    with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        ref = original('C').fit(X.iloc[a], y.iloc[a])
        new = make_pipeline(ones).fit(X.iloc[a], y.iloc[a])
        res = {}
        for part, rows in [('train', a), ('valid', b)]:
            m0 = ref['preprocess'].transform(X.iloc[rows])
            m1 = new['preprocess'].transform(X.iloc[rows])
            require(sparse.isspmatrix_csr(m1) and m1.dtype == np.float32, 'Matrix type changed')
            n0 = ref['preprocess']['columns'].get_feature_names_out().tolist()
            n1 = new['preprocess']['columns'].get_feature_names_out().tolist()
            strip = lambda n: n.split('__', 1)[1]
            names_equal = [strip(n) for n in n0] == [strip(n) for n in n1]
            pos = {strip(n): i for i, n in enumerate(n1)}
            permutation = [pos[strip(n)] for n in n0]
            identity = permutation == list(range(len(n0)))
            diff = (m0 - m1[:, permutation])
            res[part] = dict(rows=len(rows), encoded_columns=m1.shape[1], names_equal_after_prefix=names_equal,
                             permutation_is_identity=identity, matrix_max_abs_diff=float(abs(diff).max()) if diff.nnz else 0.0)
        p0, p1 = ref.predict_proba(X.iloc[b]), new.predict_proba(X.iloc[b])
        y0, y1 = ref.predict(X.iloc[b]), new.predict(X.iloc[b])
        # weight is applied as expected (one non-unit example)
        w = make_pipeline(config()['variants']['H23_025'])['preprocess'].fit(X.iloc[a])  # encoder only
        mw = w.transform(X.iloc[b])
        blocks = check_blocks(mw, w['columns'].get_feature_names_out().tolist(),
                              config()['variants']['H23_025'])
    out = dict(study_id=STUDY_ID, kind=c['kind'], counted_as_study_fit=False, classifiers_fitted=2,
               sample_rows=len(pick), train_rows=len(a), valid_rows=len(b),
               sample_ids_sha256=hashlib.sha256(ids.iloc[pick].to_numpy(dtype='<i8').tobytes()).hexdigest(),
               matrices=res, predictions_equal=bool(np.array_equal(y0, y1)),
               probability_max_abs_diff=float(np.max(np.abs(p0-p1))), probability_atol=c['probability_atol'],
               C_n_iter=ref['model'].n_iter_.tolist(), split_n_iter=new['model'].n_iter_.tolist(),
               weighted_example_blocks=blocks,
               warnings=[dict(category=x.category.__name__, message=str(x.message)) for x in caught],
               holdout_accessed=False, created_at_utc=now())
    ok = (all(r['names_equal_after_prefix'] and r['matrix_max_abs_diff'] == 0.0 for r in res.values())
          and out['predictions_equal'] and out['probability_max_abs_diff'] <= c['probability_atol'])
    out['status'] = 'PASS' if ok else 'FAIL'
    dest.mkdir(parents=True)
    write(dest/'control_check.json', out)
    print(json.dumps(out, indent=1))
    require(ok, 'Control check failed')
    return out


# ---------------------------------------------------------------- prepare/verify
def prepare(data, manifests, c_oof, output, code_commit, control):
    output = Path(output)
    require(not output.exists(), 'Do not overwrite an existing study')
    require(len(code_commit) == 40 and all(ch in '0123456789abcdef' for ch in code_commit), 'Full source commit required')
    cfg = config()
    require(t6.environment() == cfg['environment'], 'Reproduction environment differs')
    X, y, ids, old, contract = load(data)
    shas = {'stratified': sha256_file(Path(data)/'split_manifest.csv')}
    for s in ['geo2', 'geo1']:
        shas[s] = sha256_file(Path(manifests)/s/'manifest.csv')
    require(shas == cfg['manifest_sha256'], 'Split manifest SHA differs from T5 contract')
    c_sha = {s: sha256_file(c_oof[s]) for s in SCHEMES}
    require(c_sha == cfg['C_oof_sha256'], 'C OOF SHA differs')
    cc = json.loads(Path(control).read_text(encoding='utf-8'))
    require(cc['status'] == 'PASS' and cc['study_id'] == STUDY_ID, 'Control check must PASS before prepare')
    output.mkdir(parents=True)
    for s in ['geo2', 'geo1']:
        (output/s).mkdir()
        shutil.copyfile(Path(manifests)/s/'manifest.csv', output/s/'manifest.csv')
        require(sha256_file(output/s/'manifest.csv') == shas[s], 'Copied manifest SHA differs')
        m = pd.read_csv(output/s/'manifest.csv')
        np.testing.assert_array_equal(m.building_id, ids)
    (output/'backup').mkdir()
    (output/'logs').mkdir()
    design = dict(study_id=STUDY_ID, config=cfg, code_commit=code_commit, source_sha256=sources(),
        data_contract=contract, environment=t6.environment(), manifest_sha256=shas,
        c_oof_path={s: str(Path(c_oof[s]).resolve()) for s in SCHEMES}, c_oof_sha256=c_sha, c_refits=0,
        control_check=dict(path=str(Path(control).resolve()), sha256=sha256_file(control), status=cc['status']),
        max_new_fits=90, max_concurrent_fold_processes=4, threads_per_fit=1, automatic_retries=0,
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
    design = json.loads((output/'design.json').read_text(encoding='utf-8'))
    require(design['config'] == config(), 'Frozen config changed')
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
    d = Path(output)/entry['dir']
    b = Path(output)/'backup'/entry['dir']
    for name, digest in entry['artifact_sha256'].items():
        require(sha256_file(d/name) == digest, f'Artifact changed: {entry["dir"]}/{name}')
        require(sha256_file(b/name) == digest, f'Backup changed: {entry["dir"]}/{name}')


# ---------------------------------------------------------------- one fold
def fit_fold(X, y, ids, folds, output, scheme, variant, f, design):
    out = Path(output)
    k = key(scheme, variant, f)
    with locked(out):
        ledger = read_ledger(out)
        entry = ledger['folds'].get(k)
        if entry and entry['status'] in ('complete', 'complete_nonconverged'):
            check_saved(out, entry)
            done = True
        else:
            done = False
            n_fits = sum(e['attempts'] for e in ledger['folds'].values())
            require(n_fits < design['max_new_fits'], 'Fit budget exhausted')
            prior = entry['attempts'] if entry else 0
            entry = dict(dir=f'{scheme}/{variant}', scheme=scheme, variant=variant, fold=f, status='running',
                         attempts=prior+1, started_at_utc=now())
            ledger['folds'][k] = entry
            atomic_write(out/'ledger.json', ledger)
    if done:
        log(out, f'{k} already {entry["status"]}; verified hashes, not refit')
        return entry
    if prior:
        log(out, f'{k} previous attempt did not finish; manual resume attempt {prior+1}')
    rel = entry['dir']
    d = out/rel
    d.mkdir(parents=True, exist_ok=True)
    for stale in d.glob(f'fold_{f}[._]*'):  # partial files of an unfinished attempt only
        stale.rename(stale.with_name(stale.name+f'.incomplete_attempt{prior}'))
    event(out, fold=k, action='fit_start', attempt=prior+1)
    weights = config()['variants'][variant]
    a, b = np.flatnonzero(folds[scheme] != f), np.flatnonzero(folds[scheme] == f)
    pipe = variant_pipeline(variant)
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
        blocks = check_blocks(matrix, names, weights)
        require(pipe['preprocess']['columns'].transformer_weights == weights, 'Weights changed')
        model = d/f'fold_{f}.joblib'
        joblib.dump(pipe, model, compress=3)
        frame = pd.DataFrame({'building_id': ids.iloc[b].to_numpy(), 'fold': f, 'y_true': y.iloc[b].to_numpy(),
                              'y_pred': pred, **{c: prob[:, i] for i, c in enumerate(PROBS)}})
        preds = d/f'fold_{f}_pred.csv.gz'
        frame.to_csv(preds, index=False, compression={'method': 'gzip', 'mtime': 0}, float_format='%.17g')
        loaded = joblib.load(model)  # freshly generated trusted model only
        np.testing.assert_array_equal(loaded.predict(X.iloc[b]), pred)
        np.testing.assert_array_equal(loaded.predict_proba(X.iloc[b]), prob)
        back = read_pred(preds)
        np.testing.assert_array_equal(back.building_id, frame.building_id)
        np.testing.assert_array_equal(back.y_pred, pred)
        np.testing.assert_array_equal(back[PROBS].to_numpy(), prob)
    record = dict(study_id=STUDY_ID, scheme=scheme, variant=variant, fold=f, weights=weights,
                  train_rows=len(a), valid_rows=len(b), metrics=score(y.iloc[b], pred), fit_seconds=seconds,
                  n_iter=pipe['model'].n_iter_.tolist(), converged=converged,
                  warnings=[dict(category=w.category.__name__, message=str(w.message)) for w in caught],
                  median=pipe['preprocess']['age995'].median_, encoded_columns=matrix.shape[1],
                  valid_blocks=blocks, coef_blocks=coef_blocks(pipe, names),
                  feature_names_sha256=hashlib.sha256('\n'.join(names).encode()).hexdigest(),
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
                 n_iter=record['n_iter'], finished_at_utc=now())
    check_saved(out, entry)
    with locked(out):
        ledger = read_ledger(out)
        ledger['folds'][k] = entry
        atomic_write(out/'ledger.json', ledger)
    event(out, fold=k, action='fit_complete', converged=converged, macro_f1=record['metrics']['macro_f1'],
          seconds=seconds, artifact_sha256=shas)
    log(out, f'{k} F1={record["metrics"]["macro_f1"]:.9f} fit={seconds:.1f}s iter={record["n_iter"]} '
             f'converged={converged} saved+verified+backed-up')
    require(converged, 'Nonconvergence: stop without changing parameters or retrying')
    return entry


def fit_one(data, output, scheme, variant, fold, parent_pid=None):
    out = Path(output)
    lock = out/'RUNNING.lock'
    own = parent_pid is None
    if own:
        require(not lock.exists(), f'Another run holds {lock}; remove only if that process is dead')
        lock.write_text(f'pid={os.getpid()} started={now()}\n')
    else:
        require(lock.exists() and lock.read_text().startswith(f'pid={parent_pid} '), 'Parent lock missing')
    try:
        X, y, ids, folds, design = verify(data, out)
        before = pd.util.hash_pandas_object(X, index=True).to_numpy()
        fit_fold(X, y, ids, folds, out, scheme, variant, fold, design)
        np.testing.assert_array_equal(before, pd.util.hash_pandas_object(X, index=True).to_numpy())
    except Exception as exc:
        log(out, f'STOPPED {key(scheme, variant, fold)} {type(exc).__name__}: {exc}')
        event(out, fold=key(scheme, variant, fold), action='stopped', error=f'{type(exc).__name__}: {exc}',
              trace=traceback.format_exc())
        raise
    finally:
        if own:
            lock.unlink(missing_ok=True)


# ---------------------------------------------------------------- parent
def fit_all(data, output, workers):
    out = Path(output)
    require(1 <= workers <= config()['max_concurrent_fold_processes'], 'At most 4 concurrent fold processes')
    lock = out/'RUNNING.lock'
    require(not lock.exists(), f'Another run holds {lock}; remove only if that process is dead')
    lock.write_text(f'pid={os.getpid()} started={now()}\n')
    failed = []
    try:
        verify(data, out)
        targets = [(s, v, f) for s in SCHEMES for v in VARIANTS for f in range(5)]
        with locked(out):
            ledger = read_ledger(out)
        todo = []
        for t in targets:
            e = ledger['folds'].get(key(*t))
            if e and e['status'] == 'complete':
                check_saved(out, e)
            else:
                require(not (e and e['status'] == 'complete_nonconverged'), 'Nonconverged fold present; stop')
                todo.append(t)
        log(out, f'fit-all start: {len(targets)-len(todo)} complete (hash-verified, not refit), '
                 f'{len(todo)} to fit, workers={workers}')
        env = dict(os.environ, **THREAD_ENV)
        running = {}
        t0 = time.perf_counter()
        while (todo and not failed) or running:
            while todo and not failed and len(running) < workers:
                s, v, f = todo.pop(0)
                lf = open(out/'logs'/f'{s}_{v}_{f}.log', 'a', encoding='utf-8')
                cmd = [sys.executable, '-m', 'experiments.geo_hier_shrink.run', 'fit', '--data', str(data),
                       '--output', str(out), '--scheme', s, '--variant', v, '--fold', str(f),
                       '--parent-pid', str(os.getpid())]
                running[(s, v, f)] = (subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=lf, stderr=subprocess.STDOUT), lf)
            time.sleep(0.5)
            for t, (p, lf) in list(running.items()):
                if p.poll() is not None:
                    lf.close()
                    del running[t]
                    if p.returncode != 0:
                        failed.append((key(*t), p.returncode))
                        log(out, f'child {key(*t)} exited {p.returncode}; no new folds will be launched')
        log(out, f'fit-all children finished wall={time.perf_counter()-t0:.1f}s failed={failed}')
        require(not failed, f'Stopped: {failed}')
    finally:
        lock.unlink(missing_ok=True)
    finalize_all(data, out)


# ---------------------------------------------------------------- finalize/analysis
def finalize(output, y, ids, folds, scheme, variant):
    """Assemble OOF from saved fold predictions; no fitting."""
    out = Path(output)
    ledger = read_ledger(out)
    entries = [ledger['folds'].get(key(scheme, variant, f)) for f in range(5)]
    require(all(e and e['status'] == 'complete' for e in entries), f'{scheme}/{variant} incomplete')
    d = out/scheme/variant
    if (d/'metrics.json').exists():
        return json.loads((d/'metrics.json').read_text(encoding='utf-8'))
    for e in entries:
        check_saved(out, e)
    parts = pd.concat([read_pred(d/f'fold_{f}_pred.csv.gz') for f in range(5)])
    oof = parts.set_index('building_id').loc[ids.to_numpy()].reset_index()
    require(len(oof) == len(ids) and oof.building_id.is_unique, 'OOF coverage changed')
    np.testing.assert_array_equal(oof.fold.to_numpy(), folds[scheme])
    np.testing.assert_array_equal(oof.y_true.to_numpy(), y.to_numpy())
    oof.to_csv(d/'oof.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0}, float_format='%.17g')
    back = read_pred(d/'oof.csv.gz')
    np.testing.assert_array_equal(back[PROBS].to_numpy(), oof[PROBS].to_numpy())
    values = [e['macro_f1'] for e in entries]
    result = dict(study_id=STUDY_ID, scheme=scheme, variant=variant, weights=config()['variants'][variant],
                  cv_mean=float(np.mean(values)), cv_std=float(np.std(values, ddof=1)), fold_macro_f1=values,
                  pooled=score(oof.y_true, oof.y_pred), oof_sha256=sha256_file(d/'oof.csv.gz'),
                  fit_seconds=sum(e['fit_seconds'] for e in entries), finished_at_utc=now())
    write(d/'metrics.json', result)
    bd = out/'backup'/scheme/variant
    for name in ['oof.csv.gz', 'metrics.json']:
        require(not (bd/name).exists(), 'Backup already exists; refusing overwrite')
        shutil.copy2(d/name, bd/name)
    log(out, f'{scheme}/{variant} finalized mean={result["cv_mean"]:.9f} sd={result["cv_std"]:.9f}')
    return result


def c_reference(design, y, ids, folds, scheme):
    path = design['c_oof_path'][scheme]
    require(sha256_file(path) == design['c_oof_sha256'][scheme], 'C OOF changed')
    c = read_pred(path).set_index('building_id').loc[ids.to_numpy()].reset_index()
    require(len(c) == len(ids) and c.building_id.is_unique, 'C OOF coverage')
    np.testing.assert_array_equal(c.fold.to_numpy(), folds[scheme])
    np.testing.assert_array_equal(c.y_true.to_numpy(), y.to_numpy())
    values = [score(c.y_true[c.fold == f], c.y_pred[c.fold == f])['macro_f1'] for f in range(5)]
    return c, dict(cv_mean=float(np.mean(values)), cv_std=float(np.std(values, ddof=1)), fold_macro_f1=values,
                   pooled=score(c.y_true, c.y_pred), oof_sha256=design['c_oof_sha256'][scheme])


def finalize_all(data, output):
    from experiments.geo_ablation.audit import turnover
    out = Path(output)
    X, y, ids, folds, design = verify(data, out)
    if (out/'analysis.json').exists():
        return json.loads((out/'analysis.json').read_text(encoding='utf-8'))
    ledger = read_ledger(out)
    complete = sum(e['status'] == 'complete' for e in ledger['folds'].values())
    attempts = sum(e['attempts'] for e in ledger['folds'].values())
    require(complete == 90 and len(ledger['folds']) == 90, 'Ledger incomplete; not finalizing')
    rule = config()['follow_up_rule']
    res = dict(study_id=STUDY_ID, design_sha256=sha256_file(out/'design.json'), code_commit=design['code_commit'],
               ledger=dict(complete=complete, total=90, attempts=attempts, retries=attempts-complete,
                           fit_seconds_sum=float(sum(e['fit_seconds'] for e in ledger['folds'].values()))),
               c_refits=0, holdout_accessed=False, rule=rule,
               note='Selecting among six variants on these same CV folds is optimistic; descriptive only, no causal claim.',
               schemes={})
    for scheme in SCHEMES:
        c, cref = c_reference(design, y, ids, folds, scheme)
        sch = dict(C=cref, variants={})
        for v in VARIANTS:
            m = finalize(out, y, ids, folds, scheme, v)
            oof = read_pred(out/scheme/v/'oof.csv.gz')
            np.testing.assert_array_equal(oof.building_id, c.building_id)
            deltas = [a-b for a, b in zip(m['fold_macro_f1'], cref['fold_macro_f1'])]
            sch['variants'][v] = dict(weights=m['weights'], cv_mean=m['cv_mean'], cv_std=m['cv_std'],
                fold_macro_f1=m['fold_macro_f1'], delta_mean_vs_C=m['cv_mean']-cref['cv_mean'],
                fold_delta_vs_C=deltas, fold_wins_vs_C=int(sum(d > 0 for d in deltas)),
                pooled=m['pooled'], oof_sha256=m['oof_sha256'], fit_seconds=m['fit_seconds'],
                turnover_vs_C=turnover(c.y_true.to_numpy(), c.y_pred.to_numpy(), oof.y_pred.to_numpy()))
        res['schemes'][scheme] = sch
    res['follow_up_candidates'] = {}
    for v in VARIANTS:
        g2 = res['schemes']['geo2']['variants'][v]
        st = res['schemes']['stratified']['variants'][v]
        res['follow_up_candidates'][v] = dict(
            geo2_delta=g2['delta_mean_vs_C'], geo2_wins=g2['fold_wins_vs_C'], stratified_delta=st['delta_mean_vs_C'],
            meets_rule=bool(g2['delta_mean_vs_C'] >= rule['geo2_mean_delta_min']
                            and g2['fold_wins_vs_C'] >= rule['geo2_fold_wins_min']
                            and st['delta_mean_vs_C'] >= rule['stratified_mean_delta_min']))
    res['finished_at_utc'] = now()
    write(out/'analysis.json', res)
    require(not (out/'backup'/'analysis.json').exists(), 'Backup already exists; refusing overwrite')
    shutil.copy2(out/'analysis.json', out/'backup'/'analysis.json')
    log(out, 'analysis written: '+json.dumps({v: r['meets_rule'] for v, r in res['follow_up_candidates'].items()}))
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['control-check', 'prepare', 'fit', 'fit-all', 'finalize', 'status'])
    p.add_argument('--data', default='data')
    p.add_argument('--output')
    p.add_argument('--destination')
    p.add_argument('--manifests'); p.add_argument('--code-commit'); p.add_argument('--control')
    for s in SCHEMES:
        p.add_argument(f'--c-oof-{s}')
    p.add_argument('--scheme', choices=SCHEMES); p.add_argument('--variant', choices=VARIANTS)
    p.add_argument('--fold', type=int, choices=range(5))
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--parent-pid', type=int)
    a = p.parse_args()
    if a.action == 'control-check':
        control_check(a.data, a.destination)
    elif a.action == 'prepare':
        c_oof = {s: getattr(a, f'c_oof_{s}') for s in SCHEMES}
        prepare(a.data, a.manifests, c_oof, a.output, a.code_commit, a.control)
    elif a.action == 'fit':
        fit_one(a.data, a.output, a.scheme, a.variant, a.fold, a.parent_pid)
    elif a.action == 'fit-all':
        fit_all(a.data, a.output, a.workers)
    elif a.action == 'finalize':
        finalize_all(a.data, a.output)
    else:
        ledger = read_ledger(a.output)
        done = sum(e['status'] == 'complete' for e in ledger['folds'].values())
        print(json.dumps(dict(complete=done, total=90, attempts=sum(e['attempts'] for e in ledger['folds'].values()),
                              running=[k for k, e in ledger['folds'].items() if e['status'] == 'running']), indent=1))


if __name__ == '__main__':
    sys.exit(main())
