"""T6 training only: fixed C preprocessing minus selected geo OHE blocks.

prepare validates the existing evidence and seals inputs before any new fit.
fit runs five new classifiers, never C/G2. Existing output directories are refused.
All row-level outputs and saved models are private.
"""
from __future__ import annotations
import argparse
import json
import platform
import shutil
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
import threadpoolctl
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from preprocess import FEATURES, CATEGORICAL, NUMERIC, BINARY, sha256_file
from experiments.age995.run import make_pipeline as original, PARAMS, FLAG
from experiments.geo_group_cv.run import load, score, require, write, PROBS, SOURCE as T5_SOURCE

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
SOURCE = T5_SOURCE + ['configs/data_contract.json', 'experiments/geo_group_cv/audit.py',
    'experiments/age995/followup.py', 'experiments/age995/followup_v1.json',
    'experiments/geo_ablation/config.json', 'experiments/geo_ablation/run.py',
    'experiments/geo_ablation/audit.py']


def now():
    return datetime.now(timezone.utc).isoformat()


def config():
    c = json.loads((HERE/'config.json').read_text(encoding='utf-8'))
    require(c['lr'] == PARAMS, 'C hyperparameters changed')
    require(c['schemes'] == ['stratified','geo2','geo1'], 'Unplanned validation schemes')
    require(c['variants'] == {'R12':['geo_level_3_id'], 'R1':['geo_level_2_id','geo_level_3_id'],
        'R0':['geo_level_1_id','geo_level_2_id','geo_level_3_id']}, 'Unplanned feature policy')
    return c


def environment():
    return dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
        scipy=scipy.__version__, sklearn=sklearn.__version__, joblib=joblib.__version__,
        threadpoolctl=threadpoolctl.__version__)


def sources():
    return {p:sha256_file(ROOT/p) for p in SOURCE}


def make_pipeline(variant):
    """Return a new estimator; never mutate a shared C estimator or its source."""
    require(variant in config()['variants'], 'Only R12/R1/R0 may be trained in T6')
    excluded = set(config()['variants'][variant])
    pipe = original('C')
    columns = pipe['preprocess']['columns']
    require(columns.remainder == 'drop', 'Remainder must not reintroduce excluded columns')
    columns.transformers = [(name, transformer, [c for c in cols if c not in excluded])
        for name, transformer, cols in columns.transformers]
    require([name for name,_,_ in columns.transformers] == ['numeric','binary','categorical'],
        'Unexpected feature path')
    require(columns.transformers[0][2] == NUMERIC and columns.transformers[1][2] == BINARY+[FLAG],
        'Nongeographic inputs changed')
    require(columns.transformers[2][2] == [c for c in CATEGORICAL if c not in excluded],
        'Categorical policy changed')
    return pipe


def prepare(data, t5, baseline, output, code_commit):
    from experiments.geo_ablation.audit import reference, fingerprints
    output = Path(output)
    require(not output.exists(), 'Do not overwrite an existing study')
    require(len(code_commit)==40 and all(c in '0123456789abcdef' for c in code_commit),
        'Actual full training-source commit required')
    require(environment() == config()['environment'], 'Reproduction environment differs')
    before = sources()
    ref = reference(data,t5,baseline,reload_models=True,reconstruct=True)
    require(before == sources(), 'Source changed during preflight')
    output.mkdir(parents=True)
    for scheme in ['geo2','geo1']:
        (output/scheme).mkdir()
        shutil.copyfile(Path(t5)/scheme/'manifest.csv', output/scheme/'manifest.csv')
    design = dict(study_id=config()['study_id'], config=config(), code_commit=code_commit,
        source_sha256=before, data_contract=ref['contract'], environment=environment(),
        baseline_file_sha256=fingerprints(t5,baseline), baseline_models=ref['models'],
        split_checks=ref['split_checks'], created_at_utc=now(), classifiers_fitted=0,
        holdout_accessed=False)
    write(output/'design.json',design)
    write(output/'design_seal.json',dict(design_sha256=sha256_file(output/'design.json'),
        created_at_utc=now(),classifiers_fitted=0))
    return design


def verify(data,t5,baseline,output):
    from experiments.geo_ablation.audit import fingerprints
    output=Path(output)
    design=json.loads((output/'design.json').read_text())
    seal=json.loads((output/'design_seal.json').read_text())
    require(sha256_file(output/'design.json')==seal['design_sha256'], 'Design seal changed')
    require(design['config']==config() and design['source_sha256']==sources(), 'Frozen source changed')
    require(design['environment']==environment(), 'Frozen environment changed')
    require(design['baseline_file_sha256']==fingerprints(t5,baseline), 'Existing evidence changed')
    X,y,ids,old,contract=load(data)
    require(contract==design['data_contract'], 'Data contract changed')
    assignments={'stratified':old}
    for scheme in ['geo2','geo1']:
        path=output/scheme/'manifest.csv'
        require(sha256_file(path)==config()['manifest_sha256'][scheme], 'Copied split changed')
        m=pd.read_csv(path)
        np.testing.assert_array_equal(m.building_id,ids)
        assignments[scheme]=m.fold.to_numpy()
    return X,y,ids,assignments,design


def fit(data,t5,baseline,output,scheme,variant):
    require(scheme in config()['schemes'] and variant in config()['variants'], 'Unplanned fit')
    X,y,ids,assignments,design=verify(data,t5,baseline,output)
    directory=Path(output)/scheme/variant
    directory.mkdir(parents=True,exist_ok=False)
    meta=dict(status='running',scheme=scheme,variant=variant,code_commit=design['code_commit'],
        design_sha256=sha256_file(Path(output)/'design.json'),source_sha256=sources(),
        environment=environment(),started_at_utc=now(),attempted_fits=0,completed_fits=0,
        failed_fits=0,retries=0,holdout_accessed=False)
    write(directory/'metadata.json',meta)
    folds=assignments[scheme]
    pred=np.full(len(X),-1,dtype=int);prob=np.full((len(X),3),np.nan)
    seen=np.zeros(len(X),dtype=int);records=[]
    before=pd.util.hash_pandas_object(X,index=True).to_numpy()
    try:
        for f in range(5):
            a,b=np.flatnonzero(folds!=f),np.flatnonzero(folds==f)
            pipe=make_pipeline(variant)
            meta['attempted_fits']+=1
            write(directory/'metadata.json',meta)
            start=time.perf_counter()
            with threadpool_limits(limits=1),warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                pipe.fit(X.iloc[a],y.iloc[a])
            seconds=time.perf_counter()-start
            converged=not any(issubclass(w.category,ConvergenceWarning) for w in caught)
            meta['completed_fits']+=1
            np.testing.assert_array_equal(pipe['model'].classes_,[1,2,3])
            with threadpool_limits(limits=1):
                pred[b]=pipe.predict(X.iloc[b]);prob[b]=pipe.predict_proba(X.iloc[b])
                matrix=pipe['preprocess'].transform(X.iloc[b])
                require(sparse.isspmatrix_csr(matrix) and matrix.dtype==np.float32,'Matrix type changed')
                names=pipe['preprocess']['columns'].get_feature_names_out().tolist()
                for excluded in config()['variants'][variant]:
                    require(not any(excluded in name for name in names),'Excluded feature reintroduced')
                path=directory/f'fold_{f}.joblib'
                joblib.dump(pipe,path,compress=3)
                loaded=joblib.load(path)  # Our freshly generated trusted model only.
                np.testing.assert_array_equal(loaded.predict(X.iloc[b]),pred[b])
                np.testing.assert_array_equal(loaded.predict_proba(X.iloc[b]),prob[b])
            seen[b]+=1
            record=dict(fold=f,train_rows=len(a),valid_rows=len(b),metrics=score(y.iloc[b],pred[b]),
                fit_seconds=seconds,n_iter=pipe['model'].n_iter_.tolist(),converged=converged,
                warnings=[dict(category=w.category.__name__,message=str(w.message)) for w in caught],
                median=pipe['preprocess']['age995'].median_,encoded_columns=matrix.shape[1],
                model_sha256=sha256_file(path),reload_exact=True)
            records.append(record);write(directory/'folds.json',records);write(directory/'metadata.json',meta)
            print(f'{scheme}/{variant} fold={f} F1={record["metrics"]["macro_f1"]:.9f} '
                f'fit={seconds:.1f}s iter={record["n_iter"]} converged={converged}',flush=True)
            require(converged,'Nonconvergence: stop without changing parameters or retrying')
        np.testing.assert_array_equal(seen,np.ones(len(X),dtype=int))
        np.testing.assert_array_equal(before,pd.util.hash_pandas_object(X,index=True).to_numpy())
        oof=pd.DataFrame({'building_id':ids,'fold':folds,'y_true':y,'y_pred':pred,
            **{c:prob[:,i] for i,c in enumerate(PROBS)}})
        oof.to_csv(directory/'oof.csv.gz',index=False,compression={'method':'gzip','mtime':0})
        values=[r['metrics']['macro_f1'] for r in records]
        result=dict(scheme=scheme,variant=variant,cv_mean=float(np.mean(values)),
            cv_std=float(np.std(values,ddof=1)),pooled=score(y,pred),all_converged=True,
            effective_parameters=pipe['model'].get_params(),oof_sha256=sha256_file(directory/'oof.csv.gz'),
            fit_seconds=sum(r['fit_seconds'] for r in records))
        write(directory/'metrics.json',result)
        meta.update(status='complete',finished_at_utc=now(),artifact_sha256={p.name:sha256_file(p)
            for p in directory.iterdir() if p.name!='metadata.json'})
        write(directory/'metadata.json',meta)
        return result
    except Exception as exc:
        meta.update(status='failed',failed_fits=meta['attempted_fits']-meta['completed_fits'],
            error=f'{type(exc).__name__}: {exc}')
        write(directory/'metadata.json',meta)
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','fit','fit-all'])
    p.add_argument('--data',default='data');p.add_argument('--t5',required=True)
    p.add_argument('--baseline',required=True);p.add_argument('--output',required=True)
    p.add_argument('--code-commit');p.add_argument('--scheme');p.add_argument('--variant')
    a=p.parse_args()
    if a.action=='prepare':
        prepare(a.data,a.t5,a.baseline,a.output,a.code_commit)
    elif a.action=='fit':
        fit(a.data,a.t5,a.baseline,a.output,a.scheme,a.variant)
    else:
        require(not any((Path(a.output)/s/v).exists() for s in config()['schemes']
            for v in config()['variants']), 'fit-all refuses partial or existing fits')
        for scheme in config()['schemes']:
            for variant in config()['variants']:
                fit(a.data,a.t5,a.baseline,a.output,scheme,variant)

if __name__=='__main__':
    main()
