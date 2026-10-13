"""Run only the three frozen personal candidates; holdout is never opened here."""
from __future__ import annotations
import argparse, json, time, warnings, sys, gc, platform
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from personal_sampling import PersonalSamplingLR
from preprocess import ID,TARGET,sha256_file,write_json,CATEGORICAL
from scripts.audit_eda_oof import development_frame,validate_oof,verify_metrics
from baseline import metrics

MODES=['none','random_over','context_smote']

def run(mode, output, data_dir=ROOT/'data'):
    if mode not in MODES: raise ValueError(mode)
    output=Path(output)
    if output.exists() and any(output.iterdir()):raise FileExistsError('Refuse overwrite')
    output.mkdir(parents=True,exist_ok=True)
    df,contract=development_frame(data_dir)
    X=df.drop(columns=[ID,TARGET,'fold']);y=df[TARGET]
    pred=np.zeros(len(df),dtype=np.uint8);prob=np.full((len(df),3),np.nan);seen=np.zeros(len(df),dtype=int)
    metadata=dict(status='running',scope='full_development_5fold',rows=len(df),holdout_scored=False,
                  config={'name':'personal_'+mode+'_v1','mode':mode,'C':1,'seed':42,'k':5},
                  models=['lr'],data_contract=contract,source_sha256={
                    p:sha256_file(ROOT/p) for p in ['personal_sampling.py','preprocess.py','scripts/run_personal_sampling.py']},
                  python=platform.python_version())
    write_json(output/'run_metadata.json',metadata)
    df[[ID,'fold']].rename(columns={'fold':'cv_fold'}).to_csv(output/'run_manifest.csv',index=False)
    (output/'lr').mkdir();records=[]
    for f in range(5):
        train=np.flatnonzero(df.fold.to_numpy()!=f);valid=np.flatnonzero(df.fold.to_numpy()==f)
        model=PersonalSamplingLR(mode=mode)
        start=time.perf_counter()
        with warnings.catch_warnings(record=True) as caught,threadpool_limits(limits=1):
            warnings.simplefilter('always');model.fit(X.iloc[train],y.iloc[train])
        fit_seconds=time.perf_counter()-start
        with threadpool_limits(limits=1):
            pr=model.predict(X.iloc[valid]);pp=model.predict_proba(X.iloc[valid]);tp=model.predict(X.iloc[train])
        if model.classes_.tolist()!=[1,2,3]:raise AssertionError('Class order')
        for c,values in zip(CATEGORICAL,model.preprocess_['columns'].named_transformers_['categorical'].categories_):
            if set(values)!=set(X.iloc[train][c]):raise AssertionError('Train-only vocabulary')
        pred[valid]=pr;prob[valid]=pp;seen[valid]+=1
        row=dict(fold=f,train_rows=len(train),validation_rows=len(valid),train_macro_f1=metrics(y.iloc[train],tp)['macro_f1'],
                 validation=metrics(y.iloc[valid],pr),fit_seconds=fit_seconds,
                 encoded_columns=int(model.preprocess_.transform(X.iloc[valid[:1]]).shape[1]),
                 warnings=[{'category':type(w.message).__name__,'message':str(w.message)} for w in caught],
                 convergence_ok=not any(issubclass(w.category,ConvergenceWarning) for w in caught),
                 n_iter=model.model_.n_iter_.tolist(),train_only_vocabulary=True,
                 sampling=model.sampling_diagnostics_,fit_rows=model.fit_rows_)
        if f==0:
            path=output/'lr/fold0_model.joblib';joblib.dump(model,path,compress=3)
            restored=joblib.load(path)
            np.testing.assert_allclose(restored.predict_proba(X.iloc[valid[:32]]),pp[:32],atol=1e-12)
            del restored
        records.append(row);write_json(output/'lr/fold_metrics.json',records)
        print(mode,'fold',f,'F1',row['validation']['macro_f1'],'fit',round(fit_seconds,2),'sampling',row['sampling'],flush=True)
        del model;gc.collect()
    assert (seen==1).all() and np.isfinite(prob).all()
    oof=pd.DataFrame({ID:df[ID],'fold':df.fold,'y_true':y,'y_pred':pred,
                      'p_grade_1':prob[:,0],'p_grade_2':prob[:,1],'p_grade_3':prob[:,2]})
    oof.to_csv(output/'lr/oof_predictions.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    scores=[r['validation']['macro_f1'] for r in records]
    result=dict(model='lr',rows=len(df),folds=5,status='complete',holdout_scored=False,
                cv_macro_f1_mean=float(np.mean(scores)),cv_macro_f1_std=float(np.std(scores,ddof=1)),
                total_fit_seconds=sum(r['fit_seconds'] for r in records),all_folds_converged=all(r['convergence_ok'] for r in records),
                oof=metrics(y,pred),fold_results=records,exactly_one_oof_per_row=True,
                effective_parameters=PersonalSamplingLR(mode=mode).get_params(),
                oof_sha256=sha256_file(output/'lr/oof_predictions.csv.gz'))
    joined=validate_oof(df,oof);verify_metrics(joined,result)
    write_json(output/'lr/metrics.json',result);write_json(output/'results.json',[result])
    metadata['status']='complete';write_json(output/'run_metadata.json',metadata)
    print('COMPLETE',mode,round(result['cv_macro_f1_mean'],8),flush=True)
    return result

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=MODES,required=True);p.add_argument('--output',required=True)
 a=p.parse_args();run(a.mode,a.output)
