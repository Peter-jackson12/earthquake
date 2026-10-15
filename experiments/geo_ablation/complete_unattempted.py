"""Operational completion of only stratified/R12 folds3/4; never retries fold2.

Preserves every original byte. The original pre-fit design still determines
features, parameters, data and splits. This supplement is sealed before its fits.
"""
import argparse,json,warnings,time
from pathlib import Path
from datetime import datetime,timezone
import joblib,numpy as np,pandas as pd
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from experiments.geo_ablation.run import verify,make_pipeline,sources,environment,score
from experiments.geo_ablation.audit import digest,equal,need,write_json,read_json,inspect,original,arithmetic,PROBS


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['data','t5','baseline','output','destination','code-commit']:p.add_argument('--'+name,required=True)
    a=p.parse_args();out=Path(a.output);dest=Path(a.destination)
    need(not dest.exists(),'Never overwrite a completion run')
    X,y,ids,assignments,design=verify(a.data,a.t5,a.baseline,out)
    available=read_json(out/'audit_available/audit.json')
    need(available['status']=='PARTIAL_PASS_AVAILABLE_EVIDENCE' and available['new_classifier_fit_attempts']==43,'Expected audited 43-attempt prefix')
    need(len(a.code_commit)==40 and all(c in '0123456789abcdef' for c in a.code_commit),'Real supplement commit required')
    before={str(p.relative_to(out)):digest(p) for p in out.rglob('*') if p.is_file()}
    need(sum(read_json(p)['attempted_fits'] for p in out.glob('*/*/metadata.json'))==43,'Original attempts changed')
    dest.mkdir(parents=True)
    supplement=dict(folds=[3,4],scheme='stratified',variant='R12',no_retry_fold=2,
        old_design_sha256=digest(out/'design.json'),source_sha256=sources(),supplement_code_sha256=digest(__file__),
        code_commit=a.code_commit,environment=environment(),original_evidence_sha256=before,
        created_at_utc=datetime.now(timezone.utc).isoformat(),attempts_before=43,max_total_attempts=45,classifiers_fitted=0)
    write_json(dest/'supplement_design.json',supplement)
    write_json(dest/'supplement_seal.json',dict(design_sha256=digest(dest/'supplement_design.json'),created_at_utc=datetime.now(timezone.utc).isoformat(),classifiers_fitted=0))
    records=[];attempted=completed=0;predicted=[]
    for f in [3,4]:
        need(43+attempted<45,'Total attempt budget exceeded')
        need(not (out/'stratified/R12'/f'fold_{f}.joblib').exists(),'Fold already trained')
        tr,va=np.flatnonzero(assignments['stratified']!=f),np.flatnonzero(assignments['stratified']==f)
        attempted+=1;write_json(dest/'progress.json',dict(attempted=attempted,completed=completed,status='running'))
        pipe=make_pipeline('R12');start=time.perf_counter()
        with threadpool_limits(limits=1),warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always');pipe.fit(X.iloc[tr],y.iloc[tr])
        completed+=1
        need(not any(issubclass(w.category,ConvergenceWarning) for w in caught),'Nonconvergence; no retry')
        with threadpool_limits(limits=1):
            pred=pipe.predict(X.iloc[va]);prob=pipe.predict_proba(X.iloc[va])
        path=dest/f'fold_{f}.joblib';joblib.dump(pipe,path,compress=3)
        record=dict(fold=f,train_rows=len(tr),valid_rows=len(va),metrics=score(y.iloc[va],pred),model_sha256=digest(path),
            fit_and_prediction_seconds=time.perf_counter()-start,n_iter=pipe['model'].n_iter_.tolist(),
            warnings=[dict(category=w.category.__name__,message=str(w.message)) for w in caught],converged=True)
        records.append(record);write_json(dest/'folds.json',records)
        predicted.append(pd.DataFrame({'building_id':ids.iloc[va],'fold':f,'y_true':y.iloc[va],'y_pred':pred,**{p:prob[:,i] for i,p in enumerate(PROBS)}}))
        write_json(dest/'progress.json',dict(attempted=attempted,completed=completed,status='running'))
        print('SUPPLEMENT FIT',f,'COMPLETE',flush=True)
    oof=pd.concat(predicted).sort_values('building_id').reset_index(drop=True)
    oof.to_csv(dest/'partial_oof.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    # Separate reload/statistic/matrix/metric verification after both fits.
    reread=pd.read_csv(dest/'partial_oof.csv.gz',float_precision='round_trip')
    need(reread.building_id.is_unique and len(reread)==83392 and set(reread.fold)=={3,4},'Supplement partial OOF IDs')
    inspections=[]
    for f in [3,4]:
        tr,va=np.flatnonzero(assignments['stratified']!=f),np.flatnonzero(assignments['stratified']==f)
        rows=reread.loc[reread.fold.eq(f)].sort_values('building_id')
        np.testing.assert_array_equal(rows.building_id,ids.iloc[va]);np.testing.assert_array_equal(rows.y_true,y.iloc[va])
        path=dest/f'fold_{f}.joblib';equal(digest(path),records[f-3]['model_sha256'],'Supplement model SHA')
        with threadpool_limits(limits=1):
            pipe=joblib.load(path);cpre=original('C')['preprocess'].fit(X.iloc[tr])
            info=inspect(pipe,'R12',X.iloc[tr],X.iloc[va],cpre)
            np.testing.assert_array_equal(pipe['model'].classes_,[1,2,3])
            np.testing.assert_array_equal(pipe.predict(X.iloc[va]),rows.y_pred)
            np.testing.assert_allclose(pipe.predict_proba(X.iloc[va]),rows[PROBS],atol=1e-12,rtol=0)
        need(np.isfinite(rows[PROBS]).all().all() and (rows[PROBS]>=0).all().all() and (rows[PROBS]<=1).all().all(),'Probabilities')
        np.testing.assert_allclose(rows[PROBS].sum(axis=1),1,atol=1e-12,rtol=0)
        np.testing.assert_array_equal(rows[PROBS].to_numpy().argmax(1)+1,rows.y_pred)
        equal(arithmetic(rows.y_true,rows.y_pred),records[f-3]['metrics'],'Independent arithmetic')
        inspections.append(dict(fold=f,**info));print('SUPPLEMENT AUDIT',f,'PASS',flush=True)
    need(before=={str(p.relative_to(out)):digest(p) for p in out.rglob('*') if p.is_file()},'Original evidence modified')
    verify(a.data,a.t5,a.baseline,out)
    write_json(dest/'inspections.json',inspections)
    write_json(dest/'progress.json',dict(attempted=attempted,completed=completed,status='complete'))
    result=dict(status='PASS_SUPPLEMENT_ONLY',complete_study=False,attempts=2,completed_fits=2,interrupted_fits=0,
        total_attempts=45,total_completed_fits=44,total_interrupted_fits=1,total_unattempted_fits=0,
        missing_fold='stratified/R12/fold2',missing_full_cv='stratified/R12',old_classifier_refits=0,
        source_sha256=sources(),supplement_code_sha256=digest(__file__),code_commit=a.code_commit,
        full_reload_matrix_checks=2,reference_preprocessor_only_fits=2,holdout_accessed=False,
        no_full_stratified_R12_metrics=True,artifact_sha256={p.name:digest(p) for p in dest.iterdir() if p.is_file()})
    write_json(dest/'audit.json',result)

if __name__=='__main__':main()
