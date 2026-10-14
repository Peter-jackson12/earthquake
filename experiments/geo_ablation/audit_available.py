"""Audit only the explicitly documented tool-interrupted T6 execution.

Does not train, retry, impute absent predictions, or alter the frozen audit/run.
Exactly eight complete configurations and two saved partial folds are required.
The missing stratified/R12 full-CV result is never calculated or reported.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from experiments.geo_ablation.audit import (
    cfg,reference,fingerprints,inspect,turnover,original,arithmetic,check_oof,
    digest,equal,need,read_json,write_json,ROOT,PROBS)


def execute(data,t5,baseline,output,destination):
    output,destination=Path(output),Path(destination)
    need(not destination.exists(),'Existing audit must not be overwritten')
    destination.mkdir(parents=True)
    design=read_json(output/'design.json');seal=read_json(output/'design_seal.json')
    equal(digest(output/'design.json'),seal['design_sha256'],'Frozen design')
    equal(design['config'],cfg(),'Frozen config')
    need(design['classifiers_fitted']==seal['classifiers_fitted']==0,'Design not pre-fit')
    for n,h in design['source_sha256'].items():equal(digest(ROOT/n),h,'Frozen source '+n)
    equal(fingerprints(t5,baseline),design['baseline_file_sha256'],'Original evidence')
    interruption=read_json(output/'execution_interruption.json')
    need(interruption['event']=='EXECUTION_TOOL_WALL_TIME_INTERRUPT' and
         interruption['scheme']=='stratified' and interruption['variant']=='R12' and
         interruption['interrupted_fold']==2 and interruption['unattempted_folds']==[3,4] and
         interruption['saved_complete_folds']==[0,1] and interruption['automatic_retries']==0,
         'Unexpected interruption pattern')
    ref=reference(data,t5,baseline,reload_models=True,reconstruct=True)
    equal(design['baseline_models'],ref['models'],'Existing C results')
    X,y,ids=ref['X'],ref['y'],ref['ids']
    models,folds_out,predictions=ref['models'],ref['fold_rows'],ref['predictions']
    inspections=[];partial=[];partial_oof=[];counts={'attempted':0,'completed':0}
    for scheme in cfg()['schemes']:
        assignments=ref['assignments'][scheme];oofs={};records={}
        if scheme!='stratified':equal(digest(output/scheme/'manifest.csv'),cfg()['manifest_sha256'][scheme],'Used manifest')
        for variant in cfg()['variants']:
            directory=output/scheme/variant;meta=read_json(directory/'metadata.json')
            equal(meta['source_sha256'],design['source_sha256'],'Trained source')
            equal(meta['design_sha256'],seal['design_sha256'],'Trained design')
            equal(meta['environment'],design['environment'],'Trained environment')
            equal(meta['code_commit'],design['code_commit'],'Trained commit')
            need(meta['started_at_utc']>seal['created_at_utc'] and not meta['holdout_accessed'] and meta['retries']==0,'Execution boundary')
            records[variant]=read_json(directory/'folds.json')
            counts['attempted']+=meta['attempted_fits'];counts['completed']+=meta['completed_fits']
            if (scheme,variant)==('stratified','R12'):
                need(meta['status']=='running' and meta['attempted_fits']==3 and meta['completed_fits']==2,'Interrupted metadata changed')
                need([r['fold'] for r in records[variant]]==[0,1],'Partial fold set changed')
                need(not (directory/'oof.csv.gz').exists() and not (directory/'metrics.json').exists(),'Invented complete partial output')
                need(all(not (directory/f'fold_{f}.joblib').exists() for f in [2,3,4]),'Unexpected retraining')
            else:
                need(meta['status']=='complete' and meta['attempted_fits']==meta['completed_fits']==5 and meta['failed_fits']==0,'New independent run incomplete')
                need([r['fold'] for r in records[variant]]==list(range(5)),'Incomplete fold set')
                for n,h in meta['artifact_sha256'].items():equal(digest(directory/n),h,'New artifact')
                oofs[variant]=check_oof(pd.read_csv(directory/'oof.csv.gz',float_precision='round_trip'),ids,y,assignments)
        for f in range(5):
            a,b=np.flatnonzero(assignments!=f),np.flatnonzero(assignments==f)
            with threadpool_limits(limits=1):
                if scheme=='stratified':cpre=original('C')['preprocess'].fit(X.iloc[a])
                else:
                    path=Path(t5)/scheme/'C'/f'fold_{f}.joblib'
                    equal(digest(path),read_json(Path(t5)/scheme/'C'/'folds.json')[f]['model_sha256'],'C model')
                    cpre=joblib.load(path)['preprocess']
                for variant in cfg()['variants']:
                    if scheme=='stratified' and variant=='R12' and f>=2:continue
                    directory=output/scheme/variant;r=records[variant][f]
                    need(r['converged'] and r['reload_exact'],'Nonconvergence/reload failure')
                    path=directory/f'fold_{f}.joblib';equal(digest(path),r['model_sha256'],'Model SHA')
                    pipe=joblib.load(path);info=inspect(pipe,variant,X.iloc[a],X.iloc[b],cpre)
                    equal(info['median'],r['median'],'Median');equal(info['encoded_columns'],r['encoded_columns'],'Width')
                    equal(pipe['model'].n_iter_.tolist(),r['n_iter'],'Iterations')
                    np.testing.assert_array_equal(pipe['model'].classes_,[1,2,3])
                    pred=pipe.predict(X.iloc[b]);prob=pipe.predict_proba(X.iloc[b])
                    calculated=arithmetic(y.iloc[b],pred);equal(calculated,r['metrics'],'Independent fold metrics')
                    if variant in oofs:
                        np.testing.assert_array_equal(pred,oofs[variant].y_pred.iloc[b])
                        np.testing.assert_allclose(prob,oofs[variant][PROBS].iloc[b],atol=1e-12,rtol=0)
                        folds_out.append(dict(scheme=scheme,variant=variant,fold=f,**calculated))
                    else:
                        need(np.isfinite(prob).all() and (prob>=0).all() and (prob<=1).all(),'Partial probabilities')
                        np.testing.assert_allclose(prob.sum(1),1,atol=1e-12,rtol=0)
                        np.testing.assert_array_equal(pred,prob.argmax(1)+1)
                        partial.append(dict(scheme=scheme,variant=variant,fold=f,status='partial_fold_only',**calculated))
                        partial_oof.append(pd.DataFrame({'building_id':ids.iloc[b],'fold':f,'y_true':y.iloc[b],'y_pred':pred,**{p:prob[:,i] for i,p in enumerate(PROBS)}}))
                    inspections.append(dict(scheme=scheme,variant=variant,fold=f,**info))
                    print('AUDIT',scheme,variant,f,'PASS',flush=True)
        for variant,oof in oofs.items():
            directory=output/scheme/variant;pred=oof.y_pred.to_numpy()
            predictions[scheme,variant]=pred;pooled=arithmetic(y,pred)
            values=[arithmetic(y[assignments==f],pred[assignments==f])['macro_f1'] for f in range(5)]
            mean=sum(values)/5;sd=float(np.sqrt(sum((v-mean)**2 for v in values)/4))
            saved=read_json(directory/'metrics.json')
            equal(saved['cv_mean'],mean,'CV mean');equal(saved['cv_std'],sd,'Sample SD')
            equal(saved['pooled'],pooled,'Pooled metrics');need(saved['all_converged'],'Nonconvergence')
            equal(saved['effective_parameters'],original('C')['model'].get_params(),'LR parameters')
            equal(saved['oof_sha256'],digest(directory/'oof.csv.gz'),'OOF SHA')
            models.append(dict(scheme=scheme,variant=variant,cv_mean=mean,cv_std=sd,**pooled,new_fits=5,oof_sha256=digest(directory/'oof.csv.gz')))
    need(counts=={'attempted':43,'completed':42} and len(inspections)==42 and len(models)==11 and len(partial)==2,'Unexpected training budget')
    profiles=[];paired=[]
    masks={'all':np.ones(len(X),bool),'age995':X.age.eq(995).to_numpy(),'non995':X.age.ne(995).to_numpy()}
    masks.update({'geo1='+g:X.geo_level_1_id.eq(g).to_numpy() for g in sorted(X.geo_level_1_id.unique(),key=int)})
    for (scheme,variant),pred in predictions.items():
        for name,mask in masks.items():profiles.append(dict(scheme=scheme,variant=variant,group=name,**arithmetic(y[mask],pred[mask])))
    for scheme in cfg()['schemes']:
        groups={**masks,**{f'fold={f}':ref['assignments'][scheme]==f for f in range(5)}}
        for variant in cfg()['variants']:
            if (scheme,variant) not in predictions:continue
            for name,mask in groups.items():paired.append(dict(scheme=scheme,variant=variant,group=name,**turnover(y[mask],predictions[scheme,'C'][mask],predictions[scheme,variant][mask])))
    equal(fingerprints(t5,baseline),design['baseline_file_sha256'],'Final reference immutability')
    equal({n:digest(Path(data)/n) for n in ref['contract']['files']},ref['contract']['files'],'Final data')
    for n,h in design['source_sha256'].items():equal(digest(ROOT/n),h,'Final frozen source')
    for name,rows in [('models',models),('folds',folds_out),('partial_folds',partial),('profiles',profiles),('paired',paired),('preprocessing',inspections),('split_checks',ref['split_checks'])]:
        write_json(destination/(name+'.json'),rows)
        if name!='preprocessing':pd.DataFrame(rows).to_csv(destination/(name+'.csv'),index=False)
    pd.concat(partial_oof).sort_values('building_id').to_csv(destination/'partial_stratified_R12.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    result=dict(study_id=cfg()['study_id'],status='PARTIAL_PASS_AVAILABLE_EVIDENCE',complete_study=False,
        audited_at_utc=datetime.now(timezone.utc).isoformat(),code_commit=design['code_commit'],design_sha256=seal['design_sha256'],
        audit_code_sha256=digest(__file__),new_classifier_fit_attempts=43,new_classifier_fits_completed=42,
        interrupted_fits=1,unattempted_fits=2,automatic_retries=0,old_classifier_refits=0,
        complete_new_configurations=8,baseline_configurations=3,unavailable_full_cv=['stratified/R12'],
        new_models_full_validation_reloaded=42,baseline_C_models_full_validation_reloaded=10,
        reference_preprocessor_only_fits=5,full_train_valid_projection_checks=42,excluded_geography_invariance_checks=42,
        full_model_fold_metrics_checked=len(folds_out),partial_fold_metrics_checked=2,profiles=len(profiles),paired_groups=len(paired),
        new_complete_oof_rows=8*len(X),new_partial_oof_rows=sum(len(p) for p in partial_oof),reused_oof_rows=3*len(X),
        unique_development_rows=len(X),holdout_accessed=False,data_sha256=ref['contract']['files'],
        output_sha256={p.name:digest(p) for p in destination.iterdir()},
        limitations=['Stratified/R12 fold2 interrupted by execution tool, folds3/4 unattempted. No full CV mean/SD/pooled metrics.',
          'Five old stratified C classifiers not stored; OOF/config/metrics verified, model reload unavailable.',
          'Reload is not an independent classifier training reproduction; descriptive development-only analysis, no causal/deployment claim.'])
    write_json(destination/'audit.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',default='data');p.add_argument('--t5',required=True);p.add_argument('--baseline',required=True)
    p.add_argument('--output',required=True);p.add_argument('--destination',required=True)
    a=p.parse_args();execute(a.data,a.t5,a.baseline,a.output,a.destination)

if __name__=='__main__':main()
