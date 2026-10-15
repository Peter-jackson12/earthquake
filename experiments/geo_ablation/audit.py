"""Independent T6 verification and aggregate analysis; never fits a classifier.

Uses the independently implemented T5 confusion-matrix arithmetic, not the
training score function. Only checksum-verified, trusted study joblib is loaded.
"""
from __future__ import annotations
import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits
from preprocess import FEATURES, GEO, NUMERIC, BINARY, CATEGORICAL, load_development
from experiments.age995.run import make_pipeline as original, FLAG
from experiments.age995.interactions.run import verify_C
from experiments.geo_group_cv.audit import (
    arithmetic, check_oof, digest, equal, need, old_metrics, read_json, write_json,
    inspect_preprocessor as inspect_C, PROBS)

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).parent


def cfg():
    return read_json(HERE/'config.json')


def fingerprints(t5,baseline):
    t5,baseline=Path(t5),Path(baseline)
    paths=[('t5',t5,p) for p in ['design.json','design_seal.json','audit/audit.json',
        'evidence/baseline_C_verified.json','geo2/manifest.csv','geo1/manifest.csv']]
    paths += [('t5',t5,str(p.relative_to(t5))) for s in ['geo2','geo1']
        for p in sorted((t5/s/'C').iterdir()) if p.is_file()]
    paths += [('baseline',baseline,str(p.relative_to(baseline)))
        for p in sorted((baseline/'age995_abc_v1/C').rglob('*')) if p.is_file()]
    return {prefix+'/'+name:digest(root/name) for prefix,root,name in paths}


def reference(data,t5,baseline,*,reload_models,reconstruct):
    """Verify three existing C results and exact split assignments, no refitting."""
    data,t5,baseline=Path(data),Path(t5),Path(baseline)
    contract=read_json(ROOT/'configs/data_contract.json')
    equal({n:digest(data/n) for n in contract['files']},contract['files'],'data contract')
    equal(digest(data/'split_manifest.csv'),cfg()['manifest_sha256']['stratified'],'original split SHA')
    X,y,ids,cv=load_development(data)
    order=np.argsort(ids.to_numpy(),kind='stable')
    X,y,ids=(z.iloc[order].reset_index(drop=True) for z in (X,y,ids))
    old=cv.test_fold[order]
    need(len(X)==208480 and list(X)==FEATURES and ids.is_unique,'Development shape/ID mismatch')
    frame=X.copy();frame['building_id']=ids;frame['damage_grade']=y;frame['fold']=old
    verified=verify_C(frame,contract,baseline/'age995_abc_v1/C')
    equal(verified,read_json(t5/'evidence/baseline_C_verified.json'),'original C receipt')
    design=read_json(t5/'design.json');seal=read_json(t5/'design_seal.json')
    equal(digest(t5/'design.json'),seal['design_sha256'],'T5 design seal')
    equal(design['data_contract'],contract,'T5 data contract')
    equal(design['config'],read_json(ROOT/'experiments/geo_group_cv/config.json'),'T5 config')
    for name,h in design['source_sha256'].items():
        equal(digest(ROOT/name),h,'T5 dependency '+name)
    assignments={'stratified':old};split_checks=[]
    for scheme,columns in [('geo2',GEO[:2]),('geo1',GEO[:1])]:
        path=t5/scheme/'manifest.csv'
        equal(digest(path),cfg()['manifest_sha256'][scheme],'T5 manifest SHA')
        equal(design['schemes'][scheme]['manifest_sha256'],digest(path),'T5 recorded manifest')
        m=pd.read_csv(path,dtype={c:str for c in columns})
        np.testing.assert_array_equal(m.building_id,ids)
        need(m.building_id.is_unique and len(m)==len(X),'Manifest duplicates/missing IDs')
        for col in columns: np.testing.assert_array_equal(m[col],X[col])
        tuples=[tuple(int(v) for v in row) for row in X[columns].itertuples(index=False,name=None)]
        lookup={key:i for i,key in enumerate(sorted(set(tuples)))}
        g=np.array([lookup[t] for t in tuples]);f=m.fold.to_numpy()
        need(len(lookup)==(1408 if scheme=='geo2' else 31),'Group population changed')
        if reconstruct:
            recovered=np.full(len(X),-1,dtype=int)
            for fold,(_,b) in enumerate(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(X,y,g)):
                recovered[b]=fold
            np.testing.assert_array_equal(recovered,f)
            replay=pd.DataFrame({'building_id':ids,'fold':recovered,**{c:X[c] for c in columns}})
            equal(hashlib.sha256(replay.to_csv(index=False,lineterminator='\n').encode()).hexdigest(),
                cfg()['manifest_sha256'][scheme],'reconstructed file SHA')
        for fold in range(5):
            a,b=np.flatnonzero(f!=fold),np.flatnonzero(f==fold)
            need(not set(g[a])&set(g[b]) and not set(ids.iloc[a])&set(ids.iloc[b]),'Group/ID leakage')
            ta=[int((y.iloc[a]==k).sum()) for k in [1,2,3]]
            va=[int((y.iloc[b]==k).sum()) for k in [1,2,3]]
            need(min(ta+va)>=30 and .01<=len(b)/len(X)<=.5,'Split support/size gate failed')
            split_checks.append(dict(scheme=scheme,fold=fold,train_rows=len(a),valid_rows=len(b),
                train_support=ta,valid_support=va,group_overlap=0,id_overlap=0))
        need(pd.DataFrame({'g':g,'f':f}).groupby('g').f.nunique().eq(1).all(),'Group split across folds')
        assignments[scheme]=f
    models,predictions,fold_rows=[],{},[]
    expected_params=original('C')['model'].get_params()
    for scheme in cfg()['schemes']:
        directory=baseline/'age995_abc_v1/C/lr' if scheme=='stratified' else t5/scheme/'C'
        path=directory/('oof_predictions.csv.gz' if scheme=='stratified' else 'oof.csv.gz')
        equal(digest(path),cfg()['C_oof_sha256'][scheme],'C OOF pinned SHA')
        oof=check_oof(pd.read_csv(path,float_precision='round_trip'),ids,y,assignments[scheme])
        saved=read_json(directory/'metrics.json');records=read_json(directory/('fold_metrics.json' if scheme=='stratified' else 'folds.json'))
        equal(saved['effective_parameters'],expected_params,'C effective parameters')
        if scheme!='stratified':
            meta=read_json(directory/'metadata.json')
            need(meta['status']=='complete' and meta['completed_fits']==meta['attempted_fits']==5,'C training incomplete')
            need(not meta['holdout_accessed'],'C holdout flag')
            equal(meta['source_sha256'],design['source_sha256'],'C source')
            equal(meta['design_sha256'],digest(t5/'design.json'),'C trained split design')
            for n,h in meta['artifact_sha256'].items(): equal(digest(directory/n),h,'C artifact')
        scores=[]
        for f,r in enumerate(records):
            need(r['fold']==f,'C fold records out of order')
            a,b=np.flatnonzero(assignments[scheme]!=f),np.flatnonzero(assignments[scheme]==f)
            calculated=arithmetic(y.iloc[b],oof.y_pred.iloc[b]);scores.append(calculated['macro_f1'])
            if scheme=='stratified': old_metrics(r['validation'],calculated)
            else:
                equal(r['metrics'],calculated,'C fold metrics')
                if reload_models:
                    equal(digest(directory/f'fold_{f}.joblib'),r['model_sha256'],'C model SHA')
                    pipe=joblib.load(directory/f'fold_{f}.joblib')
                    equal(pipe['model'].get_params(),expected_params,'C loaded parameters')
                    with threadpool_limits(limits=1):
                        info,_=inspect_C(pipe,'C',X.iloc[a],X.iloc[b])
                        equal(r['preprocessing'],info,'C learned statistics')
                        np.testing.assert_array_equal(pipe.predict(X.iloc[b]),oof.y_pred.iloc[b])
                        np.testing.assert_allclose(pipe.predict_proba(X.iloc[b]),oof[PROBS].iloc[b],atol=1e-12,rtol=0)
            fold_rows.append(dict(scheme=scheme,variant='C',fold=f,**calculated))
        pooled=arithmetic(y,oof.y_pred)
        if scheme=='stratified': old_metrics(saved['oof'],pooled)
        else: equal(saved['pooled'],pooled,'C pooled')
        equal(saved['cv_macro_f1_mean'] if scheme=='stratified' else saved['cv_mean'],float(np.mean(scores)),'C CV mean')
        equal(saved['cv_macro_f1_std'] if scheme=='stratified' else saved['cv_std'],float(np.std(scores,ddof=1)),'C CV SD')
        predictions[scheme,'C']=oof.y_pred.to_numpy()
        models.append(dict(scheme=scheme,variant='C',cv_mean=float(np.mean(scores)),cv_std=float(np.std(scores,ddof=1)),
            **pooled,oof_sha256=digest(path),new_fits=0))
    return dict(X=X,y=y,ids=ids,assignments=assignments,contract=contract,models=models,
        predictions=predictions,fold_rows=fold_rows,split_checks=split_checks)


def inspect(pipe,variant,train,valid,reference_pre):
    """Independently inspect all retained inputs and matrix equivalence to C."""
    excluded=cfg()['variants'][variant];retained=[c for c in CATEGORICAL if c not in excluded]
    pre=pipe['preprocess'];cols=pre['columns']
    equal(pipe['model'].get_params(),original('C')['model'].get_params(),'LR unchanged')
    need([n for n,_ in pre.steps]==['schema','age995','columns'],'Unexpected feature transformation path')
    equal(pre['schema'].feature_names_in_.tolist(),FEATURES,'Raw schema')
    need(pre['schema'].n_features_in_==38 and pre['age995'].mode=='C','Input/age policy changed')
    expected={'numeric':NUMERIC,'binary':BINARY+[FLAG],'categorical':retained}
    need(cols.remainder=='drop','Remainder can leak excluded variables')
    for name,transformer,columns in cols.transformers_:
        if name=='remainder': need(transformer=='drop','Unexpected remainder');continue
        need(name in expected and list(columns)==expected[name],'Unexpected selected columns')
    names=cols.get_feature_names_out().tolist()
    original_names=reference_pre['columns'].get_feature_names_out().tolist()
    selection=[i for i,n in enumerate(original_names) if not any(n.startswith('categorical__'+c+'_') for c in excluded)]
    equal(names,[original_names[i] for i in selection],'Projection feature names')
    need(not any(c in n for n in names for c in excluded+['building_id','damage_grade','fold']),'Leaked feature name')
    median=float(np.median(train.age.to_numpy()[train.age.to_numpy()!=995]))
    need(pre['age995'].median_==median,'Median not training-only')
    numeric=train[NUMERIC].to_numpy(dtype=np.float32).copy()
    numeric[:,NUMERIC.index('age')]=np.where(train.age.to_numpy()==995,median,train.age.to_numpy())
    numeric=numeric.astype(np.float64);scaler=cols.named_transformers_['numeric']['scale']
    need(int(scaler.n_samples_seen_)==len(train),'Training-statistic denominator')
    np.testing.assert_allclose(scaler.mean_,numeric.mean(0),atol=1e-10,rtol=0)
    np.testing.assert_allclose(scaler.var_,numeric.var(0),atol=1e-8,rtol=1e-10)
    np.testing.assert_allclose(scaler.scale_,np.where(numeric.var(0)==0,1,np.sqrt(numeric.var(0))),atol=1e-9,rtol=1e-10)
    enc=cols.named_transformers_['categorical']
    need(enc.get_params()==reference_pre['columns'].named_transformers_['categorical'].get_params(),'OHE options mismatch')
    equal(enc.feature_names_in_.tolist(),retained,'OHE retained names')
    categories,offset=[],0
    encoded=enc.transform(valid[retained])
    for col in CATEGORICAL:
        unknown=~valid[col].isin(set(train[col])).to_numpy()
        used=col in retained
        if used:
            learned=enc.categories_[retained.index(col)]
            need(set(learned)==set(train[col]),'Nontraining vocabulary')
            need(encoded[unknown,offset:offset+len(learned)].nnz==0,'Unknown block not zero')
            offset+=len(learned)
        categories.append(dict(feature=col,used=used,train_categories=int(train[col].nunique()),
            valid_categories=int(valid[col].nunique()),unknown_rows=int(unknown.sum()),
            unknown_unique=int(valid.loc[unknown,col].nunique()),unknown_block_zero=True if used else None))
    before=joblib.hash(pre)
    for data in [train,valid]:
        engineered=pre['age995'].transform(pre['schema'].transform(data))
        np.testing.assert_array_equal(engineered[FLAG],data.age.to_numpy()==995)
        np.testing.assert_array_equal(engineered.age,np.where(data.age.to_numpy()==995,median,data.age.to_numpy()))
        actual=pre.transform(data);base=reference_pre.transform(data)[:,selection]
        need(sparse.isspmatrix_csr(actual) and actual.dtype==np.float32,'Matrix representation changed')
        need(actual.shape==base.shape and (actual!=base).nnz==0,'Common C matrix differs')
    changed=valid.copy()
    for col in excluded: changed[col]='2147483647'
    need((pre.transform(changed)!=actual).nnz==0,'Excluded geography changes model matrix')
    np.testing.assert_array_equal(pipe.predict(changed),pipe.predict(valid))
    np.testing.assert_array_equal(pipe.predict_proba(changed),pipe.predict_proba(valid))
    need(joblib.hash(pre)==before,'Validation changed fitted preprocessing')
    return dict(median=median,encoded_columns=actual.shape[1],full_train_valid_C_projection_equal=True,
        excluded_value_invariance=True,train_only_statistics=True,categorical=categories)


def turnover(y,c,r):
    y,c,r=np.asarray(y),np.asarray(c),np.asarray(r)
    tensor=np.bincount((9*(y-1)+3*(c-1)+r-1).astype(int),minlength=27).reshape(3,3,3)
    a,b=arithmetic(y,c),arithmetic(y,r)
    result=dict(n=len(y),macro_f1_delta=b['macro_f1']-a['macro_f1'],recall3_delta=b['recall'][2]-a['recall'][2],
        error31_delta=b['error31']-a['error31'],error32_delta=b['error32']-a['error32'],
        corrected_predictions=int(((c!=y)&(r==y)).sum()),new_errors=int(((c==y)&(r!=y)).sum()),
        prediction_agreement=float(np.mean(c==r)),transition_true_C_R=tensor.tolist())
    for label,mask_c,mask_r in [('error31',(y==3)&(c==1),(y==3)&(r==1)),('error32',(y==3)&(c==2),(y==3)&(r==2))]:
        result[label+'_removed']=int((mask_c&~mask_r).sum())
        result[label+'_new']=int((~mask_c&mask_r).sum())
        result[label+'_common']=int((mask_c&mask_r).sum())
    return result


def audit(data,t5,baseline,output,destination):
    output,destination=Path(output),Path(destination)
    need(not destination.exists(),'Do not overwrite any prior audit')
    destination.mkdir(parents=True)
    design=read_json(output/'design.json');seal=read_json(output/'design_seal.json')
    equal(digest(output/'design.json'),seal['design_sha256'],'T6 design seal')
    need(design['classifiers_fitted']==seal['classifiers_fitted']==0,'Not pre-fit design')
    equal(design['config'],cfg(),'T6 frozen config')
    for n,h in design['source_sha256'].items(): equal(digest(ROOT/n),h,'T6 source '+n)
    equal(design['baseline_file_sha256'],fingerprints(t5,baseline),'Baseline immutability')
    ref=reference(data,t5,baseline,reload_models=True,reconstruct=True)
    equal(design['baseline_models'],ref['models'],'Pre-fit C baseline')
    X,y,ids=ref['X'],ref['y'],ref['ids']
    models,fold_rows,predictions=ref['models'],ref['fold_rows'],ref['predictions']
    inspections=[];attempted=completed=0;reference_pre_fits=0
    for scheme in cfg()['schemes']:
        folds=ref['assignments'][scheme]
        if scheme!='stratified': equal(digest(output/scheme/'manifest.csv'),cfg()['manifest_sha256'][scheme],'Used split')
        oofs={};records={};metas={}
        for variant in cfg()['variants']:
            directory=output/scheme/variant
            meta=read_json(directory/'metadata.json');metas[variant]=meta
            need(meta['status']=='complete' and meta['attempted_fits']==meta['completed_fits']==5,'Incomplete T6 fit')
            need(meta['failed_fits']==meta['retries']==0 and not meta['holdout_accessed'],'Run boundary violation')
            need(meta['started_at_utc']>seal['created_at_utc'],'Fit preceded design seal')
            equal(meta['source_sha256'],design['source_sha256'],'Fitted sources')
            equal(meta['code_commit'],design['code_commit'],'Training commit')
            equal(meta['design_sha256'],digest(output/'design.json'),'Fitted design')
            equal(meta['environment'],design['environment'],'Fitted environment')
            for n,h in meta['artifact_sha256'].items(): equal(digest(directory/n),h,'T6 artifact')
            oofs[variant]=check_oof(pd.read_csv(directory/'oof.csv.gz',float_precision='round_trip'),ids,y,folds)
            records[variant]=read_json(directory/'folds.json')
            need(len(records[variant])==5,'Missing fold records')
            attempted+=meta['attempted_fits'];completed+=meta['completed_fits']
        for f in range(5):
            a,b=np.flatnonzero(folds!=f),np.flatnonzero(folds==f)
            with threadpool_limits(limits=1):
                if scheme=='stratified':
                    cpre=original('C')['preprocess'].fit(X.iloc[a]);reference_pre_fits+=1
                else:
                    cpath=Path(t5)/scheme/'C'/f'fold_{f}.joblib'
                    equal(digest(cpath),read_json(Path(t5)/scheme/'C'/'folds.json')[f]['model_sha256'],'Reference C model')
                    cpre=joblib.load(cpath)['preprocess']
                for variant in cfg()['variants']:
                    directory=output/scheme/variant;r=records[variant][f];oof=oofs[variant]
                    need(r['fold']==f and r['converged'] and r['reload_exact'],'Fold completion mismatch')
                    calculated=arithmetic(y.iloc[b],oof.y_pred.iloc[b])
                    equal(r['metrics'],calculated,'T6 fold metric')
                    equal(digest(directory/f'fold_{f}.joblib'),r['model_sha256'],'T6 model SHA')
                    pipe=joblib.load(directory/f'fold_{f}.joblib')
                    info=inspect(pipe,variant,X.iloc[a],X.iloc[b],cpre)
                    equal(r['median'],info['median'],'Recorded median')
                    equal(r['encoded_columns'],info['encoded_columns'],'Recorded width')
                    equal(r['n_iter'],pipe['model'].n_iter_.tolist(),'Recorded iterations')
                    np.testing.assert_array_equal(pipe['model'].classes_,[1,2,3])
                    np.testing.assert_array_equal(pipe.predict(X.iloc[b]),oof.y_pred.iloc[b])
                    np.testing.assert_allclose(pipe.predict_proba(X.iloc[b]),oof[PROBS].iloc[b],atol=1e-12,rtol=0)
                    inspections.append(dict(scheme=scheme,variant=variant,fold=f,**info))
                    fold_rows.append(dict(scheme=scheme,variant=variant,fold=f,**calculated))
                    print('AUDIT',scheme,variant,f,'PASS',flush=True)
        for variant in cfg()['variants']:
            directory=output/scheme/variant;oof=oofs[variant]
            pred=oof.y_pred.to_numpy();predictions[scheme,variant]=pred
            values=[arithmetic(y[folds==f],pred[folds==f])['macro_f1'] for f in range(5)]
            mean=sum(values)/5;sd=float(np.sqrt(sum((v-mean)**2 for v in values)/4))
            saved=read_json(directory/'metrics.json');pooled=arithmetic(y,pred)
            equal(saved['pooled'],pooled,'Pooled independent metric')
            equal(saved['cv_mean'],mean,'CV mean');equal(saved['cv_std'],sd,'CV sample SD')
            equal(saved['effective_parameters'],original('C')['model'].get_params(),'Saved model parameters')
            need(saved['all_converged'],'Nonconvergence')
            equal(saved['oof_sha256'],digest(directory/'oof.csv.gz'),'Declared OOF SHA')
            models.append(dict(scheme=scheme,variant=variant,cv_mean=mean,cv_std=sd,**pooled,
                new_fits=5,oof_sha256=digest(directory/'oof.csv.gz')))
    profiles=[];paired=[]
    masks={'all':np.ones(len(X),dtype=bool),'age995':X.age.eq(995).to_numpy(),'non995':X.age.ne(995).to_numpy()}
    masks.update({'geo1='+g:X.geo_level_1_id.eq(g).to_numpy() for g in sorted(X.geo_level_1_id.unique(),key=int)})
    for (scheme,variant),pred in predictions.items():
        for name,mask in masks.items(): profiles.append(dict(scheme=scheme,variant=variant,group=name,**arithmetic(y[mask],pred[mask])))
    for scheme in cfg()['schemes']:
        groups={**masks,**{f'fold={f}':ref['assignments'][scheme]==f for f in range(5)}}
        for variant in cfg()['variants']:
            for name,mask in groups.items():
                paired.append(dict(scheme=scheme,variant=variant,group=name,
                    **turnover(y[mask],predictions[scheme,'C'][mask],predictions[scheme,variant][mask])))
    need(attempted==completed==45 and len(inspections)==45,'Expected exactly 45 fits/model audits')
    equal(design['baseline_file_sha256'],fingerprints(t5,baseline),'Preserved baseline after audit')
    equal({n:digest(Path(data)/n) for n in ref['contract']['files']},ref['contract']['files'],'Preserved data')
    for name,rows in [('models',models),('folds',fold_rows),('profiles',profiles),('paired',paired),('preprocessing',inspections),('split_checks',ref['split_checks'])]:
        write_json(destination/(name+'.json'),rows)
        if name not in ['preprocessing']: pd.DataFrame(rows).to_csv(destination/(name+'.csv'),index=False)
    result=dict(study_id=cfg()['study_id'],status='PASS',audited_at_utc=datetime.now(timezone.utc).isoformat(),
        code_commit=design['code_commit'],design_sha256=digest(output/'design.json'),
        audit_code_sha256=digest(__file__),new_classifier_fits=45,old_classifier_refits=0,
        new_models_full_validation_reloaded=45,baseline_C_models_full_validation_reloaded=10,
        reference_preprocessor_only_fits=reference_pre_fits,full_train_valid_projection_checks=45,
        excluded_geography_invariance_checks=45,fold_metrics_checked=60,profiles=len(profiles),paired_groups=len(paired),
        new_oof_rows=9*len(X),reused_oof_rows=3*len(X),unique_development_rows=len(X),
        holdout_accessed=False,data_sha256=ref['contract']['files'],
        output_sha256={p.name:digest(p) for p in destination.iterdir()},
        limitations=['No independent classifier refits; reload is not an independent training reproduction.',
            'Five historical stratified C classifiers were not stored; their OOF/configuration/metrics are verified, not model reload.',
            'Development-only descriptive CV; no causal leakage proof, significance claim, external validation or deployment.'])
    write_json(destination/'audit.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',default='data');p.add_argument('--t5',required=True)
    p.add_argument('--baseline',required=True);p.add_argument('--output',required=True)
    p.add_argument('--destination',required=True)
    a=p.parse_args();audit(a.data,a.t5,a.baseline,a.output,a.destination)

if __name__=='__main__':
    main()
