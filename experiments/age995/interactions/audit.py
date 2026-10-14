"""Independent OOF arithmetic and optional private-model re-prediction audit."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from preprocess import FEATURES, ID, NUMERIC, CATEGORICAL, load_development, sha256_file, write_json
PROBS=['p_grade_1','p_grade_2','p_grade_3']


def require(ok,message):
    if not ok:raise ValueError(message)


def counts(true,pred):
    true,pred=np.asarray(true),np.asarray(pred)
    require(true.ndim==1 and true.shape==pred.shape and len(true)>0,'Empty or unaligned labels.')
    require(np.isin(true,[1,2,3]).all() and np.isin(pred,[1,2,3]).all(),'Invalid labels.')
    cm=np.bincount(3*(true.astype(int)-1)+(pred.astype(int)-1),minlength=9).reshape(3,3)
    actual,guess,hit=cm.sum(1),cm.sum(0),cm.diagonal()
    P=np.divide(hit,guess,out=np.zeros(3),where=guess!=0)
    R=np.divide(hit,actual,out=np.zeros(3),where=actual!=0)
    F=np.divide(2*hit,actual+guess,out=np.zeros(3),where=actual+guess!=0)
    row=dict(n=len(true),macro_f1=float(F.mean()),accuracy=float(hit.sum()/len(true)),
             error31=int(cm[2,0]),error32=int(cm[2,1]),grade3_n=int(actual[2]),
             rate31=float(cm[2,0]/actual[2]) if actual[2] else None,
             rate32=float(cm[2,1]/actual[2]) if actual[2] else None)
    for i in range(3):
        row.update({f'P{i+1}':float(P[i]),f'R{i+1}':float(R[i]),f'F{i+1}':float(F[i]),f'n{i+1}':int(actual[i])})
    norm=np.divide(cm,actual[:,None],out=np.zeros((3,3)),where=actual[:,None]!=0)
    return row,cm.tolist(),norm.tolist()


def check_metrics(row,cm,saved):
    np.testing.assert_array_equal(cm,saved['confusion_matrix'])
    np.testing.assert_allclose([row['macro_f1'],row['accuracy']],
                               [saved['macro_f1'],saved['accuracy']],atol=1e-12,rtol=0)
    for i in (1,2,3):
        s=saved['per_class'][str(i)]
        np.testing.assert_allclose([row[f'P{i}'],row[f'R{i}'],row[f'F{i}'],row[f'n{i}']],
                                   [s['precision'],s['recall'],s['f1-score'],s['support']],atol=1e-12,rtol=0)


def independent_terms(frame):
    # Restated with ordered comparisons, not the training pd.cut helper.
    age=frame.age.to_numpy()
    require(np.isfinite(age).all() and (age==np.floor(age)).all() and (age>=0).all() and (age<=995).all(),'Invalid age.')
    band=np.select([age==0,age<=10,age<=20,age<=30,age<=50,age<=100,age<995],
                   ['0','1-10','11-20','21-30','31-50','51-100','101-994'],default='995')
    g=frame.geo_level_1_id.astype(str).to_numpy();f=frame.foundation_type.astype(str).to_numpy()
    return pd.DataFrame({'age_band':band,'geo_foundation':[f'{a}|{b}' for a,b in zip(g,f)],
        'geo_age':[f'{a}|{b}' for a,b in zip(g,band)],'foundation_age':[f'{a}|{b}' for a,b in zip(f,band)]},index=frame.index)


def read_oof(frame,path,digest):
    require(sha256_file(path)==digest,'OOF checksum mismatch.')
    oof=pd.read_csv(path)
    require(len(oof)==len(frame) and oof[ID].is_unique,'OOF denominator/duplicate ID.')
    require(not oof[[ID,'fold','y_true','y_pred',*PROBS]].isna().any().any(),'Missing OOF value.')
    joined=frame.merge(oof,on=ID,how='outer',validate='one_to_one',suffixes=('_expected',''),indicator=True,sort=False)
    require(joined['_merge'].eq('both').all(),'OOF IDs differ.')
    np.testing.assert_array_equal(joined.y_true,joined.damage_grade)
    np.testing.assert_array_equal(joined.fold,joined.fold_expected)
    p=joined[PROBS].to_numpy()
    require(np.isfinite(p).all() and (p>=0).all() and (p<=1).all(),'Invalid probabilities.')
    np.testing.assert_allclose(p.sum(1),1,atol=1e-6,rtol=0)
    np.testing.assert_array_equal(p.argmax(1)+1,joined.y_pred)
    return joined


def paired(y,old,new):
    y,old,new=map(np.asarray,(y,old,new))
    a=(y==3)&(old==1);b=(y==3)&(new==1)
    return dict(n=len(y),disagree=int((old!=new).sum()),old_only_correct=int(((old==y)&(new!=y)).sum()),
        new_only_correct=int(((new==y)&(old!=y)).sum()),fixed31=int((a&~b).sum()),
        new31=int((~a&b).sum()),shared31=int((a&b).sum()))


def check_preprocessing(frame,record,cfg,variant,directory,models):
    train=frame.loc[frame.fold.ne(record['fold'])];valid=frame.loc[frame.fold.eq(record['fold'])]
    require(record['train_rows']==len(train) and record['valid_rows']==len(valid),'Fold denominator changed.')
    pre=record['preprocessing'];cols=cfg['variants'][variant]['extra']
    require(pre['base_C_block_exact'] and pre['train_only_statistics'] and pre['csr'],'Runtime preprocessing failure.')
    require(pre['engineered_inputs']==39 and pre['train_age995']==int(train.age.eq(995).sum())
            and pre['valid_age995']==int(valid.age.eq(995).sum()),'Base input/group count mismatch.')
    median=float(np.median(train.loc[train.age.ne(995),'age']))
    require(median==pre['median'],'Training median differs.')
    num=train[NUMERIC].copy();num['age']=num.age.astype(float).mask(num.age.eq(995),median)
    values=num.to_numpy(dtype=np.float32).astype(np.float64)
    np.testing.assert_allclose(pre['numeric_mean'],values.mean(0),atol=1e-10,rtol=0)
    np.testing.assert_allclose(pre['numeric_variance'],values.var(0),atol=1e-8,rtol=1e-10)
    base_width=5+23+sum(train[c].nunique() for c in CATEGORICAL)
    require(base_width==pre['encoded_columns'],'Base encoding width differs.')
    terms=independent_terms(train);vt=independent_terms(valid)
    require([r['feature'] for r in pre['extra_coverage']]==cols,'Wrong interaction blocks.')
    for r in pre['extra_coverage']:
        col=r['feature'];vc=terms[col].value_counts()
        require((r['categories'],r['rare_le5'],r['valid_unseen'])==
                (len(vc),int(vc.le(5).sum()),int((~vt[col].isin(vc.index)).sum())),'Extra vocabulary summary differs.')
    extra_width=sum(terms[c].nunique() for c in cols)
    require(pre['extra_columns']==extra_width and pre['total_columns']==base_width+extra_width,'Extra width differs.')
    if models:
        p=directory/f'fold_{record["fold"]}.joblib'
        require(sha256_file(p)==record['model_sha256'],'Model checksum differs.')
        model=joblib.load(p)
        require(all(model['model'].get_params()[k]==v for k,v in cfg['lr'].items()),'Saved model parameters differ.')
        np.testing.assert_array_equal(model['model'].classes_,[1,2,3])
        parts=dict(model['preprocess'].transformer_list)
        encode=parts['extra']['encode']
        for col,vocab in zip(cols,encode.categories_):
            require(set(terms[col])==set(vocab),'Saved encoder learned nontraining categories.')
        with threadpool_limits(limits=1):
            pred=model.predict(valid[FEATURES]);prob=model.predict_proba(valid[FEATURES])
        return pred,prob
    return None


def audit(data,C,runs,output,models=False):
    output=Path(output)
    if output.exists():raise FileExistsError('Use a new audit output path.')
    cfg=json.loads((Path(__file__).parent/'config.json').read_text())
    contract=json.loads((ROOT/'configs/data_contract.json').read_text())
    for p,h in contract['files'].items():require(sha256_file(Path(data)/p)==h,'Data checksum differs.')
    X,y,ids,cv=load_development(data)
    require(len(X)==contract['expected_rows'] and set(cv.test_fold)==set(range(5)),'Wrong development data.')
    frame=X.copy();frame[ID]=ids;frame['damage_grade']=y;frame['fold']=cv.test_fold
    expected_params=json.loads((Path(C)/'lr/metrics.json').read_text())['effective_parameters']
    base_env=json.loads((Path(C)/'run_metadata.json').read_text())['environment']
    models_out=[];folds_out=[];groups=[];matrices=[];profiles=[];predictions={};cover=[];receipts=[]
    for variant in ['C',*cfg['variants']]:
        old=variant=='C';d=Path(C) if old else Path(runs)/variant
        meta=json.loads((d/('run_metadata.json' if old else 'metadata.json')).read_text())
        result=json.loads((d/('lr/metrics.json' if old else 'metrics.json')).read_text())
        require(meta['status']=='complete' and meta['data_contract']==contract and not meta['holdout_scored'],'Incomplete/different contract.')
        require(result['effective_parameters']==expected_params,'Model parameters differ.')
        require(meta['rows']==len(frame) and meta['scope']=='full_development_5fold','Wrong run scope.')
        if not old:
            require(set(meta['source_sha256'])=={'preprocess.py','baseline.py','scripts/audit_eda_oof.py','experiments/age995/run.py','experiments/age995/audit_followup.py','experiments/age995/interactions/config.json','experiments/age995/interactions/run.py'},'Missing source binding.')
        for key in base_env:require(meta['environment'][key]==base_env[key],f'Environment differs: {key}')
        for p,h in meta['source_code_sha256' if old else 'source_sha256'].items():
            require(sha256_file(ROOT/p)==h,f'Source changed: {p}')
        if not old:require(meta['config']==cfg and meta['variant']==variant,'Frozen setting differs.')
        digest=result['oof_sha256']
        q=read_oof(frame,d/('lr/oof_predictions.csv.gz' if old else 'oof.csv.gz'),digest)
        predictions[variant]=q.y_pred.to_numpy()
        records=json.loads((d/('lr/fold_metrics.json' if old else 'folds.json')).read_text())
        require([r['fold'] for r in records]==list(range(5)),'Incomplete fold records.')
        scores=[]
        for f in [*range(5),'pooled']:
            sub=q if f=='pooled' else q.loc[q.fold.eq(f)]
            allrow,cm,norm=counts(sub.y_true,sub.y_pred)
            saved=result['oof' if old else 'pooled'] if f=='pooled' else records[f]['validation']
            check_metrics(allrow,cm,saved)
            for group,part in [('all',sub),('age995',sub.loc[sub.age.eq(995)]),('non995',sub.loc[sub.age.ne(995)])]:
                r,m,n=counts(part.y_true,part.y_pred)
                groups.append(dict(variant=variant,fold=f,group=group,**r))
                matrices.append(dict(variant=variant,fold=f,group=group,raw=m,row_normalized=n))
            if f!='pooled':
                rec=records[f];scores.append(allrow['macro_f1'])
                require(rec['train_rows']==166784 and len(sub)==41696,'Fixed full-fold denominator differs.')
                converged=rec['convergence_ok' if old else 'converged']
                require(converged==not_convergence_warning(rec['warnings']),'Convergence flag differs.')
                folds_out.append(dict(variant=variant,fold=f,macro_f1=allrow['macro_f1'],error31=allrow['error31'],
                    error32=allrow['error32'],train_rows=rec['train_rows'],valid_rows=len(sub),
                    fit_seconds=rec['fit_seconds'],converged=converged,n_iter=max(rec['n_iter'])))
                if not old:
                    checked=check_preprocessing(frame,rec,cfg,variant,d,models)
                    if checked is not None:
                        np.testing.assert_array_equal(checked[0],sub.y_pred)
                        np.testing.assert_allclose(checked[1],sub[PROBS],atol=1e-14,rtol=0)
                    cover.extend(dict(variant=variant,fold=f,**r) for r in rec['preprocessing']['extra_coverage'])
        mean,std=float(np.mean(scores)),float(np.std(scores,ddof=1))
        np.testing.assert_allclose([mean,std],[result['cv_macro_f1_mean' if old else 'cv_mean'],result['cv_macro_f1_std' if old else 'cv_std']],atol=1e-12,rtol=0)
        row,_,_=counts(q.y_true,q.y_pred)
        modelrow=dict(variant=variant,origin='prior_reused' if old else 'new_training',cv_mean=mean,cv_std=std,
                      pooled_f1=row['macro_f1'],error31=row['error31'],error32=row['error32'],
                      all_converged=all(r['convergence_ok' if old else 'converged'] for r in records),
                      fit_seconds=sum(r['fit_seconds'] for r in records))
        models_out.append(modelrow)
        q['age_band']=independent_terms(q).age_band
        for feature in cfg['profile_columns']:
            for category,part in q.groupby(feature,sort=True,observed=True):
                r,m,n=counts(part.y_true,part.y_pred)
                profiles.append(dict(variant=variant,feature=feature,category=str(category),small_grade3=r['grade3_n']<cfg['small_grade3_support'],**r))
        receipts.append(dict(variant=variant,code_commit=meta.get('code_commit','historical_source_sha_bound'),oof_sha256=digest,metadata_sha256=sha256_file(d/('run_metadata.json' if old else 'metadata.json'))))
    transitions=[]
    for variant in cfg['variants']:
        for reference in sorted(set(['C',cfg['variants'][variant]['reference']])):
            for group,mask in [('all',np.ones(len(frame),dtype=bool)),('age995',frame.age.eq(995).to_numpy()),('non995',frame.age.ne(995).to_numpy())]:
                transitions.append(dict(variant=variant,reference=reference,group=group,
                    **paired(y.to_numpy()[mask],predictions[reference][mask],predictions[variant][mask])))
    # Do not promote or overwrite the existing RESEARCH_CANDIDATE.json.
    c=models_out[0];cg=next(g for g in groups if g['variant']=='C' and g['fold']=='pooled' and g['group']=='age995')
    eligible=[]
    for m in models_out[1:]:
        g=next(g for g in groups if g['variant']==m['variant'] and g['fold']=='pooled' and g['group']=='age995')
        m['review_eligible']=bool(m['all_converged'] and m['cv_mean']>c['cv_mean'] and m['error31']<=c['error31'] and g['error31']<=cg['error31'])
        if m['review_eligible']:eligible.append(m)
    chosen=max(eligible,key=lambda m:m['cv_mean'])['variant'] if eligible else 'C'
    output.mkdir(parents=True)
    for name,rows in [('models',models_out),('folds',folds_out),('groups',groups),('profiles',profiles),('coverage',cover),('paired',transitions)]:
        pd.DataFrame(rows).to_csv(output/f'{name}.csv',index=False)
    write_json(output/'matrices.json',matrices)
    receipt=dict(status='PASS',method='independent bincount/direct P/R/F1, independent age/combination rules',
        predictions_checked=len(frame)*len(models_out),new_models=5,new_fit_folds=25,historical_C_refit=False,
        matrices_checked=len(matrices),profile_rows=len(profiles),private_models_repredicted=25 if models else 0,
        holdout_scored=False,candidate_for_review=chosen,existing_candidate_replaced=False,
        sources=receipts,code_sha256=sha256_file(__file__),data_contract=contract,
        limitations=['No independent re-training; development CV is reused and exploratory.',
                     'Train F1 is logged, not independently recalculated from persisted training predictions.',
                     'Optional model reload checks do not independently prove the training procedure.'])
    receipt['output_sha256']={p.name:sha256_file(p) for p in output.iterdir() if p.is_file()}
    write_json(output/'verification.json',receipt)
    print(pd.DataFrame(models_out).to_string(index=False));print(json.dumps(receipt,indent=2))
    return receipt


def not_convergence_warning(items):
    return not any(w['category']=='ConvergenceWarning' for w in items)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['data','C','runs','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--models',action='store_true')
    a=p.parse_args();audit(a.data,a.C,a.runs,a.output,a.models)
