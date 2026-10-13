"""Independent OOF/selection audit: no model fitting, no training metric helpers."""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from preprocess import CATEGORICAL, FEATURES, GEO, ID, NUMERIC, load_development, sha256_file, write_json

CONFIG = Path(__file__).with_name('followup_v1.json')
LABELS = (1,2,3)
PROBS = ('p_grade_1','p_grade_2','p_grade_3')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def close(actual, expected):
    np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=0)


def direct_stats(true, pred):
    true,pred = np.asarray(true),np.asarray(pred)
    require(true.ndim==1 and true.shape==pred.shape and len(true)>0,'Aligned nonempty labels required.')
    require(np.isin(true,LABELS).all() and np.isin(pred,LABELS).all(),'Invalid labels.')
    table = pd.crosstab(pd.Series(true),pd.Series(pred)).reindex(index=LABELS,columns=LABELS,fill_value=0)
    cm=table.to_numpy(dtype=np.int64)
    support, predicted, hit=cm.sum(1),cm.sum(0),cm.diagonal()
    precision=np.divide(hit,predicted,out=np.zeros(3),where=predicted!=0)
    recall=np.divide(hit,support,out=np.zeros(3),where=support!=0)
    f1=np.divide(2*hit,support+predicted,out=np.zeros(3),where=support+predicted!=0)
    norm=np.divide(cm,support[:,None],out=np.zeros((3,3)),where=support[:,None]!=0)
    out=dict(n=len(true),macro_f1=float(f1.mean()),accuracy=float(hit.sum()/len(true)),
        grade3_count=int(support[2]),error_3_to_1=int(cm[2,0]),error_3_to_2=int(cm[2,1]),
        rate_3_to_1=float(cm[2,0]/support[2]) if support[2] else None,
        rate_3_to_2=float(cm[2,1]/support[2]) if support[2] else None)
    for i in range(3):
        out.update({f'grade{i+1}_precision':float(precision[i]),f'grade{i+1}_recall':float(recall[i]),
                    f'grade{i+1}_f1':float(f1[i]),f'grade{i+1}_support':int(support[i])})
    return out,cm.tolist(),norm.tolist()


def check_saved_metrics(stats, cm, saved):
    np.testing.assert_array_equal(cm,saved['confusion_matrix'])
    close(stats['macro_f1'],saved['macro_f1']);close(stats['accuracy'],saved['accuracy'])
    for c in LABELS:
        for ours,theirs in [('precision','precision'),('recall','recall'),('f1','f1-score'),('support','support')]:
            close(stats[f'grade{c}_{ours}'],saved['per_class'][str(c)][theirs])
    if stats['grade3_count']:
        close(stats['rate_3_to_1'],saved['severe_as_minor_rate'])


def ordered_id_sha(values):
    return hashlib.sha256(np.asarray(values,dtype='<i8').tobytes()).hexdigest()


def independent_keep(frame,variant):
    # Deliberately do not import or call the training selection function.
    if variant=='D': drop=frame.age==995
    elif variant=='F1': drop=frame.age>55
    elif variant=='F2':
        drop=(frame.age>55)|(frame.count_floors_pre_eq>5)|(frame.area_percentage>20)|(frame.height_percentage>10)
    elif variant in ('E1','E2','E3'): drop=pd.Series(False,index=frame.index)
    else: raise ValueError('Unknown new variant.')
    return ~drop


def linear_quantile(values,q):
    ordered=np.sort(np.asarray(values,dtype=float))
    require(len(ordered)>0,'Empty cap population.')
    position=(len(ordered)-1)*q
    lower=int(np.floor(position));upper=int(np.ceil(position))
    return float(ordered[lower]+(ordered[upper]-ordered[lower])*(position-lower))


def audit_selection(frame,variant,record,cfg):
    f=record['fold'];train=frame.loc[frame.fold.ne(f)];valid=frame.loc[frame.fold.eq(f)]
    keep=independent_keep(train,variant);kept=train.loc[keep];removed=train.loc[~keep]
    sel=record['selection_audit'];pre=record['preprocessor_audit']
    require(record['original_train_rows']==len(train),'Original training denominator changed.')
    require(record['train_rows']==len(kept) and record['validation_rows']==len(valid),'Train/validation row count wrong.')
    for key,part in [('original',train),('retained',kept),('removed',removed)]:
        require(sel[f'{key}_rows']==len(part),'Selection count mismatch.')
        for c in LABELS: require(sel[f'{key}_class_counts'][str(c)]==int(part.damage_grade.eq(c).sum()),'Selected class count mismatch.')
    for key,part in [('retained',kept),('removed',removed)]:
        require(sel[f'{key}_ids_sha256']==ordered_id_sha(part[ID]),'Selected ID hash mismatch.')
    flags=pd.DataFrame({'age_eq_995':train.age.eq(995),'age_gt_55':train.age.gt(55),
        'count_floors_pre_eq_gt_5':train.count_floors_pre_eq.gt(5),
        'area_percentage_gt_20':train.area_percentage.gt(20),'height_percentage_gt_10':train.height_percentage.gt(10)},index=train.index)
    for key in flags:
        require(sel['criterion_counts'][key]==int(flags[key].sum()),'Criterion count mismatch.')
        for c in LABELS:
            require(sel['criterion_class_counts'][key][str(c)]==int((flags[key]&train.damage_grade.eq(c)).sum()),'Criterion class count mismatch.')
    broad=flags.drop(columns='age_eq_995')
    require(sel['faith_union_count']==int(broad.any(axis=1).sum()),'OR union count mismatch.')
    require(sel['faith_overlap_ge2']==int(broad.sum(axis=1).ge(2).sum()),'Overlap count mismatch.')
    columns=kept[NUMERIC].copy()
    upper=None
    train_changed=pd.Series(False,index=kept.index);valid_changed=pd.Series(False,index=valid.index)
    if variant in ('E1','E2','E3'):
        upper=linear_quantile(kept.loc[kept.age.ne(995),'age'],cfg['quantile'])
        close(upper,pre['upper'])
        train_changed=kept.age.gt(upper) if variant=='E3' else kept.age.eq(995)&kept.age.ne(upper)
        valid_changed=valid.age.gt(upper) if variant=='E3' else valid.age.eq(995)&valid.age.ne(upper)
        columns['age']=columns.age.astype(float).mask(train_changed,upper)
    else: require(pre['upper'] is None,'Unexpected cap in removal-only policy.')
    for label,part,changed in [('train',kept,train_changed),('valid',valid,valid_changed)]:
        require(pre[f'{label}_modified']==int(changed.sum()),'Transformed row count mismatch.')
        require(pre[f'{label}_non995_modified']==int((changed&part.age.ne(995)).sum()),'Normal-age changes mismatch.')
    require(pre['retained_train_age995']==int(kept.age.eq(995).sum()),'Retained995 mismatch.')
    require(pre['valid_age995']==int(valid.age.eq(995).sum()),'Validation995 denominator changed.')
    require(pre['fit_rows']==len(kept),'Scaler row count mismatch.')
    values=columns.to_numpy(dtype=np.float32).astype(np.float64)
    np.testing.assert_allclose(pre['numeric_mean'],values.mean(0),atol=1e-10,rtol=0)
    np.testing.assert_allclose(pre['numeric_variance'],values.var(0),atol=1e-8,rtol=1e-10)
    require(pre['engineered_inputs']==(39 if variant=='E2' else 38),'Unexpected engineered inputs.')
    encoded=len(NUMERIC)+22+(1 if variant=='E2' else 0)+sum(kept[c].nunique() for c in CATEGORICAL)
    require(pre['encoded_columns']==encoded,'Encoded vocabulary count mismatch.')
    require(pre['train_only_statistics'] and pre['float32_csr'],'Recorded preprocessing check failed.')
    coverage=[]
    for row in sel['category_coverage']:
        col=row['feature'];a=train[col].value_counts();b=kept[col].value_counts().reindex(a.index,fill_value=0)
        expected=dict(categories_before=len(a),categories_after=int(b.gt(0).sum()),lost_categories=int(b.eq(0).sum()),
            rare_categories_before=int(a.le(5).sum()),lost_rare_categories=int((a.le(5)&b.eq(0)).sum()),
            validation_unseen_before=int((~valid[col].isin(a.index)).sum()),
            validation_unseen_after=int((~valid[col].isin(b.index[b.gt(0)])).sum()))
        require(all(row[k]==v for k,v in expected.items()),'Category coverage mismatch.')
        coverage.append(dict(variant=variant,fold=f,**row))
    require({r['feature'] for r in coverage}==set(CATEGORICAL) and len(coverage)==len(CATEGORICAL),'Missing category coverage.')
    regions=[]
    for row in sel['regions']:
        n=int(train[GEO[0]].eq(row['region']).sum());after=int(kept[GEO[0]].eq(row['region']).sum())
        require((row['before'],row['after'],row['removed'])==(n,after,n-after),'Region count mismatch.')
        close([row['removal_rate'],row['share_before'],row['share_after']],[(n-after)/n,n/len(train),after/len(kept)])
        regions.append(dict(variant=variant,fold=f,**row))
    require(len(regions)==train[GEO[0]].nunique() and len({r['region'] for r in regions})==len(regions),'Missing/duplicate regions.')
    close(sel['region_total_variation'],sum(abs(r['share_before']-r['share_after']) for r in regions)/2)
    summary=dict(variant=variant,fold=f,original_train_rows=len(train),retained_train_rows=len(kept),
        removed_rows=len(removed),removed_rate=len(removed)/len(train),validation_rows=len(valid),cap=upper,
        train_modified=int(train_changed.sum()),valid_modified=int(valid_changed.sum()),
        train_non995_modified=int((train_changed&kept.age.ne(995)).sum()),
        valid_non995_modified=int((valid_changed&valid.age.ne(995)).sum()),region_total_variation=sel['region_total_variation'])
    for c in LABELS:
        for label,part in [('original',train),('retained',kept),('removed',removed)]:
            summary[f'{label}_grade{c}']=int(part.damage_grade.eq(c).sum())
    criteria=[dict(variant=variant,fold=f,criterion=key,count=int(flags[key].sum()),
                  **{f'grade{c}':sel['criterion_class_counts'][key][str(c)] for c in LABELS}) for key in flags]
    return summary,coverage,regions,criteria


def audit_variant(frame,directory,variant,contract,shared_params,cfg):
    directory=Path(directory)
    saved=json.loads((directory/'lr/metrics.json').read_text())
    meta=json.loads((directory/'run_metadata.json').read_text())
    require(saved['status']=='complete' and meta['status']=='complete','Incomplete run.')
    require(not saved['holdout_scored'] and not meta['holdout_scored'],'Holdout result forbidden.')
    require(meta['data_contract']==contract and meta['rows']==len(frame),'Contract mismatch.')
    require(meta['scope']=='full_development_5fold' and saved['folds']==5 and saved['rows']==len(frame),'Wrong evaluation scope.')
    require(saved['effective_parameters']==shared_params,'Effective parameters differ.')
    if variant in ('A','B','C'):
        require(meta['config']['models']['lr']==cfg['lr'],'Historical LR configuration differs.')
    else:
        require(meta['config']==cfg and meta['variant']==variant,'Frozen followup configuration differs.')
        require(meta['experiment_id']==cfg['variants'][variant]['id'],'Experiment ID mismatch.')
        require(re.fullmatch(r'[0-9a-f]{40}',meta['code_commit']) is not None,'Invalid source commit.')
    for name,digest in meta['source_code_sha256'].items():
        require(sha256_file(ROOT/name)==digest,f'Source hash differs: {name}')
    for name in ('preprocess.py','baseline.py'):
        require(name in meta['source_code_sha256'],'Missing core source binding.')
    if variant in ('B','C'):
        require('experiments/age995/run.py' in meta['source_code_sha256'],'Missing historical age source binding.')
    manifest=pd.read_csv(directory/'run_manifest.csv').sort_values(ID).reset_index(drop=True)
    expected=frame[[ID,'fold']].rename(columns={'fold':'cv_fold'}).sort_values(ID).reset_index(drop=True)
    pd.testing.assert_frame_equal(manifest,expected,check_dtype=False)
    oof_path=directory/'lr/oof_predictions.csv.gz'
    require(sha256_file(oof_path)==saved['oof_sha256'],'OOF checksum mismatch.')
    oof=pd.read_csv(oof_path)
    require(len(oof)==len(frame) and oof[ID].is_unique,'OOF denominator or ID uniqueness broken.')
    require(not oof[[ID,'fold','y_true','y_pred',*PROBS]].isna().any().any(),'Missing OOF value.')
    joined=frame[[ID,'damage_grade','fold','age']].merge(oof,on=ID,how='outer',suffixes=('_expected',''),indicator=True,validate='one_to_one')
    require(joined['_merge'].eq('both').all(),'OOF IDs differ.')
    np.testing.assert_array_equal(joined.y_true,joined.damage_grade)
    np.testing.assert_array_equal(joined.fold,joined.fold_expected)
    p=joined[list(PROBS)].to_numpy()
    require(np.isfinite(p).all() and (p>=0).all() and (p<=1).all(),'Invalid probability.')
    np.testing.assert_allclose(p.sum(1),1,atol=1e-6,rtol=0)
    np.testing.assert_array_equal(p.argmax(1)+1,joined.y_pred)
    require([r['fold'] for r in saved['fold_results']]==list(range(5)),'Invalid fold records.')
    require(json.loads((directory/'lr/fold_metrics.json').read_text())==saved['fold_results'],'Fold files disagree.')
    fold_rows=[];groups=[];matrices=[];selection=[];coverage=[];regions=[];criteria=[]
    for f in [*range(5),'pooled']:
        q=joined if f=='pooled' else joined.loc[joined.fold.eq(f)]
        all_stats=None
        for label,part in [('all',q),('age995',q.loc[q.age.eq(995)]),('non995',q.loc[q.age.ne(995)])]:
            if part.empty:continue
            stats,cm,norm=direct_stats(part.y_true,part.y_pred)
            matrices.append(dict(variant=variant,fold=f,group=label,n=len(part),raw=cm,row_normalized=norm))
            groups.append(dict(variant=variant,fold=f,group=label,**stats))
            if label=='all':all_stats=(stats,cm)
        stats,cm=all_stats
        if f=='pooled':
            check_saved_metrics(stats,cm,saved['oof'])
        else:
            record=saved['fold_results'][f]
            check_saved_metrics(stats,cm,record['validation'])
            require(record['validation_rows']==len(q),'Validation denominator changed.')
            if variant in ('A','B','C'):
                require(record['train_rows']==len(frame)-len(q),'Historical train denominator changed.')
            else:
                s,c,r,crit=audit_selection(frame,variant,record,cfg)
                selection.append(s);coverage.extend(c);regions.extend(r);criteria.extend(crit)
            require(record['convergence_ok']==(not any(w['category']=='ConvergenceWarning' for w in record['warnings'])),'Warning/convergence flag mismatch.')
            require(all(0 < n <= cfg['lr']['max_iter'] for n in record['n_iter']),'Invalid iteration count.')
            fold_rows.append(dict(variant=variant,fold=f,macro_f1=stats['macro_f1'],
                train_rows=record['train_rows'],validation_rows=len(q),fit_seconds=record['fit_seconds'],
                converged=record['convergence_ok'],n_iter_max=max(record['n_iter'])))
    scores=[r['macro_f1'] for r in fold_rows]
    close([np.mean(scores),np.std(scores,ddof=1)],[saved['cv_macro_f1_mean'],saved['cv_macro_f1_std']])
    require(saved['all_folds_converged']==all(r['convergence_ok'] for r in saved['fold_results']),'Convergence summary mismatch.')
    close(sum(r['fit_seconds'] for r in saved['fold_results']),saved['total_fit_seconds'])
    pooled=next(r for r in groups if r['fold']=='pooled' and r['group']=='all')
    model=dict(variant=variant,origin='historical_reused' if variant in ('A','B','C') else 'new_training',
        cv_mean=float(np.mean(scores)),cv_std=float(np.std(scores,ddof=1)),pooled_f1=pooled['macro_f1'],
        error_3_to_1=pooled['error_3_to_1'],error_3_to_2=pooled['error_3_to_2'],
        fit_seconds=saved['total_fit_seconds'],converged=saved['all_folds_converged'],
        oof_sha256=saved['oof_sha256'],code_commit=meta.get('code_commit',meta.get('git_head','historical_source_sha_bound')))
    return dict(model=model,folds=fold_rows,groups=groups,matrices=matrices,selection=selection,
                coverage=coverage,regions=regions,criteria=criteria,environment=meta['environment'])


def audit(data,abc,new,output,variants):
    output=Path(output)
    if output.exists():raise FileExistsError('Audit output is immutable; use a new path.')
    cfg=json.loads(CONFIG.read_text())
    require(cfg['quantile']==.99 and cfg['interpolation']=='linear','v1 cap rule changed.')
    require(cfg['faith_thresholds']==dict(age=55,count_floors_pre_eq=5,area_percentage=20,height_percentage=10),'v1 thresholds changed.')
    require(variants and len(set(variants))==len(variants) and set(variants)<=set(cfg['variants']),'Invalid variant list.')
    contract=json.loads((ROOT/'configs/data_contract.json').read_text())
    for name,digest in contract['files'].items():require(sha256_file(Path(data)/name)==digest,'Data SHA mismatch.')
    X,y,ids,cv=load_development(data)
    require(len(X)==contract['expected_rows'],'Wrong development population.')
    frame=X.copy();frame[ID]=ids;frame['damage_grade']=y;frame['fold']=cv.test_fold
    shared=json.loads((Path(abc)/'A/lr/metrics.json').read_text())['effective_parameters']
    bundles=[]
    for variant in ['A','B','C',*variants]:
        source=Path(abc)/variant if variant in ('A','B','C') else Path(new)/variant
        bundles.append(audit_variant(frame,source,variant,contract,shared,cfg))
    for bundle in bundles:
        for key in ('python','numpy','pandas','sklearn'):
            require(bundle['environment'][key]==bundles[0]['environment'][key],f'Execution environment differs: {key}')
    output.mkdir(parents=True)
    for key,name in [('model','model_summary'),('folds','fold_summary'),('groups','group_summary'),
                     ('selection','selection_summary'),('coverage','category_coverage'),('regions','region_shift'),('criteria','criteria')]:
        records=[b[key] for b in bundles] if key=='model' else [row for b in bundles for row in b[key]]
        pd.DataFrame(records).to_csv(output/f'{name}.csv',index=False,lineterminator='\n')
    matrices=[row for b in bundles for row in b['matrices']]
    write_json(output/'confusion_matrices.json',matrices)
    receipt=dict(status='PASS',method='independent pandas crosstab/direct equations and independently restated selection/quantile',
        variants=['A','B','C',*variants],historical_models_reused=3,new_models=len(variants),
        predictions_checked=len(frame)*len(bundles),folds_checked=5*len(bundles),
        matrices_checked=len(matrices),data_contract=contract,holdout_scored=False,
        model_refitting_performed=False,source_code_sha256=sha256_file(__file__),
        results=[b['model'] for b in bundles],
        limitations=['Does not independently refit fitted models or inspect model coefficients.',
                     'Persisted training predictions are unavailable; train Macro F1 is not independently recalculated.',
                     'Validation-transform immutability is checked by runtime assertions and synthetic tests; no fitted models are published.'])
    write_json(output/'verification.json',receipt)
    print(json.dumps(receipt,ensure_ascii=False,indent=2))
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,default=ROOT/'data')
    p.add_argument('--abc',type=Path,required=True)
    p.add_argument('--new',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--variants',nargs='+',default=['D','E1','E2','E3'])
    a=p.parse_args();audit(a.data_dir,a.abc,a.new,a.output,a.variants)
