"""Read-only, checksum-bound OOF diagnostics. No estimator fitting or holdout access."""
from __future__ import annotations
import argparse
import json
import platform
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from preprocess import ID, NUMERIC, load_development, sha256_file
from scripts.audit_eda_oof import age_groups

CONFIG = Path(__file__).with_name('diagnostics_v1.json')
PROBS = ['p_grade_1', 'p_grade_2', 'p_grade_3']
LABELS = [1, 2, 3]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def validate_join(frame, oof):
    """Check before coercion/reindexing; never permit a missing/duplicated building."""
    columns = [ID, 'fold', 'y_true', 'y_pred', *PROBS]
    require(not oof.columns.duplicated().any() and set(columns) <= set(oof), 'OOF schema invalid.')
    require(len(oof) == len(frame) and oof[ID].is_unique and frame[ID].is_unique, 'OOF count/ID uniqueness invalid.')
    require(not oof[columns].isna().any().any(), 'Missing OOF value.')
    require(oof.y_true.isin(LABELS).all() and oof.y_pred.isin(LABELS).all(), 'OOF labels invalid.')
    require(oof.fold.isin(range(5)).all(), 'OOF fold invalid.')
    joined = frame[[ID, 'damage_grade', 'fold']].merge(oof, on=ID, how='outer',
        suffixes=('_expected', ''), indicator=True, validate='one_to_one')
    require(joined['_merge'].eq('both').all(), 'OOF ID population differs.')
    require(np.array_equal(joined.y_true, joined.damage_grade), 'OOF truth differs.')
    require(np.array_equal(joined.fold, joined.fold_expected), 'OOF folds differ.')
    p = joined[PROBS].to_numpy(dtype=float)
    require(np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all(), 'Invalid probability.')
    require(np.allclose(p.sum(1), 1, atol=1e-6, rtol=0), 'Probability sum invalid.')
    require(np.array_equal(p.argmax(1)+1, joined.y_pred), 'Probability/label disagreement.')
    return joined.set_index(ID).reindex(frame[ID].to_numpy()).reset_index()


def summary(y, pred):
    require(len(y) > 0 and len(y) == len(pred), 'Aligned nonempty labels required.')
    require(np.isin(y, LABELS).all() and np.isin(pred, LABELS).all(), 'Invalid labels.')
    cm = np.zeros((3, 3), dtype=np.int64)
    np.add.at(cm, (np.asarray(y, dtype=int)-1, np.asarray(pred, dtype=int)-1), 1)
    support, predicted, tp = cm.sum(1), cm.sum(0), cm.diagonal()
    f = np.divide(2*tp, support+predicted, out=np.zeros(3), where=(support+predicted) > 0)
    return dict(n=int(cm.sum()), correct=int(tp.sum()), errors=int(cm.sum()-tp.sum()),
        macro_f1=float(f.mean()), grade3_support=int(support[2]),
        e31=int(cm[2, 0]), e32=int(cm[2, 1]),
        e31_rate=float(cm[2, 0]/support[2]) if support[2] else None,
        e32_rate=float(cm[2, 1]/support[2]) if support[2] else None,
        cm=cm.tolist())


def pair_summary(y, left, right):
    y, left, right = map(np.asarray, (y, left, right))
    require(len(y)>0 and y.shape == left.shape == right.shape, 'Pair shapes invalid.')
    require(all(np.isin(a, LABELS).all() for a in (y,left,right)), 'Pair labels invalid.')
    lc, rc, same = left == y, right == y, left == right
    out = dict(n=len(y), disagreements=int((~same).sum()), both_correct=int((lc & rc).sum()),
        left_only_correct=int((lc & ~rc).sum()), right_only_correct=int((~lc & rc).sum()),
        both_wrong_same=int((~lc & ~rc & same).sum()), both_wrong_different=int((~lc & ~rc & ~same).sum()),
        fixed_e31=int(((y==3)&(left==1)&(right!=1)).sum()),
        new_e31=int(((y==3)&(left!=1)&(right==1)).sum()),
        shared_e31=int(((y==3)&(left==1)&(right==1)).sum()))
    require(out['n'] == sum(out[k] for k in ('both_correct','left_only_correct','right_only_correct',
        'both_wrong_same','both_wrong_different')), 'Pair partition error.')
    require(out['disagreements'] == out['left_only_correct']+out['right_only_correct']+out['both_wrong_different'],
        'Pair disagreement partition error.')
    return out


def cost_total(stats, severe_cost):
    require(np.isfinite(severe_cost) and severe_cost >= 1, 'Illustrative severe cost must be >= 1.')
    return stats['errors']+(severe_cost-1)*stats['e31']


def age_relation(train_age, valid_age, median, cap):
    """Verify affine change of representation, not fitted-coefficient equivalence."""
    train_age, valid_age = map(lambda a: np.asarray(a,dtype=float), (train_age,valid_age))
    x_c = np.where(train_age==995, median, train_age)
    x_e = np.where(train_age==995, cap, train_age)
    mc, me, sc, se = x_c.mean(), x_e.mean(), x_c.std(ddof=0), x_e.std(ddof=0)
    require(sc>0 and se>0, 'Constant age representation.')
    alpha, beta, intercept = sc/se, (cap-median)/se, (mc-me)/se
    flag = (valid_age==995).astype(float)
    vc, ve = np.where(flag,median,valid_age), np.where(flag,cap,valid_age)
    raw_error = np.max(np.abs(ve-(vc+(cap-median)*flag)))
    scaled_error = np.max(np.abs((ve-me)/se-(alpha*(vc-mc)/sc+beta*flag+intercept)))
    require(raw_error < 1e-10 and scaled_error < 1e-10, 'Age affine identity failed.')
    return dict(median=float(median),cap=float(cap),mean_C=float(mc),mean_E2=float(me),
        scale_C=float(sc),scale_E2=float(se),alpha=float(alpha),beta=float(beta),intercept=float(intercept),
        raw_max_abs_error=float(raw_error),scaled_max_abs_error=float(scaled_error),
        interpretation='same linear score family with free coefficients/intercept; L2 penalty generally differs')


def analyze(data_dir, private_root, output):
    cfg=json.loads(CONFIG.read_text());output=Path(output)
    require(not output.exists(), 'Output exists; refusing overwrite.')
    contract=json.loads((ROOT/'configs/data_contract.json').read_text())
    watched={Path(data_dir)/k:v for k,v in contract['files'].items()}
    for path,digest in watched.items():require(sha256_file(path)==digest, f'Data hash differs: {path.name}')
    X,y,ids,cv=load_development(data_dir)
    require(len(X)==contract['expected_rows'], 'Development population mismatch.')
    frame=X.copy();frame[ID]=ids;frame['damage_grade']=y;frame['fold']=cv.test_fold
    frame['age_band']=age_groups(frame.age).astype(str)
    models={};metas={};run_dirs={};bindings={}
    published=json.loads((ROOT/'reports/age995_followup_v1/review_receipt.json').read_text())
    expected_oof={r['variant']:r['oof_sha256'] for r in published['results']}
    common_params=None
    for variant in cfg['variants']:
        parent='age995_abc_v1' if variant in ['A','B','C'] else 'outputs/age995_followup_v1_20261013'
        folder=Path(private_root)/parent/variant;run_dirs[variant]=folder
        meta=json.loads((folder/'run_metadata.json').read_text());saved=json.loads((folder/'lr/metrics.json').read_text())
        require(meta['status']==saved['status']=='complete' and not meta['holdout_scored'] and not saved['holdout_scored'], 'Invalid run scope.')
        require(meta['data_contract']==contract and meta['rows']==len(frame), 'Run contract differs.')
        for name,digest in meta['source_code_sha256'].items():
            require(sha256_file(ROOT/name)==digest, f'Historical source differs: {name}')
        if common_params is None:common_params=saved['effective_parameters']
        require(common_params==saved['effective_parameters'], 'Model parameters differ.')
        oof_path=folder/'lr/oof_predictions.csv.gz';digest=sha256_file(oof_path)
        require(digest==saved['oof_sha256']==expected_oof[variant], 'OOF checksum differs from published receipt.')
        oof=validate_join(frame,pd.read_csv(oof_path));models[variant]=oof;metas[variant]=saved
        for name in ['run_metadata.json','run_manifest.csv','lr/metrics.json','lr/fold_metrics.json','lr/oof_predictions.csv.gz']:
            watched[folder/name]=sha256_file(folder/name)
        manifest=pd.read_csv(folder/'run_manifest.csv').sort_values(ID).reset_index(drop=True)
        expected=frame[[ID,'fold']].rename(columns={'fold':'cv_fold'}).sort_values(ID).reset_index(drop=True)
        pd.testing.assert_frame_equal(manifest,expected,check_dtype=False)
        computed=summary(y,oof.y_pred);np.testing.assert_array_equal(computed['cm'],saved['oof']['confusion_matrix'])
        require(abs(computed['macro_f1']-saved['oof']['macro_f1'])<1e-12,'Published F1 differs.')
        bindings[variant]=dict(oof_sha256=digest,metadata_sha256=sha256_file(folder/'run_metadata.json'))
    frame=frame.reset_index(drop=True)
    masks={'all':np.ones(len(frame),bool),'age995':frame.age.eq(995).to_numpy(),'non995':frame.age.ne(995).to_numpy()}
    metrics=[];pair_rows=[];transitions=[];cost_rows=[];profiles=[];range_rows=[];algebra=[]
    for v,oof in models.items():
        p=oof[PROBS].to_numpy();maxp=p.max(1);margin=np.sort(p,axis=1)[:,-1]-np.sort(p,axis=1)[:,-2]
        pred=oof.y_pred.to_numpy();truth=frame.damage_grade.to_numpy()
        global_e31=summary(truth,pred)['e31']
        for group,mask in masks.items():
            s=summary(truth[mask],pred[mask]);s.pop('cm')
            high=mask & (maxp>=cfg['high_probability_threshold'])
            e31=mask & (truth==3)&(pred==1)
            metrics.append(dict(variant=v,group=group,**s,median_max_probability=float(np.median(maxp[mask])),
                high_probability_rows=int(high.sum()),high_probability_errors=int((high&(truth!=pred)).sum()),
                high_probability_accuracy=float((truth[high]==pred[high]).mean()) if high.any() else None,
                high_probability_e31=int((e31&high).sum()),
                low_margin_e31=int((e31&(margin<=cfg['low_margin_threshold'])).sum())))
            for w in cfg['illustrative_3_to_1_costs']:
                cost_rows.append(dict(variant=v,group=group,weight_3_to_1=w,total_cost=cost_total(s,w),n=s['n']))
        if v in ['A','B','C','E2']:
            for col in cfg['profile_columns']:
                for value,part in frame.groupby(col,observed=True,sort=True):
                    for fold in ['pooled',0,1,2,3,4] if col=='geo_level_1_id' else ['pooled']:
                        idx=part.index if fold=='pooled' else part.index[part.fold.eq(fold)]
                        if len(idx)==0:continue
                        s=summary(truth[idx],pred[idx]);s.pop('cm')
                        profiles.append(dict(variant=v,dimension=col,value=str(value),fold=fold,**s,
                            e31_share_all=float(s['e31']/global_e31) if global_e31 else None,
                            low_grade3_support=s['grade3_support']<cfg['rate_warning_min_grade3_support']))
    for left,right in cfg['pairs']:
        l,r=models[left],models[right]
        for group,mask in masks.items():
            for fold in ['pooled',0,1,2,3,4]:
                use=mask if fold=='pooled' else mask & frame.fold.eq(fold).to_numpy()
                if not use.any():continue
                lp,rp=l.loc[use,'y_pred'].to_numpy(),r.loc[use,'y_pred'].to_numpy();truth=frame.loc[use,'damage_grade'].to_numpy()
                s=pair_summary(truth,lp,rp)
                diff=np.abs(l.loc[use,PROBS].to_numpy()-r.loc[use,PROBS].to_numpy())
                s.update(mean_abs_probability_difference=float(diff.mean()),max_abs_probability_difference=float(diff.max()))
                pair_rows.append(dict(left=left,right=right,group=group,fold=fold,**s))
                if fold=='pooled':
                    table=pd.DataFrame({'truth':truth,'left_prediction':lp,'right_prediction':rp})
                    for key,n in table.groupby(list(table.columns)).size().items():
                        transitions.append(dict(left=left,right=right,group=group,truth=int(key[0]),
                            left_prediction=int(key[1]),right_prediction=int(key[2]),count=int(n)))
    for fold in range(5):
        train=frame.loc[frame.fold.ne(fold)];valid=frame.loc[frame.fold.eq(fold)]
        regular=train.loc[train.age.ne(995),'age'];med=float(regular.median());cap=float(regular.quantile(.99,interpolation='linear'))
        require(med==metas['C']['fold_results'][fold]['preprocessor_audit']['median'], 'C median provenance differs.')
        require(cap==metas['E2']['fold_results'][fold]['preprocessor_audit']['upper'], 'E2 cap provenance differs.')
        algebra.append(dict(fold=fold,validation_rows=len(valid),**age_relation(train.age,valid.age,med,cap)))
        for v in ['D','F1','F2']:
            drop=train.age.eq(995) if v=='D' else train.age.gt(55)
            if v=='F2':drop=drop|train.count_floors_pre_eq.gt(5)|train.area_percentage.gt(20)|train.height_percentage.gt(10)
            age=train.loc[~drop,'age'].to_numpy(dtype=float);mean,sd=float(age.mean()),float(age.std(ddof=0))
            record=metas[v]['fold_results'][fold];saved=record['preprocessor_audit'];j=NUMERIC.index('age')
            require(len(age)==record['train_rows'],'Retained count differs.')
            require(abs(saved['numeric_mean'][j]-mean)<1e-10 and abs(saved['numeric_variance'][j]-sd*sd)<1e-8,'Retained age statistics differ.')
            range_rows.append(dict(variant=v,fold=fold,retained_rows=len(age),train_age_max=float(age.max()),
                train_age_mean=mean,train_age_std=sd,validation_age995_z=(995-mean)/sd,
                train_age_max_z=(float(age.max())-mean)/sd,validation_age995=int(valid.age.eq(995).sum())))
    for p,h in watched.items():require(sha256_file(p)==h, f'Input modified: {p.name}')
    output.mkdir(parents=True)
    for name,rows in [('group_diagnostics',metrics),('paired_comparisons',pair_rows),('paired_transitions',transitions),
        ('cost_sensitivity',cost_rows),('error_profiles',profiles),('training_age_support',range_rows),('age_affine_identity',algebra)]:
        pd.DataFrame(rows).to_csv(output/f'{name}.csv',index=False,lineterminator='\n')
    source_paths=['preprocess.py','scripts/audit_eda_oof.py','experiments/age995/diagnostics.py',
                  'experiments/age995/diagnostics_v1.json','reports/age995_followup_v1/review_receipt.json']
    receipt=dict(status='PASS',study_id=cfg['study_id'],source_commit=cfg['source_commit'],
        scope=cfg['scope'],predictions_reused=len(frame)*len(models),unique_buildings=len(frame),
        new_model_fits=0,previous_models_refit=0,holdout_features_labels_read=False,
        candidate_changed=False,original_input_files_unchanged=len(watched),
        bindings=bindings,data_contract=contract,
        source_sha256={p:sha256_file(ROOT/p) for p in source_paths},
        output_sha256={p.name:sha256_file(p) for p in sorted(output.glob('*.csv'))},
        environment=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__),
        caveats=['Posthoc description of fixed OOF, not independent final evaluation or causal attribution.',
                 'Illustrative error weights are not stakeholder-approved costs.',
                 'No fitted coefficients or model binary available; extrapolation is observed but causal contribution is unverified.',
                 'Empirical probability diagnostics do not calibrate probabilities or select a decision threshold.'])
    dump(output/'verification.json',receipt)
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=ROOT/'data')
    parser.add_argument('--private-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(analyze(args.data_dir,args.private_root,args.output),ensure_ascii=False,indent=2))
