"""Freeze from CV only, then perform one delegated PERSONAL holdout evaluation.

Usage: freeze; then evaluate --selection-sha <printed SHA>. Nonempty final
output is refused. This is not team selection, main integration or deployment.
"""
from __future__ import annotations
import argparse,json,sys,time,warnings,platform
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits
from sklearn.exceptions import ConvergenceWarning
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from preprocess import ID,TARGET,FEATURES,CATEGORICAL,normalize_features,sha256_file,write_json
from scripts.audit_eda_oof import development_frame,validate_oof,verify_metrics,confusion_stats
from personal_sampling import PersonalSamplingLR
from baseline import metrics

POLICY_COMMIT='53966920c250178eb738df3856a9fd058033bac1'
SOURCES=['preprocess.py','personal_sampling.py','scripts/run_personal_sampling.py',
         'scripts/finalize_personal.py','requirements.txt','configs/baseline.json']
SELECTION=ROOT/'reports/personal_selection.json'
FINAL=ROOT/'outputs/personal_final_v1'
now=lambda:datetime.now(timezone.utc).isoformat()


def choose(records):
    eligible=[r for r in records if r['converged'] and np.isfinite(r['cv_mean'])]
    if not eligible:raise ValueError('No converged candidate')
    highest=max(r['cv_mean'] for r in eligible)
    tied=[r for r in eligible if highest-r['cv_mean']<=1e-6]
    return sorted(tied,key=lambda r:(0 if r['experiment']=='baseline_v1' and r['model']=='lr' else 1,
                                    -r['cv_mean'],r['experiment'],r['model']))[0]


def freeze():
    if SELECTION.exists() or FINAL.exists():raise FileExistsError('Selection/evaluation already exists')
    frame,contract=development_frame(ROOT/'data')
    old=json.loads((ROOT/'reports/personal_reference_snapshot.json').read_text())
    # This entry point is run on the frozen 17-setting development snapshot.
    # Restore baseline evidence without counting it as a new unique setting.
    if len(old)!=17:raise ValueError('Expected original 17-setting snapshot at freeze time')
    checked=[];current=[]
    for mode in ['none','random_over','context_smote']:
        p=ROOT/f'outputs/personal_{mode}_v1/lr';s=json.loads((p/'metrics.json').read_text())
        if sha256_file(p/'oof_predictions.csv.gz')!=s['oof_sha256']:raise ValueError('OOF hash mismatch')
        joined=validate_oof(frame,pd.read_csv(p/'oof_predictions.csv.gz'));verify_metrics(joined,s)
        check={'mode':mode,'metrics_sha256':sha256_file(p/'metrics.json'),'oof_sha256':s['oof_sha256']}
        checked.append(check)
        r=dict(experiment='baseline_v1' if mode=='none' else f'personal_{mode}_v1',model='lr',
               cv_mean=s['cv_macro_f1_mean'],cv_std=s['cv_macro_f1_std'],
               converged=s['all_folds_converged'],mode=mode,metrics_sha256=check['metrics_sha256'])
        if mode=='none':
            baseline=next(x for x in old if x['experiment']=='baseline_v1' and x['model']=='lr')
            if baseline['matrix']!=s['oof']['confusion_matrix']:raise ValueError('Baseline CM did not reproduce')
            np.testing.assert_allclose(baseline['fold_scores'],[x['validation']['macro_f1'] for x in s['fold_results']],atol=1e-12,rtol=0)
            # Same candidate, stronger evidence. Preserve historical metrics file.
            old=[x for x in old if (x['experiment'],x['model'])!=('baseline_v1','lr')]
        current.append(r)
    pool=old+current
    selected=choose(pool)
    if 'mode' not in selected:raise ValueError('Winner lacks the freshly verified executable path; manual review needed')
    artifact={'schema_version':1,'scope':'personal_final_not_team', 'status':'FROZEN_BEFORE_HOLDOUT',
              'frozen_at_utc':now(),'policy_commit':POLICY_COMMIT,
              'delegation':'User delegated recommended processing and end-to-end personal completion.',
              'selected':selected,'unique_configurations':len(pool),
              'selection_rule':'Highest converged development CV Macro F1; <=1e-6 tie prefers original unweighted LR.',
              'no_post_holdout_tuning':True,'evaluation_limit':1,'holdout_opened':False,
              'data_contract':contract,'source_sha256':{p:sha256_file(ROOT/p) for p in SOURCES},
              'input_comparison_sha256':sha256_file(ROOT/'reports/personal_reference_snapshot.json'),
              'verified_candidates':checked,
              'candidate_scores':[{'experiment':r['experiment'],'model':r['model'],'cv_mean':r['cv_mean'],'converged':r['converged']} for r in pool]}
    write_json(SELECTION,artifact);print('FROZEN',selected,sha256_file(SELECTION),flush=True)
    return artifact


def validate_holdout_tables(features,labels,expected_ids,development_ids):
    if list(features.columns)!=[ID]+FEATURES or list(labels.columns)!=[ID,TARGET]:
        raise ValueError('Unexpected holdout schema')
    for df in [features,labels]:
        if df[ID].duplicated().any() or df[ID].isna().any():raise ValueError('Nonunique holdout IDs')
        if set(df[ID])!=set(expected_ids):raise ValueError('Holdout IDs differ from frozen manifest')
        if set(df[ID])&set(development_ids):raise ValueError('Development/holdout overlap')
    if not labels[TARGET].isin([1,2,3]).all():raise ValueError('Invalid holdout label')
    return features[[ID]].merge(labels,on=ID,how='left',validate='one_to_one')[TARGET]


def evaluate(selection_sha):
    # Authorization/digest gates run BEFORE any final-output or holdout operation.
    if not SELECTION.is_file() or sha256_file(SELECTION)!=selection_sha:
        raise ValueError('Exact frozen selection SHA required')
    frozen=json.loads(SELECTION.read_text())
    if frozen['status']!='FROZEN_BEFORE_HOLDOUT' or frozen['evaluation_limit']!=1:
        raise ValueError('Invalid final evaluation policy')
    for p,digest in frozen['source_sha256'].items():
        if sha256_file(ROOT/p)!=digest:raise ValueError(f'Frozen code changed: {p}')
    if FINAL.exists():raise FileExistsError('Final evaluation already started; no repeat scoring')
    frame,contract=development_frame(ROOT/'data')
    if contract!=frozen['data_contract']:raise ValueError('Data contract changed')
    FINAL.mkdir(parents=True)
    write_json(FINAL/'evaluation_started.json',{'selection_sha256':selection_sha,'started_at_utc':now()})
    X=frame[FEATURES];y=frame[TARGET]
    model=PersonalSamplingLR(mode=frozen['selected']['mode'])
    start=time.perf_counter()
    with warnings.catch_warnings(record=True) as caught,threadpool_limits(limits=1):
        warnings.simplefilter('always');model.fit(X,y)
    train_seconds=time.perf_counter()-start
    if any(issubclass(w.category,ConvergenceWarning) for w in caught):
        raise ValueError('Final fit did not converge; holdout NOT read')
    fitted_at=now();model_path=FINAL/'model.joblib';joblib.dump(model,model_path,compress=3)
    checksums=json.loads((ROOT/'data/checksums.json').read_text())
    # Fixed named private files, no arbitrary web model or untrusted pickle input.
    names=['final_holdout/features.csv','final_holdout/labels_DO_NOT_USE_FOR_TUNING.csv']
    actual_hashes={name:sha256_file(ROOT/'data'/name) for name in names}
    if any(actual_hashes[name]!=checksums[name] for name in names):raise ValueError('Holdout checksum mismatch')
    holdout_opened_at=now()
    features=pd.read_csv(ROOT/'data'/names[0],dtype={c:str for c in CATEGORICAL})
    labels=pd.read_csv(ROOT/'data'/names[1])
    manifest=pd.read_csv(ROOT/'data/split_manifest.csv')
    expected=manifest.loc[manifest['split'].eq('final_holdout'),ID]
    truth=validate_holdout_tables(features,labels,expected,frame[ID])
    if len(truth)!=52121:raise ValueError('Expected 52121 final buildings')
    Xh=normalize_features(features[FEATURES])
    with threadpool_limits(limits=1):
        pred=model.predict(Xh);proba=model.predict_proba(Xh)
        loaded=joblib.load(model_path)
        np.testing.assert_allclose(loaded.predict_proba(Xh.iloc[:128]),proba[:128],atol=1e-12)
    if model.classes_.tolist()!=[1,2,3]:raise ValueError('Wrong probability column order')
    if not np.isfinite(proba).all() or ((proba<0)|(proba>1)).any():raise ValueError('Bad probabilities')
    np.testing.assert_allclose(proba.sum(1),1,atol=1e-6)
    np.testing.assert_array_equal(proba.argmax(1)+1,pred)
    result=metrics(truth,pred)
    independent=confusion_stats(truth,pred)
    np.testing.assert_allclose(result['macro_f1'],independent['macro_f1'],atol=1e-12,rtol=0)
    oof=pd.DataFrame({ID:features[ID],'y_true':truth,'y_pred':pred,
                      'p_grade_1':proba[:,0],'p_grade_2':proba[:,1],'p_grade_3':proba[:,2]})
    path=FINAL/'holdout_predictions.csv.gz';oof.to_csv(path,index=False,compression={'method':'gzip','mtime':0})
    public={'schema_version':1,'scope':'personal_final_not_team','status':'COMPLETE',
            'selected_experiment':frozen['selected']['experiment'],'model':'lr','mode':frozen['selected']['mode'],
            'development_rows':len(frame),'holdout_rows':len(truth),'evaluation_count':1,
            'frozen_at_utc':frozen['frozen_at_utc'],'model_fitted_at_utc':fitted_at,
            'holdout_opened_at_utc':holdout_opened_at,'completed_at_utc':now(),
            'selection_sha256':selection_sha,'policy_commit':POLICY_COMMIT,
            'model_sha256':sha256_file(model_path),'predictions_sha256':sha256_file(path),
            'source_sha256':frozen['source_sha256'],'data_contract':contract,'holdout_sha256':actual_hashes,
            'metrics':result,'independent_confusion_recalculation':independent,
            'fit_seconds':train_seconds,'n_iter':model.model_.n_iter_.tolist(),
            'warnings':[{'category':type(w.message).__name__,'message':str(w.message)} for w in caught],
            'reload_checked_rows':128,'post_holdout_tuning':False,
            'limitation':'One personal random holdout within the same earthquake; not new-region/future-earthquake validation. This shared holdout is now exposed and must not be reused to choose a team model.'}
    write_json(FINAL/'metrics.json',public);write_json(ROOT/'reports/personal_final_holdout.json',public)
    print('FINAL COMPLETE',result['macro_f1'],result['confusion_matrix'],flush=True)
    return public


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['freeze','evaluate'])
    p.add_argument('--selection-sha');a=p.parse_args()
    if a.stage=='freeze':freeze()
    else:evaluate(a.selection_sha)
