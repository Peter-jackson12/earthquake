"""Read-only public aggregate contract for the Streamlit dashboard.

This module never imports modeling code and never reads data/, OOF rows, models,
or holdout labels. Rates and class metrics are recalculated from integer counts.
"""
from __future__ import annotations
import json
import hashlib
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parent
REPORTS = ROOT/'reports'
EVIDENCE_LABELS = {
    'legacy_aggregate_checked_raw_oof_unavailable': '집계 재계산 · 과거 행별 OOF 미확인',
    'row_oof_checked': '행별 OOF 검증 완료',
}


def error_stats(record: dict) -> dict:
    cm = np.asarray(record['matrix'],dtype=np.int64)
    true, pred, tp = cm.sum(1), cm.sum(0), cm.diagonal()
    precision = np.divide(tp,pred,out=np.zeros(3),where=pred!=0)
    recall = np.divide(tp,true,out=np.zeros(3),where=true!=0)
    f1 = np.divide(2*tp,true+pred,out=np.zeros(3),where=(true+pred)!=0)
    severe = int(true[2])
    return {'matrix': cm, 'support': true, 'precision': precision, 'recall':recall, 'f1':f1,
            'oof_macro_f1':float(f1.mean()),'accuracy':float(tp.sum()/cm.sum()),
            'grade3_count':severe,'error_3_to_1':int(cm[2,0]),'error_3_to_2':int(cm[2,1]),
            'rate_3_to_1':float(cm[2,0]/severe) if severe else None,
            'rate_3_to_2':float(cm[2,1]/severe) if severe else None}


def validate_personal_final(bundle):
    final=bundle.get('holdout_result')
    selection=bundle.get('personal_selection')
    if not isinstance(final,dict) or not isinstance(selection,dict):
        raise ValueError('Verified personal result and frozen selection required')
    if (final.get('scope')!='personal_final_not_team' or final.get('status')!='COMPLETE'
            or final.get('evaluation_count')!=1 or final.get('post_holdout_tuning') is not False):
        raise ValueError('Invalid personal evaluation scope')
    if selection.get('status')!='FROZEN_BEFORE_HOLDOUT' or selection.get('evaluation_limit')!=1:
        raise ValueError('Frozen selection required')
    if final.get('selection_sha256')!=bundle.get('personal_selection_sha256'):
        raise ValueError('Final result/selection digest mismatch')
    if (final.get('selected_experiment')!=selection['selected']['experiment']
            or final.get('mode')!=selection['selected']['mode']):
        raise ValueError('Final model differs from frozen choice')
    if final.get('holdout_rows')!=bundle['holdout_rows'] or final.get('development_rows')!=bundle['development_rows']:
        raise ValueError('Incorrect final population')
    times=[datetime.fromisoformat(final[x]) for x in
           ['frozen_at_utc','model_fitted_at_utc','holdout_opened_at_utc','completed_at_utc']]
    if times!=sorted(times) or final['frozen_at_utc']!=selection['frozen_at_utc']:
        raise ValueError('Selection must precede final fit and holdout opening')
    cm=np.asarray(final['metrics']['confusion_matrix'])
    if cm.shape!=(3,3) or not np.isfinite(cm).all() or (cm<0).any() or (cm!=np.floor(cm)).any():
        raise ValueError('Bad final confusion counts')
    if cm.sum()!=bundle['holdout_rows']:
        raise ValueError('Final confusion population mismatch')
    stats=error_stats({'matrix':cm})
    if not np.isclose(stats['oof_macro_f1'],final['metrics']['macro_f1'],rtol=0,atol=1e-12):
        raise ValueError('Final Macro F1 mismatch')
    if not np.isclose(stats['accuracy'],final['metrics']['accuracy'],rtol=0,atol=1e-12):
        raise ValueError('Final Accuracy mismatch')
    for c in range(3):
        row=final['metrics']['per_class'][str(c+1)]
        for key,values in [('precision',stats['precision']),('recall',stats['recall']),('f1-score',stats['f1']),('support',stats['support'])]:
            if not np.isclose(row[key],values[c],rtol=0,atol=1e-12):
                raise ValueError('Final class metric mismatch')


def validate_bundle(bundle: dict) -> dict:
    version = bundle.get('schema_version')
    if version not in (1, 2):
        raise ValueError('Unknown aggregate schema')
    if version == 1 and bundle.get('holdout_result') is not None:
        raise ValueError('Legacy snapshot cannot contain final results')
    if version == 2:
        validate_personal_final(bundle)
    n=bundle['development_rows']; support=np.asarray(bundle['class_support'])
    if n!=208480 or support.tolist()!=[20099,118607,69774] or support.sum()!=n:
        raise ValueError('Development population differs from the frozen contract.')
    seen=set()
    for record in bundle['models']:
        identity=(record['experiment'],record['model'])
        if identity in seen: raise ValueError('Duplicate model configuration.')
        seen.add(identity)
        cm=np.asarray(record['matrix'])
        if cm.shape!=(3,3) or not np.isfinite(cm).all() or (cm<0).any() or (cm!=np.floor(cm)).any():
            raise ValueError('Confusion counts must be nonnegative 3x3 integers.')
        if not np.array_equal(cm.sum(1),support): raise ValueError('Incorrect confusion support.')
        scores=np.asarray(record['fold_scores'],dtype=float)
        if scores.shape!=(5,) or not np.isfinite(scores).all() or ((scores<0)|(scores>1)).any():
            raise ValueError('Five valid fold metrics required.')
        if not np.isclose(scores.mean(),record['cv_mean'],atol=1e-12,rtol=0):
            raise ValueError('CV mean mismatch.')
        if not np.isclose(scores.std(ddof=1),record['cv_std'],atol=1e-12,rtol=0):
            raise ValueError('CV sample standard deviation mismatch.')
        if not np.isclose(error_stats(record)['oof_macro_f1'],record['oof_macro_f1'],atol=1e-12,rtol=0):
            raise ValueError('Pooled OOF F1 mismatch.')
        if record['evidence_level'] not in EVIDENCE_LABELS: raise ValueError('Unknown evidence level.')
    eda=pd.DataFrame(bundle['eda'])
    for feature,part in eda.groupby('feature'):
        if part.category.duplicated().any(): raise ValueError('Duplicate EDA category.')
        if ((part.n<0)|(part.grade3_count<0)|(part.grade3_count>part.n)).any():
            raise ValueError('Invalid EDA denominator or numerator.')
        if feature!='region_foundation' and (part.n.sum()!=n or part.grade3_count.sum()!=support[2]):
            raise ValueError('EDA partition totals disagree.')
        expected=np.divide(part.grade3_count,part.n,out=np.zeros(len(part)),where=part.n.to_numpy()!=0)
        if not np.allclose(expected,part.grade3_rate.astype(float),atol=1e-12,rtol=0,equal_nan=True):
            raise ValueError('EDA rates disagree.')
    regions=pd.DataFrame(bundle['regions'])
    if len(regions):
        for identity,group in regions.groupby(['experiment','model']):
            if identity not in seen or group.region.duplicated().any(): raise ValueError('Invalid region identity.')
            if len(group)!=31 or group.n.sum()!=n or group.grade3_count.sum()!=support[2]:
                raise ValueError('Regional populations do not reconcile.')
            rec=next(x for x in bundle['models'] if (x['experiment'],x['model'])==identity)
            if rec['evidence_level']!='row_oof_checked': raise ValueError('Unverified legacy regional results must not be imported.')
            stats=error_stats(rec)
            for key in ['error_3_to_1','error_3_to_2']:
                if group[key].sum()!=stats[key]: raise ValueError('Regional error counts do not reconcile.')
            if ((group.error_3_to_1+group.error_3_to_2)>group.grade3_count).any():
                raise ValueError('More severe errors than severe cases.')
    return bundle


def load_bundle(directory: Path = REPORTS) -> dict:
    directory=Path(directory)
    def read(name):
        with (directory/name).open(encoding='utf-8') as f:
            return json.load(f)
    bundle=read('dashboard_context.json')
    if bundle.get('schema_version')==2:
        for filename,key in [('personal_selection.json','personal_selection_sha256'),
                             ('personal_final_holdout.json','personal_final_sha256')]:
            with (directory/filename).open('rb') as handle:
                data=handle.read()
            if hashlib.sha256(data).hexdigest()!=bundle.get(key):
                raise ValueError('Public final provenance checksum mismatch: '+filename)
            bundle['personal_selection' if filename=='personal_selection.json' else 'holdout_result']=json.loads(data)
    bundle['models']=read('dashboard_models.json')
    packed=read('dashboard_regions.json')
    bundle['regions']=[]
    for identity, rows in packed.items():
        experiment,model=identity.split('|')
        for region,n,g3,e1,e2 in rows:
            bundle['regions'].append(dict(experiment=experiment,model=model,region=str(region),
                                          n=n,grade3_count=g3,error_3_to_1=e1,error_3_to_2=e2))
    bundle['eda']=pd.read_csv(directory/'eda_summary.csv',dtype={'category':str}).to_dict('records')
    return validate_bundle(bundle)


def model_table(bundle: dict) -> pd.DataFrame:
    records=[]
    for m in bundle['models']:
        stats=error_stats(m)
        records.append({'experiment':m['experiment'],'model':m['model'],'label':m['label'],'stage':m['stage'],
                        'cv_macro_f1':m['cv_mean'],'fold_std':m['cv_std'],
                        'oof_macro_f1':stats['oof_macro_f1'],'accuracy':stats['accuracy'],
                        'error_3_to_1':stats['error_3_to_1'],'error_3_to_2':stats['error_3_to_2'],
                        'rate_3_to_1':stats['rate_3_to_1'],'grade3_recall':float(stats['recall'][2]),
                        'converged':m['converged'],'evidence':EVIDENCE_LABELS[m['evidence_level']]})
    return pd.DataFrame(records).sort_values('cv_macro_f1',ascending=False,kind='stable').reset_index(drop=True)


def region_table(bundle: dict, experiment: str, model: str) -> pd.DataFrame:
    data=pd.DataFrame([x for x in bundle['regions'] if x['experiment']==experiment and x['model']==model])
    if data.empty: return data
    for error in ['1','2']:
        data[f'rate_3_to_{error}']=np.divide(data[f'error_3_to_{error}'],data.grade3_count,
                                          out=np.full(len(data),np.nan),where=data.grade3_count.to_numpy()!=0)
    data['grade3_recall']=1-data.rate_3_to_1-data.rate_3_to_2
    return data.sort_values('rate_3_to_1',ascending=False,na_position='last')
