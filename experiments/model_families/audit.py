"""Independent OOF arithmetic and native-model replay; never fits a predictive model.

Only load joblib preprocessors created by this study in a trusted private directory.
The trainer uses sklearn metrics; this file uses crosstab and direct 3x3 arithmetic.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from preprocess import FEATURES, ID, TARGET, NUMERIC, CATEGORICAL, sha256_file, write_json
from scripts.audit_eda_oof import development_frame

LABELS = [1, 2, 3]
PROBS = ['p_grade_1', 'p_grade_2', 'p_grade_3']
NEW = ['XGB_A_v1', 'XGB_C_v1', 'LGBM_A_v1', 'LGBM_C_v1']
ALL = ['LR_A', 'LR_C', *NEW]
PINNED = {'LR_A': 'e8dbaa962fe4b2e2fdad160d25f72cbfd0816d73aafbc0419149c53b8a3de221',
          'LR_C': 'de43f9974e536b3a9c5ebac35b984ed886ba2d5426d59dda989ca12f366a2cbc'}
PAIRS = [('LR_A','LR_C'), ('LR_A','XGB_A_v1'), ('LR_C','XGB_C_v1'),
         ('LR_A','LGBM_A_v1'), ('LR_C','LGBM_C_v1'), ('XGB_A_v1','XGB_C_v1'),
         ('LGBM_A_v1','LGBM_C_v1'), ('XGB_A_v1','LGBM_A_v1'), ('XGB_C_v1','LGBM_C_v1')]


def close(a, b, atol=1e-12):
    np.testing.assert_allclose(a, b, atol=atol, rtol=0)


def integers(values, name):
    arr = pd.to_numeric(values, errors='raise').to_numpy(dtype=float)
    if not np.isfinite(arr).all() or not np.equal(arr, np.floor(arr)).all():
        raise ValueError(f'Noninteger or nonfinite {name}.')
    return arr.astype(np.int64)


def aligned(frame, raw):
    cols = [ID, 'fold', 'y_true', 'y_pred', *PROBS]
    if len(raw) != len(frame) or not set(cols).issubset(raw.columns):
        raise ValueError('OOF coverage/schema mismatch.')
    raw = raw[cols].copy()
    for key in (ID, 'fold', 'y_true', 'y_pred'):
        raw[key] = integers(raw[key], key)
    if raw[ID].duplicated().any() or set(raw[ID]) != set(frame[ID]):
        raise ValueError('OOF ID membership/uniqueness mismatch.')
    raw = raw.set_index(ID).loc[frame[ID]].reset_index()
    if not np.array_equal(raw.fold, frame['fold']) or not np.array_equal(raw.y_true, frame[TARGET]):
        raise ValueError('OOF labels or frozen folds disagree.')
    if not np.isin(raw.y_true, LABELS).all() or not np.isin(raw.y_pred, LABELS).all():
        raise ValueError('Invalid external labels.')
    p = raw[PROBS].to_numpy(dtype=float)
    if not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1):
        raise ValueError('Invalid probabilities.')
    close(p.sum(axis=1), 1, atol=1e-6)
    if not np.array_equal(p.argmax(axis=1)+1, raw.y_pred):
        raise ValueError('Prediction/argmax/class order mismatch.')
    return raw


def statistics(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    if y.ndim != 1 or y.shape != pred.shape or not len(y):
        raise ValueError('Aligned nonempty observations required.')
    if not np.isin(y, LABELS).all() or not np.isin(pred, LABELS).all():
        raise ValueError('Invalid label.')
    cm = pd.crosstab(pd.Series(y, name='truth'), pd.Series(pred, name='pred')).reindex(
        index=LABELS, columns=LABELS, fill_value=0).fillna(0).to_numpy(dtype=np.int64)
    support, predicted = cm.sum(axis=1), cm.sum(axis=0)
    tp = cm.diagonal()
    p = np.divide(tp, predicted, out=np.zeros(3), where=predicted != 0)
    r = np.divide(tp, support, out=np.zeros(3), where=support != 0)
    f = np.divide(2*tp, support+predicted, out=np.zeros(3), where=(support+predicted) != 0)
    norm = [[float(cm[i,j]/support[i]) if support[i] else None for j in range(3)] for i in range(3)]
    out = {'n': int(support.sum()), 'macro_f1': float(f.mean()),
           'accuracy': float(tp.sum()/support.sum()), 'grade3_count': int(support[2]),
           'error_3_to_1': int(cm[2,0]), 'error_3_to_2': int(cm[2,1]),
           'rate_3_to_1': float(cm[2,0]/support[2]) if support[2] else None,
           'rate_3_to_2': float(cm[2,1]/support[2]) if support[2] else None,
           'matrix': cm.tolist(), 'matrix_row_normalized': norm}
    for i in range(3):
        for key, value in [('support',support[i]),('predicted',predicted[i]),
                           ('precision',p[i]),('recall',r[i]),('f1',f[i])]:
            out[f'grade{i+1}_{key}'] = int(value) if key in ('support','predicted') else float(value)
    return out


def compare_metric(s, saved):
    if s['matrix'] != saved['confusion_matrix']:
        raise ValueError('Confusion matrix mismatch.')
    for key in ('macro_f1','accuracy'):
        close(s[key], saved[key])
    close(s['rate_3_to_1'], saved['severe_as_minor_rate'])
    for c in LABELS:
        for key, old in [('precision','precision'),('recall','recall'),('f1','f1-score'),('support','support')]:
            close(s[f'grade{c}_{key}'], saved['per_class'][str(c)][old])


def verify_saved(oof, saved):
    if (saved['status'] != 'complete' or saved['rows'] != len(oof) or saved['folds'] != 5
            or saved['holdout_scored'] or [r['fold'] for r in saved['fold_results']] != list(range(5))):
        raise ValueError('Incomplete or incompatible saved result.')
    fold_rows = []
    for f, rec in enumerate(saved['fold_results']):
        part = oof.loc[oof.fold.eq(f)]
        if not len(part) or rec['train_rows'] != len(oof)-len(part) or rec['validation_rows'] != len(part):
            raise ValueError('Train/validation denominator mismatch.')
        s = statistics(part.y_true, part.y_pred)
        compare_metric(s, rec['validation'])
        fold_rows.append({'fold': f, 'macro_f1': s['macro_f1'], 'grade3_recall': s['grade3_recall'],
                          'error_3_to_1': s['error_3_to_1'], 'error_3_to_2': s['error_3_to_2'],
                          'fit_seconds': rec['fit_seconds'], 'train_rows': rec['train_rows'],
                          'validation_rows': rec['validation_rows']})
    pooled = statistics(oof.y_true, oof.y_pred)
    compare_metric(pooled, saved['oof'])
    scores = [r['macro_f1'] for r in fold_rows]
    close(np.mean(scores), saved['cv_macro_f1_mean'])
    close(np.std(scores, ddof=1), saved['cv_macro_f1_std'])
    return pooled, fold_rows


def flat(s):
    return {k:v for k,v in s.items() if not k.startswith('matrix')}


def transitions(y, before, after):
    y,b,a = map(np.asarray, (y,before,after))
    old, new = (y==3)&(b==1), (y==3)&(a==1)
    return {'n': len(y), 'changed': int(np.sum(b!=a)),
            'before_only_correct': int(np.sum((y==b)&(y!=a))),
            'after_only_correct': int(np.sum((y!=b)&(y==a))),
            'both_wrong_changed': int(np.sum((y!=a)&(y!=b)&(a!=b))),
            'severe_fixed': int(np.sum(old&~new)), 'severe_new': int(np.sum(~old&new)),
            'severe_retained': int(np.sum(old&new))}


def partitions(frame):
    allmask = np.ones(len(frame),dtype=bool)
    age995 = frame.age.eq(995).to_numpy()
    groups = [('group','all',allmask),('group','age995',age995),('group','non995',~age995)]
    for field in ['geo_level_1_id', 'foundation_type']:
        for v in sorted(frame[field].unique()):
            groups.append((field,str(v),frame[field].eq(v).to_numpy()))
    bins = pd.cut(frame.age, [-1,0,10,20,30,50,100,994,995],
                  labels=['0','1-10','11-20','21-30','31-50','51-100','101-994','995'])
    for label in bins.cat.categories:
        groups.append(('age_group',label,bins.eq(label).to_numpy()))
    return groups


def encoded_digest(matrix):
    # Independently spelled out to avoid using trainer's hashing helper.
    from scipy import sparse
    if not sparse.isspmatrix_csr(matrix) or str(matrix.dtype) != 'float32':
        raise ValueError('Wrong sparse representation.')
    h = hashlib.sha256()
    for item, dtype in [(matrix.shape,'<i8'),(matrix.indptr,'<i8'),(matrix.indices,'<i8'),(matrix.data,'<f4')]:
        h.update(np.asarray(item, dtype=dtype).tobytes())
    return h.hexdigest()


def replay(frame, root, oo, saved):
    import joblib
    import xgboost as xgb
    import lightgbm as lgb
    from scipy import sparse
    evidence=[]
    for mode in ('A','C'):
        for f in range(5):
            tr = np.flatnonzero(frame['fold'].ne(f)); va = np.flatnonzero(frame['fold'].eq(f))
            xr, xv = frame.iloc[tr][FEATURES], frame.iloc[va][FEATURES]
            recs = [saved[f'{family}_{mode}_v1']['fold_results'][f] for family in ('XGB','LGBM')]
            path = root/recs[0]['preprocessor_file']
            if any(sha256_file(path)!=r['preprocessor_sha256'] for r in recs):
                raise ValueError('Preprocessor SHA mismatch.')
            pre = joblib.load(path)  # Our own checksum-verified, trusted artifact only.
            for name, vocab in zip(CATEGORICAL,pre['columns'].named_transformers_['categorical'].categories_):
                if set(vocab) != set(xr[name]): raise ValueError('Vocabulary not training-only.')
            expected = xr[NUMERIC].copy()
            if mode=='C':
                med = float(xr.loc[xr.age.ne(995),'age'].median())
                close(pre['age995'].median_,med)
                expected['age']=expected['age'].mask(expected.age.eq(995),med)
            mean=expected.to_numpy(dtype='float32').mean(axis=0,dtype='float64')
            close(mean,pre['columns'].named_transformers_['numeric']['scale'].mean_,atol=1e-10)
            mt,mv=pre.transform(xr),pre.transform(xv)
            actual={'train':encoded_digest(mt),'validation':encoded_digest(mv)}
            if any(actual!=r['encoded_sha256'] for r in recs): raise ValueError('Matrix SHA mismatch.')
            for family, rec in zip(('XGB','LGBM'),recs):
                name=f'{family}_{mode}_v1'; model_path=root/rec['model_file']
                if sha256_file(model_path)!=rec['model_sha256']: raise ValueError('Model SHA mismatch.')
                if family=='XGB':
                    model=xgb.Booster(params={'nthread':2});model.load_model(model_path)
                    p=model.predict(xgb.DMatrix(mv,nthread=2)); rounds=model.num_boosted_rounds()
                else:
                    model=lgb.Booster(model_file=str(model_path))
                    p=model.predict(mv,num_threads=2);rounds=model.current_iteration()
                if rounds!=300 or rec['rounds_completed']!=300: raise ValueError('Budget mismatch.')
                expected_p=oo[name].iloc[va][PROBS].to_numpy()
                close(p,expected_p,atol=1e-7)
                if not np.array_equal(p.argmax(1)+1,oo[name].iloc[va].y_pred):
                    raise ValueError('Native model replay changes predictions.')
                evidence.append({'experiment':name,'fold':f,'rows':len(va),'rounds':rounds,
                    'max_probability_difference':float(np.max(np.abs(p-expected_p))),
                    'model_sha256':rec['model_sha256'],'preprocessor_sha256':rec['preprocessor_sha256'],
                    'validation_matrix_sha256':actual['validation'], 'predictions_exactly_equal':True})
                del model
            del pre,mt,mv
            gc.collect()
    return evidence


def audit(data_dir, root, lr_root, output):
    data_dir,root,lr_root,output=map(Path,(data_dir,root,lr_root,output))
    if output.exists() and any(output.iterdir()): raise FileExistsError('Use a fresh audit directory.')
    frame,contract=development_frame(data_dir)
    meta=json.loads((root/'run_metadata.json').read_text())
    if (meta['status']!='TRAINED_PENDING_INDEPENDENT_AUDIT' or meta['new_full_cv_fits']!=20
            or meta['holdout_opened'] or meta['data_contract']!=contract):
        raise ValueError('Training completion/contract is not valid.')
    protected={str(p):sha256_file(p) for folder in [root,lr_root] for p in folder.rglob('*') if p.is_file()}
    for name,digest in meta['source_code_sha256'].items():
        if sha256_file(ROOT/name)!=digest: raise ValueError('Execution source changed.')
    if sha256_file(ROOT/'experiments/model_families/config.json')!=meta['config_sha256']:
        raise ValueError('Configuration changed.')
    oo,saved,sources={}, {}, []
    summaries,folds,groups,profiles,matrices=[],[],[],[],[]
    parts=partitions(frame)
    for name in ALL:
        path=lr_root/name[-1]/'lr' if name.startswith('LR_') else root/name
        saved[name]=json.loads((path/'metrics.json').read_text())
        oof_path=path/'oof_predictions.csv.gz'; digest=sha256_file(oof_path)
        if saved[name]['oof_sha256']!=digest or (name in PINNED and digest!=PINNED[name]):
            raise ValueError('OOF SHA mismatch.')
        if name in PINNED:
            ref=meta['lr_reuse'][name[-1]]
            lrmeta=lr_root/name[-1]/'run_metadata.json'
            if (sha256_file(lrmeta)!=ref['metadata_sha256'] or
                    sha256_file(path/'metrics.json')!=ref['metrics_sha256'] or ref['refit_in_this_study']):
                raise ValueError('Historical metadata/metric receipt changed.')
        elif saved[name]['parameters']!=meta['config']['models'][name.split('_')[0]]:
            raise ValueError('Recorded estimator parameters differ from frozen configuration.')
        oo[name]=aligned(frame,pd.read_csv(oof_path))
        pooled,ff=verify_saved(oo[name],saved[name])
        for r in ff: folds.append({'experiment':name, **r})
        summaries.append({'experiment':name,'historical_reuse':name in PINNED,
            'cv_mean':saved[name]['cv_macro_f1_mean'],'cv_std_ddof1':saved[name]['cv_macro_f1_std'],
            'pooled_f1':pooled['macro_f1'], 'total_fit_seconds':sum(r['fit_seconds'] for r in ff), **flat(pooled)})
        for dimension,category,mask in parts:
            if not np.any(mask): continue
            s=statistics(oo[name].loc[mask,'y_true'],oo[name].loc[mask,'y_pred'])
            row={'experiment':name,'dimension':dimension,'category':category,**flat(s)}
            (groups if dimension=='group' else profiles).append(row)
            matrices.append({'experiment':name,'dimension':dimension,'category':category,
                             'matrix':s['matrix'],'matrix_row_normalized':s['matrix_row_normalized']})
        sources.append({'experiment':name,'oof_sha256':digest,'metrics_sha256':sha256_file(path/'metrics.json'),
                        'historical_reuse':name in PINNED})
    replays=replay(frame,root,oo,saved)
    paired,delta,segment_delta=[],[],[]
    for before,after in PAIRS:
        for dimension,category,mask in parts:
            if not np.any(mask): continue
            b,a=oo[before].loc[mask],oo[after].loc[mask]
            bs,aa=statistics(b.y_true,b.y_pred),statistics(a.y_true,a.y_pred)
            d={'before':before,'after':after,'dimension':dimension,'category':category,'n':len(b),
               'grade3_count':bs['grade3_count'], 'delta_f1':aa['macro_f1']-bs['macro_f1'],
               'delta_recall3':aa['grade3_recall']-bs['grade3_recall'],
               'delta_3_to_1':aa['error_3_to_1']-bs['error_3_to_1'],
               'delta_3_to_2':aa['error_3_to_2']-bs['error_3_to_2']}
            segment_delta.append(d)
            if dimension=='group':
                paired.append({**d,**transitions(b.y_true,b.y_pred,a.y_pred)})
        for f in range(5):
            b=[r for r in folds if r['experiment']==before and r['fold']==f][0]
            a=[r for r in folds if r['experiment']==after and r['fold']==f][0]
            delta.append({'before':before,'after':after,'fold':f,'delta_macro_f1':a['macro_f1']-b['macro_f1']})
    if protected!={p:sha256_file(p) for p in protected}: raise ValueError('Private evidence mutated.')
    for name,digest in contract['files'].items():
        if sha256_file(data_dir/name)!=digest: raise ValueError('Data mutated.')
    output.mkdir(parents=True,exist_ok=True)
    tables={'model_summary.csv':summaries,'fold_summary.csv':folds,'group_summary.csv':groups,
            'error_profiles.csv':profiles,'paired_transitions.csv':paired,'paired_fold_deltas.csv':delta,
            'segment_deltas.csv':segment_delta}
    for filename,rows in tables.items(): pd.DataFrame(rows).to_csv(output/filename,index=False)
    (output/'confusion_matrices.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in matrices))
    write_json(output/'replay_validation.json',replays)
    receipt={'status':'PASS','completed_at_utc':datetime.now(timezone.utc).isoformat(),
             'base_commit':meta['base_commit'],'preregistration_commit':'e0dd14630b9d20aa116fee57aa6555102c15c01c',
             'data_contract':contract,'raw_source_sha_verified_in_this_study':False,
             'development_rows':len(frame),'models_compared':6,'new_predictive_fits':20,'lr_refits':0,
             'verified_oof_rows':sum(len(o) for o in oo.values()),'replayed_models':len(replays),
             'replayed_validation_predictions':sum(r['rows'] for r in replays),'holdout_opened':False,
             'private_files_unchanged':len(protected),'sources':sources,
             'environment':meta['environment'],'config_sha256':meta['config_sha256'],
             'source_code_sha256':{**meta['source_code_sha256'],'experiments/model_families/audit.py':sha256_file(__file__)},
             'tables':{name:len(rows) for name,rows in tables.items()},
             'aggregate_sha256':{p.name:sha256_file(p) for p in output.iterdir() if p.is_file()},
             'limitations':['Same historical development folds, not independent holdout/generalization evidence.',
                            'Independent arithmetic and full validation replay, not independent model retraining.',
                            'Fixed untuned implementations; not equal capacity or compute-time matching.',
                            'CSR implicit-zero semantics differ across XGBoost and LightGBM.']}
    write_json(output/'verification.json',receipt)
    print(json.dumps({'status':'PASS','new_fits':20,'models':6,'oof_rows':receipt['verified_oof_rows'],
                      'replayed_predictions':receipt['replayed_validation_predictions']}),flush=True)
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for arg in ['data-dir','root','lr-root','output']: parser.add_argument('--'+arg,type=Path,required=True)
    a=parser.parse_args()
    with threadpool_limits(limits=2): audit(a.data_dir,a.root,a.lr_root,a.output)
