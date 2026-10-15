"""Complete RandomForest five-fold CV using prior verified fold0 without refitting fold0.
Development-only, same predeclared Stratified manifest. Nested OOF geographic TE;
train-only ROS. No holdout reading and no Git writes.
"""
from pathlib import Path
import gc,json,time,hashlib,warnings
import numpy as np,pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,confusion_matrix,precision_recall_fscore_support
from imblearn.over_sampling import RandomOverSampler
ROOT=Path(__file__).parent
start=time.monotonic()
def msg(*xs):print(f'[{time.monotonic()-start:.1f}s]',*xs,flush=True)
def target_enc(xtr,ytr,xva,cols,m=20):
    xtr=xtr.copy();xva=xva.copy()
    Y=pd.get_dummies(ytr).reindex(columns=range(3),fill_value=False).astype(float)
    splits=list(StratifiedKFold(5,shuffle=True,random_state=0).split(xtr,ytr))
    for c in cols:
        for k in (0,2):
            oof=np.zeros(len(xtr),dtype=float)
            for a,b in splits:
                prior=float(Y.iloc[a][k].mean());g=Y.iloc[a].groupby(xtr[c].iloc[a].values)[k].agg(['sum','count'])
                sm=(g['sum']+m*prior)/(g['count']+m)
                oof[b]=xtr[c].iloc[b].map(sm).fillna(prior).to_numpy()
            name=f'{c}_te{k}';xtr[name]=oof
            prior=float(Y[k].mean());g=Y.groupby(xtr[c].values)[k].agg(['sum','count']);sm=(g['sum']+m*prior)/(g['count']+m)
            xva[name]=xva[c].map(sm).fillna(prior).to_numpy()
        count=xtr[c].value_counts()
        xtr[c+'_cnt']=xtr[c].map(count).astype(float).to_numpy();xva[c+'_cnt']=xva[c].map(count).fillna(0).astype(float).to_numpy()
    return xtr,xva
original=json.loads((ROOT/'te_ros_fold0_results.json').read_text())
fold0=next(row for row in original['results'] if row['model']=='RandomForest')
d=pd.read_csv(ROOT/'development.csv',encoding='utf-8-sig').sort_values('building_id').reset_index(drop=True)
s=pd.read_csv(ROOT/'split_manifest.csv').query("split=='development'").sort_values('building_id').reset_index(drop=True)
np.testing.assert_array_equal(d.building_id,s.building_id)
X=d.drop(columns=['building_id','damage_grade']);y=d.damage_grade-1
cats=X.select_dtypes(include=['object']).columns.tolist()
for c in cats:X[c]=X[c].astype('category')
geo=['geo_level_1_id','geo_level_2_id','geo_level_3_id']
cv=[{'fold':0,'macro_f1':fold0['macro_f1'],'micro_f1':fold0['micro_f1'],
     'confusion':fold0['confusion'],'grade_recall':fold0['grade_recall'],
     'grade_f1':fold0['grade_f1'],'grade_precision':fold0['grade_precision'],
     'source':'reused original exact fold0 run result; no refit'}]
for fold in [1,2,3,4]:
    tr=s.cv_fold.ne(fold).to_numpy();va=~tr
    msg('Fold',fold,'train',tr.sum(),'validation',va.sum())
    Xtr,Xva=X.loc[tr].copy(),X.loc[va].copy()
    ytr,yva=y.loc[tr].copy(),y.loc[va].copy()
    A,B=target_enc(Xtr,ytr,Xva,geo)
    Ao=pd.get_dummies(A,columns=cats,dtype=int);Bo=pd.get_dummies(B,columns=cats,dtype=int).reindex(columns=Ao.columns,fill_value=0)
    assert len(Ao.columns)==original['encoded_feature_count'],f"feature count drift {len(Ao.columns)}"
    Ao=Ao.astype('float32');Bo=Bo.astype('float32')
    Xos,yos=RandomOverSampler(random_state=0).fit_resample(Ao,ytr)
    del A,B,Xtr,Xva,Ao,ytr;gc.collect()
    model=RandomForestClassifier(n_jobs=4,random_state=0)
    tic=time.monotonic();model.fit(Xos,yos);pred=model.predict(Bo)
    p,r,f,sp=precision_recall_fscore_support(yva,pred,labels=[0,1,2],zero_division=0)
    cm=confusion_matrix(yva,pred,labels=[0,1,2])
    record={'fold':fold,'macro_f1':float(f1_score(yva,pred,average='macro')),'micro_f1':float(f1_score(yva,pred,average='micro')),
            'confusion':cm.tolist(),'grade_recall':r.tolist(),'grade_f1':f.tolist(),'grade_precision':p.tolist(),
            'train_rows':int(tr.sum()),'valid_rows':int(va.sum()),'ros_train_rows':int(len(yos)),'fit_predict_sec':round(time.monotonic()-tic,2)}
    cv.append(record)
    with (ROOT/'rf_5fold_progress.json').open('w') as out:json.dump(cv,out,ensure_ascii=False,indent=2)
    msg('DONE fold',fold,'macro',record['macro_f1'],'micro',record['micro_f1'],'3Recall',record['grade_recall'][2],'seconds',record['fit_predict_sec'])
    del Xos,yos,Bo,model,pred;gc.collect()
cm=np.array([r['confusion'] for r in cv]).sum(axis=0)
P=cm.sum(axis=0);S=cm.sum(axis=1);TP=np.diag(cm)
F1=np.divide(2*TP,P+S,out=np.zeros(3),where=(P+S)!=0)
prec=np.divide(TP,P,out=np.zeros(3),where=P!=0)
rec=np.divide(TP,S,out=np.zeros(3),where=S!=0)
aggregate={'model':'RandomForest_TE_ROS','status':'5_FOLD_DEV_ONLY_COMPLETE','n_valid_total':int(cm.sum()),
           'cv_macro_f1_mean':float(np.mean([r['macro_f1'] for r in cv])),
           'cv_macro_f1_std_ddof1':float(np.std([r['macro_f1'] for r in cv],ddof=1)),
           'cv_micro_f1_mean':float(np.mean([r['micro_f1'] for r in cv])),
           'pooled_macro_f1_from_5fold_confusion':float(F1.mean()),
           'pooled_micro_f1_from_5fold_confusion':float(TP.sum()/cm.sum()),
           'pooled_grade_precision':prec.tolist(),'pooled_grade_recall':rec.tolist(),'pooled_grade_f1':F1.tolist(),
           'pooled_confusion_matrix':cm.tolist(),'folds':cv,
           'holdout_evaluated':False,'development_sha256':original['development_sha256'],'split_sha256':original['split_sha256'],
           'note':'Original user code exploratory TE+ROS; not T6 geo ablation; only RF has 5fold result.'}
with (ROOT/'rf_5fold_result.json').open('w') as out:json.dump(aggregate,out,ensure_ascii=False,indent=2)
msg('FINAL CV',round(aggregate['cv_macro_f1_mean'],8), 'SD',round(aggregate['cv_macro_f1_std_ddof1'],8),
    'pooled_macro',round(aggregate['pooled_macro_f1_from_5fold_confusion'],8), 'pooled_micro',round(aggregate['pooled_micro_f1_from_5fold_confusion'],8))
