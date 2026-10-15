"""Exploratory TE+ROS model comparison: DEVELOPMENT fixed Stratified fold0 ONLY.
No final holdout, no 5-fold OOF, not T6 ablation. Stored privately; no Git writes.
"""
from pathlib import Path
import gc,hashlib,json,time,warnings
import pandas as pd,numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
import xgboost as xgb
import lightgbm as lgb
ROOT=Path(__file__).parent
start=time.monotonic()
def msg(*xs):print(f'[{time.monotonic()-start:.1f}s]',*xs,flush=True)
def target_enc(xtr,ytr,xva,cols,m=20):
    xtr=xtr.copy();xva=xva.copy()
    Y=pd.get_dummies(ytr).reindex(columns=range(3),fill_value=False).astype(float)
    skf=StratifiedKFold(5,shuffle=True,random_state=0)
    splits=list(skf.split(xtr,ytr))
    for c in cols:
        for k in (0,2):
            name=f'{c}_te{k}';oof=np.zeros(len(xtr),dtype=float)
            for a,b in splits:
                prior_fold=float(Y.iloc[a][k].mean())
                g=Y.iloc[a].groupby(xtr[c].iloc[a].values)[k].agg(['sum','count'])
                smooth=(g['sum']+m*prior_fold)/(g['count']+m)
                oof[b]=xtr[c].iloc[b].map(smooth).fillna(prior_fold).to_numpy()
            xtr[name]=oof
            global_prior=float(Y[k].mean())
            g=Y.groupby(xtr[c].values)[k].agg(['sum','count'])
            smooth=(g['sum']+m*global_prior)/(g['count']+m)
            xva[name]=xva[c].map(smooth).fillna(global_prior).to_numpy()
        count=xtr[c].value_counts()
        xtr[c+'_cnt']=xtr[c].map(count).astype(float).to_numpy()
        xva[c+'_cnt']=xva[c].map(count).fillna(0).astype(float).to_numpy()
    return xtr,xva
D=ROOT/'development.csv'; S=ROOT/'split_manifest.csv'
expected={D.name:'8ef0302a90fc66a49f20fdaefc8e7302ef4bbd3ca2b8b7b741a52f0503ce31b1',S.name:'7673dbeb136e9a916c67a3b19928b3fbc621ae627a5667a677a1ca6eae838721'}
for path in (D,S):
    with path.open('rb') as file:sha=hashlib.file_digest(file,'sha256').hexdigest()
    assert sha==expected[path.name], f'{path.name} SHA mismatch';msg('Verified',path.name)
raw=pd.read_csv(D,encoding='utf-8-sig').sort_values('building_id').reset_index(drop=True)
manifest=pd.read_csv(S).query("split=='development'").sort_values('building_id').reset_index(drop=True)
np.testing.assert_array_equal(raw.building_id,manifest.building_id)
fold=0
train=manifest.cv_fold.ne(fold).to_numpy();valid=~train
X=raw.drop(columns=['building_id','damage_grade']);y=raw.damage_grade-1
# pandas 2.2.3 rejects `include=['object','str']`; use valid include `object`.
cats=X.select_dtypes(include=['object']).columns.tolist()
for c in cats: X[c]=X[c].astype('category')
Xtr,Xva=X.loc[train].copy(),X.loc[valid].copy();ytr,yva=y.loc[train].copy(),y.loc[valid].copy()
msg('split',len(Xtr),len(Xva),'classes train',ytr.value_counts().sort_index().to_dict(),'valid',yva.value_counts().sort_index().to_dict())
geo=['geo_level_1_id','geo_level_2_id','geo_level_3_id']
A,B=target_enc(Xtr,ytr,Xva,geo)
Ao=pd.get_dummies(A,columns=cats,dtype=int);Bo=pd.get_dummies(B,columns=cats,dtype=int).reindex(columns=Ao.columns,fill_value=0)
selected=Ao.columns.tolist()
msg('feature columns',len(selected),'TE/count',[c for c in selected if c.endswith(('_te0','_te2','_cnt'))])
msg('raw geo still present',geo)
Ao=Ao.astype('float32');Bo=Bo.astype('float32')
del A,B,Xtr,Xva,X,raw;gc.collect()
msg('matrixMB',Ao.memory_usage(deep=True).sum()/1e6,Bo.memory_usage(deep=True).sum()/1e6)
Xos,yos=RandomOverSampler(random_state=0).fit_resample(Ao[selected],ytr)
msg('ROS rows',len(Xos),'distribution',yos.value_counts().sort_index().to_dict())
del Ao,ytr;gc.collect()
models={
 'LogisticRegression':make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000)),
 'DecisionTree':DecisionTreeClassifier(random_state=0),
 'RandomForest':RandomForestClassifier(n_jobs=4,random_state=0),
 'XGBoost':xgb.XGBClassifier(subsample=.8,colsample_bytree=.7,n_jobs=4,random_state=0),
 'LightGBM':lgb.LGBMClassifier(verbose=-1,n_jobs=4,random_state=0),
}
result={'scope':'development fixed Stratified fold0 (not original holdout, not full CV, not T6 region ablation)',
 'fold':0,'development_sha256':expected[D.name],'split_sha256':expected[S.name],
 'changes':['pandas object dtype correction','inner-OOF prior from inner training fold, eliminating label prior leak','threads=4 and float32 memory safety'],
 'n_train':int(train.sum()),'n_valid':int(valid.sum()),'n_train_ros':len(Xos),'encoded_feature_count':len(selected),
 'te_features':[c for c in selected if c.endswith(('_te0','_te2','_cnt'))],
 'ros_class_counts':{str(k):int(v) for k,v in yos.value_counts().sort_index().to_dict().items()},'results':[]}
for name,m in models.items():
    t=time.monotonic();msg('START',name)
    try:
        with warnings.catch_warnings(record=True) as warning_records:
            warnings.simplefilter('always');m.fit(Xos,yos)
        p=m.predict(Bo[selected])
        pr,rc,f,support=precision_recall_fscore_support(yva,p,labels=[0,1,2],zero_division=0)
        cm=confusion_matrix(yva,p,labels=[0,1,2]);n3=int((yva==2).sum())
        row={'model':name,'micro_f1':float(f1_score(yva,p,average='micro')),'macro_f1':float(f1_score(yva,p,average='macro')),
             'grade_precision':pr.tolist(),'grade_recall':rc.tolist(),'grade_f1':f.tolist(),'support':support.tolist(),
             'confusion':cm.tolist(),'n_true_grade3':n3,'error31':int(cm[2,0]),'error32':int(cm[2,1]),
             'runtime_sec':round(time.monotonic()-t,2),'warnings':[str(w.message) for w in warning_records]}
        result['results'].append(row)
        with (ROOT/'te_ros_fold0_results.json').open('w') as fp:json.dump(result,fp,ensure_ascii=False,indent=2)
        msg('DONE',name,'macro',round(row['macro_f1'],6),'micro',round(row['micro_f1'],6),'3Recall',round(float(rc[2]),6),
            '3->1',row['error31'],'3->2',row['error32'],'sec',row['runtime_sec']);msg('confusion',cm.tolist())
    except Exception as e:
        msg('ERROR',name,type(e).__name__,str(e));result.setdefault('errors',[]).append({'model':name,'type':type(e).__name__,'error':str(e)})
    del m;gc.collect()
msg('Finished',len(result['results']),'models')
