"""Independent crosstab arithmetic over saved age995 OOF; never fits a model."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from preprocess import ID, load_development, sha256_file, write_json


def recalculate(true, pred):
    cm=pd.crosstab(pd.Series(np.asarray(true),name='true'),pd.Series(np.asarray(pred),name='pred'))
    m=cm.reindex(index=[1,2,3],columns=[1,2,3],fill_value=0).to_numpy(dtype=int)
    row,col,hit=m.sum(1),m.sum(0),m.diagonal()
    f=np.divide(2*hit,row+col,out=np.zeros(3),where=(row+col)!=0)
    return m, float(f.mean())


def audit(data, output, receipt):
    contract=json.loads((ROOT/'configs/data_contract.json').read_text())
    for name,sha in contract['files'].items():
        if sha256_file(data/name)!=sha:raise ValueError('Data hash mismatch.')
    X,y,ids,cv=load_development(data)
    expected=pd.DataFrame({ID:ids,'expected_y':y,'expected_fold':cv.test_fold,'age':X.age})
    summary=pd.read_csv(output/'age995_model_summary.csv')
    group_summary=pd.read_csv(output/'age995_group_summary.csv')
    receipts=[]
    shared_params=None
    for mode in ('A','B','C'):
        source=output/mode/'lr/oof_predictions.csv.gz'
        o=pd.read_csv(source)
        saved=json.loads((output/mode/'lr/metrics.json').read_text())
        if sha256_file(source)!=saved['oof_sha256']:raise ValueError('OOF checksum mismatch.')
        if len(o)!=len(expected) or not o[ID].is_unique:raise ValueError('Missing or duplicated predictions.')
        joined=expected.merge(o,on=ID,how='outer',validate='one_to_one',indicator=True)
        assert joined['_merge'].eq('both').all()
        np.testing.assert_array_equal(joined.y_true,joined.expected_y)
        np.testing.assert_array_equal(joined.fold,joined.expected_fold)
        p=joined[['p_grade_1','p_grade_2','p_grade_3']].to_numpy()
        assert np.isfinite(p).all() and (p>=0).all() and (p<=1).all()
        np.testing.assert_allclose(p.sum(1),1,atol=1e-6,rtol=0)
        np.testing.assert_array_equal(p.argmax(1)+1,joined.y_pred)
        if shared_params is None:shared_params=saved['effective_parameters']
        assert saved['effective_parameters']==shared_params
        fold_scores=[];fold_matrix=np.zeros((3,3),dtype=int)
        for fold in range(5):
            q=joined.loc[joined.fold.eq(fold)]
            m,f=recalculate(q.y_true,q.y_pred)
            np.testing.assert_array_equal(m,saved['fold_results'][fold]['validation']['confusion_matrix'])
            np.testing.assert_allclose(f,saved['fold_results'][fold]['validation']['macro_f1'],atol=1e-12,rtol=0)
            assert len(q)==41696
            fold_scores.append(f);fold_matrix+=m
        whole,pooled=recalculate(joined.y_true,joined.y_pred)
        np.testing.assert_array_equal(whole,fold_matrix)
        np.testing.assert_array_equal(whole,saved['oof']['confusion_matrix'])
        mrow=summary.loc[summary.variant.eq(mode)].iloc[0]
        np.testing.assert_allclose([np.mean(fold_scores),np.std(fold_scores,ddof=1),pooled],
                                  [mrow.cv_mean,mrow.cv_std,mrow.pooled_f1],atol=1e-12,rtol=0)
        for label,mask in [('all',np.ones(len(joined),dtype=bool)),('age995',joined.age.eq(995)),('non995',joined.age.ne(995))]:
            q=joined.loc[mask];m,f=recalculate(q.y_true,q.y_pred)
            grow=group_summary.loc[group_summary.variant.eq(mode)&group_summary['group'].eq(label)].iloc[0]
            assert grow.n==len(q) and grow.grade3_count==m[2].sum()
            assert grow.error_3_to_1==m[2,0] and grow.error_3_to_2==m[2,1]
            np.testing.assert_allclose(grow.macro_f1,f,atol=1e-12,rtol=0)
            for c in range(3):
                precision=m[c,c]/m[:,c].sum() if m[:,c].sum() else 0.
                recall=m[c,c]/m[c].sum() if m[c].sum() else 0.
                f1=2*m[c,c]/(m[c].sum()+m[:,c].sum()) if m[c].sum()+m[:,c].sum() else 0.
                np.testing.assert_allclose([grow[f'grade{c+1}_precision'],grow[f'grade{c+1}_recall'],grow[f'grade{c+1}_f1']],
                    [precision,recall,f1],atol=1e-12,rtol=0)
        receipts.append(dict(variant=mode,oof_sha256=sha256_file(source),cv_mean=float(np.mean(fold_scores)),
                            cv_std=float(np.std(fold_scores,ddof=1)),pooled_f1=pooled,matrix=whole.tolist()))
    result=dict(status='PASS',models=3,fold_matrices=15,group_rows=9,rows_verified=3*len(X),
                holdout_scored=False,method='independent pandas crosstab and direct metric equations',
                source_code_sha256=sha256_file(__file__),results=receipts)
    if receipt.exists():raise FileExistsError('Do not overwrite a prior audit receipt.')
    write_json(receipt,result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,default=ROOT/'data')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--receipt',type=Path,required=True)
    a=p.parse_args();audit(a.data_dir,a.output,a.receipt)
