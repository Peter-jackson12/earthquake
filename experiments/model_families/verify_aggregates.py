"""Check published aggregate arithmetic against private OOF using sklearn.

This read-only check is independent from audit.py's crosstab/formula calculation.
No predictive fitting, holdout access, or raw-row publication.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from preprocess import ID, TARGET, sha256_file, load_development


def verify(data_dir: Path, run: Path, lr_root: Path, aggregates: Path) -> dict:
    receipt = json.loads((aggregates / 'verification.json').read_text())
    if receipt['status'] != 'PASS' or receipt['holdout_opened']:
        raise ValueError('A completed development-only audit is required.')
    for name, digest in receipt['aggregate_sha256'].items():
        if sha256_file(aggregates / name) != digest:
            raise ValueError(f'Aggregate checksum changed: {name}')
    for name, digest in receipt['data_contract']['files'].items():
        if sha256_file(data_dir / name) != digest:
            raise ValueError(f'Data checksum changed: {name}')
    X, y, ids, cv = load_development(data_dir)
    frames = {}
    for source in receipt['sources']:
        name = source['experiment']
        p = lr_root / name[-1] / 'lr' if name.startswith('LR_') else run / name
        if sha256_file(p / 'oof_predictions.csv.gz') != source['oof_sha256']:
            raise ValueError(f'OOF changed: {name}')
        q = pd.read_csv(p / 'oof_predictions.csv.gz').set_index(ID).loc[ids].reset_index()
        np.testing.assert_array_equal(q.y_true, y)
        np.testing.assert_array_equal(q.fold, cv.test_fold)
        frames[name] = q
    group = pd.concat([pd.read_csv(aggregates / 'group_summary.csv', dtype={'category':str}),
                       pd.read_csv(aggregates / 'error_profiles.csv', dtype={'category':str})], ignore_index=True)
    age_group = pd.cut(X.age, [-1,0,10,20,30,50,100,994,995],
                      labels=['0','1-10','11-20','21-30','31-50','51-100','101-994','995']).astype(str)
    def mask_for(dimension, category):
        if dimension == 'group':
            if category == 'all': return np.ones(len(X), dtype=bool)
            if category == 'age995': return X.age.eq(995).to_numpy()
            if category == 'non995': return X.age.ne(995).to_numpy()
            raise ValueError('Unknown group.')
        s = age_group if dimension == 'age_group' else X[dimension].astype(str)
        return s.eq(str(category)).to_numpy()
    group_reference = {}
    matrix_reference = {}
    for row in group.to_dict('records'):
        q = frames[row['experiment']].loc[mask_for(row['dimension'], row['category'])]
        yy, pp = q.y_true, q.y_pred
        cm = confusion_matrix(yy, pp, labels=[1,2,3])
        report = classification_report(yy, pp, labels=[1,2,3], output_dict=True, zero_division=0)
        actual = {'n':len(q),'macro_f1':f1_score(yy,pp,average='macro',labels=[1,2,3],zero_division=0),
                  'accuracy':accuracy_score(yy,pp),'grade3_count':int(cm[2].sum()),
                  'error_3_to_1':int(cm[2,0]),'error_3_to_2':int(cm[2,1])}
        for c in range(1,4):
            for key, field in [('precision','precision'),('recall','recall'),('f1','f1-score'),('support','support')]:
                actual[f'grade{c}_{key}'] = report[str(c)][field]
            actual[f'grade{c}_predicted'] = int(cm[:,c-1].sum())
        key3 = (row['experiment'], row['dimension'], row['category'])
        if key3 in group_reference: raise ValueError('Duplicate group.')
        group_reference[key3] = actual
        matrix_reference[key3] = cm
        for key, value in actual.items():
            np.testing.assert_allclose(row[key],value,rtol=0,atol=1e-12,err_msg=f'{row["experiment"]}/{row["dimension"]}/{row["category"]}/{key}')
        for c in (1,2):
            expected = cm[2,c-1] / cm[2].sum() if cm[2].sum() else np.nan
            np.testing.assert_allclose(row[f'rate_3_to_{c}'],expected,rtol=0,atol=1e-12,equal_nan=True)
    pairs = pd.read_csv(aggregates / 'paired_transitions.csv', dtype={'category':str})
    for row in pairs.to_dict('records'):
        mask = mask_for(row['dimension'],row['category'])
        b,a = frames[row['before']].loc[mask],frames[row['after']].loc[mask]
        # 3-way contingency table, not audit.py's Boolean-expression helper.
        cube=np.zeros((3,3,3),dtype=np.int64)
        np.add.at(cube,(b.y_true.to_numpy()-1,b.y_pred.to_numpy()-1,a.y_pred.to_numpy()-1),1)
        counts={'severe_fixed':int(cube[2,0,1:].sum()),'severe_new':int(cube[2,1:,0].sum()),
                'severe_retained':int(cube[2,0,0]),'changed':int(cube.sum()-sum(cube[:,i,i].sum() for i in range(3)))}
        for key,value in counts.items():
            if row[key]!=value: raise ValueError(f'Pair transition mismatch: {key}')
    folds = pd.read_csv(aggregates / 'fold_summary.csv')
    for row in folds.to_dict('records'):
        q = frames[row['experiment']].loc[cv.test_fold == row['fold']]
        score = f1_score(q.y_true,q.y_pred,average='macro',labels=[1,2,3],zero_division=0)
        np.testing.assert_allclose(row['macro_f1'],score,rtol=0,atol=1e-12)
    summaries = pd.read_csv(aggregates / 'model_summary.csv')
    if set(summaries.experiment) != set(frames) or summaries.experiment.duplicated().any():
        raise ValueError('Model summary coverage changed.')
    for row in summaries.to_dict('records'):
        ref = group_reference[(row['experiment'],'group','all')]
        for key,value in ref.items():
            np.testing.assert_allclose(row[key],value,rtol=0,atol=1e-12)
        scores=folds.loc[folds.experiment.eq(row['experiment']),'macro_f1'].to_numpy()
        np.testing.assert_allclose([row['cv_mean'],row['cv_std_ddof1'],row['pooled_f1']],
            [scores.mean(),scores.std(ddof=1),ref['macro_f1']],rtol=0,atol=1e-12)
    seen=set()
    for line in (aggregates/'confusion_matrices.jsonl').read_text().splitlines():
        row=json.loads(line); key=(row['experiment'],row['dimension'],row['category'])
        if key in seen: raise ValueError('Duplicate confusion matrix.')
        seen.add(key); cm=matrix_reference[key]
        np.testing.assert_array_equal(row['matrix'],cm)
        support=cm.sum(axis=1)
        normalized=np.divide(cm,support[:,None],out=np.full((3,3),np.nan),where=support[:,None]!=0)
        np.testing.assert_allclose(np.asarray(row['matrix_row_normalized'],dtype=float),normalized,rtol=0,atol=1e-12,equal_nan=True)
    if seen != set(matrix_reference): raise ValueError('Confusion matrix coverage changed.')
    return {'status':'PASS','group_profile_rows_checked':len(group),'pair_transition_rows_checked':len(pairs),
            'fold_rows_checked':len(folds),'model_rows_checked':len(summaries),'confusion_matrices_checked':len(seen),'oof_rows_checked':sum(len(q) for q in frames.values()),
            'holdout_opened':False,'new_predictive_fits':0,'method':'sklearn metrics and independent 3-way contingency cubes',
            'verification_input_sha256':sha256_file(aggregates/'verification.json'),
            'verifier_sha256':sha256_file(__file__)}


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for arg in ['data-dir','run','lr-root','aggregates']: ap.add_argument('--'+arg,type=Path,required=True)
    ap.add_argument('--receipt',type=Path,required=True)
    args=ap.parse_args()
    if args.receipt.exists(): raise FileExistsError('Use a new receipt path.')
    result=verify(args.data_dir,args.run,args.lr_root,args.aggregates)
    args.receipt.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))
