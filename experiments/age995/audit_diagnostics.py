"""Independent read-only verification of diagnostics against existing private OOF."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from preprocess import load_development,sha256_file


def audit(data_dir,private_root,report_dir):
    report_dir=Path(report_dir)
    receipt=json.loads((report_dir/'verification.json').read_text())
    if receipt['status']!='PASS' or receipt['new_model_fits']!=0:raise ValueError('Unexpected run status.')
    for rel,digest in receipt['source_sha256'].items():
        if sha256_file(ROOT/rel)!=digest:raise ValueError('Analysis source changed.')
    for name,digest in receipt['output_sha256'].items():
        if sha256_file(report_dir/name)!=digest:raise ValueError('Output checksum mismatch.')
    for name,digest in receipt['data_contract']['files'].items():
        if sha256_file(Path(data_dir)/name)!=digest:raise ValueError('Data checksum mismatch.')
    X,y,ids,cv=load_development(data_dir);frame=X.copy()
    frame['truth']=y;frame['fold']=cv.test_fold
    frame['age_band']=pd.cut(frame.age,[-1,0,10,20,30,50,100,994,995],
        labels=['0','1-10','11-20','21-30','31-50','51-100','101-994','995'],right=True).astype(str)
    oofs={}
    for variant,binding in receipt['bindings'].items():
        parent='age995_abc_v1' if variant in ['A','B','C'] else 'outputs/age995_followup_v1_20261013'
        path=Path(private_root)/parent/variant/'lr/oof_predictions.csv.gz'
        if sha256_file(path)!=binding['oof_sha256']:raise ValueError('OOF checksum mismatch.')
        raw=pd.read_csv(path)
        if raw.building_id.duplicated().any() or set(raw.building_id)!=set(ids):raise ValueError('OOF IDs invalid.')
        raw=raw.set_index('building_id').loc[ids.to_numpy()].reset_index()
        np.testing.assert_array_equal(raw.y_true,y);np.testing.assert_array_equal(raw.fold,cv.test_fold)
        oofs[variant]=raw
    def mask(group):
        if group=='all':return np.ones(len(frame),dtype=bool)
        if group=='age995':return frame.age.eq(995).to_numpy()
        if group=='non995':return frame.age.ne(995).to_numpy()
        raise ValueError('Unknown group.')
    def near(a,b):np.testing.assert_allclose(a,b,atol=2e-12,rtol=0,equal_nan=True)
    groups=pd.read_csv(report_dir/'group_diagnostics.csv')
    if len(groups)!=27 or groups.duplicated(['variant','group']).any():raise ValueError('Group completeness failed.')
    checked=0
    for row in groups.itertuples(index=False):
        use=mask(row.group);o=oofs[row.variant];truth=o.loc[use,'y_true'].to_numpy();pred=o.loc[use,'y_pred'].to_numpy()
        p=o.loc[use,['p_grade_1','p_grade_2','p_grade_3']].to_numpy();maxp=p.max(1)
        high=maxp>=.8;margin=np.partition(p,1,axis=1)[:,-1]-np.partition(p,1,axis=1)[:,-2]
        e31=(truth==3)&(pred==1);e32=(truth==3)&(pred==2);s3=(truth==3).sum()
        near([row.n,row.correct,row.errors,row.macro_f1,row.grade3_support,row.e31,row.e32],
             [len(truth),(truth==pred).sum(),(truth!=pred).sum(),f1_score(truth,pred,labels=[1,2,3],average='macro',zero_division=0),s3,e31.sum(),e32.sum()])
        near([row.e31_rate,row.e32_rate],[e31.sum()/s3 if s3 else np.nan,e32.sum()/s3 if s3 else np.nan])
        near([row.median_max_probability,row.high_probability_rows,row.high_probability_errors,
              row.high_probability_accuracy,row.high_probability_e31,row.low_margin_e31],
             [np.median(maxp),high.sum(),(high&(truth!=pred)).sum(),(truth[high]==pred[high]).mean() if high.any() else np.nan,
              (high&e31).sum(),((margin<=.1)&e31).sum()]);checked+=1
    pairs=pd.read_csv(report_dir/'paired_comparisons.csv');transitions=pd.read_csv(report_dir/'paired_transitions.csv')
    if len(pairs)!=54 or pairs.duplicated(['left','right','group','fold']).any():raise ValueError('Pair completeness failed.')
    if transitions.duplicated(['left','right','group','truth','left_prediction','right_prediction']).any():raise ValueError('Duplicate transitions.')
    transition_rows=0
    for row in pairs.itertuples(index=False):
        use=mask(row.group)
        if row.fold!='pooled':use=use & frame.fold.eq(int(row.fold)).to_numpy()
        l=oofs[row.left].loc[use,'y_pred'].to_numpy();r=oofs[row.right].loc[use,'y_pred'].to_numpy();y=frame.loc[use,'truth'].to_numpy()
        cube=np.zeros((3,3,3),dtype=int)
        for a in range(1,4):
            for b in range(1,4):
                for c in range(1,4):cube[a-1,b-1,c-1]=((y==a)&(l==b)&(r==c)).sum()
        both=sum(cube[i,i,i] for i in range(3));lc=sum(cube[i,i,:].sum()-cube[i,i,i] for i in range(3))
        rc=sum(cube[i,:,i].sum()-cube[i,i,i] for i in range(3))
        samewrong=sum(cube[i,j,j] for i in range(3) for j in range(3) if i!=j)
        diffwrong=int(cube.sum())-both-lc-rc-samewrong
        near([row.n,row.disagreements,row.both_correct,row.left_only_correct,row.right_only_correct,
              row.both_wrong_same,row.both_wrong_different,row.fixed_e31,row.new_e31,row.shared_e31],
             [len(y),lc+rc+diffwrong,both,lc,rc,samewrong,diffwrong,cube[2,0,1:].sum(),cube[2,1:,0].sum(),cube[2,0,0]])
        pcols=['p_grade_1','p_grade_2','p_grade_3'];diff=abs(oofs[row.left].loc[use,pcols].to_numpy()-oofs[row.right].loc[use,pcols].to_numpy())
        near([row.mean_abs_probability_difference,row.max_abs_probability_difference],[diff.mean(),diff.max()])
        if row.fold=='pooled':
            rows=transitions.query('left==@row.left and right==@row.right and group==@row.group')
            if len(rows)!=np.count_nonzero(cube):raise ValueError('Transition completeness failed.')
            for t in rows.itertuples(index=False):near(t.count,cube[t.truth-1,t.left_prediction-1,t.right_prediction-1])
            transition_rows+=len(rows)
    if transition_rows!=len(transitions):raise ValueError('Unexpected transition groups.')
    profiles=pd.read_csv(report_dir/'error_profiles.csv',dtype={'value':str,'fold':str})
    if profiles.duplicated(['variant','dimension','value','fold']).any():raise ValueError('Duplicate profile.')
    expected_profiles=0
    for v in ['A','B','C','E2']:
        for col in ['geo_level_1_id','foundation_type','roof_type','ground_floor_type','other_floor_type','age_band']:
            expected_profiles+=frame[col].nunique()
            if col=='geo_level_1_id':expected_profiles+=sum(frame.loc[frame.fold.eq(k),col].nunique() for k in range(5))
    if len(profiles)!=expected_profiles:raise ValueError('Missing/extra profile rows.')
    for row in profiles.itertuples(index=False):
        use=frame[row.dimension].astype(str).eq(row.value)
        if row.fold!='pooled':use &=frame.fold.eq(int(row.fold))
        truth=frame.loc[use,'truth'];pred=oofs[row.variant].loc[use,'y_pred'];e31=((truth==3)&(pred==1)).sum();e32=((truth==3)&(pred==2)).sum();s3=(truth==3).sum()
        near([row.n,row.correct,row.errors,row.macro_f1,row.grade3_support,row.e31,row.e32],
            [len(truth),(truth==pred).sum(),(truth!=pred).sum(),f1_score(truth,pred,labels=[1,2,3],average='macro',zero_division=0),s3,e31,e32])
        near([row.e31_rate,row.e32_rate],[e31/s3 if s3 else np.nan,e32/s3 if s3 else np.nan])
        total=((frame.truth==3)&(oofs[row.variant].y_pred==1)).sum()
        near(row.e31_share_all,e31/total if total else np.nan)
        if row.low_grade3_support != (s3<30):raise ValueError('Small-group flag invalid.')
    costs=pd.read_csv(report_dir/'cost_sensitivity.csv')
    if len(costs)!=162 or costs.duplicated(['variant','group','weight_3_to_1']).any():raise ValueError('Cost completeness failed.')
    for row in costs.itertuples(index=False):
        o=oofs[row.variant].loc[mask(row.group)];v=np.where(o.y_true==o.y_pred,0,1)
        v=np.where((o.y_true==3)&(o.y_pred==1),row.weight_3_to_1,v)
        near([row.n,row.total_cost],[len(o),v.sum()])
    return {'status':'PASS','groups_checked':checked,'paired_slices_checked':len(pairs),
        'transition_rows_checked':len(transitions),'profile_rows_checked':len(profiles),'cost_rows_checked':len(costs),
        'predictions_checked':sum(len(o) for o in oofs.values()),'model_fits':0,
        'scope':'Independent OOF aggregation via sklearn F1 and 3x3x3 transition counts; affine identity covered by unit tests and direct equations.',
        'auditor_sha256':sha256_file(__file__)}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-dir',type=Path,default=ROOT/'data')
    p.add_argument('--private-root',type=Path,required=True);p.add_argument('--report-dir',type=Path,required=True)
    a=p.parse_args();print(json.dumps(audit(a.data_dir,a.private_root,a.report_dir),ensure_ascii=False,indent=2))
