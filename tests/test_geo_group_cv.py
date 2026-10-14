"""T5 synthetic contracts; not evidence of real-data predictive performance."""
import copy
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))
from test_modeling import synthetic
from experiments.geo_group_cv.run import cfg,groups,validate_assignment,score,preprocessing_evidence,make_C,make_G2

class SplitTests(unittest.TestCase):
    def setUp(self):
        self.X,self.y,self.ids,_=synthetic(900)
        self.X['geo_level_1_id']=np.repeat(np.arange(30),30).astype(str)
        self.X['geo_level_2_id']=np.repeat(np.arange(90),10).astype(str)
        self.y=pd.Series(np.tile([1,2,3],300))
        self.cols=['geo_level_1_id']
        g,_=groups(self.X,self.cols)
        self.folds=np.full(900,-1)
        for f,(_,b) in enumerate(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(self.X,self.y,g)):
            self.folds[b]=f
    def test_valid_partition(self):
        result=validate_assignment(self.X,self.y,self.ids,self.folds,self.cols,cfg())
        self.assertEqual(result['status'],'PASS')
        self.assertEqual(sum(x['valid_rows'] for x in result['folds']),900)
    def test_composite_keys_keep_repeated_local_codes_separate(self):
        x=pd.DataFrame({'geo_level_1_id':['1','2','1'],'geo_level_2_id':['7','7','8']})
        g,k=groups(x,list(x))
        self.assertEqual(len(k),3);self.assertNotEqual(g[0],g[1])
    def test_numeric_tuple_sort(self):
        x=pd.DataFrame({'a':['10','2','1'],'b':['1','1','1']})
        _,k=groups(x,list(x));self.assertEqual(k[:,0].tolist(),[1,2,10])
    def test_group_overlap_rejected(self):
        f=self.folds.copy();f[0]=(f[0]+1)%5
        with self.assertRaises(ValueError):validate_assignment(self.X,self.y,self.ids,f,self.cols,cfg())
    def test_duplicate_id_rejected(self):
        ids=self.ids.copy();ids.iloc[0]=ids.iloc[1]
        with self.assertRaises(ValueError):validate_assignment(self.X,self.y,ids,self.folds,self.cols,cfg())
    def test_missing_fold_rejected(self):
        f=self.folds.copy();f[f==4]=3
        with self.assertRaises(ValueError):validate_assignment(self.X,self.y,self.ids,f,self.cols,cfg())
    def test_fractional_fold_rejected(self):
        f=self.folds.astype(float);f[0]=0.5
        with self.assertRaises(ValueError):validate_assignment(self.X,self.y,self.ids,f,self.cols,cfg())
    def test_missing_class_rejected(self):
        y=self.y.copy();y[y==1]=2
        with self.assertRaises(ValueError):validate_assignment(self.X,y,self.ids,self.folds,self.cols,cfg())
    def test_support_threshold_rejected(self):
        c=cfg();c['min_class_support']=1000
        with self.assertRaises(ValueError):validate_assignment(self.X,self.y,self.ids,self.folds,self.cols,c)
    def test_validation_fraction_rejected(self):
        c=cfg();c['max_validation_fraction']=0.15
        with self.assertRaises(ValueError):validate_assignment(self.X,self.y,self.ids,self.folds,self.cols,c)
    def test_fixed_models(self):
        self.assertEqual(make_C('C')['model'].get_params(),make_G2('G2')['model'].get_params())
    def test_known_metrics(self):
        s=score([1,1,2,2,3,3],[1,2,2,2,1,3]);self.assertEqual(s['error31'],1);self.assertEqual(s['rate31'],0.5)
        self.assertEqual(s['support'],[2,2,2]);self.assertEqual(s['confusion'],[[1,1,0],[0,2,0],[1,0,1]])
    def test_absent_class_in_diagnostic_has_null_rate(self):
        self.assertIsNone(score([1,2],[1,2])['rate31'])
    def test_unseen_geography_preprocessing_and_no_mutation(self):
        X,y,_,_=synthetic(150);X.loc[0,'age']=995
        train,valid=X.iloc[:120].copy(),X.iloc[120:].copy()
        valid['geo_level_1_id']='99999';valid['geo_level_2_id']='99998';valid['geo_level_3_id']='99997'
        before=valid.copy(deep=True)
        for v,factory in [('C',lambda:make_C('C')),('G2',lambda:make_G2('G2'))]:
            with threadpool_limits(limits=1):
                p=factory().fit(train,y.iloc[:120]);r=preprocessing_evidence(p,v,train,valid)
            for item in r['categorical'][:3]:self.assertEqual(item['unknown_rows'],30)
            self.assertTrue(r['preprocessing_before_after_equal'])
            if v=='G2':self.assertEqual(r['extra'][1]['unknown_rows'],30)
        pd.testing.assert_frame_equal(before,valid)
    def test_no_ID_predictor(self):
        X,y,_,_=synthetic(60);X['building_id']=np.arange(60)
        with self.assertRaises(ValueError):make_C('C').fit(X,y)


class AuditTests(unittest.TestCase):
    def setUp(self):
        from experiments.geo_group_cv.audit import check_oof,arithmetic,equal
        self.check,self.calc,self.equal=check_oof,arithmetic,equal
        self.ids=pd.Series([1,2,3,4,5,6])
        self.y=pd.Series([1,1,2,2,3,3])
        self.folds=np.array([0,1,2,3,4,0])
        prediction=np.array([1,2,2,2,1,3])
        probabilities=np.full((6,3),.1)
        probabilities[np.arange(6),prediction-1]=.8
        self.frame=pd.DataFrame({'building_id':self.ids,'fold':self.folds,'y_true':self.y,'y_pred':prediction,
            **{c:probabilities[:,i] for i,c in enumerate(['p_grade_1','p_grade_2','p_grade_3'])}})
    def reject(self,frame):
        with self.assertRaises((ValueError,AssertionError)):
            self.check(frame,self.ids,self.y,self.folds)
    def test_oof_correct_and_row_order_independent(self):
        result=self.check(self.frame.iloc[::-1],self.ids,self.y,self.folds)
        self.assertEqual(result.building_id.tolist(),self.ids.tolist())
    def test_duplicate_id_corruption(self):
        frame=self.frame.copy();frame.loc[0,'building_id']=2;self.reject(frame)
    def test_answer_corruption(self):
        frame=self.frame.copy();frame.loc[0,'y_true']=3;self.reject(frame)
    def test_fold_corruption(self):
        frame=self.frame.copy();frame.loc[0,'fold']=4;self.reject(frame)
    def test_probability_nan(self):
        frame=self.frame.copy();frame.loc[0,'p_grade_1']=np.nan;self.reject(frame)
    def test_probability_negative(self):
        frame=self.frame.copy();frame.loc[0,'p_grade_1']=-.1;self.reject(frame)
    def test_probability_sum(self):
        frame=self.frame.copy();frame.loc[0,'p_grade_1']=.6;self.reject(frame)
    def test_probability_argmax(self):
        frame=self.frame.copy();frame.loc[0,['p_grade_1','p_grade_3']]=[.1,.8];self.reject(frame)
    def test_independent_confusion_and_f1(self):
        computed=self.calc(self.frame.y_true,self.frame.y_pred)
        expected=score(self.frame.y_true,self.frame.y_pred)
        self.equal(computed,expected)
    def test_rehashed_metrics_are_not_sufficient(self):
        expected=self.calc(self.frame.y_true,self.frame.y_pred)
        damaged=copy.deepcopy(expected);damaged['error31']=0
        with self.assertRaises(ValueError):self.equal(damaged,expected)
    def test_absent_grade3_is_not_zero_rate(self):
        computed=self.calc([1,2],[1,2]);self.assertIsNone(computed['rate31'])
    def test_age_bands_independent_at_boundaries(self):
        from experiments.geo_group_cv.audit import independent_terms
        from experiments.age995.interactions.run import raw_terms
        X,_,_,_=synthetic(16)
        X['age']=[0,1,10,11,20,21,30,31,50,51,100,101,200,994,995,15]
        pd.testing.assert_frame_equal(independent_terms(X),raw_terms(X)[['age_band','geo_foundation']])
    def test_training_statistics_and_tampered_median(self):
        from experiments.geo_group_cv.audit import inspect_preprocessor
        X,y,_,_=synthetic(150);X.loc[0,'age']=995
        with threadpool_limits(limits=1):
            p=make_C('C').fit(X.iloc[:120],y.iloc[:120])
            info,_=inspect_preprocessor(p,'C',X.iloc[:120],X.iloc[120:])
        self.assertEqual(info['n_samples_seen'],120)
        p['preprocess']['age995'].median_+=1
        with self.assertRaises(ValueError):inspect_preprocessor(p,'C',X.iloc[:120],X.iloc[120:])
    def test_training_statistics_tampered_scaler(self):
        from experiments.geo_group_cv.audit import inspect_preprocessor
        X,y,_,_=synthetic(150)
        with threadpool_limits(limits=1):p=make_C('C').fit(X.iloc[:120],y.iloc[:120])
        p['preprocess']['columns'].named_transformers_['numeric']['scale'].mean_[0]+=1
        with self.assertRaises(AssertionError):inspect_preprocessor(p,'C',X.iloc[:120],X.iloc[120:])

if __name__=='__main__':unittest.main()
