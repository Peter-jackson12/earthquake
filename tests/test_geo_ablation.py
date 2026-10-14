"""Synthetic contracts only: no real development data, no holdout access."""
import copy
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import clone
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))
from test_modeling import synthetic
from preprocess import FEATURES, CATEGORICAL
from experiments.age995.run import make_pipeline as original
from experiments.geo_ablation.run import make_pipeline, config
from experiments.geo_ablation.audit import inspect,turnover,arithmetic,check_oof,PROBS

class FeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        X,y,_,_=synthetic(180)
        X.loc[[0,17,155],'age']=995
        cls.train=X.iloc[:150].copy();cls.valid=X.iloc[150:].copy();cls.y=y.iloc[:150]
        cls.valid['geo_level_1_id']='90001';cls.valid['geo_level_2_id']='90002';cls.valid['geo_level_3_id']='90003'
        with threadpool_limits(limits=1):
            cls.reference=original('C')['preprocess'].fit(cls.train)
            cls.pipes={v:make_pipeline(v).fit(cls.train,cls.y) for v in ['R12','R1','R0']}
    def test_R12_full_parity(self):
        with threadpool_limits(limits=1):r=inspect(self.pipes['R12'],'R12',self.train,self.valid,self.reference)
        self.assertTrue(r['full_train_valid_C_projection_equal'])
    def test_R1_full_parity(self):
        with threadpool_limits(limits=1):r=inspect(self.pipes['R1'],'R1',self.train,self.valid,self.reference)
        self.assertTrue(r['full_train_valid_C_projection_equal'])
    def test_R0_full_parity(self):
        with threadpool_limits(limits=1):r=inspect(self.pipes['R0'],'R0',self.train,self.valid,self.reference)
        self.assertTrue(r['full_train_valid_C_projection_equal'])
    def test_fresh_C_not_mutated(self):
        self.assertEqual(original('C')['preprocess']['columns'].transformers[2][2],CATEGORICAL)
        self.assertEqual(original('C')['preprocess']['age995'].mode,'C')
    def test_unknown_variant_rejected(self):
        with self.assertRaises(ValueError):make_pipeline('G2')
    def test_C_training_rejected(self):
        with self.assertRaises(ValueError):make_pipeline('C')
    def test_cloneable(self):
        self.assertEqual(clone(self.pipes['R0'])['model'].get_params(),original('C')['model'].get_params())
    def test_raw_38_preserved(self):
        for p in self.pipes.values():self.assertEqual(p['preprocess']['schema'].feature_names_in_.tolist(),FEATURES)
    def test_missing_excluded_input_still_rejected(self):
        with self.assertRaises(ValueError):self.pipes['R0'].predict(self.valid.drop(columns='geo_level_3_id'))
    def test_ID_predictor_rejected(self):
        x=self.valid.assign(building_id=np.arange(len(self.valid)))
        with self.assertRaises(ValueError):self.pipes['R0'].predict(x)
    def test_target_predictor_rejected(self):
        with self.assertRaises(ValueError):self.pipes['R0'].predict(self.valid.assign(damage_grade=3))
    def test_fold_predictor_rejected(self):
        with self.assertRaises(ValueError):self.pipes['R0'].predict(self.valid.assign(fold=0))
    def test_invalid_removed_geo_still_schema_error(self):
        x=self.valid.copy();x['geo_level_3_id']='-1'
        with self.assertRaises(ValueError):self.pipes['R0'].predict(x)
    def test_all995_training_rejected(self):
        x=self.train.copy();x['age']=995
        with self.assertRaises(ValueError):make_pipeline('R0')['preprocess'].fit(x)
    def test_geography_injection_audit_rejected(self):
        with self.assertRaises(ValueError):inspect(self.pipes['R12'],'R0',self.train,self.valid,self.reference)
    def test_weight_mutation_rejected(self):
        p=copy.deepcopy(self.pipes['R0']);p['model'].class_weight='balanced'
        with self.assertRaises(ValueError):inspect(p,'R0',self.train,self.valid,self.reference)
    def test_mean_contamination_rejected(self):
        p=copy.deepcopy(self.pipes['R0']);p['preprocess']['columns'].named_transformers_['numeric']['scale'].mean_[0]+=1
        with self.assertRaises(AssertionError):inspect(p,'R0',self.train,self.valid,self.reference)
    def test_median_contamination_rejected(self):
        p=copy.deepcopy(self.pipes['R0']);p['preprocess']['age995'].median_+=1
        with self.assertRaises(ValueError):inspect(p,'R0',self.train,self.valid,self.reference)
    def test_remainder_injection_rejected(self):
        p=copy.deepcopy(self.pipes['R0']);p['preprocess']['columns'].remainder='passthrough'
        with self.assertRaises(ValueError):inspect(p,'R0',self.train,self.valid,self.reference)
    def test_unknown_policy_mutation_rejected(self):
        p=copy.deepcopy(self.pipes['R0']);p['preprocess']['columns'].named_transformers_['categorical'].handle_unknown='error'
        with self.assertRaises(ValueError):inspect(p,'R0',self.train,self.valid,self.reference)
    def test_indicator_keeps_raw995_meaning(self):
        p=self.pipes['R0']['preprocess'];x=p['age995'].transform(p['schema'].transform(self.valid))
        np.testing.assert_array_equal(x.age_is_995,self.valid.age.eq(995))
    def test_fixed_budget_and_parameters(self):
        self.assertEqual(config()['max_new_classifier_fits'],45)
        for p in self.pipes.values():self.assertEqual(p['model'].get_params(),original('C')['model'].get_params())

class MetricTests(unittest.TestCase):
    def setUp(self):
        self.y=np.array([1,2,3,3,3,3]);self.c=np.array([1,2,1,1,2,3]);self.r=np.array([2,2,2,3,3,1])
        self.ids=pd.Series(np.arange(6));self.f=np.array([0,1,2,3,4,0])
        p=np.full((6,3),.1);p[np.arange(6),self.r-1]=.8
        self.oof=pd.DataFrame({'building_id':self.ids,'fold':self.f,'y_true':self.y,'y_pred':self.r,
            **{name:p[:,i] for i,name in enumerate(PROBS)}})
    def test_denominator_is_actual_grade3(self):
        a=arithmetic(self.y,self.r);self.assertEqual(a['rate31'],.25);self.assertEqual(a['rate32'],.25)
    def test_transition_distinguishes_31_to_32_from_correct(self):
        t=turnover(self.y,self.c,self.r)
        self.assertEqual(t['error31_removed'],2);self.assertEqual(t['transition_true_C_R'][2][0][2],1)
        self.assertEqual(t['transition_true_C_R'][2][0][1],1)
    def test_turnover_conservation(self):
        t=turnover(self.y,self.c,self.r)
        self.assertEqual(t['error31_new']-t['error31_removed'],t['error31_delta'])
        self.assertEqual(t['corrected_predictions']-t['new_errors'],int((self.r==self.y).sum()-(self.c==self.y).sum()))
        self.assertEqual(np.array(t['transition_true_C_R']).sum(),6)
    def test_no_grade3_rate_is_null(self):self.assertIsNone(arithmetic([1,2],[1,2])['rate31'])
    def test_empty_group_rejected(self):
        with self.assertRaises(ValueError):arithmetic([],[])
    def test_shuffled_OOF_sorted_back(self):
        result=check_oof(self.oof.iloc[::-1],self.ids,self.y,self.f)
        np.testing.assert_array_equal(result.y_pred,self.r)
    def reject(self,oof):
        with self.assertRaises((AssertionError,ValueError)):check_oof(oof,self.ids,self.y,self.f)
    def test_duplicate_id_rejected(self):
        o=self.oof.copy();o.loc[0,'building_id']=1;self.reject(o)
    def test_changed_label_rejected(self):
        o=self.oof.copy();o.loc[0,'y_true']=3;self.reject(o)
    def test_changed_fold_rejected(self):
        o=self.oof.copy();o.loc[0,'fold']=1;self.reject(o)
    def test_probability_nan_rejected(self):
        o=self.oof.copy();o.loc[0,'p_grade_1']=np.nan;self.reject(o)
    def test_probability_sum_rejected(self):
        o=self.oof.copy();o.loc[0,'p_grade_1']=.9;self.reject(o)
    def test_prediction_argmax_rejected(self):
        o=self.oof.copy();o.loc[0,'y_pred']=1;self.reject(o)
    def test_removed_row_rejected(self):self.reject(self.oof.iloc[1:])

if __name__=='__main__':unittest.main()
