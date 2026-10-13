"""Tuning and synthetic-sample safety contracts, not predictive performance tests."""
import copy
import json
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_limited_tuning import CANDIDATES,validate_candidate
from scripts.pilot_smotenc import synthesize,inspect_synthetic,CATS
from preprocess import NUMERIC,BINARY,GEO,normalize_features
from test_modeling import synthetic

class TuningContractTests(unittest.TestCase):
    def test_exact_five_predeclared_candidates(self):
        self.assertEqual(len(CANDIDATES),5)
        for key in CANDIDATES:
            c=json.loads((ROOT/'configs/tuning'/f'{key}.json').read_text())
            self.assertIn(validate_candidate(key,c),['lr','rf'])
    def test_reused_baselines_are_not_retrained(self):
        self.assertEqual({v[1]['C'] for v in CANDIDATES.values() if v[0]=='lr'},{.1,10.})
        self.assertNotIn((.1,2),{(v[1]['max_features'],v[1]['min_samples_leaf']) for v in CANDIDATES.values() if v[0]=='rf'})
    def test_rf_weight_fixed(self):
        for key in CANDIDATES:
            if key.startswith('rf'):
                c=json.loads((ROOT/'configs/tuning'/f'{key}.json').read_text())
                self.assertEqual(c['models']['rf']['class_weight'],'balanced')
    def test_undeclared_changes_rejected(self):
        c=json.loads((ROOT/'configs/tuning/lr_c01.json').read_text())
        for change in [{'max_iter':3000},{'class_weight':'balanced'}]:
            bad=copy.deepcopy(c);bad['models']['lr'].update(change)
            with self.assertRaises(ValueError): validate_candidate('lr_c01',bad)
    def test_seed_sampling_and_identity_rejected(self):
        c=json.loads((ROOT/'configs/tuning/lr_c10.json').read_text())
        for key,value in [('seed',43),('name','other'),('sampling',{'lr':'random_under'})]:
            bad=copy.deepcopy(c);bad[key]=value
            with self.assertRaises(ValueError): validate_candidate('lr_c10',bad)

class SmotencSafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X,_,_,_=synthetic(150)
        cls.y=pd.Series([1]*20+[2]*90+[3]*40)
        cls.before=cls.X.copy(deep=True)
        cls.original,cls.out,cls.labels,cls.scaler=synthesize(cls.X,cls.y)
        cls.new=cls.out.iloc[150:].reset_index(drop=True)
    def test_original_data_not_mutated(self):
        pd.testing.assert_frame_equal(self.X,self.before)
        np.testing.assert_allclose(self.out.iloc[:150][NUMERIC].astype(float),self.X[NUMERIC].astype(float),atol=1e-8)
    def test_counts_and_categories(self):
        self.assertEqual(pd.Series(self.labels).value_counts().tolist(),[90,90,90])
        self.assertEqual(len(self.new),120)
        for col in CATS:
            self.assertTrue(self.out[col].astype(str).isin(self.X[col].astype(str).unique()).all())
    def test_float_synthesis_not_silently_rounded(self):
        report=inspect_synthetic(self.original,self.new,self.original)
        self.assertGreater(report['counts']['fractional_numeric_rows'],0)
        self.assertFalse(report['full_experiment_allowed'])
        with self.assertRaises(ValueError): normalize_features(self.new)
    def test_scaler_uses_only_provided_training_rows(self):
        np.testing.assert_allclose(self.scaler.mean_,self.X[NUMERIC].mean())
    def test_too_few_minority_rows_rejected(self):
        y=pd.Series([1]*5+[2]*100+[3]*45)
        with self.assertRaises(ValueError): synthesize(self.X,y)
    def test_geographic_joint_conflict_detected(self):
        source=self.X.iloc[:2].copy()
        source.loc[:,GEO]=[['1','10','100'],['2','20','200']]
        new=source.iloc[:1].copy();new.loc[:,GEO]=[['2','10','100']]
        report=inspect_synthetic(source,new,source)
        self.assertEqual(report['counts']['unobserved_geo_triple_rows'],1)
        self.assertEqual(report['counts']['geo_level_2_id_parent_conflict_rows'],1)
        self.assertFalse(report['full_experiment_allowed'])
    def test_invalid_binary_is_blocking(self):
        new=self.out.iloc[:1].copy();new[BINARY[0]]='2'
        report=inspect_synthetic(self.original,new,self.original)
        self.assertEqual(report['counts']['invalid_binary_rows'],1)
        self.assertFalse(report['full_experiment_allowed'])
    def test_age_special_interpolation_is_flagged(self):
        source=self.X.iloc[:2].copy();source['age']=[10,995]
        new=source.iloc[:1].copy();new['age']=500.
        report=inspect_synthetic(source,new,source)
        self.assertEqual(report['counts']['age_between_regular_max_and_995_rows'],1)

if __name__=='__main__': unittest.main()
