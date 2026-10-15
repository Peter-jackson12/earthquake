"""Synthetic safety tests, not measured real-data performance."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))
from test_modeling import synthetic
from preprocess import BINARY, FEATURES, ID, NUMERIC
from experiments.age995.run import make_pipeline as abc_pipeline, PARAMS, FLAG
from experiments.age995.followup import (UpperCap, conditions, training_mask, make_pipeline,
    preprocessor_audit, selection_audit, fit_variant, run, config)


class FollowupTests(unittest.TestCase):
    def setUp(self):
        self.X,self.y,self.ids,self.folds=synthetic(150)
        self.X.loc[[0,7,121],'age']=995
        self.X.loc[1,'age']=0

    def test_frozen_spec(self):
        c=config()
        self.assertEqual(c['quantile'],.99)
        self.assertEqual(c['interpolation'],'linear')
        self.assertEqual(c['lr'],PARAMS)
        self.assertEqual(set(c['variants']),{'D','E1','E2','E3','F1','F2'})
        self.assertEqual(len({v['id'] for v in c['variants'].values()}),6)

    def test_transform_requires_fit(self):
        with self.assertRaises(NotFittedError):UpperCap().transform(self.X)

    def test_fractional_cap_is_internal_and_includes_zero(self):
        X=self.X.iloc[:4].copy();X['age']=[0,10,20,995]
        p=UpperCap(quantile=.75).fit(X)
        self.assertEqual(p.upper_,15.)
        self.assertEqual(p.transform(X).age.tolist(),[0.,10.,20.,15.])

    def test_cap_does_not_use_labels(self):
        self.assertEqual(UpperCap().fit(self.X,self.y).upper_,
                         UpperCap().fit(self.X,self.y.iloc[::-1]).upper_)

    def test_non995_ages_unchanged_in_targeted_cap(self):
        p=UpperCap().fit(self.X)
        a=p.transform(self.X)
        np.testing.assert_array_equal(a.loc[self.X.age.ne(995),'age'],self.X.loc[self.X.age.ne(995),'age'])

    def test_all_cap_changes_normal_old_buildings(self):
        X=self.X.iloc[:4].copy();X['age']=[0,10,20,995]
        p=UpperCap('all_above_cap',quantile=.5).fit(X)
        self.assertEqual(p.transform(X).age.tolist(),[0.,10.,10.,10.])

    def test_flag_is_from_original_age(self):
        p=UpperCap(flag=True).fit(self.X)
        np.testing.assert_array_equal(p.transform(self.X)[FLAG],self.X.age.eq(995).astype(int))

    def test_validation_cannot_fit_the_cap(self):
        p=UpperCap().fit(self.X.iloc[:100]);before=p.upper_
        valid=self.X.iloc[100:].copy();valid['age']=995
        self.assertTrue(p.transform(valid).age.eq(before).all())
        self.assertEqual(p.upper_,before)

    def test_all995_rejected(self):
        X=self.X.copy();X['age']=995
        with self.assertRaises(ValueError):UpperCap().fit(X)

    def test_empty_regular_subset_and_invalid_rules(self):
        for q in (0,-.1,1.1):
            with self.assertRaises(ValueError):UpperCap(quantile=q).fit(self.X)
        with self.assertRaises(ValueError):UpperCap(scope='bad').fit(self.X)

    def test_input_and_other_features_unchanged(self):
        before=self.X.copy(deep=True)
        out=UpperCap(flag=True).fit_transform(self.X)
        pd.testing.assert_frame_equal(out.drop(columns=['age',FLAG]),before.drop(columns='age'))
        pd.testing.assert_frame_equal(before,self.X)

    def test_no_double_flag_or_schema_change(self):
        X=self.X.copy();X[FLAG]=0
        with self.assertRaises(ValueError):UpperCap().fit(X)
        p=UpperCap().fit(self.X)
        with self.assertRaises(ValueError):p.transform(X)

    def test_original_integer_validation_not_relaxed(self):
        X=self.X.copy();X['age']=X.age.astype(float);X.loc[0,'age']=10.5
        for mode in ('E1','E2','E3'):
            with self.assertRaises(ValueError):make_pipeline(mode).fit(X,self.y)

    def test_E1_E2_E3_same_training_cap(self):
        caps=[]
        for mode in ('E1','E2','E3'):
            p=make_pipeline(mode)['preprocess'].fit(self.X)
            caps.append(p['cap'].upper_)
        self.assertEqual(len(set(caps)),1)

    def test_delete995_only_training_mask(self):
        before=self.X.copy(deep=True)
        keep=training_mask(self.X,'D')
        np.testing.assert_array_equal(keep,self.X.age.ne(995))
        pd.testing.assert_frame_equal(before,self.X)
        self.assertEqual(int((~keep).sum()),3)

    def test_strict_threshold_boundaries(self):
        X=self.X.iloc[:3].copy()
        for c,t in config()['faith_thresholds'].items():X[c]=[t,t+1,t]
        self.assertEqual(training_mask(X,'F1').tolist(),[True,False,True])
        self.assertEqual(training_mask(X,'F2').tolist(),[True,False,True])

    def test_faith_union_not_intersection(self):
        X=self.X.iloc[:5].copy()
        for c,t in config()['faith_thresholds'].items():X[c]=t
        for i,(c,t) in enumerate(config()['faith_thresholds'].items()):X.iloc[i,X.columns.get_loc(c)]=t+1
        self.assertEqual(training_mask(X,'F2').tolist(),[False,False,False,False,True])
        self.assertEqual(training_mask(X,'F1').tolist(),[False,True,True,True,True])

    def test_no_training_rows_rejected(self):
        X=self.X.copy();X['age']=995
        for mode in ('D','F1','F2'):
            with self.assertRaises(ValueError):training_mask(X,mode)

    def test_E_variants_remove_nothing(self):
        for mode in ('E1','E2','E3'):self.assertTrue(training_mask(self.X,mode).all())

    def test_lr_parameters_unchanged_and_cloneable(self):
        for mode in config()['variants']:
            p=make_pipeline(mode)
            self.assertEqual(p['model'].get_params(),abc_pipeline('A')['model'].get_params())
            self.assertFalse(hasattr(clone(p)['model'],'classes_'))

    def test_scaler_encoder_use_retained_training_only(self):
        for mode in config()['variants']:
            X=self.X.iloc[:120];valid=self.X.iloc[120:]
            kept=X.loc[training_mask(X,mode)]
            with threadpool_limits(limits=1):p=make_pipeline(mode).fit(kept,self.y.loc[kept.index])
            result=preprocessor_audit(p,kept,valid)
            self.assertTrue(result['train_only_statistics'])
            self.assertEqual(result['fit_rows'],len(kept))
            self.assertEqual(result['engineered_inputs'],39 if mode=='E2' else 38)
            self.assertEqual(result['valid_age995'],1)

    def test_new_category_is_not_learned(self):
        train=self.X.iloc[:120];valid=self.X.iloc[120:].copy();valid['geo_level_1_id']='999999'
        p=make_pipeline('D')['preprocess'].fit(train.loc[training_mask(train,'D')])
        before=[x.copy() for x in p['columns'].named_transformers_['categorical'].categories_]
        self.assertEqual(p.transform(valid).shape[0],len(valid))
        for a,b in zip(before,p['columns'].named_transformers_['categorical'].categories_):np.testing.assert_array_equal(a,b)

    def test_flag_is_unscaled_binary_and_sparse(self):
        p=make_pipeline('E2')['preprocess'].fit(self.X.iloc[:120]);a=p.transform(self.X.iloc[120:])
        self.assertTrue(sparse.isspmatrix_csr(a));self.assertEqual(a.dtype,np.float32)
        np.testing.assert_array_equal(a[:,len(NUMERIC)+len(BINARY)].toarray().ravel(),self.X.iloc[120:].age.eq(995).astype(int))

    def test_selection_records_overlap_and_classes(self):
        keep=training_mask(self.X,'F2')
        audit=selection_audit(self.X,self.y,keep,self.X.iloc[:20])
        self.assertEqual(sum(audit['removed_class_counts'].values()),audit['removed_rows'])
        self.assertEqual(audit['faith_union_count'],audit['removed_rows'])
        self.assertEqual(sum(r['removed'] for r in audit['regions']),audit['removed_rows'])
        self.assertEqual(len(audit['category_coverage']),11)

    def test_output_collision_refused_before_read(self):
        with tempfile.TemporaryDirectory() as t:
            (Path(t)/'D').mkdir()
            with self.assertRaises(FileExistsError):run('missing',t,['D'],'a'*40)

    def test_duplicate_variants_and_bad_commit_refused(self):
        for v,c in [(['D','D'],'a'*40),(['Z'],'a'*40),(['D'],'unknown')]:
            with self.assertRaises(ValueError):run('missing','unused',v,c)

    def test_all_synthetic_variants_keep_entire_validation(self):
        frame=self.X.copy();frame[ID]=self.ids;frame['damage_grade']=self.y;frame['fold']=self.folds
        original=frame.copy(deep=True)
        with tempfile.TemporaryDirectory() as t:
            for mode in config()['variants']:
                result=fit_variant(mode,frame,{'n_folds':5},Path(t)/mode,'a'*40)
                self.assertEqual(result['rows'],150)
                self.assertEqual(sum(r['validation_rows'] for r in result['fold_results']),150)
                for r in result['fold_results']:
                    self.assertEqual(r['original_train_rows'],120)
                    self.assertEqual(r['validation_rows'],30)
                    self.assertEqual(r['train_rows']+r['selection_audit']['removed_rows'],120)
                oof=pd.read_csv(Path(t)/mode/'lr/oof_predictions.csv.gz')
                self.assertEqual(len(oof),150);self.assertTrue(oof[ID].is_unique)
                self.assertEqual(int(oof[ID].isin(self.ids.loc[self.X.age.eq(995)]).sum()),3)
                with self.assertRaises(FileExistsError):fit_variant(mode,frame,{'n_folds':5},Path(t)/mode,'a'*40)
        pd.testing.assert_frame_equal(frame,original)


if __name__=='__main__':unittest.main()
