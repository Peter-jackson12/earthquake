"""Synthetic contract tests; no real-data performance is inferred here."""
import copy
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))
from test_modeling import synthetic
from preprocess import FEATURES, ID, normalize_features
from experiments.age995.run import make_pipeline as C
from experiments.age995.interactions.run import config, Terms, raw_terms, make_pipeline, fitted_checks, fit_one, run


class InteractionTests(unittest.TestCase):
    def setUp(self):
        self.X,self.y,self.ids,self.folds=synthetic(150)
        self.X.loc[[0,7,121],'age']=995

    def test_five_prespecified_variants(self):
        self.assertEqual(list(config()['variants']),['G1','G2','G3','G4','G5'])
        self.assertEqual([len(v['extra']) for v in config()['variants'].values()],[1,2,2,2,4])

    def test_boundaries_and_original995(self):
        X=self.X.iloc[:12].copy();X['age']=[0,1,10,11,20,21,30,50,100,101,994,995]
        self.assertEqual(raw_terms(X).age_band.tolist(),['0','1-10','1-10','11-20','11-20','21-30','21-30','31-50','51-100','101-994','101-994','995'])

    def test_label_blind_terms(self):
        a=Terms('G5').fit_transform(self.X,self.y)
        b=Terms('G5').fit_transform(self.X,self.y[::-1])
        pd.testing.assert_frame_equal(a,b)

    def test_no_input_mutation(self):
        before=self.X.copy(deep=True);Terms('G5').fit_transform(self.X)
        pd.testing.assert_frame_equal(before,self.X)

    def test_unfitted_refused(self):
        with self.assertRaises(NotFittedError):Terms().transform(self.X)

    def test_unknown_variant_refused(self):
        for action in [lambda:Terms('bad').fit(self.X),lambda:make_pipeline('bad')]:
            with self.assertRaises(ValueError):action()

    def test_schema_reordering_rejected_by_terms(self):
        p=Terms().fit(self.X)
        with self.assertRaises(ValueError):p.transform(self.X.iloc[:,::-1])

    def test_age995_category_not_replaced_by_median(self):
        out=Terms('G5').fit_transform(self.X)
        self.assertTrue(out.loc[self.X.age.eq(995),'geo_age'].str.endswith('|995').all())

    def test_no_special_region_rules(self):
        X=self.X.iloc[:4].copy();X['geo_level_1_id']=['26','20','13','987']
        X['age']=10;X['foundation_type']='r'
        self.assertEqual(raw_terms(X).geo_foundation.tolist(),['26|r','20|r','13|r','987|r'])

    def test_exact_parameters_and_clone(self):
        for v in config()['variants']:
            p=make_pipeline(v)
            self.assertEqual(p['model'].get_params(),C('C')['model'].get_params())
            self.assertFalse(hasattr(clone(p)['model'],'classes_'))

    def test_id_and_label_never_inputs(self):
        for column in [ID,'damage_grade']:
            X=self.X.copy();X[column]=1
            with self.assertRaises(ValueError):make_pipeline('G1').fit(X,self.y)

    def test_fractional_age_and_bad_binary_rejected(self):
        for col,value in [('age',1.5),('has_secondary_use',3)]:
            X=self.X.copy();X[col]=X[col].astype(float);X.loc[0,col]=value
            with self.assertRaises(ValueError):make_pipeline('G1').fit(X,self.y)

    def test_no_out_of_range_age_binning(self):
        X=self.X.copy();X.loc[0,'age']=996
        with self.assertRaises(ValueError):make_pipeline('G1').fit(X,self.y)

    def test_base_block_exact_on_all_synthetic_rows(self):
        for v in config()['variants']:
            p=make_pipeline(v)['preprocess'].fit(self.X.iloc[:120],self.y.iloc[:120])
            reference=C('C')['preprocess'].fit(self.X.iloc[:120])
            a=reference.transform(self.X);b=p.transform(self.X)[:,:a.shape[1]]
            self.assertEqual((a!=b).nnz,0)

    def test_sparse_float32_and_row_preservation(self):
        for v in config()['variants']:
            m=make_pipeline(v)['preprocess'].fit_transform(self.X,self.y)
            self.assertTrue(sparse.isspmatrix_csr(m));self.assertEqual(m.dtype,np.float32)
            self.assertEqual(m.shape[0],150)

    def test_train_only_extra_vocabulary(self):
        train=self.X.iloc[:120];valid=self.X.iloc[120:].copy();valid['geo_level_1_id']='999999'
        pre=make_pipeline('G5')['preprocess'].fit(train)
        extra=dict(pre.transformer_list)['extra'];before=copy.deepcopy(extra['encode'].categories_)
        extra.transform(valid)
        for old,new in zip(before,extra['encode'].categories_):np.testing.assert_array_equal(old,new)
        self.assertFalse(any('999999' in str(x) for cat in before for x in cat))

    def test_unseen_combo_zero_keeps_main_features(self):
        train=self.X.iloc[:120].copy();train['geo_level_1_id']='1';train['foundation_type']='r'
        valid=train.iloc[:2].copy();valid['geo_level_1_id']='2'
        pre=make_pipeline('G2')['preprocess'].fit(train)
        extra=dict(pre.transformer_list)['extra'];m=extra.transform(valid)
        n=len(extra['encode'].categories_[0]);self.assertEqual(m[:,n:].nnz,0)
        self.assertGreater(pre.transform(valid).nnz,0)

    def test_fitted_checks_and_reload(self):
        with threadpool_limits(limits=1),tempfile.TemporaryDirectory() as t:
            p=make_pipeline('G5').fit(self.X.iloc[:120],self.y.iloc[:120])
            check=fitted_checks(p,self.X.iloc[:120],self.X.iloc[120:])
            self.assertTrue(check['base_C_block_exact']);self.assertEqual(check['valid_age995'],1)
            path=Path(t)/'m.joblib';joblib.dump(p,path);q=joblib.load(path)
            np.testing.assert_array_equal(p.predict(self.X),q.predict(self.X))

    def test_duplicate_variants_or_fake_commit_rejected(self):
        for v,c in [(['G1','G1'],'a'*40),(['x'],'a'*40),(['G1'],'fake')]:
            with self.assertRaises(ValueError):run('missing','missing','missing',c,v)

    def test_existing_output_refused_before_reading(self):
        with tempfile.TemporaryDirectory() as t:
            (Path(t)/'G1').mkdir()
            with self.assertRaises(FileExistsError):run('missing','missing',t,'a'*40,['G1'])

    def test_end_to_end_five_synthetic_variants(self):
        frame=self.X.copy();frame[ID]=self.ids;frame['damage_grade']=self.y;frame['fold']=self.folds
        before=frame.copy(deep=True)
        with tempfile.TemporaryDirectory() as t,contextlib.redirect_stdout(io.StringIO()):
            for v in config()['variants']:
                target=Path(t)/v
                r=fit_one(v,frame,{'n_folds':5},target,'a'*40)
                self.assertTrue(r['all_converged'])
                rows=pd.read_csv(target/'oof.csv.gz')
                self.assertEqual(len(rows),150);self.assertTrue(rows[ID].is_unique)
                self.assertEqual(set(rows.fold),set(range(5)))
                with self.assertRaises(FileExistsError):fit_one(v,frame,{},target,'a'*40)
        pd.testing.assert_frame_equal(before,frame)


if __name__=='__main__':unittest.main()
