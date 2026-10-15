"""Synthetic contract tests; these are not full-data model performance results."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))
from test_modeling import synthetic
from preprocess import NUMERIC, FEATURES, normalize_features
from experiments.age995.run import Age995Treatment, make_pipeline, run, FLAG, preprocessor_check


class Age995Tests(unittest.TestCase):
    def setUp(self):
        self.X,self.y,_,_=synthetic(150)
        self.X.loc[[0,7,121], 'age']=995
        self.X.loc[1,'age']=0

    def test_unknown_variant_rejected(self):
        with self.assertRaises(ValueError):make_pipeline('bad')

    def test_transform_requires_fit(self):
        with self.assertRaises(NotFittedError):Age995Treatment('C').transform(self.X)

    def test_raw_schema_is_unchanged(self):
        X=self.X.copy();X['age']=X['age'].astype(float);X.loc[0,'age']=10.5
        with self.assertRaises(ValueError):make_pipeline('C').fit(X,self.y)

    def test_median_excludes995_includes_zero(self):
        X=self.X.iloc[:4].copy();X['age']=[0,10,995,30]
        p=Age995Treatment('C').fit(X)
        self.assertEqual(p.median_,10)
        self.assertEqual(p.transform(X)['age'].tolist(),[0,10,10,30])
        self.assertEqual(p.transform(X)[FLAG].tolist(),[0,0,1,0])

    def test_fractional_median_internal_only(self):
        X=self.X.iloc[:3].copy();X['age']=[10,15,995]
        p=Age995Treatment('C').fit(X)
        self.assertEqual(p.median_,12.5)
        self.assertEqual(p.transform(X).age.iloc[-1],12.5)
        np.testing.assert_array_equal(X.age,[10,15,995])

    def test_validation_does_not_change_statistics(self):
        p=Age995Treatment('C').fit(self.X.iloc[:120])
        expected=p.median_
        altered=self.X.iloc[120:].copy();altered['age']=995
        self.assertTrue(p.transform(altered).age.eq(expected).all())
        self.assertEqual(p.median_,expected)

    def test_flag_does_not_replace(self):
        X=Age995Treatment('B').fit_transform(self.X)
        np.testing.assert_array_equal(X.age,self.X.age)
        self.assertEqual(int(X[FLAG].sum()),3)

    def test_other_columns_and_input_preserved(self):
        before=self.X.copy(deep=True)
        for mode in ('B','C'):
            out=Age995Treatment(mode).fit_transform(self.X)
            pd.testing.assert_frame_equal(out.drop(columns=['age',FLAG]),before.drop(columns='age'))
            self.assertEqual(len(out),len(before))
        pd.testing.assert_frame_equal(self.X,before)

    def test_labels_not_used_for_median(self):
        a=Age995Treatment('C').fit(self.X,self.y)
        b=Age995Treatment('C').fit(self.X,self.y.iloc[::-1])
        self.assertEqual(a.median_,b.median_)

    def test_all995_rejected_only_for_median(self):
        X=self.X.copy();X['age']=995
        with self.assertRaises(ValueError):Age995Treatment('C').fit(X)
        self.assertTrue(Age995Treatment('B').fit_transform(X)[FLAG].eq(1).all())

    def test_indicator_is_unscaled_binary(self):
        p=make_pipeline('C')['preprocess'].fit(self.X.iloc[:120])
        array=p.transform(self.X.iloc[120:])
        self.assertTrue(sparse.isspmatrix_csr(array));self.assertEqual(array.dtype,np.float32)
        np.testing.assert_array_equal(array[:,27].toarray().ravel(),self.X.iloc[120:].age.eq(995).astype(int))

    def test_fold_statistics_all_modes(self):
        for mode in ('A','B','C'):
            p=make_pipeline(mode).fit(self.X.iloc[:120],self.y.iloc[:120])
            result=preprocessor_check(p,self.X.iloc[:120],self.X.iloc[120:],mode)
            self.assertTrue(result['train_only_statistics'])
            self.assertEqual(result['engineered_inputs'],38 if mode=='A' else 39)
            self.assertFalse(hasattr(clone(p)['model'],'classes_'))

    def test_no_double_engineering(self):
        X=self.X.copy();X[FLAG]=0
        with self.assertRaises(ValueError):Age995Treatment('C').fit(X)
        with self.assertRaises(ValueError):make_pipeline('C').fit(X,self.y)

    def test_novel_category_does_not_refit(self):
        p=make_pipeline('C')['preprocess'].fit(self.X.iloc[:120])
        vocab=[v.copy() for v in p['columns'].named_transformers_['categorical'].categories_]
        changed=self.X.iloc[120:].copy();changed['geo_level_1_id']='999999'
        p.transform(changed)
        for a,b in zip(vocab,p['columns'].named_transformers_['categorical'].categories_):
            np.testing.assert_array_equal(a,b)

    def test_unseen995_still_flagged_in_validation(self):
        train=self.X.loc[self.X.age.ne(995)]
        p=Age995Treatment('C').fit(train)
        new=self.X.iloc[:1].copy();new['age']=995
        self.assertEqual(p.transform(new)[FLAG].iloc[0],1)
        self.assertEqual(p.transform(new).age.iloc[0],train.age.median())

    def test_A_matches_unchanged_preprocessing(self):
        from preprocess import make_preprocessor
        a=make_pipeline('A')['preprocess'].fit_transform(self.X)
        b=make_preprocessor(scale_numeric=True).fit_transform(self.X)
        self.assertEqual((a-b).nnz,0)

    def test_nonempty_output_refused_before_data_read(self):
        with tempfile.TemporaryDirectory() as t:
            (Path(t)/'existing.txt').write_text('keep')
            with self.assertRaises(FileExistsError):run('not_available',t)


if __name__=='__main__':unittest.main()
