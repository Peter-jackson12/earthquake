"""Independent arithmetic and corrupted OOF rejection tests."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score,confusion_matrix
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))
from test_modeling import synthetic
from preprocess import ID,sha256_file
from experiments.age995.interactions.audit import counts,read_oof,independent_terms,paired,check_metrics
from experiments.age995.interactions.run import raw_terms


class AuditTests(unittest.TestCase):
    def setUp(self):
        X,y,ids,folds=synthetic(150)
        self.frame=X.copy();self.frame[ID]=ids;self.frame['damage_grade']=y;self.frame['fold']=folds
        self.oof=pd.DataFrame({ID:ids,'fold':folds,'y_true':y,'y_pred':y,
            **{f'p_grade_{i}':np.where(y==i,.8,.1) for i in (1,2,3)}})
        self.temp=tempfile.TemporaryDirectory();self.path=Path(self.temp.name)/'oof.csv.gz'

    def tearDown(self):self.temp.cleanup()

    def check(self,oof=None):
        (self.oof if oof is None else oof).to_csv(self.path,index=False,compression={'method':'gzip','mtime':0})
        return read_oof(self.frame,self.path,sha256_file(self.path))

    def test_metric_matches_sklearn_with_missing_predictions(self):
        true=[1,1,2,2,3,3];pred=[3]*6
        r,m,n=counts(true,pred)
        self.assertAlmostEqual(r['macro_f1'],f1_score(true,pred,labels=[1,2,3],average='macro',zero_division=0))
        np.testing.assert_array_equal(m,confusion_matrix(true,pred,labels=[1,2,3]))
        self.assertEqual(n,[[0,0,1]]*3)

    def test_missing_grade3_rate_is_undefined(self):
        r,_,_=counts([1,2],[2,2]);self.assertIsNone(r['rate31']);self.assertEqual(r['grade3_n'],0)

    def test_invalid_labels_and_shapes_rejected(self):
        for a,b in [([],[]),([1],[0]),([1],[4]),([1,2],[1])]:
            with self.assertRaises(ValueError):counts(a,b)

    def test_independent_terms_match_boundaries(self):
        frame=self.frame.iloc[:14].copy();frame['age']=[0,1,10,11,20,21,30,31,50,51,100,101,994,995]
        pd.testing.assert_frame_equal(raw_terms(frame),independent_terms(frame))

    def test_invalid_ages_rejected(self):
        for a in [-1,996,np.inf,1.5]:
            frame=self.frame.copy();frame['age']=frame.age.astype(float);frame.loc[0,'age']=a
            with self.assertRaises(ValueError):independent_terms(frame)

    def test_shuffled_oof_keeps_id_alignment(self):
        q=self.check(self.oof.sample(frac=1,random_state=71))
        np.testing.assert_array_equal(q[ID],self.frame[ID]);np.testing.assert_array_equal(q.y_pred,self.frame.damage_grade)

    def test_duplicate_id_rejected_even_with_new_checksum(self):
        self.oof.loc[0,ID]=self.oof.loc[1,ID]
        with self.assertRaises(ValueError):self.check()

    def test_missing_row_rejected(self):
        with self.assertRaises(ValueError):self.check(self.oof.iloc[:-1])

    def test_new_wrong_id_rejected(self):
        self.oof.loc[0,ID]=999999
        with self.assertRaises(ValueError):self.check()

    def test_wrong_label_rejected(self):
        self.oof.loc[0,'y_true']=2
        with self.assertRaises(AssertionError):self.check()

    def test_wrong_fold_rejected(self):
        self.oof.loc[0,'fold']=1
        with self.assertRaises(AssertionError):self.check()

    def test_probability_range_sum_argmax(self):
        for edits in [{'p_grade_1':1.2},{'p_grade_1':.4},{'y_pred':2}]:
            with self.subTest(edits=edits):
                o=self.oof.copy()
                for c,v in edits.items():o.loc[0,c]=v
                with self.assertRaises((ValueError,AssertionError)):self.check(o)

    def test_stale_checksum_rejected(self):
        self.check()
        with self.assertRaises(ValueError):read_oof(self.frame,self.path,'0'*64)

    def test_paired_errors_reconciled(self):
        r=paired([3,3,3,1],[1,1,2,1],[2,1,1,2])
        self.assertEqual((r['fixed31'],r['new31'],r['shared31']),(1,1,1))
        self.assertEqual(r['disagree'],3)


if __name__=='__main__':unittest.main()
