"""Synthetic corruption tests for the independent follow-up auditor."""
import contextlib
import copy
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))
from test_modeling import synthetic
from preprocess import ID, sha256_file
from experiments.age995.followup import fit_variant, make_pipeline, config
from experiments.age995.audit_followup import audit_variant, direct_stats, linear_quantile, independent_keep


class FollowupAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory()
        cls.source=Path(cls.tmp.name)/'original'
        X,y,ids,folds=synthetic(150)
        X.loc[[0,7,121],'age']=995
        cls.frame=X.copy();cls.frame[ID]=ids;cls.frame['damage_grade']=y;cls.frame['fold']=folds
        cls.contract={'n_folds':5}
        with contextlib.redirect_stdout(io.StringIO()):
            fit_variant('D',cls.frame,cls.contract,cls.source,'a'*40)
        cls.params=make_pipeline('D')['model'].get_params()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        self.case=tempfile.TemporaryDirectory()
        self.path=Path(self.case.name)/'D'
        shutil.copytree(self.source,self.path)

    def tearDown(self):
        self.case.cleanup()

    def check(self):
        return audit_variant(self.frame,self.path,'D',self.contract,self.params,config())

    def edit_json(self,relative,mutator):
        path=self.path/relative
        value=json.loads(path.read_text());mutator(value);path.write_text(json.dumps(value))

    def edit_record(self,mutator):
        self.edit_json('lr/metrics.json',lambda v:mutator(v['fold_results'][0]))
        # Keep the duplicated fold file in agreement, so the independent check must detect it.
        self.edit_json('lr/fold_metrics.json',lambda v:mutator(v[0]))

    def edit_oof(self,mutator):
        path=self.path/'lr/oof_predictions.csv.gz'
        value=pd.read_csv(path);mutator(value);value.to_csv(path,index=False,compression={'method':'gzip','mtime':0})
        self.edit_json('lr/metrics.json',lambda v:v.update(oof_sha256=sha256_file(path)))

    def test_original_audits_with_deleted_training_and_full_validation(self):
        result=self.check()
        self.assertEqual(sum(r['validation_rows'] for r in result['folds']),150)
        self.assertTrue(any(r['train_rows']<120 for r in result['folds']))
        self.assertEqual(result['model']['code_commit'],'a'*40)

    def test_equations_equal_reference_for_missing_predicted_class(self):
        y=[1,1,2,2,3,3];p=[3,3,3,3,3,3]
        result,cm,norm=direct_stats(y,p)
        self.assertAlmostEqual(result['macro_f1'],f1_score(y,p,labels=[1,2,3],average='macro',zero_division=0))
        np.testing.assert_array_equal(cm,confusion_matrix(y,p,labels=[1,2,3]))
        np.testing.assert_array_equal(norm,[[0,0,1]]*3)
        self.assertEqual(result['grade1_recall'],0)

    def test_no_grade3_has_no_error_rate_denominator(self):
        result,_,_=direct_stats([1,2],[2,2])
        self.assertIsNone(result['rate_3_to_1']);self.assertEqual(result['grade3_support'],0)

    def test_invalid_and_empty_labels_rejected(self):
        for y,p in [([],[]),([0],[1]),([1],[4]),([1,2],[1])]:
            with self.assertRaises(ValueError):direct_stats(y,p)

    def test_independent_linear_quantile(self):
        for q in [0,.25,.99,1]:
            self.assertEqual(linear_quantile([0,10,30],q),float(pd.Series([0,10,30]).quantile(q)))
        with self.assertRaises(ValueError):linear_quantile([],.99)

    def test_independent_faith_union_and_strict_boundary(self):
        X=self.frame.iloc[:4].copy()
        X['age']=[55,56,55,55];X['count_floors_pre_eq']=[5,5,6,5]
        X['area_percentage']=[20,20,20,21];X['height_percentage']=10
        self.assertEqual(independent_keep(X,'F2').tolist(),[True,False,False,False])

    def test_changed_oof_bytes_rejected(self):
        with open(self.path/'lr/oof_predictions.csv.gz','ab') as f:f.write(b'changed')
        with self.assertRaisesRegex(ValueError,'checksum'):self.check()

    def test_duplicate_oof_id_rejected_even_with_updated_checksum(self):
        self.edit_oof(lambda d:d.__setitem__(ID,[d[ID].iloc[0]]*len(d)))
        with self.assertRaisesRegex(ValueError,'uniqueness'):self.check()

    def test_relabelled_oof_rejected_even_with_updated_checksum(self):
        self.edit_oof(lambda d:d.__setitem__('y_true',d.y_true%3+1))
        with self.assertRaises(AssertionError):self.check()

    def test_changed_validation_folds_rejected(self):
        self.edit_oof(lambda d:d.__setitem__('fold',(d.fold+1)%5))
        with self.assertRaises(AssertionError):self.check()

    def test_invalid_probabilities_rejected(self):
        self.edit_oof(lambda d:d.__setitem__('p_grade_1',2.0))
        with self.assertRaisesRegex(ValueError,'probability'):self.check()

    def test_validation_removal_rejected(self):
        self.edit_record(lambda r:r.update(validation_rows=r['validation_rows']-1))
        with self.assertRaisesRegex(ValueError,'denominator'):self.check()

    def test_fake_retained_training_count_rejected(self):
        self.edit_record(lambda r:r.update(train_rows=120))
        with self.assertRaisesRegex(ValueError,'row count'):self.check()

    def test_fake_removed_class_count_rejected(self):
        self.edit_record(lambda r:r['selection_audit']['removed_class_counts'].update({'1':999}))
        with self.assertRaisesRegex(ValueError,'class count'):self.check()

    def test_fake_selection_id_hash_rejected(self):
        self.edit_record(lambda r:r['selection_audit'].update(retained_ids_sha256='0'*64))
        with self.assertRaisesRegex(ValueError,'ID hash'):self.check()

    def test_fake_preprocessor_statistics_rejected(self):
        self.edit_record(lambda r:r['preprocessor_audit'].update(numeric_mean=[0]*5))
        with self.assertRaises(AssertionError):self.check()

    def test_missing_category_coverage_rejected(self):
        self.edit_record(lambda r:r['selection_audit']['category_coverage'].pop())
        with self.assertRaisesRegex(ValueError,'coverage'):self.check()

    def test_forged_classification_metrics_rejected(self):
        self.edit_record(lambda r:r['validation'].update(macro_f1=.99))
        with self.assertRaises(AssertionError):self.check()

    def test_source_hash_change_rejected(self):
        self.edit_json('run_metadata.json',lambda m:m['source_code_sha256'].update({'preprocess.py':'0'*64}))
        with self.assertRaisesRegex(ValueError,'Source hash'):self.check()

    def test_hyperparameter_change_rejected(self):
        self.edit_json('lr/metrics.json',lambda m:m['effective_parameters'].update(C=10))
        with self.assertRaisesRegex(ValueError,'parameters'):self.check()

    def test_holdout_or_incomplete_status_rejected(self):
        self.edit_json('run_metadata.json',lambda m:m.update(holdout_scored=True))
        with self.assertRaisesRegex(ValueError,'Holdout'):self.check()


if __name__=='__main__':unittest.main()
