"""Synthetic adversarial audit tests; never opens real development or holdout files."""
import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.model_families.audit import aligned, statistics, verify_saved, transitions, encoded_digest, audit
if importlib.util.find_spec('psutil') is None:
    raise unittest.SkipTest('Optional model-family requirements not installed: psutil')
from experiments.model_families.run import matrix_sha


def sample():
    y=np.tile([1,2,3],10); pred=y.copy();pred[[0,3,6]]=2;pred[[2,5]]=1
    frame=pd.DataFrame({'building_id':np.arange(30)+100,'damage_grade':y,'fold':np.repeat(np.arange(5),6)})
    prob=np.full((30,3),.1);prob[np.arange(30),pred-1]=.8
    raw=pd.DataFrame({'building_id':frame.building_id,'fold':frame.fold,'y_true':y,'y_pred':pred,
                      **{f'p_grade_{c+1}':prob[:,c] for c in range(3)}})
    return frame,raw


def skmetrics(y,p):
    cm=confusion_matrix(y,p,labels=[1,2,3])
    return {'macro_f1':f1_score(y,p,labels=[1,2,3],average='macro',zero_division=0),
            'accuracy':accuracy_score(y,p),'confusion_matrix':cm.tolist(),
            'severe_as_minor_rate':cm[2,0]/cm[2].sum(),
            'per_class':classification_report(y,p,labels=[1,2,3],output_dict=True,zero_division=0)}


def saved_fixture(raw):
    rec=[]
    for f in range(5):
        r=raw[raw.fold.eq(f)]
        rec.append({'fold':f,'train_rows':24,'validation_rows':6,'fit_seconds':0,
                    'validation':skmetrics(r.y_true,r.y_pred)})
    scores=[r['validation']['macro_f1'] for r in rec]
    return {'status':'complete','rows':30,'folds':5,'holdout_scored':False,'fold_results':rec,
            'oof':skmetrics(raw.y_true,raw.y_pred),'cv_macro_f1_mean':float(np.mean(scores)),
            'cv_macro_f1_std':float(np.std(scores,ddof=1))}


class AuditContracts(unittest.TestCase):
    def setUp(self): self.frame,self.raw=sample()
    def test_valid_permuted_rows(self):
        pd.testing.assert_frame_equal(aligned(self.frame,self.raw.iloc[::-1]),self.raw)
    def test_duplicate_id(self):
        self.raw.loc[0,'building_id']=self.raw.loc[1,'building_id']
        with self.assertRaises(ValueError):aligned(self.frame,self.raw)
    def test_wrong_id_set(self):
        self.raw.loc[0,'building_id']=9999
        with self.assertRaises(ValueError):aligned(self.frame,self.raw)
    def test_missing_row(self):
        with self.assertRaises(ValueError):aligned(self.frame,self.raw.iloc[:-1])
    def test_fractional_integer_fields(self):
        for col in ['building_id','fold','y_true','y_pred']:
            r=self.raw.astype({col:float});r.loc[0,col]+=.5
            with self.assertRaises(ValueError):aligned(self.frame,r)
    def test_nonfinite_integer_fields(self):
        for value in [np.nan,np.inf]:
            r=self.raw.astype({'fold':float});r.loc[0,'fold']=value
            with self.assertRaises(ValueError):aligned(self.frame,r)
    def test_wrong_frozen_fold(self):
        self.raw.loc[0,'fold']=4
        with self.assertRaises(ValueError):aligned(self.frame,self.raw)
    def test_wrong_truth(self):
        self.raw.loc[0,'y_true']=3
        with self.assertRaises(ValueError):aligned(self.frame,self.raw)
    def test_invalid_prediction_label(self):
        self.raw.loc[0,'y_pred']=0
        with self.assertRaises(ValueError):aligned(self.frame,self.raw)
    def test_probability_outside_domain(self):
        for value in [np.nan,np.inf,-.1,1.1]:
            r=self.raw.copy();r.loc[0,'p_grade_1']=value
            with self.assertRaises(ValueError):aligned(self.frame,r)
    def test_probability_sum(self):
        self.raw.loc[0,'p_grade_1']=.2
        with self.assertRaises(AssertionError):aligned(self.frame,self.raw)
    def test_probability_order_argmax(self):
        r=self.raw.copy();r[['p_grade_1','p_grade_2']]=r[['p_grade_2','p_grade_1']].to_numpy()
        with self.assertRaises(ValueError):aligned(self.frame,r)
    def test_all_metrics_against_sklearn(self):
        saved=saved_fixture(self.raw)
        s,folds=verify_saved(self.raw,saved)
        self.assertEqual(s['n'],30);self.assertEqual(len(folds),5)
        self.assertEqual(s['grade3_count'],10);self.assertEqual(s['error_3_to_1'],2)
        self.assertAlmostEqual(s['rate_3_to_1'],.2)
    def test_no_grade3_denominator_is_missing_not_zero_rate(self):
        s=statistics([1,2],[1,2]);self.assertIsNone(s['rate_3_to_1'])
        self.assertEqual(s['grade3_support'],0);self.assertIsNone(s['matrix_row_normalized'][2][0])
    def test_empty_or_bad_labels(self):
        for y,p in [([],[]),([1],[2,3]),([1],[0]),([1.5],[1])]:
            with self.assertRaises(ValueError):statistics(y,p)
    def test_wrong_saved_macro_even_with_valid_oof_hash(self):
        s=saved_fixture(self.raw);s['cv_macro_f1_mean']+=.01
        with self.assertRaises(AssertionError):verify_saved(self.raw,s)
    def test_population_sd_instead_of_sample_sd(self):
        s=saved_fixture(self.raw);s['cv_macro_f1_std']=float(np.std([r['validation']['macro_f1'] for r in s['fold_results']],ddof=0))
        with self.assertRaises(AssertionError):verify_saved(self.raw,s)
    def test_wrong_saved_matrix(self):
        s=saved_fixture(self.raw);s['oof']['confusion_matrix'][0][0]+=1
        with self.assertRaises(ValueError):verify_saved(self.raw,s)
    def test_wrong_saved_support_precision(self):
        for key in ['support','precision','recall','f1-score']:
            s=saved_fixture(self.raw);s['oof']['per_class']['3'][key]+=1
            with self.assertRaises(AssertionError):verify_saved(self.raw,s)
    def test_wrong_train_valid_counts(self):
        s=saved_fixture(self.raw);s['fold_results'][0]['train_rows']=25
        with self.assertRaises(ValueError):verify_saved(self.raw,s)
    def test_fold_missing_duplicate(self):
        s=saved_fixture(self.raw);s['fold_results'][4]['fold']=0
        with self.assertRaises(ValueError):verify_saved(self.raw,s)
    def test_holdout_result_forbidden(self):
        s=saved_fixture(self.raw);s['holdout_scored']=True
        with self.assertRaises(ValueError):verify_saved(self.raw,s)
    def test_transitions_count_fixed_new_retained(self):
        s=transitions([3,3,3,1,2],[1,1,2,1,1],[3,1,1,2,2])
        self.assertEqual((s['severe_fixed'],s['severe_new'],s['severe_retained']),(1,1,1))
        self.assertEqual(s['changed'],s['before_only_correct']+s['after_only_correct']+s['both_wrong_changed'])
    def test_independent_digest_matches(self):
        from scipy import sparse
        a=sparse.csr_matrix(np.eye(4,dtype=np.float32));self.assertEqual(encoded_digest(a),matrix_sha(a))
    def test_nonempty_audit_output_refused(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'keep';p.write_text('original')
            with self.assertRaises(FileExistsError):audit('missing','missing','missing',t)
            self.assertEqual(p.read_text(),'original')


if __name__=='__main__':unittest.main()
