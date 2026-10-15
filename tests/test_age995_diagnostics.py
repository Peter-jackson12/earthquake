"""Synthetic contract tests only; these tests never fit an estimator."""
import ast
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score,confusion_matrix
from experiments.age995.diagnostics import (validate_join,summary,pair_summary,cost_total,
    age_relation,analyze,CONFIG,PROBS)


class DiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.frame=pd.DataFrame({'building_id':[10,20,30,40,50,60],
            'damage_grade':[1,1,2,2,3,3],'fold':[0,1,2,3,4,0]})
        self.oof=pd.DataFrame({'building_id':[10,20,30,40,50,60],
            'y_true':[1,1,2,2,3,3],'y_pred':[1,2,2,3,1,3],'fold':[0,1,2,3,4,0]})
        p=np.full((6,3),.1);p[np.arange(6),self.oof.y_pred-1]=.8
        self.oof[PROBS]=p

    def test_join_is_id_based(self):
        a=validate_join(self.frame,self.oof.sample(frac=1,random_state=12))
        np.testing.assert_array_equal(a.y_pred,self.oof.y_pred)

    def test_duplicate_id_rejected(self):
        self.oof.loc[1,'building_id']=10
        with self.assertRaisesRegex(ValueError,'uniqueness'):validate_join(self.frame,self.oof)

    def test_missing_row_rejected(self):
        with self.assertRaises(ValueError):validate_join(self.frame,self.oof.iloc[:-1])

    def test_foreign_id_rejected(self):
        self.oof.loc[1,'building_id']=999
        with self.assertRaisesRegex(ValueError,'population'):validate_join(self.frame,self.oof)

    def test_truth_corruption_rejected(self):
        self.oof.loc[0,'y_true']=3
        with self.assertRaisesRegex(ValueError,'truth'):validate_join(self.frame,self.oof)

    def test_fold_corruption_rejected(self):
        self.oof.loc[0,'fold']=1
        with self.assertRaisesRegex(ValueError,'folds'):validate_join(self.frame,self.oof)

    def test_fractional_or_holdout_fold_rejected(self):
        for f in [-1,.5,5,float('inf')]:
            o=self.oof.copy();o['fold']=o.fold.astype(float);o.loc[0,'fold']=f
            with self.assertRaises(ValueError):validate_join(self.frame,o)

    def test_fractional_label_rejected(self):
        self.oof['y_pred']=self.oof.y_pred.astype(float);self.oof.loc[0,'y_pred']=1.5
        with self.assertRaises(ValueError):validate_join(self.frame,self.oof)

    def test_missing_and_nonfinite_probability_rejected(self):
        for v in [np.nan,np.inf,-.1,1.1]:
            o=self.oof.copy();o.loc[0,PROBS[0]]=v
            with self.assertRaises(ValueError):validate_join(self.frame,o)

    def test_bad_probability_sum_rejected(self):
        self.oof.loc[0,PROBS]=[.6,.1,.1]
        with self.assertRaisesRegex(ValueError,'sum'):validate_join(self.frame,self.oof)

    def test_swapped_probability_columns_rejected(self):
        self.oof[PROBS]=self.oof[PROBS[::-1]].to_numpy()
        with self.assertRaisesRegex(ValueError,'disagreement'):validate_join(self.frame,self.oof)

    def test_duplicate_column_rejected(self):
        o=pd.concat([self.oof,self.oof[['fold']]],axis=1)
        with self.assertRaisesRegex(ValueError,'schema'):validate_join(self.frame,o)

    def test_metrics_match_external_reference(self):
        y=self.oof.y_true.to_numpy();p=self.oof.y_pred.to_numpy();s=summary(y,p)
        self.assertAlmostEqual(s['macro_f1'],f1_score(y,p,labels=[1,2,3],average='macro',zero_division=0))
        np.testing.assert_array_equal(s['cm'],confusion_matrix(y,p,labels=[1,2,3]))
        self.assertEqual(s['e31'],1);self.assertEqual(s['grade3_support'],2);self.assertEqual(s['e31_rate'],.5)

    def test_no_grade3_rates_are_missing(self):
        s=summary([1,2],[1,1]);self.assertIsNone(s['e31_rate']);self.assertIsNone(s['e32_rate'])

    def test_all_predicted_three_not_perfect(self):
        s=summary([1,2,3],[3,3,3]);self.assertAlmostEqual(s['macro_f1'],1/6)
        self.assertEqual(s['errors'],2);self.assertEqual(s['e31'],0)

    def test_empty_or_bad_labels_rejected(self):
        for y,p in [([],[]),([1],[0]),([1],[4]),([1],[1,2])]:
            with self.assertRaises(ValueError):summary(y,p)

    def test_pairs_partition_and_severe_error_turnover(self):
        s=pair_summary([1,1,2,2,3,3],[1,2,1,1,1,2],[1,1,2,3,2,1])
        self.assertEqual(s['both_correct'],1);self.assertEqual(s['right_only_correct'],2)
        self.assertEqual(s['both_wrong_different'],3);self.assertEqual(s['fixed_e31'],1);self.assertEqual(s['new_e31'],1)
        self.assertEqual(s['disagreements'],5)

    def test_identical_predictions(self):
        s=pair_summary([1,2,3],[1,1,1],[1,1,1]);self.assertEqual(s['disagreements'],0);self.assertEqual(s['shared_e31'],1)

    def test_pair_labels_and_shapes_rejected(self):
        with self.assertRaises(ValueError):pair_summary([1],[1],[1,2])
        with self.assertRaises(ValueError):pair_summary([1],[1],[4])

    def test_cost_counts_not_macro_f1(self):
        self.assertEqual(cost_total({'errors':10,'e31':2},5),18)
        for w in [-1,.5,np.inf]:
            with self.assertRaises(ValueError):cost_total({'errors':10,'e31':2},w)

    def test_affine_identity_with_zero_and_unseen995(self):
        r=age_relation([0,5,10,20,30],[995,0,10,200],10,29)
        self.assertLess(r['scaled_max_abs_error'],1e-10)

    def test_affine_identity_and_free_linear_score_mapping(self):
        train=np.array([0,10,30,50,995]);valid=np.array([0,20,200,995])
        r=age_relation(train,valid,20,48)
        flag=valid==995;z_c=(np.where(flag,20,valid)-r['mean_C'])/r['scale_C']
        z_e=(np.where(flag,48,valid)-r['mean_E2'])/r['scale_E2']
        w,b,intercept=.8,-.6,.4
        a=r['alpha'];beta=r['beta'];k=r['intercept']
        np.testing.assert_allclose(w*z_e+b*flag+intercept,w*a*z_c+(w*beta+b)*flag+intercept+w*k)
        self.assertNotAlmostEqual(w*w+b*b,(w*a)**2+(w*beta+b)**2)

    def test_constant_age_relation_rejected(self):
        with self.assertRaises(ValueError):age_relation([0,0],[0],0,0)

    def test_nonempty_output_rejected_before_data_read(self):
        with tempfile.TemporaryDirectory() as d,patch('experiments.age995.diagnostics.load_development',side_effect=AssertionError):
            with self.assertRaisesRegex(ValueError,'Output exists'):analyze('missing','missing',d)

    def test_no_model_fit_or_holdout_path(self):
        import experiments.age995.diagnostics as module
        tree=ast.parse(Path(module.__file__).read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
                self.assertNotIn(node.func.attr,{'fit','fit_transform','predict','predict_proba'})
        cfg=json.loads(CONFIG.read_text());self.assertFalse(cfg['model_refit']);self.assertFalse(cfg['holdout_access'])
        self.assertEqual(cfg['scope'],'posthoc_descriptive_development_oof_only')


if __name__=='__main__':unittest.main()
