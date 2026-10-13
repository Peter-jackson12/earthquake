"""Small deterministic tests: not evidence of real-data predictive performance."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baseline import build_pipeline, evaluate_model, fitted_preprocessor, training_parameters
from preprocess import ID, TARGET, NUMERIC
from scripts.audit_eda_oof import age_groups, confusion_stats, eda_summary, validate_oof, verify_metrics
from test_modeling import synthetic


class EdaOofTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({ID: [4, 5, 6], TARGET: [1, 2, 3], 'fold': [0, 1, 2],
                                   'geo_level_1_id': ['26', '26', '17']})
        self.oof = pd.DataFrame({ID: [4, 5, 6], 'y_true': [1, 2, 3], 'fold': [0, 1, 2],
                                'y_pred': [1, 2, 3], 'p_grade_1': [1., 0., 0.],
                                'p_grade_2': [0., 1., 0.], 'p_grade_3': [0., 0., 1.]})

    def test_age_boundaries_exhaustive(self):
        result = age_groups(pd.Series(np.arange(996)))
        self.assertFalse(result.isna().any())
        self.assertEqual(result.value_counts(sort=False).tolist(), [1,10,10,10,20,50,894,1])
        self.assertEqual(result.iloc[995], '995')

    def test_invalid_ages_rejected(self):
        for value in [-1, 996, np.nan, np.inf, 10.5]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                age_groups(pd.Series([value]))

    def test_shuffled_oof_join_is_id_based(self):
        result = validate_oof(self.frame, self.oof.iloc[::-1])
        self.assertEqual(result[ID].tolist(), [4,5,6])
        self.assertEqual(result.y_true.tolist(), [1,2,3])

    def test_bad_id_fold_label_rejected(self):
        for column, value in [(ID, 999), ('fold', 0.5), ('fold', 4), ('y_true', 3)]:
            bad = self.oof.astype({'fold':float}).copy()
            bad.loc[0, column] = value
            with self.subTest(column=column, value=value), self.assertRaises(ValueError):
                validate_oof(self.frame, bad)

    def test_duplicate_and_missing_rows_rejected(self):
        for bad in [self.oof.iloc[:2], pd.concat([self.oof.iloc[:2], self.oof.iloc[:1]])]:
            with self.assertRaises(ValueError):
                validate_oof(self.frame, bad)

    def test_bad_probabilities_rejected(self):
        for value in [-.1, 1.1, np.nan, np.inf]:
            bad = self.oof.copy();bad.loc[0, 'p_grade_1'] = value
            with self.subTest(value=value), self.assertRaises((ValueError, AssertionError)):
                validate_oof(self.frame, bad)
        bad = self.oof.copy();bad.loc[0, 'p_grade_1'] = .8
        with self.assertRaises(AssertionError):
            validate_oof(self.frame, bad)

    def test_swapped_probability_columns_rejected(self):
        bad = self.oof.copy()
        bad[['p_grade_1','p_grade_3']] = bad[['p_grade_3','p_grade_1']].to_numpy()
        with self.assertRaises(ValueError):
            validate_oof(self.frame, bad)

    def test_no_grade3_is_undefined_not_zero(self):
        result = confusion_stats([1,2], [1,1])
        self.assertIsNone(result['rate_3_to_1'])
        self.assertIsNone(result['rate_3_to_2'])
        self.assertEqual(result['grade3_count'], 0)

    def test_severe_denominator_and_hand_f1(self):
        result = confusion_stats([1,1,2,2,3,3], [1,2,2,2,3,1])
        self.assertAlmostEqual(result['macro_f1'], (.5+.8+2/3)/3)
        self.assertEqual(result['rate_3_to_1'], .5)

    def test_eda_partitions_cover_all_rows(self):
        X,y,ids,folds = synthetic(150)
        X[ID],X[TARGET],X['fold'] = ids,y,folds
        result = eda_summary(X)
        for _,group in result.loc[result.feature!='region_foundation'].groupby('feature'):
            self.assertEqual(group.n.sum(),len(X))
            self.assertEqual(group.grade3_count.sum(),int(y.eq(3).sum()))


class TreatmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = json.loads((ROOT/'configs/baseline.json').read_text())
        cls.weighted = json.loads((ROOT/'configs/class_weights.json').read_text())
        cls.rus = json.loads((ROOT/'configs/lr_rus.json').read_text())

    def test_gbc_weights_use_only_passed_train_labels(self):
        train = pd.Series([1]*2+[2]*6+[3]*4)
        weights = training_parameters('gbc', self.weighted, train)['model__sample_weight']
        np.testing.assert_allclose(weights, [2.]*2+[2/3]*6+[1.]*4)
        self.assertEqual(training_parameters('lr', self.base, train), {})

    def test_single_factor_configs(self):
        feature = json.loads((ROOT/'configs/feature_budget.json').read_text())
        for model, reference in [('dt',self.base),('rf',feature)]:
            actual = self.weighted['models'][model].copy()
            self.assertEqual(actual.pop('class_weight'), 'balanced')
            self.assertEqual(actual,reference['models'][model])
        self.assertEqual(self.weighted['models']['gbc'],feature['models']['gbc'])
        self.assertEqual(self.rus['models']['lr'],self.base['models']['lr'])

    def test_invalid_policy_and_combined_changes_rejected(self):
        bad = copy.deepcopy(self.rus);bad['models']['lr']['class_weight']='balanced'
        with self.assertRaises(ValueError): build_pipeline('lr',bad)
        bad = copy.deepcopy(self.base);bad['sample_weight']={'lr':'balanced'}
        with self.assertRaises(ValueError): training_parameters('lr',bad,pd.Series([1,2,3]))

    def test_rus_train_only_sparse_and_scaler_before_sampling(self):
        X,_,_,_ = synthetic(210)
        y = pd.Series([1]*20+[2]*100+[3]*30)
        model = build_pipeline('lr',self.rus,jobs=1).fit(X.iloc[:150],y)
        kept = model['sampler'].sample_indices_.copy()
        self.assertEqual(len(kept),60)
        self.assertEqual(y.iloc[kept].value_counts().sort_index().tolist(),[20,20,20])
        pre = fitted_preprocessor(model)
        np.testing.assert_allclose(pre['columns'].named_transformers_['numeric']['scale'].mean_,
                                   X.iloc[:150][NUMERIC].mean())
        self.assertTrue(sparse.issparse(pre.transform(X.iloc[150:])) )
        self.assertEqual(len(model.predict(X.iloc[150:])),60)
        np.testing.assert_array_equal(kept,model['sampler'].sample_indices_)
        cloned = build_pipeline('lr',self.rus,jobs=1).fit(X.iloc[:150],y)
        np.testing.assert_array_equal(kept,cloned['sampler'].sample_indices_)

    def test_gbc_sample_weight_reaches_fit(self):
        X,y,_,_=synthetic(60)
        config=copy.deepcopy(self.weighted);config['models']['gbc']['n_estimators']=2
        pipeline=build_pipeline('gbc',config,jobs=1)
        weights=training_parameters('gbc',config,y)
        with patch.object(pipeline['model'],'fit', wraps=pipeline['model'].fit) as fit:
            pipeline.fit(X,y,**weights)
            np.testing.assert_array_equal(fit.call_args.kwargs['sample_weight'],weights['model__sample_weight'])

    def test_all_new_paths_oof_and_reload(self):
        X,y,ids,folds=synthetic(150)
        frame=X.copy();frame[ID],frame[TARGET],frame['fold']=ids,y,folds
        with tempfile.TemporaryDirectory() as temp:
            for name,raw in [('dt',self.weighted),('rf',self.weighted),('gbc',self.weighted),('lr',self.rus)]:
                config=copy.deepcopy(raw)
                for key in ['rf','gbc']:config['models'][key]['n_estimators']=2
                directory=Path(temp)/name
                result=evaluate_model(name,X,y,ids,folds,config,directory,jobs=1)
                self.assertTrue(result['fold_results'][0]['reload_prediction_checked'])
                oof=pd.read_csv(directory/'oof_predictions.csv.gz')
                joined=validate_oof(frame,oof)
                verify_metrics(joined,result)
                if name=='lr':
                    self.assertTrue(all(r['sampling'] for r in result['fold_results']))
                else:
                    self.assertTrue(all(r['class_weights'] for r in result['fold_results']))



@unittest.skipUnless((ROOT/'data/development.csv').exists(), 'Private development data not installed.')
class RealDevelopmentTests(unittest.TestCase):
    def test_eda_published_counts_recomputed(self):
        from scripts.audit_eda_oof import development_frame
        frame,_=development_frame(ROOT/'data')
        summary=eda_summary(frame).set_index(['feature','category'])
        checks=[('foundation_type','r',175259,66352),('foundation_type','i',8465,182),
                ('roof_type','q',49319,18902),('roof_type','x',12976,583),
                ('geo_level_1_id','17',17370,14013),('geo_level_1_id','26',18038,1532),
                ('age_group','1-10',58051,16972),('age_group','31-50',26475,10374),
                ('age_group','995',1125,317)]
        for feature,category,n,severe in checks:
            with self.subTest(feature=feature,category=category):
                row=summary.loc[(feature,category)]
                self.assertEqual(row['n'],n);self.assertEqual(row.grade3_count,severe)
                self.assertAlmostEqual(row.grade3_rate,severe/n)

    def test_actual_five_folds_refit_both_preprocessing_paths(self):
        from preprocess import CATEGORICAL,load_development,make_preprocessor
        X,y,ids,cv=load_development(ROOT/'data')
        for scale in [False,True]:
            for f,(train,valid) in enumerate(cv.split()):
                with self.subTest(scale=scale,fold=f):
                    pre=make_preprocessor(scale)
                    encoded=pre.fit_transform(X.iloc[train])
                    self.assertTrue(sparse.isspmatrix_csr(encoded))
                    self.assertEqual(encoded.dtype,np.float32)
                    for c,learned in zip(CATEGORICAL,pre['columns'].named_transformers_['categorical'].categories_):
                        self.assertEqual(set(learned),set(X.iloc[train][c]))
                    if scale:
                        np.testing.assert_allclose(pre['columns'].named_transformers_['numeric']['scale'].mean_,
                                                   X.iloc[train][NUMERIC].mean())
                    transformed=pre.transform(X.iloc[valid])
                    self.assertTrue(sparse.isspmatrix_csr(transformed))
                    self.assertEqual(transformed.shape[1],encoded.shape[1])

    def test_development_loader_does_not_open_holdout(self):
        from scripts.audit_eda_oof import development_frame
        original=pd.read_csv
        paths=[]
        def checked(path,*args,**kwargs):
            paths.append(str(path))
            if 'final_holdout' in str(path):raise AssertionError('Holdout access forbidden')
            return original(path,*args,**kwargs)
        with patch('pandas.read_csv',side_effect=checked):development_frame(ROOT/'data')
        self.assertEqual(len(paths),2)

if __name__=='__main__': unittest.main()
