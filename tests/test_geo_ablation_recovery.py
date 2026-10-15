"""Recovery contracts on synthetic data only; four preprocessing-only fits.

No real-data loading, no classifier fitting and no holdout reads in this file.
"""
import ast
import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))
from test_modeling import synthetic
from preprocess import FEATURES, CATEGORICAL
from experiments.age995.run import make_pipeline as original
from experiments.geo_ablation.run import make_pipeline, config
from experiments.geo_ablation import recovery


class RecoveryMetricTests(unittest.TestCase):
    def setUp(self):
        self.ids = np.arange(6)
        self.y = np.array([1, 2, 3, 3, 3, 3])
        self.pred = np.array([1, 2, 1, 2, 3, 3])
        self.folds = np.array([0, 1, 2, 3, 4, 0])
        p = np.full((6, 3), 0.1)
        p[np.arange(6), self.pred - 1] = 0.8
        self.frame = pd.DataFrame(dict(building_id=self.ids, fold=self.folds,
                                      y_true=self.y, y_pred=self.pred,
                                      **{name: p[:, i] for i, name in enumerate(recovery.PROBS)}))

    def test_independent_f1_matches_reference(self):
        got = recovery.arithmetic(self.y, self.pred)
        self.assertAlmostEqual(got['macro_f1'], f1_score(self.y, self.pred, labels=[1, 2, 3], average='macro'))
        np.testing.assert_array_equal(got['confusion'], confusion_matrix(self.y, self.pred, labels=[1, 2, 3]))

    def test_error_denominator_is_actual_grade3(self):
        got = recovery.arithmetic(self.y, self.pred)
        self.assertEqual(got['rate31'], 0.25)
        self.assertEqual(got['rate32'], 0.25)
        self.assertEqual(got['recall'][2], 0.5)

    def test_no_grade3_rate_is_null(self):
        self.assertIsNone(recovery.arithmetic([1, 2], [1, 2])['rate31'])

    def test_empty_metrics_rejected(self):
        with self.assertRaises(ValueError):
            recovery.arithmetic([], [])

    def test_fractional_label_rejected(self):
        with self.assertRaises(ValueError):
            recovery.arithmetic([1, 2.5], [1, 2])

    def test_invalid_label_rejected(self):
        with self.assertRaises(ValueError):
            recovery.arithmetic([0, 2], [1, 2])

    def test_unaligned_metrics_rejected(self):
        with self.assertRaises(ValueError):
            recovery.arithmetic([1, 2], [1])

    def check(self, frame):
        return recovery.check_oof(frame, self.ids, self.y, self.folds)

    def test_shuffled_rows_realign(self):
        np.testing.assert_array_equal(self.check(self.frame.iloc[::-1]).y_pred, self.pred)

    def test_duplicate_ids_rejected(self):
        frame = self.frame.copy(); frame.loc[0, 'building_id'] = 1
        with self.assertRaises(ValueError): self.check(frame)

    def test_missing_row_rejected(self):
        with self.assertRaises(ValueError): self.check(self.frame.iloc[:-1])

    def test_changed_truth_rejected(self):
        frame = self.frame.copy(); frame.loc[0, 'y_true'] = 3
        with self.assertRaises(AssertionError): self.check(frame)

    def test_changed_fold_rejected(self):
        frame = self.frame.copy(); frame.loc[0, 'fold'] = 4
        with self.assertRaises(AssertionError): self.check(frame)

    def test_nan_probability_rejected(self):
        frame = self.frame.copy(); frame.loc[0, 'p_grade_1'] = np.nan
        with self.assertRaises(ValueError): self.check(frame)

    def test_bad_probability_sum_rejected(self):
        frame = self.frame.copy(); frame.loc[0, 'p_grade_1'] = 0.5
        with self.assertRaises(AssertionError): self.check(frame)

    def test_wrong_argmax_rejected(self):
        frame = self.frame.copy(); frame.loc[0, 'y_pred'] = 2
        with self.assertRaises(AssertionError): self.check(frame)

    def test_missing_probability_column_rejected(self):
        with self.assertRaises(ValueError): self.check(self.frame.drop(columns='p_grade_3'))


class RecoverySafetyTests(unittest.TestCase):
    def test_absence_is_unknown_not_zero_historical_fits(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = recovery.inventory(tmp)
        self.assertIsNone(result['t6_completed_fits_verified'])
        self.assertFalse(result['training_allowed'])
        self.assertFalse(result['complete_study'])

    def test_file_presence_does_not_prove_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for path in ['outputs/geo_ablation_v1/design.json', 'outputs/geo_ablation_v1/design_seal.json',
                         'outputs/geo_ablation_v1/audit_available/audit.json',
                         'outputs/geo_ablation_unattempted_v1/audit.json']:
                p = root / path; p.parent.mkdir(parents=True, exist_ok=True); p.write_text('{}')
            result = recovery.inventory(root)
        self.assertEqual(result['status'], 'RECOVERED_FILES_REQUIRE_FULL_AUDIT')
        self.assertIsNone(result['t6_attempts_verified'])
        self.assertFalse(result['training_allowed'])

    def test_never_overwrite_existing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'audit.json'
            recovery.write_new(path, {'value': 1})
            with self.assertRaises(FileExistsError): recovery.write_new(path, {'value': 2})
            self.assertEqual(json.loads(path.read_text()), {'value': 1})

    def test_protected_evidence_destination_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(ValueError):
                recovery.write_new(root / 'private' / 'audit.json', {}, (root / 'private',))
            self.assertFalse((root / 'private').exists())

    def test_nan_output_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError): recovery.write_new(Path(tmp) / 'audit.json', {'x': float('nan')})

    def test_recovery_has_no_fit_or_fit_transform_calls(self):
        tree = ast.parse(inspect.getsource(recovery))
        forbidden = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and
                     isinstance(n.func, ast.Attribute) and n.func.attr in {'fit', 'fit_transform', 'partial_fit'}]
        self.assertEqual(forbidden, [])


class RecoveryFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        X, _, _, _ = synthetic(120)
        X.loc[[0, 5, 100], 'age'] = 995
        cls.train, cls.valid = X.iloc[:90].copy(), X.iloc[90:].copy()
        cls.valid['geo_level_1_id'] = '90001'
        cls.valid['geo_level_2_id'] = '90002'
        cls.valid['geo_level_3_id'] = '90003'
        with threadpool_limits(limits=1):
            cls.base = original('C')['preprocess'].fit(cls.train)
            cls.ablation = {v: make_pipeline(v)['preprocess'].fit(cls.train) for v in ('R12', 'R1', 'R0')}

    def projection(self, variant):
        excluded = config()['variants'][variant]
        names = self.base['columns'].get_feature_names_out()
        keep = np.array([not any(column in name for column in excluded) for name in names])
        pre = self.ablation[variant]
        np.testing.assert_array_equal(pre['columns'].get_feature_names_out(), names[keep])
        for frame in [self.train, self.valid]:
            self.assertEqual((pre.transform(frame) != self.base.transform(frame)[:, keep]).nnz, 0)

    def test_R12_matrix_is_C_projection(self): self.projection('R12')
    def test_R1_matrix_is_C_projection(self): self.projection('R1')
    def test_R0_matrix_is_C_projection(self): self.projection('R0')

    def test_excluded_geography_change_invariant(self):
        for variant, pre in self.ablation.items():
            changed = self.valid.copy()
            for column in config()['variants'][variant]: changed[column] = '99999'
            self.assertEqual((pre.transform(self.valid) != pre.transform(changed)).nnz, 0)

    def test_original_C_factory_not_mutated(self):
        self.assertEqual(original('C')['preprocess']['columns'].transformers[2][2], CATEGORICAL)

    def test_model_parameters_unchanged(self):
        for variant in self.ablation:
            self.assertEqual(make_pipeline(variant)['model'].get_params(), original('C')['model'].get_params())

    def test_raw_38_column_contract_preserved(self):
        for pre in self.ablation.values():
            self.assertEqual(pre['schema'].feature_names_in_.tolist(), FEATURES)
            with self.assertRaises(ValueError): pre.transform(self.valid.drop(columns='geo_level_3_id'))

    def test_ID_and_target_and_fold_rejected(self):
        for column in ('building_id', 'damage_grade', 'fold'):
            with self.assertRaises(ValueError): self.ablation['R0'].transform(self.valid.assign(**{column: 1}))

    def test_age_median_and_raw_flag_match_C(self):
        for pre in self.ablation.values():
            self.assertEqual(pre['age995'].median_, float(self.train.loc[self.train.age.ne(995), 'age'].median()))
            transformed = pre['age995'].transform(pre['schema'].transform(self.valid))
            np.testing.assert_array_equal(transformed.age_is_995, self.valid.age.eq(995))

    def test_C_is_not_an_allowed_new_training_variant(self):
        with self.assertRaises(ValueError): make_pipeline('C')


if __name__ == '__main__':
    unittest.main()
