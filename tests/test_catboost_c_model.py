"""T25 모의 CatBoost·합성 C 검사. 연구 자료와 실제 분류기 학습은 사용하지 않는다."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import joblib
import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'tests'))
from test_modeling import synthetic
from experiments.lgbm_native_cat.run import make_preprocessor
from experiments.catboost_c import model as adapter


class FakeCatBoost:
    calls = []
    saved = {}
    mutation = False
    invalid_probability = False
    wrong_classes = False
    fewer_trees = False
    reload_shift = 0.

    def __init__(self, **params):
        self.params = dict(params)
        self.reloaded = False

    def fit(self, matrix, target):
        self.__class__.calls.append((self, matrix, target.copy()))
        self.classes_ = np.array([0, 2, 1] if self.wrong_classes else [0, 1, 2])
        self.tree_count_ = 299 if self.fewer_trees else 300
        self.feature_names_ = [str(i) for i in range(matrix.shape[1])]
        if self.mutation:
            matrix.data[0] += 1
        return self

    def predict_proba(self, matrix, thread_count):
        if thread_count != 2:
            raise AssertionError('두 스레드만 허용')
        p = np.tile([.2, .3, .5], (matrix.shape[0], 1))
        if self.invalid_probability:
            p[0, 0] = np.nan
        if self.reloaded:
            p[:, 0] += self.reload_shift
            p[:, 1] -= self.reload_shift
        return p

    def get_all_params(self):
        result = {k: v for k, v in self.params.items()
                  if k not in ('verbose', 'thread_count', 'allow_writing_files')}
        result['learning_rate'] = float(np.float32(result['learning_rate']))
        return result

    def get_params(self):
        return dict(self.params)

    def save_model(self, path, format):
        if format != 'cbm':
            raise AssertionError('native cbm만 허용')
        self.__class__.saved[path] = copy.deepcopy(self)
        Path(path).write_bytes(b'synthetic-cbm')

    def load_model(self, path, format):
        if format != 'cbm':
            raise AssertionError('native cbm만 허용')
        self.__dict__.update(copy.deepcopy(self.__class__.saved[path].__dict__))
        self.reloaded = True
        return self


class CatBoostCModelTests(unittest.TestCase):
    def setUp(self):
        FakeCatBoost.calls, FakeCatBoost.saved = [], {}
        FakeCatBoost.mutation = FakeCatBoost.invalid_probability = False
        FakeCatBoost.wrong_classes = FakeCatBoost.fewer_trees = False
        FakeCatBoost.reload_shift = 0.
        X, y, _, _ = synthetic(60)
        X.loc[:8, 'age'] = 995
        self.Xa, self.Xb, self.y = X.iloc[:45].copy(), X.iloc[45:].copy(), y.iloc[:45].copy()
        self.pre = make_preprocessor('O')
        self.mt = self.pre.fit_transform(self.Xa)
        self.mv = self.pre.transform(self.Xb)
        self.patch = mock.patch.object(adapter, '_catboost_classifier', return_value=FakeCatBoost)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def fit(self, **kwargs):
        return adapter.fit_catboost_c(self.Xa, self.y, self.Xb, dict(adapter.CATBOOST_PARAMS),
                                    prepared=(self.pre, self.mt, self.mv), **kwargs)

    def test_fixed_parameters_csr_identity_targets_and_no_fit_extras(self):
        before_t, before_v = adapter.matrix_evidence(self.mt), adapter.matrix_evidence(self.mv)
        result = self.fit()
        model, matrix, target = FakeCatBoost.calls[0]
        self.assertIs(matrix, self.mt)
        self.assertEqual(model.params, adapter.CATBOOST_PARAMS)
        np.testing.assert_array_equal(target, self.y.to_numpy()-1)
        self.assertEqual(result['info']['damage_labels'], [1, 2, 3])
        self.assertEqual(result['info']['internal_classes'], [0, 1, 2])
        self.assertEqual(result['info']['tree_count'], 300)
        self.assertEqual(result['info']['matrix_evidence'], dict(train=before_t, valid=before_v))
        self.assertEqual(adapter.matrix_evidence(self.mt), before_t)
        self.assertEqual(adapter.matrix_evidence(self.mv), before_v)
        np.testing.assert_array_equal(result['p'], np.tile([.2, .3, .5], (len(self.Xb), 1)))
        json.dumps(result['info'], allow_nan=False)

    def test_unprepared_path_uses_exact_original_c_and_train_only_median(self):
        result = adapter.fit_catboost_c(self.Xa, self.y, self.Xb, dict(adapter.CATBOOST_PARAMS))
        self.assertEqual(result['info']['matrix_evidence'], dict(train=adapter.matrix_evidence(self.mt),
                                                               valid=adapter.matrix_evidence(self.mv)))
        regular = self.Xa.loc[self.Xa.age != 995, 'age']
        self.assertEqual(result['info']['age995_median'], float(regular.median()))
        changed = self.Xb.copy()
        changed.loc[:, 'age'] = 995
        second = adapter.fit_catboost_c(self.Xa, self.y, changed, dict(adapter.CATBOOST_PARAMS))
        self.assertEqual(result['info']['preprocessor_state'], second['info']['preprocessor_state'])

    def test_save_native_and_preprocessor_reload_exactly_and_reject_overwrite(self):
        result = self.fit()
        with tempfile.TemporaryDirectory() as d:
            paths = result['save'](Path(d)/'fit')
            self.assertEqual([p.name for p in paths], ['fit_model.cbm', 'fit_preprocessor.joblib'])
            self.assertTrue(all(p.is_file() for p in paths))
            self.assertEqual(result['verify'](paths, self.Xb, result['p']), 0.)
            with self.assertRaises(ValueError):
                result['save'](Path(d)/'fit')

    def test_partial_existing_checkpoint_is_never_overwritten(self):
        result = self.fit()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'fit_preprocessor.joblib'
            p.write_bytes(b'preserve')
            with self.assertRaises(ValueError): result['save'](Path(d)/'fit')
            self.assertEqual(p.read_bytes(), b'preserve')
            self.assertFalse((Path(d)/'fit_model.cbm').exists())

    def test_reload_probability_and_preprocessor_matrix_tampering_rejected(self):
        result = self.fit()
        with tempfile.TemporaryDirectory() as d:
            paths = result['save'](Path(d)/'fit')
            FakeCatBoost.reload_shift = 1e-8
            with self.assertRaises(ValueError): result['verify'](paths, self.Xb, result['p'])
            FakeCatBoost.reload_shift = 0.
            changed = self.Xb.copy()
            changed.iloc[0, changed.columns.get_loc('age')] = 900
            with self.assertRaises(ValueError): result['verify'](paths, changed, result['p'])
            pre = joblib.load(paths[1])
            pre['age995'].median_ += 1
            joblib.dump(pre, paths[1])
            with self.assertRaises(ValueError): result['verify'](paths, self.Xb, result['p'])

    def test_reload_class_and_effective_parameter_tampering_rejected(self):
        result = self.fit()
        with tempfile.TemporaryDirectory() as d:
            paths = result['save'](Path(d)/'fit')
            saved = FakeCatBoost.saved[str(paths[0])]
            saved.classes_ = np.array([2, 1, 0])
            with self.assertRaises(ValueError): result['verify'](paths, self.Xb, result['p'])
            saved.classes_ = np.array([0, 1, 2])
            saved.params['depth'] = 7
            with self.assertRaises(ValueError): result['verify'](paths, self.Xb, result['p'])

    def test_parameter_changes_extra_fit_policy_and_boolean_alias_rejected(self):
        for key, value in [('iterations', 299), ('sample_weight', None), ('cat_features', []),
                           ('early_stopping_rounds', 3), ('thread_count', 4), ('random_seed', 101),
                           ('depth', 6.), ('bagging_temperature', True)]:
            params = dict(adapter.CATBOOST_PARAMS)
            params[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                adapter.fit_catboost_c(self.Xa, self.y, self.Xb, params,
                                      prepared=(self.pre, self.mt, self.mv))
        self.assertEqual(FakeCatBoost.calls, [])

    def test_invalid_targets_rejected_before_any_fit(self):
        for y in [self.y.iloc[:-1], np.ones(len(self.y)), np.zeros(len(self.y)),
                  self.y.astype(str), self.y.to_numpy()[:, None],
                  np.where(np.arange(len(self.y)) == 0, np.nan, self.y),
                  np.where(np.arange(len(self.y)) == 0, 1.5, self.y)]:
            with self.assertRaises(ValueError):
                adapter.fit_catboost_c(self.Xa, y, self.Xb, dict(adapter.CATBOOST_PARAMS),
                                      prepared=(self.pre, self.mt, self.mv))
        self.assertEqual(FakeCatBoost.calls, [])

    def test_noncsr_float64_nonfinite_and_misaligned_matrices_rejected(self):
        bad = self.mt.copy()
        bad.data[0] = np.inf
        for matrix in [self.mt.tocsc(), self.mt.astype(np.float64), bad, self.mt[:-1], self.mt[:, :-1]]:
            with self.assertRaises(ValueError):
                adapter.fit_catboost_c(self.Xa, self.y, self.Xb, dict(adapter.CATBOOST_PARAMS),
                                      prepared=(self.pre, matrix, self.mv))
        self.assertEqual(FakeCatBoost.calls, [])

    def test_original_inputs_preprocessor_and_sparse_arrays_are_not_changed(self):
        Xa, Xb, y = self.Xa.copy(deep=True), self.Xb.copy(deep=True), self.y.copy()
        state = adapter.preprocessor_state(self.pre)
        self.fit()
        self.assertTrue(self.Xa.equals(Xa) and self.Xb.equals(Xb) and self.y.equals(y))
        self.assertEqual(adapter.preprocessor_state(self.pre), state)

    def test_model_mutation_invalid_classes_rounds_and_probability_rejected(self):
        for attr in ['mutation', 'wrong_classes', 'fewer_trees', 'invalid_probability']:
            with self.subTest(attr=attr):
                setattr(FakeCatBoost, attr, True)
                with self.assertRaises(ValueError): self.fit()
                setattr(FakeCatBoost, attr, False)

    def test_probabilities_no_clipping_or_renormalizing_and_class_dimensions(self):
        for p in [[[.2, .3, .4]], [[-.1, .5, .6]], [[0, 0, np.inf]], [[.5, .5]],
                  [[True, False, False]], [['0', '0', '1']]]:
            with self.assertRaises(ValueError): adapter.probabilities(p, 1)
        p = np.array([[0., .5, .5]])
        np.testing.assert_array_equal(adapter.probabilities(p, 1), p)

    def test_import_does_not_load_optional_catboost_or_runner(self):
        with mock.patch.dict(sys.modules, {'catboost': None}):
            spec = importlib.util.spec_from_file_location('t25_model_only', adapter.__file__)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.assertEqual(module.CATBOOST_PARAMS, adapter.CATBOOST_PARAMS)
            with self.assertRaises(ModuleNotFoundError): module._catboost_classifier()

    def test_native_json_float32_equivalent_rounding_only_tolerance(self):
        state = dict(depth=6, learning_rate=float(np.float32(.05)), classes=[0, 1, 2])
        equivalent = copy.deepcopy(state)
        equivalent['learning_rate'] = float(np.nextafter(state['learning_rate'], 1.))
        self.assertTrue(adapter._same_native_state(state, equivalent))
        equivalent['learning_rate'] = .05000000075
        self.assertTrue(adapter._same_native_state(state, equivalent))
        self.assertTrue(adapter._same_native_state(.10000000149011612, .1000000015))
        self.assertFalse(adapter._same_native_state(1e6, 1e6+1e-6))
        for key, value in [('learning_rate', .0500001), ('depth', 6.), ('classes', [2, 1, 0])]:
            changed = dict(state, **{key: value})
            self.assertFalse(adapter._same_native_state(state, changed))


if __name__ == '__main__':
    unittest.main()
