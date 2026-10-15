"""T26: 합성 자료의 C head 동일성·원래 범주 전달·모의 native 저장 검사."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'tests'))
from test_modeling import synthetic
from experiments.lgbm_native_cat.run import make_preprocessor as original_c
from experiments.catboost_native_c import model as adapter
from preprocess import CATEGORICAL


class FakeCatBoost:
    calls = []
    saved = {}
    fault = None

    def __init__(self, **params):
        self.params = dict(params)
        self.reloaded = False

    def fit(self, frame, target, *, cat_features):
        self.__class__.calls.append((frame, target.copy(), list(cat_features)))
        self.classes_ = np.array([0, 1, 2])
        self.tree_count_ = 300
        self.feature_names_ = frame.columns.tolist()
        self.cat_indices = list(cat_features)
        if self.fault == 'mutate': frame.iloc[0, 0] += 1
        if self.fault == 'classes': self.classes_ = np.array([1, 0, 2])
        if self.fault == 'trees': self.tree_count_ = 299
        if self.fault == 'cat_indices': self.cat_indices[0] = 0
        return self

    def get_cat_feature_indices(self):
        return list(self.cat_indices)

    def predict_proba(self, frame, thread_count):
        if thread_count != 2: raise AssertionError('두 스레드만 허용')
        p = np.tile([.2, .3, .5], (len(frame), 1))
        if self.fault == 'probability': p[0, 0] = np.nan
        if self.reloaded and self.fault == 'reload_probability':
            p[:, 0] += 1e-8
            p[:, 1] -= 1e-8
        return p

    def get_all_params(self):
        return {k: v for k, v in self.params.items() if k not in ('verbose', 'thread_count', 'allow_writing_files')}

    def get_params(self):
        return dict(self.params)

    def save_model(self, path, format):
        if format != 'cbm': raise AssertionError('native cbm 필수')
        self.__class__.saved[path] = copy.deepcopy(self)
        Path(path).write_bytes(b'synthetic-cbm')

    def load_model(self, path, format):
        if format != 'cbm': raise AssertionError('native cbm 필수')
        self.__dict__.update(copy.deepcopy(self.__class__.saved[path].__dict__))
        self.reloaded = True
        return self


class NativeCModelTests(unittest.TestCase):
    def setUp(self):
        X, y, _, _ = synthetic(90)
        X.loc[:8, 'age'] = 995
        X.loc[9:10, 'age'] = 0
        self.Xa, self.Xb, self.y = X.iloc[:72].copy(), X.iloc[72:].copy(), y.iloc[:72].copy()
        FakeCatBoost.calls, FakeCatBoost.saved, FakeCatBoost.fault = [], {}, None

    def fit(self):
        with mock.patch.object(adapter, '_catboost_classifier', return_value=FakeCatBoost):
            return adapter.fit_catboost_native_c(self.Xa, self.y, self.Xb, dict(adapter.CATBOOST_PARAMS))

    def test_numeric_head_exactly_matches_original_c_train_and_valid_bytes(self):
        base = original_c('O')
        expected_train = base.fit_transform(self.Xa)[:, :28].toarray()
        expected_valid = base.transform(self.Xb)[:, :28].toarray()
        pre, train, valid = adapter.prepare_native_c(self.Xa, self.Xb)
        for frame, expected in [(train, expected_train), (valid, expected_valid)]:
            actual = frame[adapter.HEAD_COLUMNS].to_numpy()
            self.assertEqual(actual.dtype, np.float32)
            self.assertEqual(actual.tobytes(), expected.tobytes())
            self.assertEqual(adapter.input_evidence(frame)['numeric_head']['sha256'], hashlib.sha256(expected.tobytes()).hexdigest())
        self.assertEqual(pre.head_['age995'].median_, base['age995'].median_)
        self.assertEqual(adapter.HEAD_COLUMNS, base.get_feature_names_out()[:28].tolist())

    def test_original_11_normalized_strings_pass_through_and_exact_order(self):
        pre, train, valid = adapter.prepare_native_c(self.Xa, self.Xb)
        expected = pre.head_['schema'].transform(self.Xb)[CATEGORICAL].astype(str)
        self.assertEqual(valid.columns.tolist(), adapter.HEAD_COLUMNS+CATEGORICAL)
        self.assertEqual(valid.shape, (18, 39))
        self.assertTrue(all(valid[c].dtype == object for c in CATEGORICAL))
        np.testing.assert_array_equal(valid[CATEGORICAL].to_numpy(), expected.to_numpy())
        payload = json.dumps(expected.values.tolist(), ensure_ascii=False, separators=(',', ':')).encode('utf8')
        self.assertEqual(adapter.input_evidence(valid)['categorical']['sha256'], hashlib.sha256(payload).hexdigest())
        np.testing.assert_array_equal(pre.get_feature_names_out(), valid.columns)

    def test_unseen_categories_remain_strings_and_train_only_unknown_counts(self):
        pre = adapter.make_preprocessor().fit(self.Xa)
        state = pre.metadata()
        changed = self.Xb.copy()
        changed.loc[:, 'geo_level_2_id'] = '999999'
        changed.loc[:, 'foundation_type'] = '새범주|문자열,[]'
        transformed = pre.transform(changed)
        self.assertEqual(transformed['foundation_type'].tolist(), ['새범주|문자열,[]']*18)
        counts = pre.unknown_counts(changed)
        for c in ['geo_level_2_id', 'foundation_type']:
            self.assertEqual(counts[c], dict(count=18, rate=1., rows=18))
        self.assertTrue(all(v['count'] == 0 for v in pre.unknown_counts(self.Xa).values()))
        self.assertEqual(pre.metadata(), state)

    def test_age995_training_median_includes_zero_and_validation_never_changes_state(self):
        Xa = self.Xa.copy()
        Xa.loc[:, 'age'] = 995
        Xa.iloc[:4, Xa.columns.get_loc('age')] = np.array([0, 0, 10, 20], dtype=Xa.age.dtype)
        pre = adapter.make_preprocessor().fit(Xa, np.full(len(Xa), 999))
        self.assertEqual(pre.metadata()['age995_median'], 5.)
        state = pre.metadata()
        changed = self.Xb.copy()
        changed.loc[:, 'age'] = 995
        pre.transform(changed)
        self.assertEqual(pre.metadata(), state)
        second = adapter.make_preprocessor().fit(Xa, np.arange(len(Xa)))
        self.assertEqual(second.metadata(), state)

    def test_input_index_row_order_and_original_frame_are_preserved(self):
        Xa, Xb = self.Xa.copy(deep=True), self.Xb.copy(deep=True)
        pre = adapter.make_preprocessor().fit(self.Xa)
        order = np.arange(len(self.Xb))[::-1]
        actual = pre.transform(self.Xb.iloc[order])
        expected = pre.transform(self.Xb).iloc[order]
        pd.testing.assert_frame_equal(actual, expected)
        self.assertTrue(self.Xa.equals(Xa) and self.Xb.equals(Xb))

    def test_empty_transform_and_unknown_are_defined_empty_fit_is_rejected(self):
        pre = adapter.make_preprocessor().fit(self.Xa)
        result = pre.transform(self.Xa.iloc[:0])
        self.assertEqual(result.shape, (0, 39))
        self.assertEqual(adapter.input_evidence(result)['numeric_head']['shape'], [0, 28])
        self.assertTrue(all(v == dict(count=0, rate=None, rows=0) for v in pre.unknown_counts(self.Xa.iloc[:0]).values()))
        with self.assertRaises(ValueError): adapter.make_preprocessor().fit(self.Xa.iloc[:0])

    def test_schema_invalid_numeric_and_all995_train_are_rejected(self):
        pre = adapter.make_preprocessor().fit(self.Xa)
        for X in [self.Xb.iloc[:, ::-1], self.Xb.drop(columns=['age']), self.Xb.assign(extra=1)]:
            with self.assertRaises(ValueError): pre.transform(X)
        for value in [np.inf, -1, 1.5]:
            X = self.Xb.copy()
            X['count_floors_pre_eq'] = X['count_floors_pre_eq'].astype(float)
            X.iloc[0, X.columns.get_loc('count_floors_pre_eq')] = value
            with self.assertRaises(ValueError): pre.transform(X)
        with self.assertRaises(ValueError): adapter.make_preprocessor().fit(self.Xa.assign(age=995))

    def test_input_evidence_rejects_numeric_dtype_cat_type_nonfinite_or_reordered_columns(self):
        _, train, valid = adapter.prepare_native_c(self.Xa, self.Xb)
        bad64 = valid.copy(); bad64[adapter.HEAD_COLUMNS[0]] = bad64[adapter.HEAD_COLUMNS[0]].astype(float)
        badcat = valid.copy(); badcat['geo_level_1_id'] = np.arange(len(valid))
        badinf = valid.copy(); badinf.iloc[0, 0] = np.inf
        for frame in [bad64, badcat, badinf, valid.iloc[:, ::-1], valid.iloc[:, :-1]]:
            with self.assertRaises(ValueError): adapter.input_evidence(frame)

    def test_metadata_is_json_and_returned_state_cannot_mutate_training_vocabulary(self):
        pre = adapter.make_preprocessor().fit(self.Xa)
        state = pre.metadata()
        json.dumps(state, ensure_ascii=False, allow_nan=False)
        state['categorical_categories'][0].append('new')
        state['output_columns'].append('new')
        self.assertNotEqual(pre.metadata(), state)

    def test_preprocessor_pickle_in_fresh_process_reproduces_native_primitive_hashes(self):
        pre, _, valid = adapter.prepare_native_c(self.Xa, self.Xb)
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            joblib.dump(pre, root/'pre.joblib'); joblib.dump(self.Xb, root/'raw.joblib')
            script = 'import joblib,json;from experiments.catboost_native_c.model import input_evidence;p=joblib.load(__import__("sys").argv[1]);x=joblib.load(__import__("sys").argv[2]);print(json.dumps(input_evidence(p.transform(x))))'
            proc = subprocess.run([sys.executable, '-c', script, str(root/'pre.joblib'), str(root/'raw.joblib')],
                                  cwd=ROOT, capture_output=True, text=True, encoding='utf8', check=True)
            self.assertEqual(json.loads(proc.stdout), adapter.input_evidence(valid))

    def test_fit_uses_only_outer_train_labels_and_exact_native_indices(self):
        result = self.fit()
        frame, target, cats = FakeCatBoost.calls[0]
        self.assertEqual(cats, list(range(28, 39)))
        self.assertEqual(frame.shape, (72, 39))
        np.testing.assert_array_equal(target, self.y.to_numpy()-1)
        self.assertEqual(result['info']['internal_classes'], [0, 1, 2])
        self.assertEqual(result['info']['cat_feature_indices'], cats)
        self.assertEqual(result['info']['numeric_head_columns'], 28)
        self.assertEqual(result['p'].shape, (18, 3))
        json.dumps(result['info'], ensure_ascii=False, allow_nan=False)

    def test_save_reload_two_artifacts_and_no_overwrite(self):
        result = self.fit()
        with tempfile.TemporaryDirectory() as d:
            paths = result['save'](Path(d)/'fit')
            self.assertEqual([p.name for p in paths], ['fit_model.cbm', 'fit_preprocessor.joblib'])
            self.assertEqual(result['verify'](paths, self.Xb, result['p']), 0.)
            with self.assertRaises(ValueError): result['save'](Path(d)/'fit')
            FakeCatBoost.fault = 'reload_probability'
            with self.assertRaises(ValueError): result['verify'](paths, self.Xb, result['p'])

    def test_model_rejects_input_mutation_wrong_classes_trees_catindices_or_probability(self):
        for fault in ['mutate', 'classes', 'trees', 'cat_indices', 'probability']:
            FakeCatBoost.fault = fault
            with self.subTest(fault=fault), self.assertRaises(ValueError): self.fit()

    def test_fit_rejects_parameter_and_target_drift_before_classifier(self):
        for key, value in [('iterations', 299), ('sample_weight', None), ('eval_set', []), ('cat_features', [0])]:
            params = dict(adapter.CATBOOST_PARAMS, **{key: value})
            with mock.patch.object(adapter, '_catboost_classifier') as classifier:
                with self.assertRaises(ValueError): adapter.fit_catboost_native_c(self.Xa, self.y, self.Xb, params)
                classifier.assert_not_called()
        for y in [self.y.iloc[:-1], np.ones(72), self.y.astype(str), np.full(72, np.nan)]:
            with self.assertRaises(ValueError): adapter.fit_catboost_native_c(self.Xa, y, self.Xb, dict(adapter.CATBOOST_PARAMS))

    def test_optional_dependency_lazy_and_no_t25_import(self):
        text = Path(adapter.__file__).read_text(encoding='utf8')
        self.assertNotIn('experiments.catboost_c', text)
        with mock.patch.dict(sys.modules, {'catboost': None}):
            spec = importlib.util.spec_from_file_location('t26_model_only', adapter.__file__)
            module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
            self.assertEqual(module.CATBOOST_PARAMS, adapter.CATBOOST_PARAMS)
            with self.assertRaises(ModuleNotFoundError): module._catboost_classifier()

    def test_native_float32_rounding_policy_is_strict_for_other_values(self):
        self.assertTrue(adapter._same_native_state(.05000000074505806, .05000000075))
        self.assertFalse(adapter._same_native_state(.05000000074505806, .0500001))
        self.assertFalse(adapter._same_native_state(6, 6.))
        self.assertFalse(adapter._same_native_state({'depth':6}, {'depth':7}))


if __name__ == '__main__':
    unittest.main()
