"""T24 학습 지역 중앙값·차이의 누수 방지와 희소 추가 블록 합성 검사."""
import copy
import json
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from experiments.geo_context_features.features import (
    CONTEXT_COLUMNS, GROUP_COLUMN, OUTPUT_FEATURES, GeoContextFeatures,
)


def sample():
    return pd.DataFrame(dict(geo_level_1_id=['2', '1', '2', '1', '2'],
        count_floors_pre_eq=[0, 1, 2, 4, 8], height_percentage=[10, 2, 20, 8, 30],
        area_percentage=[5, 2, 9, 8, 13]), index=[71, 3, 85, 22, 9])


class GeoContextFeaturesTest(unittest.TestCase):
    def test_odd_even_group_medians_and_six_columns_by_hand(self):
        x = sample()
        f = GeoContextFeatures().fit(x)
        expected = np.array([[2, 20, 9, -2, -10, -4], [2.5, 5, 5, -1.5, -3, -3],
            [2, 20, 9, 0, 0, 0], [2.5, 5, 5, 1.5, 3, 3],
            [2, 20, 9, 6, 10, 4]], dtype=np.float32)
        np.testing.assert_array_equal(f.transform(x).toarray(), expected)
        np.testing.assert_array_equal(f.global_medians_, [2, 10, 8])
        self.assertEqual(f.metadata()['groups'], [dict(geo_level_1_id=1, rows=2, medians=[2.5, 5., 5.]),
            dict(geo_level_1_id=2, rows=3, medians=[2., 20., 9.])])
        self.assertEqual(f.get_feature_names_out().tolist(), list(OUTPUT_FEATURES))
        self.assertEqual(len(OUTPUT_FEATURES), 6)
        self.assertEqual(f.metadata()['context_columns'], list(CONTEXT_COLUMNS))

    def test_unseen_geo_uses_training_global_medians_and_all_row_denominator(self):
        f = GeoContextFeatures().fit(sample())
        valid = pd.DataFrame(dict(geo_level_1_id=['0', '99', '1'], count_floors_pre_eq=[0, 10, 5],
                                 height_percentage=[0, 100, 8], area_percentage=[0, 20, 9]))
        expected = np.array([[2, 10, 8, -2, -10, -8], [2, 10, 8, 8, 90, 12],
                             [2.5, 5, 5, 2.5, 3, 4]], dtype=np.float32)
        np.testing.assert_array_equal(f.transform(valid).toarray(), expected)
        self.assertEqual(f.unknown_counts(valid), {GROUP_COLUMN: dict(count=2, rate=2/3, rows=3)})

    def test_global_even_median_uses_all_rows_not_average_of_group_medians(self):
        x = sample().iloc[:4].copy()
        f = GeoContextFeatures().fit(x)
        np.testing.assert_array_equal(f.global_medians_, [1.5, 9, 6.5])
        # 그룹별 중앙값의 평균과 전체 중앙값은 일반적으로 같지 않다.
        full = GeoContextFeatures().fit(sample())
        self.assertNotEqual(full.global_medians_[0], np.mean([v[0] for v in full.group_medians_.values()]))

    def test_training_rows_include_self_and_singleton_has_zero_differences(self):
        x = sample().iloc[:1].copy()
        f = GeoContextFeatures().fit(x)
        np.testing.assert_array_equal(f.transform(x).toarray(), [[0, 10, 5, 0, 0, 0]])
        self.assertTrue(f.metadata()['training_rows_include_self'])
        self.assertEqual(sum(row['rows'] for row in f.metadata()['groups']), len(x))

    def test_validation_changes_cannot_update_training_state(self):
        f = GeoContextFeatures().fit(sample())
        state = copy.deepcopy(f.metadata())
        valid = sample().copy()
        valid.loc[:, 'height_percentage'] = [1000, 0, 100, 0, 10000]
        valid.loc[:, 'geo_level_1_id'] = ['99', '1', '100', '1', '2']
        before = valid.copy(deep=True)
        f.transform(valid)
        f.unknown_counts(valid)
        self.assertEqual(state, f.metadata())
        pd.testing.assert_frame_equal(valid, before)

    def test_y_is_not_read_and_input_is_not_mutated(self):
        class UnreadableTarget:
            def __array__(self, *args, **kwargs):
                raise AssertionError('정답을 읽었습니다.')
        x = sample()
        before = x.copy(deep=True)
        f = GeoContextFeatures().fit(x, UnreadableTarget())
        f.transform(x)
        pd.testing.assert_frame_equal(x, before)
        self.assertFalse(f.metadata()['target_used'])

    def test_row_order_csr_float32_finite_and_zero_inputs(self):
        x = sample()
        f = GeoContextFeatures().fit(x)
        expected = f.transform(x)[[4, 0, 3, 1, 2]]
        actual = f.transform(x.iloc[[4, 0, 3, 1, 2]])
        self.assertEqual((actual != expected).nnz, 0)
        self.assertTrue(sparse.isspmatrix_csr(actual))
        self.assertEqual(actual.dtype, np.float32)
        self.assertTrue(np.isfinite(actual.data).all())
        zero = sample().copy()
        zero.loc[:, list(CONTEXT_COLUMNS)] = 0
        zero[GROUP_COLUMN] = '0'
        self.assertEqual(GeoContextFeatures().fit(zero).transform(zero).nnz, 0)

    def test_invalid_geo_or_context_numbers_rejected_before_fit_or_transform(self):
        fitted = GeoContextFeatures().fit(sample())
        for column in (GROUP_COLUMN, *CONTEXT_COLUMNS):
            for value in [np.nan, np.inf, -.1, .5]:
                x = sample().copy()
                x[column] = pd.to_numeric(x[column]).astype(float)
                x.iloc[0, x.columns.get_loc(column)] = value
                with self.subTest(column=column, value=value):
                    with self.assertRaises(ValueError):
                        GeoContextFeatures().fit(x)
                    with self.assertRaises(ValueError):
                        fitted.transform(x)

    def test_large_integer_group_ids_are_not_merged_by_float_conversion(self):
        x = sample().iloc[:2].copy()
        x[GROUP_COLUMN] = pd.Series([2**53, 2**53+1], index=x.index, dtype=np.uint64)
        f = GeoContextFeatures().fit(x)
        self.assertEqual([g[GROUP_COLUMN] for g in f.metadata()['groups']], [2**53, 2**53+1])
        np.testing.assert_array_equal(f.transform(x).toarray()[:, 3:], np.zeros((2, 3)))

    def test_float32_overflow_rejected_for_fit_and_validation(self):
        x = sample().copy()
        x['height_percentage'] = 1e40
        with self.assertRaises(ValueError):
            GeoContextFeatures().fit(x)
        f = GeoContextFeatures().fit(sample())
        with self.assertRaises(ValueError):
            f.transform(x)

    def test_schema_duplicate_reordered_missing_and_changed_arm_rejected(self):
        x = sample()
        f = GeoContextFeatures().fit(x)
        for invalid in [x.iloc[:, :-1], x.iloc[:, ::-1], pd.concat([x, x.iloc[:, :1]], axis=1)]:
            with self.assertRaises(ValueError):
                f.transform(invalid)
        with self.assertRaises(ValueError):
            GeoContextFeatures('wrong').fit(x)
        with self.assertRaises(ValueError):
            f.get_feature_names_out(['wrong'])
        f.arm = 'changed'
        with self.assertRaises(ValueError):
            f.metadata()

    def test_empty_fit_rejected_empty_transform_and_unknown_allowed(self):
        x = sample()
        with self.assertRaises(ValueError):
            GeoContextFeatures().fit(x.iloc[:0])
        f = GeoContextFeatures().fit(x)
        block = f.transform(x.iloc[:0])
        self.assertEqual(block.shape, (0, 6))
        self.assertEqual(block.dtype, np.float32)
        self.assertTrue(sparse.isspmatrix_csr(block))
        self.assertEqual(f.unknown_counts(x.iloc[:0]), {GROUP_COLUMN: dict(count=0, rate=None, rows=0)})

    def test_json_state_copies_clone_and_repeated_transform(self):
        x = sample()
        f = GeoContextFeatures().fit(x)
        state = f.metadata()
        self.assertEqual(json.loads(json.dumps(state, allow_nan=False)), state)
        state['groups'][0]['medians'][0] = -999
        state['global_medians'][0] = -999
        self.assertGreaterEqual(f.metadata()['groups'][0]['medians'][0], 0)
        self.assertGreaterEqual(f.metadata()['global_medians'][0], 0)
        self.assertEqual((f.transform(x) != f.transform(x)).nnz, 0)
        cloned = clone(f)
        with self.assertRaises(NotFittedError):
            cloned.transform(x)

    def test_fresh_process_pickle_without_execution_module_import(self):
        x = sample()
        f = GeoContextFeatures().fit(x)
        expected = f.transform(x)
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'synthetic.pkl'
            path.write_bytes(pickle.dumps(dict(transformer=f, frame=x)))
            code = ('import json,pickle,sys; from pathlib import Path; '
                'd=pickle.loads(Path(sys.argv[1]).read_bytes()); f=d["transformer"]; '
                'm=f.transform(d["frame"]); '
                'assert "experiments.geo_context_features.run" not in sys.modules; '
                'print(json.dumps(dict(shape=m.shape,data=m.data.tolist(),indices=m.indices.tolist(),'
                'indptr=m.indptr.tolist(),metadata=f.metadata()),ensure_ascii=False))')
            run = subprocess.run([sys.executable, '-c', code, str(path)], cwd=root,
                                 capture_output=True, encoding='utf-8', text=True, check=True)
            got = json.loads(run.stdout)
            self.assertEqual(got['shape'], list(expected.shape))
            self.assertEqual(got['data'], expected.data.tolist())
            self.assertEqual(got['indices'], expected.indices.tolist())
            self.assertEqual(got['indptr'], expected.indptr.tolist())
            self.assertEqual(got['metadata'], f.metadata())


if __name__ == '__main__':
    unittest.main()
