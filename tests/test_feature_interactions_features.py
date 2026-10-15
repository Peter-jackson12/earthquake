"""T21 변환기의 학습 통계·충돌 방지·희소 표현·직렬화를 합성 행으로 확인한다."""
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

from experiments.feature_interactions.features import ARMS, FeatureInteractions


def numeric():
    return pd.DataFrame(dict(age=[0, 10, 995], height_percentage=[1, 2, 3],
                             count_floors_pre_eq=[1, 1, 2]), index=[13, 7, 99])


def categorical():
    return pd.DataFrame(dict(foundation_type=['a', 'b', 'a'], roof_type=['x', 'y', 'x'],
                             ground_floor_type=['g', 'h', 'g'], other_floor_type=['p', 'q', 'q']),
                        index=[13, 7, 99])


class FeatureInteractionsTest(unittest.TestCase):
    def test_age_training_median_includes_zero_and_exact_scaled_product(self):
        x = numeric()
        before = x.copy(deep=True)
        f = FeatureInteractions(ARMS[0]).fit(x)
        self.assertEqual(f.median_, 5.)
        raw = np.array([0., 20., 30.])
        expected = ((raw-raw.mean())/np.sqrt(np.mean((raw-raw.mean())**2))).astype(np.float32)
        got = f.transform(x)
        self.assertTrue(sparse.isspmatrix_csr(got))
        self.assertEqual(got.dtype, np.float32)
        np.testing.assert_array_equal(got.toarray().ravel(), expected)
        self.assertEqual(f.metadata()['scaler']['mean'], [float(raw.mean())])
        pd.testing.assert_frame_equal(x, before)

    def test_validation_cannot_change_median_scaler_or_input(self):
        f = FeatureInteractions().fit(numeric())
        old = copy.deepcopy(f.metadata())
        valid = pd.DataFrame(dict(age=[995, 100], height_percentage=[4, 9], count_floors_pre_eq=[2, 3]))
        before = valid.copy(deep=True)
        result = f.transform(valid).toarray()
        expected_product = np.array([[40.], [2700.]])
        expected = ((expected_product-f.scaler_.mean_)/f.scaler_.scale_).astype(np.float32)
        np.testing.assert_array_equal(result, expected)
        self.assertEqual(f.metadata(), old)
        pd.testing.assert_frame_equal(valid, before)

    def test_zero_product_constant_scaler_and_all995_rejection(self):
        x = numeric()
        x['age'] = [0, 995, 0]
        x['count_floors_pre_eq'] = [0, 2, 1]
        f = FeatureInteractions().fit(x)
        self.assertEqual(f.median_, 0.)
        self.assertEqual(f.metadata()['scaler']['variance'], [0.])
        self.assertEqual(f.metadata()['scaler']['scale'], [1.])
        np.testing.assert_array_equal(f.transform(x).toarray(), np.zeros((3, 1), dtype=np.float32))
        x['age'] = 995
        with self.assertRaises(ValueError):
            FeatureInteractions().fit(x)

    def test_targets_are_not_read(self):
        class UnreadableTarget:
            def __array__(self, *args, **kwargs):
                raise AssertionError('목표변수에 접근했습니다.')
        for arm, x in zip(ARMS, [numeric(), categorical()]):
            f = FeatureInteractions(arm).fit(x, UnreadableTarget())
            self.assertEqual(f.metadata()['n_train_rows'], 3)

    def test_known_pairs_exact_one_per_block_and_unknown_zero(self):
        train = categorical()
        f = FeatureInteractions(ARMS[1]).fit(train)
        valid = pd.DataFrame(dict(foundation_type=['a', 'a', 'b'], roof_type=['x', 'y', 'z'],
                                 ground_floor_type=['g', 'g', 'h'], other_floor_type=['p', 'q', 'p']))
        matrix = f.transform(valid)
        np.testing.assert_array_equal(matrix.sum(axis=1).A.ravel(), [2., 1., 0.])
        state = f.metadata()
        width = len(state['pairs'][0]['vocabulary'])
        self.assertEqual(matrix[1, :width].nnz, 0)
        self.assertEqual(matrix[1, width:].nnz, 1)
        self.assertEqual(f.unknown_counts(valid), {
            'foundation_roof': dict(count=2, rate=2/3, rows=3),
            'ground_other_floor': dict(count=1, rate=1/3, rows=3)})
        self.assertEqual(len(f.get_feature_names_out()), matrix.shape[1])

    def test_tuple_delimiter_and_quote_collisions_do_not_merge(self):
        x = categorical()
        x['foundation_type'] = ['a|b', 'a', '["a","b"]']
        x['roof_type'] = ['c', 'b|c', '\\"한글']
        before = x.copy(deep=True)
        f = FeatureInteractions(ARMS[1]).fit(x)
        vocab = f.metadata()['pairs'][0]['vocabulary']
        self.assertEqual(len(vocab), 3)
        self.assertIn(['a|b', 'c'], vocab)
        self.assertIn(['a', 'b|c'], vocab)
        first = f.transform(x)[:, :len(vocab)].toarray()
        self.assertEqual(len(np.unique(first, axis=0)), 3)
        pd.testing.assert_frame_equal(x, before)

    def test_row_order_preserved_and_validation_vocabulary_unchanged(self):
        for arm, x in zip(ARMS, [numeric(), categorical()]):
            f = FeatureInteractions(arm).fit(x)
            state = copy.deepcopy(f.metadata())
            expected = f.transform(x)[[2, 0, 1]]
            got = f.transform(x.iloc[[2, 0, 1]])
            self.assertEqual((got != expected).nnz, 0)
            self.assertEqual(state, f.metadata())

    def test_missing_schema_reordered_columns_mutated_arm_rejected(self):
        for arm, x in zip(ARMS, [numeric(), categorical()]):
            f = FeatureInteractions(arm).fit(x)
            with self.assertRaises(ValueError):
                f.transform(x.iloc[:, :-1])
            with self.assertRaises(ValueError):
                f.transform(x.iloc[:, ::-1])
            with self.assertRaises(ValueError):
                f.get_feature_names_out(['wrong'])
            f.arm = ARMS[1] if arm == ARMS[0] else ARMS[0]
            with self.assertRaises(ValueError):
                f.transform(x)

    def test_nonfinite_fractional_negative_numeric_and_invalid_categories(self):
        for value in [np.nan, np.inf, -.1, .5]:
            x = numeric().astype(float)
            x.loc[x.index[0], 'age'] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                FeatureInteractions().fit(x)
        for value in [None, np.nan, 1]:
            x = categorical()
            x.loc[x.index[0], 'roof_type'] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                FeatureInteractions(ARMS[1]).fit(x)

    def test_metadata_roundtrip_copy_and_sklearn_clone(self):
        for arm, x in zip(ARMS, [numeric(), categorical()]):
            f = FeatureInteractions(arm).fit(x)
            state = f.metadata()
            self.assertEqual(json.loads(json.dumps(state, ensure_ascii=False, allow_nan=False)), state)
            state['input_columns'].append('mutated')
            self.assertNotIn('mutated', f.metadata()['input_columns'])
            names = f.get_feature_names_out()
            names[0] = 'mutated'
            self.assertNotIn('mutated', f.get_feature_names_out())
            other = clone(f)
            self.assertEqual(other.arm, arm)
            with self.assertRaises(NotFittedError):
                other.transform(x)

    def test_empty_train_rejected_empty_transform_has_fixed_width(self):
        for arm, x in zip(ARMS, [numeric(), categorical()]):
            with self.assertRaises(ValueError):
                FeatureInteractions(arm).fit(x.iloc[:0])
            f = FeatureInteractions(arm).fit(x)
            got = f.transform(x.iloc[:0])
            self.assertTrue(sparse.isspmatrix_csr(got))
            self.assertEqual(got.shape, (0, len(f.get_feature_names_out())))
            self.assertEqual(got.dtype, np.float32)

    def test_pickle_fresh_process_without_runner_import(self):
        root = Path(__file__).resolve().parents[1]
        for arm, x in zip(ARMS, [numeric(), categorical()]):
            f = FeatureInteractions(arm).fit(x)
            expected = f.transform(x)
            with tempfile.TemporaryDirectory() as folder:
                p = Path(folder)/'synthetic.pkl'
                p.write_bytes(pickle.dumps(dict(transformer=f, frame=x)))
                program = ('import pickle,sys,json; from pathlib import Path; '
                    'd=pickle.loads(Path(sys.argv[1]).read_bytes()); f=d["transformer"]; '
                    'm=f.transform(d["frame"]); '
                    'assert "experiments.feature_interactions.run" not in sys.modules; '
                    'print(json.dumps(dict(shape=m.shape,data=m.data.tolist(),indices=m.indices.tolist(),'
                    'indptr=m.indptr.tolist(),metadata=f.metadata()),ensure_ascii=False))')
                result = subprocess.run([sys.executable, '-c', program, str(p)], cwd=root,
                                        capture_output=True, text=True, encoding='utf-8', check=True)
                got = json.loads(result.stdout)
                self.assertEqual(got['shape'], list(expected.shape))
                self.assertEqual(got['data'], expected.data.tolist())
                self.assertEqual(got['indices'], expected.indices.tolist())
                self.assertEqual(got['indptr'], expected.indptr.tolist())
                self.assertEqual(got['metadata'], f.metadata())


if __name__ == '__main__':
    unittest.main()
