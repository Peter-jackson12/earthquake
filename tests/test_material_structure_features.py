"""T22 추가 블록의 고정 공식·학습 어휘·희소 표현을 합성 행으로 검증한다."""
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

from experiments.material_structure_interactions.features import (
    ARMS, MATERIAL_COLUMNS, MATERIAL_PAIRS, MaterialStructureInteractions,
)


def materials():
    x = pd.DataFrame(0, index=[19, 3, 44], columns=list(MATERIAL_COLUMNS), dtype=np.uint8)
    x.loc[19, [MATERIAL_COLUMNS[0], MATERIAL_COLUMNS[1]]] = 1
    x.loc[3, [MATERIAL_COLUMNS[0], MATERIAL_COLUMNS[2], MATERIAL_COLUMNS[10]]] = 1
    return x


def structure():
    return pd.DataFrame(dict(foundation_type=['b', 'a', 'a'], other_floor_type=['z', 'y', 'z'],
                             count_floors_pre_eq=[2, 3, 0]), index=[19, 3, 44])


class MaterialStructureFeaturesTest(unittest.TestCase):
    def test_fixed_material_order_55_unordered_pairs_and_hand_values(self):
        self.assertEqual(len(MATERIAL_COLUMNS), 11)
        self.assertEqual(MATERIAL_COLUMNS[0], 'has_superstructure_adobe_mud')
        self.assertEqual(MATERIAL_COLUMNS[-1], 'has_superstructure_other')
        self.assertEqual(len(MATERIAL_PAIRS), 55)
        self.assertEqual(len(set(MATERIAL_PAIRS)), 55)
        self.assertEqual(MATERIAL_PAIRS[0], (MATERIAL_COLUMNS[0], MATERIAL_COLUMNS[1]))
        self.assertEqual(MATERIAL_PAIRS[-1], (MATERIAL_COLUMNS[9], MATERIAL_COLUMNS[10]))
        x = materials()
        f = MaterialStructureInteractions(ARMS[0]).fit(x)
        actual = f.transform(x).toarray()
        expected = np.zeros((3, 55), dtype=np.float32)
        expected[0, 0] = 1
        for pair in [(0, 2), (0, 10), (2, 10)]:
            expected[1, MATERIAL_PAIRS.index((MATERIAL_COLUMNS[pair[0]], MATERIAL_COLUMNS[pair[1]]))] = 1
        np.testing.assert_array_equal(actual, expected)

    def test_all_pairs_retained_even_when_never_active_in_training(self):
        train = materials()*0
        valid = materials()*0+1
        f = MaterialStructureInteractions().fit(train)
        matrix = f.transform(valid)
        self.assertEqual(matrix.shape, (3, 55))
        self.assertEqual(matrix.nnz, 165)
        np.testing.assert_array_equal(matrix.data, np.ones(165, dtype=np.float32))
        self.assertEqual(f.unknown_counts(valid), {})

    def test_structure_vocabulary_sort_and_raw_floor_multiplication(self):
        x = structure()
        f = MaterialStructureInteractions(ARMS[1]).fit(x)
        self.assertEqual(f.metadata()['categories'], [
            dict(column='foundation_type', vocabulary=['a', 'b']),
            dict(column='other_floor_type', vocabulary=['y', 'z'])])
        expected = np.array([[0, 2, 0, 2], [3, 0, 3, 0], [0, 0, 0, 0]], dtype=np.float32)
        np.testing.assert_array_equal(f.transform(x).toarray(), expected)
        self.assertEqual(f.transform(x)[2].nnz, 0)
        self.assertEqual(f.metadata()['scaling'], 'none')
        self.assertEqual(f.get_feature_names_out().tolist(), [
            'foundation_type__"a"__x_floors', 'foundation_type__"b"__x_floors',
            'other_floor_type__"y"__x_floors', 'other_floor_type__"z"__x_floors'])

    def test_zero_floor_training_rows_still_define_vocabulary(self):
        x = structure()
        x['count_floors_pre_eq'] = 0
        f = MaterialStructureInteractions(ARMS[1]).fit(x)
        valid = x.copy(deep=True)
        valid['count_floors_pre_eq'] = 4
        np.testing.assert_array_equal(f.transform(valid).sum(axis=1).A.ravel(), [8, 8, 8])

    def test_unknown_categories_zero_and_counts_use_all_rows(self):
        f = MaterialStructureInteractions(ARMS[1]).fit(structure())
        valid = pd.DataFrame(dict(foundation_type=['new', 'a', 'new'],
            other_floor_type=['z', 'new', 'new'], count_floors_pre_eq=[4, 3, 0]))
        np.testing.assert_array_equal(f.transform(valid).toarray(),
            np.array([[0, 0, 0, 4], [3, 0, 0, 0], [0, 0, 0, 0]], dtype=np.float32))
        self.assertEqual(f.unknown_counts(valid), dict(
            foundation_type=dict(count=2, rate=2/3, rows=3),
            other_floor_type=dict(count=2, rate=2/3, rows=3)))

    def test_exact_binary_rejects_invalid_fit_and_transform(self):
        fitted = MaterialStructureInteractions().fit(materials())
        for value in [2, -1, .5, np.nan, np.inf]:
            x = materials().astype(float)
            x.iloc[0, 0] = value
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    MaterialStructureInteractions().fit(x)
                with self.assertRaises(ValueError):
                    fitted.transform(x)

    def test_nonnegative_integer_floors_and_category_validation(self):
        fitted = MaterialStructureInteractions(ARMS[1]).fit(structure())
        for value in [-1, .5, np.nan, np.inf]:
            x = structure().astype({ 'count_floors_pre_eq': float })
            x.iloc[0, 2] = value
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    MaterialStructureInteractions(ARMS[1]).fit(x)
                with self.assertRaises(ValueError):
                    fitted.transform(x)
        for value in [None, np.nan, 7]:
            x = structure()
            x.iloc[0, 0] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                MaterialStructureInteractions(ARMS[1]).fit(x)

    def test_input_and_fit_state_unchanged_row_order_preserved(self):
        for arm, x in zip(ARMS, [materials(), structure()]):
            before = x.copy(deep=True)
            f = MaterialStructureInteractions(arm).fit(x)
            state = copy.deepcopy(f.metadata())
            expected = f.transform(x)[[2, 0, 1]]
            actual = f.transform(x.iloc[[2, 0, 1]])
            self.assertEqual((expected != actual).nnz, 0)
            self.assertEqual(state, f.metadata())
            pd.testing.assert_frame_equal(x, before)
            self.assertTrue(sparse.isspmatrix_csr(actual))
            self.assertEqual(actual.dtype, np.float32)
            self.assertTrue(np.isfinite(actual.data).all())

    def test_y_is_not_used(self):
        class UnreadableTarget:
            def __array__(self, *args, **kwargs):
                raise AssertionError('정답을 읽었습니다.')
        for arm, x in zip(ARMS, [materials(), structure()]):
            f = MaterialStructureInteractions(arm).fit(x, UnreadableTarget())
            self.assertEqual(f.metadata()['n_train_rows'], len(x))

    def test_schema_missing_reordered_duplicates_and_changed_arm(self):
        for arm, x in zip(ARMS, [materials(), structure()]):
            f = MaterialStructureInteractions(arm).fit(x)
            for invalid in [x.iloc[:, :-1], x.iloc[:, ::-1], pd.concat([x, x.iloc[:, :1]], axis=1)]:
                with self.assertRaises(ValueError):
                    f.transform(invalid)
            with self.assertRaises(ValueError):
                f.get_feature_names_out(['wrong'])
            f.arm = ARMS[1] if arm == ARMS[0] else ARMS[0]
            with self.assertRaises(ValueError):
                f.metadata()
        with self.assertRaises(ValueError):
            MaterialStructureInteractions('unknown').fit(materials())

    def test_empty_fit_rejected_empty_transform_retains_width(self):
        for arm, x in zip(ARMS, [materials(), structure()]):
            with self.assertRaises(ValueError):
                MaterialStructureInteractions(arm).fit(x.iloc[:0])
            f = MaterialStructureInteractions(arm).fit(x)
            matrix = f.transform(x.iloc[:0])
            self.assertEqual(matrix.shape, (0, len(f.get_feature_names_out())))
            self.assertTrue(sparse.isspmatrix_csr(matrix))
            self.assertEqual(matrix.dtype, np.float32)
            if arm == ARMS[1]:
                self.assertEqual(f.unknown_counts(x.iloc[:0])['foundation_type'],
                                 dict(count=0, rate=None, rows=0))

    def test_metadata_json_roundtrip_copies_and_clone(self):
        for arm, x in zip(ARMS, [materials(), structure()]):
            f = MaterialStructureInteractions(arm).fit(x)
            state = f.metadata()
            self.assertEqual(json.loads(json.dumps(state, ensure_ascii=False, allow_nan=False)), state)
            state['input_columns'].append('mutated')
            state['pairs'].append(['mutated', 'mutated'])
            self.assertNotIn('mutated', f.metadata()['input_columns'])
            self.assertNotIn(['mutated', 'mutated'], f.metadata()['pairs'])
            other = clone(f)
            with self.assertRaises(NotFittedError):
                other.transform(x)

    def test_fresh_process_pickle_without_execution_module_import(self):
        root = Path(__file__).resolve().parents[1]
        for arm, x in zip(ARMS, [materials(), structure()]):
            f = MaterialStructureInteractions(arm).fit(x)
            expected = f.transform(x)
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder)/'synthetic.pkl'
                path.write_bytes(pickle.dumps(dict(transformer=f, frame=x)))
                code = ('import json,pickle,sys; from pathlib import Path; '
                    'x=pickle.loads(Path(sys.argv[1]).read_bytes()); f=x["transformer"]; '
                    'a=f.transform(x["frame"]); '
                    'assert "experiments.material_structure_interactions.run" not in sys.modules; '
                    'assert "experiments.feature_interactions.run" not in sys.modules; '
                    'print(json.dumps(dict(shape=a.shape,data=a.data.tolist(),indices=a.indices.tolist(),'
                    'indptr=a.indptr.tolist(),metadata=f.metadata()),ensure_ascii=False))')
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
