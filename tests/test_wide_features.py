"""T27의 수기 산식·train-only 정리·저장 복원 합성 검사. 연구 자료는 읽지 않는다."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import joblib
import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'tests'))
from test_modeling import synthetic
from experiments.wide_features import features as f


def hand_frame():
    X = synthetic(6)[0]
    for c in f.BINARY:
        X[c] = np.zeros(6, dtype=np.uint8)
    for c, values in {'age':[995,0,10,20,995,30], 'count_floors_pre_eq':[0,1,2,3,4,5],
                      'height_percentage':[2,4,6,8,10,12], 'area_percentage':[0,2,4,6,8,10],
                      'count_families':[0,1,2,3,4,5]}.items():
        X[c] = np.asarray(values, dtype=np.int64)
    X['geo_level_1_id'] = ['1','1','1','2','2','2']
    for i, values in enumerate([[1,1,0,1,0,0],[1,0,1,1,0,0],[0,0,0,1,1,0]]):
        X[f.MATERIAL_COLUMNS[i]] = np.asarray(values, dtype=np.uint8)
    X['has_secondary_use'] = np.ones(6, dtype=np.uint8)
    for c in f.SECONDARY_DETAIL[:2]:
        X[c] = np.asarray([1,0,0,0,0,0], dtype=np.uint8)
    return X


class WideFeaturesTests(unittest.TestCase):
    def test_frozen_schema_and_candidate_counts_order(self):
        from preprocess import FEATURES, SUPERSTRUCTURE, SECONDARY_DETAIL
        self.assertEqual(f.FEATURES, FEATURES)
        self.assertEqual(f.MATERIAL_COLUMNS, tuple(SUPERSTRUCTURE))
        self.assertEqual(f.SECONDARY_DETAIL, SECONDARY_DETAIL)
        self.assertEqual(len(f.NUMERIC_NAMES), 95)
        self.assertEqual(len(set(f.NUMERIC_NAMES)), 95)
        self.assertEqual(f.NUMERIC_NAMES[:4], ('age_clean__x__floors','age_clean__x__height',
                                               'age_clean__x__area','age_clean__x__families'))
        self.assertEqual(sum(p[3] for p in f.PAIRS), 127)
        self.assertEqual(f.NUMERIC_NAMES[-6:], ('geo1_median_count_floors_pre_eq','geo1_median_height_percentage',
            'geo1_median_area_percentage','geo1_difference_count_floors_pre_eq',
            'geo1_difference_height_percentage','geo1_difference_area_percentage'))

    def test_numeric_95_hand_values_and_zero_denominator_convention(self):
        X = hand_frame(); feature = f.WideFeatures().fit(X)
        groups, values = f._raw(X)
        raw = feature._numeric(X, groups, values)
        self.assertEqual(feature.median_, 15.)
        np.testing.assert_array_equal(raw[0,:10], [0,30,0,0,0,0,0,0,0,0])
        np.testing.assert_array_equal(raw[1,:10], [0,0,0,0,4,2,1,8,4,2])
        np.testing.assert_array_equal(raw[0,10:16], [0,0,0,0,0,0])
        np.testing.assert_array_equal(raw[1,10:16], [4,2,1,2,.5,2])
        np.testing.assert_array_equal(raw[0,16:19], [1,1,1])
        self.assertEqual(raw[0,19], 1.)
        np.testing.assert_array_equal(raw[0,74:85], np.zeros(11))
        np.testing.assert_array_equal(raw[0,85:89], [2,0,1,2])
        np.testing.assert_array_equal(raw[0,89:95], [1,4,2,-1,-2,-2])
        self.assertEqual(raw.dtype, np.float64)

    def test_geo_odd_even_self_median_and_unseen_global_fallback(self):
        X = hand_frame(); feature = f.WideFeatures().fit(X)
        np.testing.assert_array_equal(feature.global_medians_, [2.5,7,5])
        np.testing.assert_array_equal(feature.group_medians_[1], [1,4,2])
        np.testing.assert_array_equal(feature.group_medians_[2], [4,10,8])
        valid = X.iloc[[0]].copy(); valid['geo_level_1_id'] = '999'
        groups, values = f._raw(valid)
        raw = feature._numeric(valid, groups, values)
        np.testing.assert_array_equal(raw[0,89:92], [2.5,7,5])
        np.testing.assert_array_equal(raw[0,92:95], [-2.5,-5,-5])
        self.assertEqual(feature.unknown_counts(valid)['geo_level_1_id'], dict(count=1,rate=1.,rows=1))
        even = X.iloc[:4].copy(); even['geo_level_1_id'] = '1'
        self.assertEqual(f.WideFeatures().fit(even).group_medians_[1].tolist(), [1.5,5.,3.])

    def test_age_median_matches_unchanged_c_including_zero_and_ignores_y(self):
        from experiments.lgbm_native_cat.run import make_preprocessor
        X = synthetic(90)[0]; X['age'] = 995
        X.iloc[:4, X.columns.get_loc('age')] = np.array([0,0,10,20], dtype=X.age.dtype)
        class UnreadableY:
            def __array__(self, *a, **k): raise AssertionError('y를 읽으면 안 됩니다.')
        feature = f.WideFeatures().fit(X, UnreadableY())
        base = make_preprocessor('O').fit(X)
        self.assertEqual(feature.median_, base['age995'].median_)
        self.assertEqual(feature.median_, 5.)
        with self.assertRaises(ValueError): f.WideFeatures().fit(X.assign(age=995))

    def test_canonical_filter_zero_nonzero_constant_duplicate_rare_and_float32_tie(self):
        block = np.column_stack([[0,0,0,0],[5,5,5,5],[1,0,-1,0],[1,0,-1,0],
                                 [1,0,0,-1],[1+1e-9,0,-1-1e-9,0],[0,0,0,7],
                                 [1,1+1e-9,1,1+1e-9]])
        keep, records = f.filter_candidates(block, [f'c{i}' for i in range(8)])
        np.testing.assert_array_equal(keep, [2,4,6])
        self.assertEqual([r['removal_reason'] for r in records], ['constant_zero','constant_nonzero',None,
            'exact_duplicate',None,'exact_duplicate',None,'constant_nonzero'])
        self.assertEqual(records[3]['duplicate_of_index'], 2)
        self.assertEqual(records[5]['duplicate_of_index'], 2)
        self.assertEqual(records[2]['support'], 2)  # 음수도 비0 지원수에 포함
        self.assertEqual(records[6]['support'], 1)  # 희귀 빈도로 제거하지 않음
        self.assertEqual(records[7]['constant_value'], 1.)
        self.assertEqual(records[6]['retained_index'], 2)

    def test_sparse_canonical_cancels_explicit_zero_and_duplicate_entries(self):
        block = sparse.coo_matrix(([1.,-1.,-0.,2.,2.], ([0,0,1,1,2],[0,0,0,1,1])),shape=(3,2))
        canonical = f.canonical_csr(block)
        self.assertTrue(canonical.has_canonical_format)
        self.assertEqual(canonical.nnz, 2)
        keep, records = f.filter_candidates(canonical, ['zero','variable'])
        np.testing.assert_array_equal(keep, [1])
        self.assertEqual(records[0]['removal_reason'], 'constant_zero')

    def test_transform_keeps_only_training_indices_without_refitting_filter(self):
        X = hand_frame(); feature = f.WideFeatures().fit(X)
        state = copy.deepcopy(feature.metadata())
        valid = X.copy(); valid['height_percentage'] = np.array([0,0,0,0,0,0],dtype=np.int64)
        valid['age'] = np.array([900,900,900,900,900,900],dtype=np.int64)
        output = feature.transform(valid)
        groups, values = f._raw(valid)
        expected = feature._candidates(valid, groups, values)[:, feature.keep_indices_]
        self.assertEqual((output != expected).nnz, 0)
        self.assertEqual(feature.metadata(), state)
        self.assertEqual(output.shape[1], len(state['keep_indices']))

    def test_tuple_serialization_has_no_delimiter_collision_and_unknown_is_zero(self):
        X = hand_frame()
        X['foundation_type'] = ['a|b','a','a|b','a','a|b','a']
        X['roof_type'] = ['c','b|c','c','b|c','c','b|c']
        feature = f.WideFeatures().fit(X)
        vocab = feature.encoder_.categories_[0].tolist()
        self.assertEqual(len(vocab), 2)
        self.assertEqual({tuple(json.loads(v)) for v in vocab}, {('a|b','c'),('a','b|c')})
        valid = X.iloc[:1].copy(); valid['foundation_type'] = '미관측,[]'
        encoded = feature.encoder_.transform(f._pair_values(valid))
        self.assertEqual(encoded[:, :len(vocab)].nnz, 0)
        self.assertEqual(feature.unknown_counts(valid)['foundation_roof'], dict(count=1,rate=1.,rows=1))
        self.assertEqual(len(feature.unknown_counts(valid)), 7)

    def test_per_pair_vocabulary_caps_reject_more_than_fixed_upper_bound(self):
        X = synthetic(32)[0]
        X['foundation_type'] = [str(i) for i in range(32)]
        X['roof_type'] = 'q'
        with self.assertRaisesRegex(ValueError, '상한'): f.WideFeatures().fit(X)

    def test_train_and_valid_whitespace_normalization_matches_original_c_categories(self):
        X = synthetic(90)[0]
        padded = X.copy(deep=True)
        for column in f.TEXT:
            padded[column] = padded[column].map(lambda value: ' \t'+value+'  ')
        clean_feature = f.WideFeatures().fit(X.iloc[:72])
        padded_feature = f.WideFeatures().fit(padded.iloc[:72])
        self.assertEqual(clean_feature.metadata(), padded_feature.metadata())
        clean = clean_feature.transform(X.iloc[72:])
        for feature, frame in [(clean_feature,padded.iloc[72:]),(padded_feature,X.iloc[72:]),
                               (padded_feature,padded.iloc[72:])]:
            self.assertEqual((feature.transform(frame) != clean).nnz, 0)
            self.assertEqual(feature.unknown_counts(frame), clean_feature.unknown_counts(X.iloc[72:]))
        self.assertTrue(padded['foundation_type'].str.startswith(' \t').all())

    def test_csr_metadata_candidate_decisions_and_support_are_consistent(self):
        X = synthetic(90)[0]; feature = f.WideFeatures().fit(X)
        result, state = feature.transform(X), feature.metadata()
        self.assertTrue(sparse.isspmatrix_csr(result) and result.has_canonical_format)
        self.assertEqual(result.dtype, np.float32)
        self.assertTrue(np.isfinite(result.data).all())
        self.assertEqual(state['candidate_logical_count'], 101)
        self.assertEqual(state['candidate_numeric_count'], 95)
        self.assertEqual(state['candidate_categorical_count'], 6)
        self.assertEqual(state['candidate_encoded_count'], 95+sum(len(p['vocabulary']) for p in state['pairs']))
        self.assertEqual(state['retained_count'], result.shape[1])
        self.assertEqual([r['index'] for r in state['candidates'] if r['removal_reason'] is None],state['keep_indices'])
        for j, source in enumerate(state['keep_indices']):
            self.assertEqual(state['candidates'][source]['support'], result.getcol(j).nnz)
        self.assertFalse(state['filtering']['compare_with_C'])
        self.assertIsNone(state['filtering']['rare_frequency_threshold'])
        json.dumps(state, ensure_ascii=False, allow_nan=False)

    def test_input_and_metadata_copies_do_not_mutate_fitted_state(self):
        X = hand_frame(); before = X.copy(deep=True)
        feature = f.WideFeatures().fit(X)
        state = feature.metadata(); pristine = copy.deepcopy(state)
        feature.transform(X)
        self.assertTrue(X.equals(before))
        state['pairs'][0]['vocabulary'][0].append('mutation')
        state['candidates'][0]['support'] = -1
        state['geo_context']['groups'][0]['medians'][0] = -1
        self.assertEqual(feature.metadata(), pristine)

    def test_row_order_unknown_and_validation_values_never_change_training_state(self):
        X = synthetic(90)[0]; feature = f.WideFeatures().fit(X.iloc[:72])
        state = feature.metadata(); valid = X.iloc[72:].copy()
        order = np.arange(18)[::-1]
        expected = feature.transform(valid)[order]
        self.assertEqual((feature.transform(valid.iloc[order]) != expected).nnz,0)
        valid['geo_level_1_id'] = '999999'
        valid['foundation_type'] = '미관측'
        valid['age'] = 995
        feature.transform(valid); feature.unknown_counts(valid)
        self.assertEqual(feature.metadata(), state)

    def test_empty_transform_zero_width_and_schema_arm_changes(self):
        X = hand_frame(); feature = f.WideFeatures().fit(X)
        self.assertEqual(feature.transform(X.iloc[:0]).shape, (0,len(feature.keep_indices_)))
        self.assertTrue(all(v == dict(count=0,rate=None,rows=0) for v in feature.unknown_counts(X.iloc[:0]).values()))
        with self.assertRaises(ValueError): f.WideFeatures().fit(X.iloc[:0])
        for changed in [X.iloc[:,::-1],X.drop(columns='age'),X.assign(extra=1)]:
            with self.assertRaises(ValueError): feature.transform(changed)
        feature.arm = 'OTHER'
        with self.assertRaises(ValueError): feature.transform(X)
        with self.assertRaises(ValueError): feature.metadata()
        one = f.WideFeatures().fit(X.iloc[[1]])
        self.assertEqual(one.transform(X.iloc[:2]).shape, (2,0))

    def test_invalid_binary_fraction_negative_missing_nonfinite_and_overflow(self):
        X = hand_frame(); feature = f.WideFeatures().fit(X)
        for column, value in [('age',np.inf),('age',-1),('age',1.5),('count_families',np.nan),
                              (f.MATERIAL_COLUMNS[0],2),('foundation_type','')]:
            invalid = X.copy()
            if column in f.NUMERIC: invalid[column] = invalid[column].astype(float)
            invalid.loc[0,column] = value
            with self.subTest(column=column,value=value),self.assertRaises(ValueError): feature.transform(invalid)
        huge = X.copy(); huge['age'] = np.full(6,1e200)
        with self.assertRaises(ValueError): feature.transform(huge)
        with self.assertRaises(ValueError): f.canonical_csr(np.array([[1e100]]))

    def test_joblib_fresh_process_matches_csr_arrays_names_and_full_metadata(self):
        X = synthetic(90)[0]; feature = f.WideFeatures().fit(X.iloc[:72]); valid = X.iloc[72:]
        def evidence(matrix):
            import hashlib
            return dict(shape=list(matrix.shape),arrays={k:hashlib.sha256(getattr(matrix,k).tobytes()).hexdigest()
                        for k in ['data','indices','indptr']})
        expected = dict(csr=evidence(feature.transform(valid)),state=feature.metadata(),unknown=feature.unknown_counts(valid))
        with tempfile.TemporaryDirectory() as d:
            root = Path(d);joblib.dump(feature,root/'feature.joblib');joblib.dump(valid,root/'input.joblib')
            script = ('import joblib,json,hashlib,sys;f=joblib.load(sys.argv[1]);x=joblib.load(sys.argv[2]);m=f.transform(x);'
                      'e=dict(shape=list(m.shape),arrays={k:hashlib.sha256(getattr(m,k).tobytes()).hexdigest() for k in ["data","indices","indptr"]});'
                      'print(json.dumps(dict(csr=e,state=f.metadata(),unknown=f.unknown_counts(x))))')
            proc = subprocess.run([sys.executable,'-c',script,str(root/'feature.joblib'),str(root/'input.joblib')],
                                  cwd=ROOT,capture_output=True,text=True,encoding='utf8',check=True)
            self.assertEqual(json.loads(proc.stdout),expected)


if __name__ == '__main__':
    unittest.main()
