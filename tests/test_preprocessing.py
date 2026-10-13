"""Unit/integrity tests only. No predictive model is fitted by this test suite."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from preprocess import (ID, TARGET, FEATURES, NUMERIC, BINARY, CATEGORICAL, GEO, TEXT,
                        make_preprocessor, make_model_pipeline, normalize_features,
                        load_development, make_splits, sha256_file)


@unittest.skipUnless((ROOT / "data/development.csv").is_file(), "Private prepared data not installed; integration suite skipped.")
class PreprocessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_dir = ROOT / 'data'
        cls.X, cls.y, cls.ids, cls.cv = load_development(cls.data_dir)
        cls.audit = json.loads((cls.data_dir / 'qa/audit.json').read_text())
        cls.train_small = cls.X.iloc[:400].copy()
        cls.valid_small = cls.X.iloc[400:450].copy()

    def test_feature_partition_is_exact(self):
        self.assertEqual(len(FEATURES), 38)
        self.assertEqual((len(NUMERIC), len(BINARY), len(CATEGORICAL)), (5, 22, 11))
        self.assertEqual(len(set(NUMERIC + BINARY + CATEGORICAL)), 38)
        self.assertEqual(set(FEATURES), set(NUMERIC + BINARY + CATEGORICAL))

    def test_actual_counts_and_no_loss(self):
        self.assertEqual(len(self.X), 208480)
        self.assertEqual(self.audit['holdout_rows'], 52121)
        self.assertEqual(self.audit['deleted_rows'], 0)
        self.assertEqual(self.audit['imputed_cells'], 0)
        self.assertEqual(self.audit['duplicate_ids'], 0)
        self.assertFalse(self.audit['predictive_model_trained'])

    def test_geography_is_categorical_text(self):
        self.assertTrue(all(isinstance(v, str) for c in GEO for v in self.X[c].head(10)))

    def test_fixed_folds_cover_each_row_once(self):
        coverage = np.zeros(len(self.X), dtype=int)
        for train, valid in self.cv.split():
            self.assertEqual(len(train), 166784)
            self.assertEqual(len(valid), 41696)
            self.assertEqual(len(np.intersect1d(train, valid)), 0)
            self.assertEqual(set(self.y.iloc[valid]), {1, 2, 3})
            coverage[valid] += 1
        np.testing.assert_array_equal(coverage, np.ones(len(self.X), dtype=int))

    def test_holdout_and_development_ids_disjoint(self):
        manifest = pd.read_csv(self.data_dir / 'split_manifest.csv')
        a = set(manifest.loc[manifest.split.eq('development'), ID])
        b = set(manifest.loc[manifest.split.eq('final_holdout'), ID])
        self.assertFalse(a & b)
        self.assertEqual(len(a | b), 260601)
        self.assertTrue(manifest.loc[manifest.split.eq('final_holdout'), 'cv_fold'].eq(-1).all())

    def test_reordered_input_preserves_split(self):
        small = self.train_small.copy()
        small.insert(0, ID, self.ids.iloc[:len(small)].to_numpy())
        small[TARGET] = self.y.iloc[:len(small)].to_numpy()
        _, _, a = make_splits(small)
        _, _, b = make_splits(small.iloc[::-1])
        pd.testing.assert_frame_equal(a, b)

    def test_sparse_float32_train_validation_consistency(self):
        pre = make_preprocessor()
        a = pre.fit_transform(self.train_small)
        b = pre.transform(self.valid_small)
        self.assertTrue(sparse.issparse(a) and sparse.issparse(b))
        self.assertEqual(a.dtype, np.float32)
        self.assertEqual(a.shape[1], b.shape[1])
        self.assertEqual(a.shape[1], len(pre.get_feature_names_out()))
        self.assertTrue(np.isfinite(a.data).all() and np.isfinite(b.data).all())

    def test_unknown_categories_do_not_create_columns_or_refit(self):
        pre = make_preprocessor().fit(self.train_small)
        ohe = pre['columns'].named_transformers_['categorical']
        before = [set(c) for c in ohe.categories_]
        new = self.valid_small.iloc[:1].copy()
        for c in GEO:
            new[c] = '999999'
        for c in TEXT:
            new[c] = 'UNSEEN_TEST_CATEGORY'
        result = pre.transform(new)
        # All categorical blocks are zero for this deliberately novel row.
        self.assertEqual(result[:, len(NUMERIC) + len(BINARY):].nnz, 0)
        self.assertEqual(before, [set(c) for c in ohe.categories_])

    def test_vocabulary_contains_only_train_values(self):
        pre = make_preprocessor().fit(self.train_small)
        ohe = pre['columns'].named_transformers_['categorical']
        for c, learned in zip(CATEGORICAL, ohe.categories_):
            self.assertEqual(set(learned), set(self.train_small[c]))

    def test_scaler_fits_train_only(self):
        pre = make_preprocessor(scale_numeric=True).fit(self.train_small)
        scaler = pre['columns'].named_transformers_['numeric']['scale']
        np.testing.assert_allclose(scaler.mean_, self.train_small[NUMERIC].to_numpy(dtype=float).mean(axis=0))
        before = scaler.mean_.copy()
        new = self.valid_small.copy()
        new['age'] = 995
        pre.transform(new)
        np.testing.assert_array_equal(before, scaler.mean_)

    def test_identifier_and_target_are_rejected_as_inputs(self):
        for extra in [ID, TARGET]:
            bad = self.train_small.copy()
            bad[extra] = 1
            with self.assertRaises(ValueError):
                normalize_features(bad)

    def test_invalid_binary_rejected(self):
        bad = self.train_small.copy()
        bad.loc[bad.index[0], BINARY[0]] = 2
        with self.assertRaises(ValueError):
            normalize_features(bad)

    def test_missing_and_blank_rejected(self):
        for value in [None, '   ']:
            bad = self.train_small.copy()
            bad.loc[bad.index[0], TEXT[0]] = value
            with self.assertRaises(ValueError):
                normalize_features(bad)

    def test_fractional_infinite_negative_rejected(self):
        for value in [1.5, np.inf, -1]:
            bad = self.train_small.copy()
            bad['age'] = bad['age'].astype(float)
            bad.loc[bad.index[0], 'age'] = value
            with self.assertRaises(ValueError):
                normalize_features(bad)

    def test_special_values_are_preserved(self):
        original = self.train_small.iloc[:2].copy()
        original['age'] = [0, 995]
        original['count_families'] = 0
        normalized = normalize_features(original)
        self.assertEqual(normalized.age.tolist(), [0, 995])
        self.assertEqual(normalized.count_families.tolist(), [0, 0])

    def test_pipeline_clone_is_unfitted(self):
        model = make_model_pipeline(RandomForestClassifier(n_estimators=2, random_state=42))
        copied = clone(model)
        self.assertFalse(hasattr(copied['model'], 'estimators_'))
        self.assertNotIn(ID, copied['preprocess']['columns'].transformers[0][2])

    def test_actual_five_fold_encoding_passes(self):
        checks = json.loads((self.data_dir / 'qa/fold_preprocessing_validation.json').read_text())
        self.assertEqual(len(checks), 5)
        self.assertTrue(all(row['status'] == 'PASS' and row['train_only_vocabulary_checked'] for row in checks))

    def test_published_data_checksums(self):
        for relative, expected in json.loads((self.data_dir / 'checksums.json').read_text()).items():
            with self.subTest(file=relative):
                self.assertEqual(sha256_file(self.data_dir / relative), expected)

    def test_manifest_alignment_survives_row_shuffle(self):
        manifest = pd.read_csv(self.data_dir / 'split_manifest.csv').set_index(ID)
        shuffled_ids = self.ids.sample(frac=1, random_state=7)
        expected = manifest.loc[self.ids, 'cv_fold']
        actual = manifest.loc[shuffled_ids, 'cv_fold'].reindex(self.ids)
        np.testing.assert_array_equal(expected.to_numpy(), actual.to_numpy())


if __name__ == '__main__':
    unittest.main(verbosity=2)
