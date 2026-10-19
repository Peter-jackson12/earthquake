"""Regression tests for validation-before-cast in load_development.

Synthetic file fixtures only. No model training and no data mutations outside
TemporaryDirectory; run together with the existing 19 tests.
"""
from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from preprocess import (
    ALLOWED_TEXT, BINARY, GEO, ID, NUMERIC, SOURCE_COLUMNS, TARGET,
    load_development,
)


class CvFoldValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        n = 50
        values = {ID: np.arange(1, n + 1)}
        values.update({col: np.arange(n) % 3 for col in GEO})
        values.update(dict(zip(NUMERIC, [2, 20, 8, 5, 1])))
        values.update({col: 0 for col in BINARY})
        values.update({col: accepted[0] for col, accepted in ALLOWED_TEXT.items()})
        values[TARGET] = np.arange(n) % 3 + 1
        self.frame = pd.DataFrame(values).loc[:, SOURCE_COLUMNS]
        self.frame.to_csv(self.path / 'development.csv', index=False)
        self.expected = np.tile(np.arange(5), n // 5)
        self.manifest = pd.DataFrame({
            ID: np.arange(1, n + 1), 'split': 'development',
            'cv_fold': self.expected.astype(object),
        })
        # Loader must tolerate holdout's legitimate -1 and must not open labels.
        self.manifest.loc[n] = [9999, 'final_holdout', -1]
        self.write_manifest(self.manifest)

    def write_manifest(self, manifest):
        manifest.to_csv(self.path / 'split_manifest.csv', index=False)

    def test_valid_integer_folds_preserved(self):
        X, y, ids, cv = load_development(self.path)
        self.assertEqual(X.shape, (50, 38))
        np.testing.assert_array_equal(cv.test_fold, self.expected)
        np.testing.assert_array_equal(ids.to_numpy(), self.frame[ID].to_numpy())
        self.assertEqual(set(y), {1, 2, 3})

    def test_integral_decimal_representation_accepted(self):
        manifest = self.manifest.copy()
        manifest['cv_fold'] = manifest['cv_fold'].astype(float)
        self.write_manifest(manifest)
        _, _, _, cv = load_development(self.path)
        np.testing.assert_array_equal(cv.test_fold, self.expected)

    def test_fractional_folds_rejected_before_cast(self):
        for value in [0.5, 1.5, 4.9, -0.5]:
            with self.subTest(value=value):
                manifest = self.manifest.copy()
                manifest.loc[0, 'cv_fold'] = value
                self.write_manifest(manifest)
                with self.assertRaisesRegex(ValueError, 'cv_fold'):
                    load_development(self.path)

    def test_missing_nonfinite_and_non_numeric_folds_rejected(self):
        for value in [None, '', np.inf, -np.inf, 'not-a-fold']:
            with self.subTest(value=str(value)):
                manifest = self.manifest.copy()
                manifest.loc[0, 'cv_fold'] = value
                self.write_manifest(manifest)
                with self.assertRaises(ValueError):
                    load_development(self.path)

    def test_negative_out_of_range_and_missing_fold_rejected(self):
        for value in [-1, 5, 2**31]:
            with self.subTest(value=value):
                manifest = self.manifest.copy()
                manifest.loc[0, 'cv_fold'] = value
                self.write_manifest(manifest)
                with self.assertRaises(ValueError):
                    load_development(self.path)
        manifest = self.manifest.copy()
        manifest.loc[manifest['cv_fold'].eq(4), 'cv_fold'] = 3
        self.write_manifest(manifest)
        with self.assertRaisesRegex(ValueError, 'five development folds'):
            load_development(self.path)

    def test_holdout_minus_one_does_not_enter_cv(self):
        self.assertFalse((self.path / 'final_holdout').exists())
        _, _, ids, cv = load_development(self.path)
        self.assertNotIn(9999, set(ids))
        self.assertNotIn(-1, set(cv.test_fold))
        self.assertEqual(cv.get_n_splits(), 5)

    def test_manifest_and_data_row_shuffle_preserves_id_assignment(self):
        shuffled = self.frame.sample(frac=1, random_state=17)
        shuffled.to_csv(self.path / 'development.csv', index=False)
        self.write_manifest(self.manifest.sample(frac=1, random_state=23))
        _, _, ids, cv = load_development(self.path)
        by_id = pd.Series(self.expected, index=self.frame[ID])
        np.testing.assert_array_equal(cv.test_fold, by_id.loc[ids].to_numpy())


if __name__ == '__main__':
    unittest.main(verbosity=2)
