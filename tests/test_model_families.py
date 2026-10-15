"""Synthetic contracts only; no earthquake performance claims and no holdout access."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import sys
from unittest.mock import patch
import numpy as np
import pandas as pd
from scipy import sparse
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from preprocess import FEATURES, GEO, NUMERIC, BINARY, TEXT, ALLOWED_TEXT, normalize_features
from experiments.age995.run import make_pipeline, Age995Treatment, preprocessor_check
if importlib.util.find_spec('psutil') is None:
    raise unittest.SkipTest('Optional model-family requirements not installed: psutil')
from experiments.model_families.run import (load_config, make_estimator, checked_probability,
                                            matrix_sha, protected_sources, run)


def synthetic(n=150):
    rng = np.random.default_rng(902)
    x = pd.DataFrame(index=np.arange(n))
    for c in GEO:
        x[c] = rng.integers(0, 7, n).astype(str)
    for c in NUMERIC:
        x[c] = rng.integers(1, 6, n)
    x['age'] = rng.choice([0, 5, 10, 15, 30, 995], n)
    for c in BINARY:
        x[c] = rng.integers(0, 2, n)
    for c in TEXT:
        x[c] = rng.choice(ALLOWED_TEXT[c], n)
    y = np.tile([1, 2, 3], (n+2)//3)[:n]
    return normalize_features(x[FEATURES]), y


class ModelFamilyContracts(unittest.TestCase):
    def setUp(self):
        self.x, self.y = synthetic()
        self.c = load_config()

    def test_ids_and_budget(self):
        self.assertEqual(len(self.c['experiment_ids']), 4)
        for p in self.c['models'].values():
            self.assertEqual(p['n_estimators'], 300)
            self.assertEqual(p['learning_rate'], .05)
            self.assertEqual(p['subsample'], 1.0)
            self.assertEqual(p['colsample_bytree'], 1.0)
            self.assertEqual(p['n_jobs'], 2)

    def test_no_stopping_weights_sampling(self):
        self.assertFalse(self.c['early_stopping'])
        self.assertIsNone(self.c['sampling'])
        self.assertIsNone(self.c['class_weight'])
        for p in self.c['models'].values():
            self.assertNotIn('early_stopping_rounds', p)
            self.assertNotIn('callbacks', p)
            self.assertNotIn('class_weight', p)

    def test_bad_config_rejected(self):
        for key, val in [('early_stopping', True), ('seed', 1), ('sampling', 'SMOTE')]:
            c = copy.deepcopy(self.c); c[key] = val
            with tempfile.TemporaryDirectory() as t:
                p = Path(t)/'c.json'; p.write_text(json.dumps(c))
                with self.assertRaises(ValueError): load_config(p)

    def test_unknown_family(self):
        with self.assertRaises(ValueError): make_estimator('LR', self.c)

    def test_inherited_sources_intact(self):
        self.assertIn('preprocess.py', protected_sources())

    def test_A_identical_to_historical_preprocessor(self):
        from preprocess import make_preprocessor
        a = make_pipeline('A')['preprocess'].fit_transform(self.x)
        b = make_preprocessor(scale_numeric=True).fit_transform(self.x)
        self.assertEqual(matrix_sha(a), matrix_sha(b))

    def test_C_train_median_including_zero(self):
        t = self.x.iloc[:4].copy(); t['age'] = [0, 10, 30, 995]
        p = Age995Treatment('C').fit(t)
        self.assertEqual(p.median_, 10)
        self.assertEqual(p.transform(t)['age_is_995'].tolist(), [0, 0, 0, 1])
        self.assertEqual(p.transform(t).age.tolist(), [0, 10, 30, 10])

    def test_fractional_median_does_not_relax_raw_schema(self):
        t = self.x.iloc[:3].copy(); t['age'] = [10, 15, 995]
        self.assertEqual(Age995Treatment('C').fit(t).median_, 12.5)
        t['age'] = [10, 12.5, 995]
        with self.assertRaises(ValueError): make_pipeline('C')['preprocess'].fit(t)

    def test_unseen_category_no_leak_and_row_preservation(self):
        for mode in ('A', 'C'):
            pre = make_pipeline(mode)['preprocess'].fit(self.x.iloc[:120])
            before = self.x.copy(deep=True)
            v = self.x.iloc[120:].copy(); v[GEO[0]] = '99999'; v['age'] = 995
            a = pre.transform(v)
            self.assertEqual(a.shape[0], 30)
            self.assertNotIn('99999', pre['columns'].named_transformers_['categorical'].categories_[0])
            preprocessor_check({'preprocess': pre}, self.x.iloc[:120], v, mode)
            pd.testing.assert_frame_equal(self.x, before)

    def test_id_label_and_flag_rejected(self):
        for extra in ['building_id', 'damage_grade', 'age_is_995']:
            bad = self.x.copy(); bad[extra] = 1
            with self.assertRaises(ValueError): make_pipeline('C')['preprocess'].fit(bad)

    def test_all995_train_rejected_for_C(self):
        bad = self.x.copy(); bad['age'] = 995
        with self.assertRaises(ValueError): make_pipeline('C')['preprocess'].fit(bad)

    def test_non_csr_rejected(self):
        with self.assertRaises(ValueError): matrix_sha(np.eye(3, dtype=np.float32))

    def test_nonfinite_csr_rejected(self):
        with self.assertRaises(ValueError): matrix_sha(sparse.csr_matrix([[np.nan]], dtype=np.float32))

    def test_matrix_mutation_detected(self):
        a = sparse.csr_matrix(np.eye(3, dtype=np.float32)); b = a.copy(); b.data[0] = 2
        self.assertNotEqual(matrix_sha(a), matrix_sha(b))

    def test_nonempty_output_rejected_before_read(self):
        with tempfile.TemporaryDirectory() as t:
            (Path(t)/'keep').write_text('unchanged')
            with self.assertRaises(FileExistsError): run(Path('absent'), Path(t), Path('absent'))
            self.assertEqual((Path(t)/'keep').read_text(), 'unchanged')

    @unittest.skipUnless(importlib.util.find_spec('xgboost') and importlib.util.find_spec('lightgbm'), 'Optional model-family dependencies not installed')
    def test_four_synthetic_fits_probability_order_and_shared_matrix(self):
        for mode in ('A', 'C'):
            pre = make_pipeline(mode)['preprocess']
            a = pre.fit_transform(self.x.iloc[:120]); b = pre.transform(self.x.iloc[120:])
            hashes = (matrix_sha(a), matrix_sha(b))
            for family in ('XGB', 'LGBM'):
                c = copy.deepcopy(self.c); c['models'][family]['n_estimators'] = 3
                model = make_estimator(family, c)
                model.fit(a, self.y[:120]-1)
                p = checked_probability(model, b)
                self.assertEqual(p.shape, (30, 3))
                self.assertEqual(hashes, (matrix_sha(a), matrix_sha(b)))
                self.assertTrue(np.isin(p.argmax(1)+1, [1,2,3]).all())
                with patch.object(model, 'predict_proba', return_value=np.full((30,3), .9)):
                    with self.assertRaises(AssertionError): checked_probability(model, b)


if __name__ == '__main__': unittest.main()
