"""T8 synthetic contracts only: no real development data, no holdout access.

Fits here are synthetic sample checks, never study training runs.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'tests'))
from test_modeling import synthetic  # noqa: E402
from preprocess import BINARY, CATEGORICAL, NUMERIC  # noqa: E402

try:
    import lightgbm  # noqa: F401
    from experiments.lgbm_native_cat import run as t8
    HAVE_LGBM = True
except ImportError:  # pragma: no cover
    HAVE_LGBM = False


def data(n=400, seed=7):
    X, _, _, _ = synthetic(n)
    X.loc[[0, 17, 155], 'age'] = 995
    rng = np.random.default_rng(seed)
    geo = X.geo_level_1_id.astype(int).to_numpy()
    y = pd.Series(np.clip(1 + (geo % 3) + (rng.random(n) < 0.2), 1, 3).astype(int))
    ids = pd.Series(np.arange(1, n + 1), name='building_id')
    return X, y, ids


@unittest.skipUnless(HAVE_LGBM, 'lightgbm not installed')
class EncodingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.y, cls.ids = data()
        cls.train, cls.valid = cls.X.iloc[:320].copy(), cls.X.iloc[320:].copy()
        cls.valid.loc[cls.valid.index[:5], 'geo_level_3_id'] = '99999'  # unseen fine region
        cls.valid.loc[cls.valid.index[5:7], 'geo_level_2_id'] = '88888'

    def test_vocabulary_from_training_fold_only(self):
        pre = t8.make_preprocessor('N').fit(self.train)
        enc = pre['columns'].named_transformers_['categorical']
        for col, learned in zip(CATEGORICAL, enc.categories_):
            self.assertEqual(set(learned), set(self.train[col].astype(str).str.strip()))
        self.assertNotIn('99999', enc.categories_[CATEGORICAL.index('geo_level_3_id')])

    def test_unseen_categories_become_nan(self):
        pre = t8.make_preprocessor('N').fit(self.train)
        m = pre.transform(self.valid)
        t8.check_matrix('N', m)
        g3 = t8.categorical_indices()[CATEGORICAL.index('geo_level_3_id')]
        g2 = t8.categorical_indices()[CATEGORICAL.index('geo_level_2_id')]
        self.assertTrue(np.isnan(m[:5, g3]).all())
        self.assertTrue(np.isnan(m[5:7, g2]).all())
        self.assertFalse(np.isnan(pre.transform(self.train)).any())
        expected_nan = (~self.valid.geo_level_3_id.isin(set(self.train.geo_level_3_id))).sum()
        self.assertEqual(int(np.isnan(m[:, g3]).sum()), int(expected_nan))

    def test_categorical_feature_indices(self):
        pre = t8.make_preprocessor('N').fit(self.train)
        names = [n.split('__', 1)[1] for n in pre['columns'].get_feature_names_out()]
        idx = t8.categorical_indices()
        self.assertEqual(idx, list(range(28, 39)))
        self.assertEqual(idx, t8.config()['categorical_feature_indices_N'])
        self.assertEqual([names[i] for i in idx], list(CATEGORICAL))
        self.assertEqual(len(names), len(NUMERIC) + len(BINARY) + 1 + len(CATEGORICAL))
        self.assertEqual(names[27], t8.FLAG)

    def test_numeric_binary_columns_identical_to_O(self):
        o = t8.make_preprocessor('O').fit(self.train)
        n = t8.make_preprocessor('N').fit(self.train)
        mo, mn = o.transform(self.valid), n.transform(self.valid)
        self.assertTrue(sparse.isspmatrix_csr(mo))
        self.assertTrue(t8.same_head(mo, mn))
        bad = mn.copy(); bad[0, 1] += 1
        self.assertFalse(t8.same_head(mo, bad))

    def test_O_is_unchanged_C_preprocessor(self):
        from experiments.age995.run import make_pipeline
        a = make_pipeline('C')['preprocess'].fit(self.train).transform(self.valid)
        b = t8.make_preprocessor('O').fit(self.train).transform(self.valid)
        self.assertEqual((a != b).nnz, 0)

    def test_native_fit_declares_categoricals(self):
        params = dict(t8.config()['lgbm_params'], n_estimators=5, verbosity=-1)
        pre, model, p, pt, mt, mv, info = t8.fit_arm('N', self.train, self.y.iloc[:320], self.valid, params, 1)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'m.txt'
            model.booster_.save_model(str(path))
            params, _ = t8.model_text_parameters(path)
            self.assertEqual(params['categorical_feature'], ','.join(map(str, t8.categorical_indices())))
            eff, retained = t8.categorical_effective(path)
            self.assertEqual(eff, {k: float(v) for k, v in
                                   t8.config()['lightgbm_categorical_defaults_expected'].items()})
            self.assertEqual(set(retained), set(CATEGORICAL))
        self.assertEqual(p.shape, (80, 3))


@unittest.skipUnless(HAVE_LGBM, 'lightgbm not installed')
class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.X, self.y, self.ids = data(300)
        self.folds = {'stratified': np.arange(300) % 5}
        cfg = t8.config()
        cfg['lgbm_params'] = dict(cfg['lgbm_params'], n_estimators=4, verbosity=-1)
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)
        (self.out/'backup').mkdir()
        t8.atomic_write(self.out/'ledger.json', dict(study_id=t8.STUDY_ID, folds={}))
        self.design = dict(config=cfg, study_targets=['stratified/N/0', 'stratified/N/1'], max_new_study_fits=25,
                           lightgbm_categorical_effective_defaults={
                               k: float(v) for k, v in cfg['lightgbm_categorical_defaults_expected'].items()})

    def tearDown(self):
        self.tmp.cleanup()

    def test_resume_skips_completed_folds(self):
        entry, fitted = t8.fit_fold(self.X, self.y, self.ids, self.folds, self.out, 'stratified', 'N', 0, self.design)
        self.assertTrue(fitted)
        self.assertEqual(entry['status'], 'complete')
        with mock.patch.object(t8, 'fit_arm', side_effect=AssertionError('must not refit')):
            again, fitted2 = t8.fit_fold(self.X, self.y, self.ids, self.folds, self.out, 'stratified', 'N', 0,
                                         self.design)
        self.assertFalse(fitted2)
        ledger = t8.read_ledger(self.out)
        self.assertEqual(ledger['folds']['stratified/N/0']['attempts'], 1)
        for name in entry['artifact_sha256']:
            self.assertTrue((self.out/'backup'/'stratified'/'N'/name).exists())

    def test_tampered_artifact_detected_on_resume(self):
        t8.fit_fold(self.X, self.y, self.ids, self.folds, self.out, 'stratified', 'N', 0, self.design)
        path = self.out/'stratified'/'N'/'fold_0.json'
        path.write_text(path.read_text() + ' ', encoding='utf-8')
        with self.assertRaises(ValueError):
            t8.fit_fold(self.X, self.y, self.ids, self.folds, self.out, 'stratified', 'N', 0, self.design)

    def test_unregistered_target_refused(self):
        with self.assertRaises(ValueError):
            t8.fit_fold(self.X, self.y, self.ids, self.folds, self.out, 'stratified', 'O', 0, self.design)

    def test_fold_csv_round_trip_exact(self):
        t8.fit_fold(self.X, self.y, self.ids, self.folds, self.out, 'stratified', 'N', 1, self.design)
        back = t8.read_pred(self.out/'stratified'/'N'/'fold_1_pred.csv.gz')
        self.assertEqual(list(back.columns), ['building_id', 'fold', 'y_true', 'y_pred', *t8.PROBS])
        rng = np.random.default_rng(3)
        p = rng.dirichlet([1, 1, 1], 50)
        frame = pd.DataFrame({'building_id': np.arange(50), 'fold': 0, 'y_true': 1, 'y_pred': p.argmax(1) + 1,
                              **{n: p[:, i] for i, n in enumerate(t8.PROBS)}})
        path = self.out/'rt.csv.gz'
        t8.write_pred(path, frame)
        np.testing.assert_array_equal(t8.read_pred(path)[t8.PROBS].to_numpy(), p)


@unittest.skipUnless(HAVE_LGBM, 'lightgbm not installed')
class ConfigTests(unittest.TestCase):
    def test_params_equal_T4_and_plan_fixed(self):
        from experiments.model_families.run import load_config
        c = t8.config()
        self.assertEqual(c['lgbm_params'], load_config()['models']['LGBM'])
        self.assertEqual(len(t8.targets()), 25)
        self.assertNotIn(('stratified', 'O', 0), t8.targets())
        self.assertEqual(c['automatic_retries'], 0)
        self.assertFalse(c['holdout_access'])


if __name__ == '__main__':
    unittest.main()
