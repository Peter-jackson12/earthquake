"""T9 synthetic contracts only: no real development data, no holdout access.

Fits here are synthetic sample checks, never study training runs.
"""
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
from preprocess import BINARY, GEO, NUMERIC, TEXT  # noqa: E402

try:
    import lightgbm  # noqa: F401
    from experiments.lgbm_partial_native import run as t9
    from experiments.lgbm_native_cat import run as t8
    HAVE_LGBM = True
except ImportError:  # pragma: no cover
    HAVE_LGBM = False


def data(n=400, seed=7):
    X, _, _, _ = synthetic(n)
    X.loc[[0, 17, 155], 'age'] = 995
    rng = np.random.default_rng(seed)
    geo = X.geo_level_1_id.astype(int).to_numpy()
    txt = (X[TEXT[0]].astype(str).str.len().to_numpy() % 2)
    y = pd.Series(np.clip(1 + (geo % 2) + txt * (rng.random(n) < 0.5), 1, 3).astype(int))
    ids = pd.Series(np.arange(1, n + 1), name='building_id')
    return X, y, ids


def with_unseen_text(train, valid):
    """Valid rows whose text value never appears in the training fold (simulated by
    dropping those training rows), so the encoder must map them to NaN."""
    col = TEXT[1]
    rare = valid[col].iloc[0]
    keep = train[col] != rare
    return train[keep].copy(), valid, col, rare


@unittest.skipUnless(HAVE_LGBM, 'lightgbm not installed')
class EncodingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.y, cls.ids = data()
        cls.train, cls.valid = cls.X.iloc[:320].copy(), cls.X.iloc[320:].copy()
        cls.valid.loc[cls.valid.index[:5], 'geo_level_3_id'] = '99999'  # unseen fine region (one-hot ignore)

    def test_vocabulary_from_training_fold_only(self):
        train, valid, col, rare = with_unseen_text(self.train, self.valid)
        pre = t9.make_preprocessor().fit(train)
        enc = pre['columns'].named_transformers_['text_codes']
        geo = pre['columns'].named_transformers_['geo_onehot']
        schema = pre['schema'].transform(train)
        for c, learned in zip(TEXT, enc.categories_):
            self.assertEqual(set(learned), set(schema[c]))
        for c, learned in zip(GEO, geo.categories_):
            self.assertEqual(set(learned), set(schema[c]))
        self.assertNotIn('99999', geo.categories_[GEO.index('geo_level_3_id')])
        self.assertNotIn(str(rare).strip(), enc.categories_[TEXT.index(col)])

    def test_unseen_text_is_explicit_nan_in_csr(self):
        train, valid, col, rare = with_unseen_text(self.train, self.valid)
        pre = t9.make_preprocessor().fit(train)
        m = pre.transform(valid)
        t9.check_matrix(m)
        self.assertTrue(sparse.isspmatrix_csr(m))
        codes = t9.code_block(m)
        j = TEXT.index(col)
        expected = int((valid[col] == rare).sum())
        self.assertGreater(expected, 0)
        self.assertEqual(int(np.isnan(codes[:, j]).sum()), expected)
        # NaN cannot be an implicit entry: every NaN code is a stored value of the CSR.
        self.assertEqual(int(np.isnan(m.data).sum()), int(np.isnan(codes).sum()))
        self.assertEqual(t9.expected_unseen(pre, train, valid)[col], expected)
        self.assertFalse(np.isnan(t9.code_block(pre.transform(train))).any())
        # zero_as_missing=false: code 0 is a valid category stored implicitly.
        self.assertTrue((codes == 0).any())
        self.assertFalse(t9.config()['lgbm_params']['zero_as_missing'])

    def test_categorical_indices_point_to_text_codes(self):
        pre = t9.make_preprocessor().fit(self.train)
        names = [n.split('__', 1)[1] for n in pre['columns'].get_feature_names_out()]
        idx = t9.categorical_indices()
        self.assertEqual(idx, list(range(28, 36)))
        self.assertEqual(idx, t9.config()['categorical_feature_indices_P'])
        self.assertEqual([names[i] for i in idx], list(TEXT))
        self.assertEqual(names[27], t9.FLAG)
        self.assertTrue(all(n.split('_')[0] == 'geo' for n in names[36:]))
        n_geo = sum(len(v) for v in pre['columns'].named_transformers_['geo_onehot'].categories_)
        self.assertEqual(len(names), len(NUMERIC) + len(BINARY) + 1 + len(TEXT) + n_geo)

    def test_head_and_geo_blocks_identical_to_O(self):
        o = t8.make_preprocessor('O').fit(self.train)
        p = t9.make_preprocessor().fit(self.train)
        n_geo = sum(len(v) for v in p['columns'].named_transformers_['geo_onehot'].categories_)
        mo, mp = o.transform(self.valid), p.transform(self.valid)
        self.assertEqual(t9.o_blocks_equal(mo, mp, n_geo), (True, True))
        bad = mp.tolil(); bad[0, 1] = bad[0, 1] + 1; bad = bad.tocsr()
        self.assertEqual(t9.o_blocks_equal(mo, bad, n_geo)[0], False)

    def test_csr_with_codes_equals_dense_lightgbm(self):
        from lightgbm import LGBMClassifier
        train, valid, col, rare = with_unseen_text(self.train, self.valid)
        pre = t9.make_preprocessor().fit(train)
        mt, mv = pre.transform(train), pre.transform(valid)
        params = dict(t9.config()['lgbm_params'], n_estimators=20, verbosity=-1, min_child_samples=5)
        ytr = self.y.loc[train.index].to_numpy() - 1
        a = LGBMClassifier(**params).fit(mt, ytr, categorical_feature=t9.categorical_indices())
        b = LGBMClassifier(**params).fit(mt.toarray(), ytr, categorical_feature=t9.categorical_indices())
        np.testing.assert_array_equal(a.predict_proba(mv), b.predict_proba(mv.toarray()))
        np.testing.assert_array_equal(a.predict_proba(mv), a.predict_proba(mv.toarray()))

    def test_fit_declares_text_categoricals(self):
        params = dict(t9.config()['lgbm_params'], n_estimators=5, verbosity=-1)
        train, valid, col, rare = with_unseen_text(self.train, self.valid)
        pre, model, p, pt, mt, mv, info = t9.fit_arm(train, self.y.loc[train.index], valid, params, 1)
        self.assertGreater(info['valid_unseen_nan'][col], 0)
        self.assertEqual(info['valid_nan_stored'], sum(info['valid_unseen_nan'].values()))
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'m.txt'
            model.booster_.save_model(str(path))
            eff, retained, cat = t9.categorical_effective(path)
            self.assertEqual(cat, ','.join(map(str, t9.categorical_indices())))
            self.assertEqual(eff, {k: float(v) for k, v in
                                   t9.config()['lightgbm_categorical_defaults_expected'].items()})
            self.assertEqual(set(retained), set(TEXT))
        self.assertEqual(p.shape, (len(valid), 3))


@unittest.skipUnless(HAVE_LGBM, 'lightgbm not installed')
class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.X, self.y, self.ids = data(300)
        self.folds = {'stratified': np.arange(300) % 5}
        cfg = t9.config()
        cfg['lgbm_params'] = dict(cfg['lgbm_params'], n_estimators=4, verbosity=-1)
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)
        (self.out/'backup').mkdir()
        t9.atomic_write(self.out/'ledger.json', dict(study_id=t9.STUDY_ID, folds={}))
        self.design = dict(config=cfg, study_targets=['stratified/P/0', 'stratified/P/1'], max_new_study_fits=15,
                           lightgbm_categorical_effective_defaults={
                               k: float(v) for k, v in cfg['lightgbm_categorical_defaults_expected'].items()})

    def tearDown(self):
        self.tmp.cleanup()

    def test_resume_skips_completed_folds(self):
        entry, fitted = t9.fit_fold(self.X, self.y, self.ids, self.folds, self.out, 'stratified', 'P', 0, self.design)
        self.assertTrue(fitted)
        self.assertEqual(entry['status'], 'complete')
        with mock.patch.object(t9, 'fit_arm', side_effect=AssertionError('must not refit')):
            again, fitted2 = t9.fit_fold(self.X, self.y, self.ids, self.folds, self.out, 'stratified', 'P', 0,
                                         self.design)
        self.assertFalse(fitted2)
        ledger = t9.read_ledger(self.out)
        self.assertEqual(ledger['folds']['stratified/P/0']['attempts'], 1)
        for name in entry['artifact_sha256']:
            self.assertTrue((self.out/'backup'/'stratified'/'P'/name).exists())
        rec = (self.out/'stratified'/'P'/'fold_0.json').read_text(encoding='utf-8')
        self.assertIn('"geo_onehot_columns_equal_O": true', rec)

    def test_tampered_artifact_detected_on_resume(self):
        t9.fit_fold(self.X, self.y, self.ids, self.folds, self.out, 'stratified', 'P', 0, self.design)
        path = self.out/'stratified'/'P'/'fold_0.json'
        path.write_text(path.read_text() + ' ', encoding='utf-8')
        with self.assertRaises(ValueError):
            t9.fit_fold(self.X, self.y, self.ids, self.folds, self.out, 'stratified', 'P', 0, self.design)

    def test_unregistered_target_refused(self):
        with self.assertRaises(ValueError):
            t9.fit_fold(self.X, self.y, self.ids, self.folds, self.out, 'geo2', 'P', 0, self.design)

    def test_fold_csv_round_trip_exact(self):
        t9.fit_fold(self.X, self.y, self.ids, self.folds, self.out, 'stratified', 'P', 1, self.design)
        back = t9.read_pred(self.out/'stratified'/'P'/'fold_1_pred.csv.gz')
        self.assertEqual(list(back.columns), ['building_id', 'fold', 'y_true', 'y_pred', *t9.PROBS])
        rng = np.random.default_rng(3)
        p = rng.dirichlet([1, 1, 1], 50)
        frame = pd.DataFrame({'building_id': np.arange(50), 'fold': 0, 'y_true': 1, 'y_pred': p.argmax(1) + 1,
                              **{n: p[:, i] for i, n in enumerate(t9.PROBS)}})
        path = self.out/'rt.csv.gz'
        t9.write_pred(path, frame)
        np.testing.assert_array_equal(t9.read_pred(path)[t9.PROBS].to_numpy(), p)


@unittest.skipUnless(HAVE_LGBM, 'lightgbm not installed')
class ConfigTests(unittest.TestCase):
    def test_plan_fixed(self):
        c = t9.config()
        self.assertEqual(c['lgbm_params'], t8.config()['lgbm_params'])
        self.assertEqual(len(t9.targets()), 15)
        self.assertEqual({a for _, a, _ in t9.targets()}, {'P'})
        self.assertEqual(c['automatic_retries'], 0)
        self.assertFalse(c['holdout_access'])
        r = c['rules']['combined_candidate']
        self.assertEqual((r['stratified_mean_P_gt'], r['geo2_mean_P_ge']), (0.688687, 0.565146))
        self.assertEqual((r['stratified_fold_wins_vs_LR_C_min'], r['geo2_fold_wins_vs_LR_C_min']), (4, 4))

    def test_decomposition_share(self):
        self.assertAlmostEqual(t9.decomposition_share(0.68, 0.67, 0.70), 1/3)
        self.assertIsNone(t9.decomposition_share(0.68, 0.67, 0.67))


if __name__ == '__main__':
    unittest.main()
