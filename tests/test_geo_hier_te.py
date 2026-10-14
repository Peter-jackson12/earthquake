"""T10 synthetic contracts only: no real development data, no holdout access.

Fits here are synthetic sample checks, never study training runs.
"""
import inspect
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
from preprocess import GEO  # noqa: E402

try:
    import lightgbm  # noqa: F401
    from experiments.geo_hier_te import run as t10
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
    def test_hierarchical_formula_hand_computed(self):
        G, y, Q, m, expected = t10.tiny_case()
        got = t10.encode(t10.fit_stats(G, y, m), Q)
        np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)
        np.testing.assert_allclose(t10.reference_encode(G, y, Q, m), expected, rtol=0, atol=1e-12)

    def test_unseen_falls_back_to_parent(self):
        G, y, Q, m, expected = t10.tiny_case()
        got, unseen = t10.encode(t10.fit_stats(G, y, m), Q, return_unseen=True)
        np.testing.assert_array_equal(got[1, 6:8], got[1, 3:5])   # unseen geo3 -> geo2
        self.assertEqual(got[1, 8], 0.0)
        np.testing.assert_array_equal(got[2, 3:5], got[2, 0:2])   # unseen geo2 -> geo1
        np.testing.assert_allclose(got[3, 0:2], [2/6, 3/6], atol=1e-15)  # unseen geo1 -> prior
        np.testing.assert_array_equal(got[4, 3:5], got[4, 0:2])   # composite key: '10' under geo1 '2' unseen
        self.assertEqual(unseen, {'geo1': 1, 'geo2': 3, 'geo3': 4})

    def test_vectorized_equals_bruteforce_reference(self):
        X, y, _ = data(400)
        G = X[GEO].reset_index(drop=True)
        tr, va = np.arange(300), np.arange(300, 400)
        got = t10.encode(t10.fit_stats(G.iloc[tr], y.iloc[tr], 20.0), G.iloc[va])
        ref = t10.reference_encode(G.iloc[tr], y.iloc[tr], G.iloc[va], 20.0)
        np.testing.assert_allclose(got, ref, rtol=0, atol=1e-12)

    def test_inner_oof_never_uses_own_label(self):
        X, y, _ = data(400)
        G, yy = X[GEO].reset_index(drop=True), y.to_numpy()
        folds = t10.inner_fold_ids(yy)
        base = t10.oof_encode(G, yy, folds, 20.0)
        for i in [0, 51, 299]:
            mask = folds == folds[i]
            y2 = yy.copy(); y2[mask] = (y2[mask] % 3) + 1
            e2 = t10.oof_encode(G, y2, folds, 20.0)
            np.testing.assert_array_equal(e2[mask], base[mask])
            self.assertFalse(np.array_equal(e2[~mask], base[~mask]))
        # Transformer fit_transform returns exactly the inner-OOF encodings (float32), not in-sample ones.
        enc = t10.HierGeoTargetEncoder(m=20.0)
        out = enc.fit_transform(G, yy)
        np.testing.assert_array_equal(out, base.astype(np.float32))
        self.assertFalse(np.array_equal(out, enc.transform(G)))

    def test_inner_oof_prior_and_counts_from_inner_train_only(self):
        X, y, _ = data(400)
        G, yy = X[GEO].reset_index(drop=True), y.to_numpy()
        folds = t10.inner_fold_ids(yy)
        out = t10.oof_encode(G, yy, folds, 20.0)
        i = int(np.flatnonzero(folds == 2)[0])
        tr = folds != 2
        k1 = G.geo_level_1_id.to_numpy()
        self.assertAlmostEqual(out[i, 2], np.log1p((k1[tr] == k1[i]).sum()))
        prior = np.mean(yy[tr] == 1)
        Q = pd.DataFrame({GEO[0]: ['nope'], GEO[1]: ['x'], GEO[2]: ['z']})
        self.assertAlmostEqual(t10.encode(t10.fit_stats(G.iloc[np.flatnonzero(tr)], yy[tr], 20.0), Q)[0, 0], prior)

    def test_validation_label_independence(self):
        X, y, _ = data(400)
        self.assertNotIn('y_valid', inspect.signature(t10.fit_arm).parameters)
        a, b = np.arange(320), np.arange(320, 400)
        yp = y.copy(); yp.iloc[b] = np.random.default_rng(0).permutation(y.iloc[b].to_numpy())
        for build in (lambda: t10.make_lr_pipeline()['preprocess'], t10.make_lgbm_preprocessor):
            mats = []
            for labels in (y, yp):
                pre = build()
                pre.fit_transform(X.iloc[a], labels.iloc[a])
                mats.append(pre.transform(X.iloc[b]))
            self.assertEqual((mats[0] != mats[1]).nnz, 0)

    def test_no_raw_geo_columns_and_blocks_equal_C(self):
        from experiments.age995.run import make_pipeline
        X, y, _ = data(400)
        a, b = np.arange(320), np.arange(320, 400)
        for arm, pre in [('LR_TE', t10.make_lr_pipeline()['preprocess']), ('LGBM_TE', t10.make_lgbm_preprocessor())]:
            mt = pre.fit_transform(X.iloc[a], y.iloc[a])
            mv = pre.transform(X.iloc[b])
            names = t10.check_matrix(pre, mv)
            self.assertFalse(any('geo_level' in n for n in names))
            self.assertEqual(names[28:37], ['geo_te__' + n for n in t10.TE_NAMES])
            block = mv[:, 28:37].toarray()  # 9-column slice only
            for g in GEO:
                raw = X.iloc[b][g].astype(float).to_numpy()
                self.assertFalse(any(np.array_equal(block[:, j], raw) for j in range(9)))
            pre_c = make_pipeline('C')['preprocess'].fit(X.iloc[a])
            ok, n_geo = t10.same_blocks_as_C(pre_c, pre_c.transform(X.iloc[b]), mv, names)
            self.assertTrue(ok)
            self.assertGreater(n_geo, 0)
            ok_t, _ = t10.same_blocks_as_C(pre_c, pre_c.transform(X.iloc[a]), mt, names)
            self.assertTrue(ok_t)
            bad = mv.copy().tolil(); bad[0, 1] = bad[0, 1] + 1
            self.assertFalse(t10.same_blocks_as_C(pre_c, pre_c.transform(X.iloc[b]), sparse.csr_matrix(bad, dtype=np.float32), names)[0])
        lr = t10.make_lr_pipeline()['preprocess']
        lr.fit_transform(X.iloc[a], y.iloc[a])
        self.assertIsInstance(lr['columns'].named_transformers_['geo_te'][-1], t10.StandardScaler)

    def test_requires_labels(self):
        X, _, _ = data(200)
        with self.assertRaises(ValueError):
            t10.HierGeoTargetEncoder().fit(X[GEO])


@unittest.skipUnless(HAVE_LGBM, 'lightgbm not installed')
class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.X, self.y, self.ids = data(300)
        self.folds = {'stratified': np.arange(300) % 5}
        cfg = t10.config()
        cfg['lgbm_params'] = dict(cfg['lgbm_params'], n_estimators=4, verbosity=-1)
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)
        (self.out/'backup').mkdir()
        t10.atomic_write(self.out/'ledger.json', dict(study_id=t10.STUDY_ID, folds={}))
        self.design = dict(config=cfg, study_targets=['stratified/LGBM_TE/0', 'stratified/LGBM_TE/1',
                                                      'stratified/LR_TE/0'],
                           max_new_study_fits=30, valid_unseen_geo_keys=t10.unseen_counts(self.X, self.folds))

    def tearDown(self):
        self.tmp.cleanup()

    def test_resume_skips_completed_folds(self):
        for arm in ['LGBM_TE', 'LR_TE']:
            entry, fitted = t10.fit_fold(self.X, self.y, self.ids, self.folds, self.out, 'stratified', arm, 0, self.design)
            self.assertTrue(fitted)
            self.assertEqual(entry['status'], 'complete')
            with mock.patch.object(t10, 'fit_arm', side_effect=AssertionError('must not refit')):
                again, fitted2 = t10.fit_fold(self.X, self.y, self.ids, self.folds, self.out, 'stratified', arm, 0,
                                              self.design)
            self.assertFalse(fitted2)
            self.assertEqual(t10.read_ledger(self.out)['folds'][f'stratified/{arm}/0']['attempts'], 1)
            for name in entry['artifact_sha256']:
                self.assertTrue((self.out/'backup'/'stratified'/arm/name).exists())

    def test_tampered_artifact_detected_on_resume(self):
        t10.fit_fold(self.X, self.y, self.ids, self.folds, self.out, 'stratified', 'LGBM_TE', 0, self.design)
        path = self.out/'stratified'/'LGBM_TE'/'fold_0.json'
        path.write_text(path.read_text() + ' ', encoding='utf-8')
        with self.assertRaises(ValueError):
            t10.fit_fold(self.X, self.y, self.ids, self.folds, self.out, 'stratified', 'LGBM_TE', 0, self.design)

    def test_unregistered_target_refused(self):
        with self.assertRaises(ValueError):
            t10.fit_fold(self.X, self.y, self.ids, self.folds, self.out, 'geo2', 'LGBM_TE', 0, self.design)

    def test_fold_csv_round_trip_exact(self):
        t10.fit_fold(self.X, self.y, self.ids, self.folds, self.out, 'stratified', 'LGBM_TE', 1, self.design)
        back = t10.read_pred(self.out/'stratified'/'LGBM_TE'/'fold_1_pred.csv.gz')
        self.assertEqual(list(back.columns), ['building_id', 'fold', 'y_true', 'y_pred', *t10.PROBS])
        rng = np.random.default_rng(3)
        p = rng.dirichlet([1, 1, 1], 50)
        frame = pd.DataFrame({'building_id': np.arange(50), 'fold': 0, 'y_true': 1, 'y_pred': p.argmax(1) + 1,
                              **{n: p[:, i] for i, n in enumerate(t10.PROBS)}})
        path = self.out/'rt.csv.gz'
        t10.write_pred(path, frame)
        np.testing.assert_array_equal(t10.read_pred(path)[t10.PROBS].to_numpy(), p)


@unittest.skipUnless(HAVE_LGBM, 'lightgbm not installed')
class ConfigTests(unittest.TestCase):
    def test_plan_fixed(self):
        from experiments.model_families.run import load_config
        from experiments.age995.run import PARAMS
        c = t10.config()
        self.assertEqual(c['lgbm_params'], load_config()['models']['LGBM'])
        self.assertEqual(c['lr_params'], PARAMS)
        self.assertEqual(len(t10.targets()), 30)
        self.assertEqual(len(set(t10.targets())), 30)
        self.assertEqual(c['automatic_retries'], 0)
        self.assertFalse(c['holdout_access'])
        self.assertEqual(c['te']['smoothing_m'], 20.0)


if __name__ == '__main__':
    unittest.main()
