"""T11 synthetic contracts only: no real development data, no holdout access.

Fits here are synthetic sample checks, never study training runs.
"""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'tests'))
from test_modeling import synthetic  # noqa: E402
from preprocess import GEO  # noqa: E402

try:
    import lightgbm  # noqa: F401
    from experiments.geo_hier_te import run as t10
    from experiments.geo_te_group_inner import encoder as enc_mod
    from experiments.geo_te_group_inner import run as t11
    HAVE_LGBM = True
except ImportError:  # pragma: no cover
    HAVE_LGBM = False


def data(n=600, seed=7):
    """geo1 in 0..19, geo2 nested under geo1 (5 per geo1), geo3 nested under geo2 (3 per geo2)."""
    X, _, _, _ = synthetic(n)
    rng = np.random.default_rng(seed)
    g1 = rng.integers(0, 20, n)
    g2 = g1*5 + rng.integers(0, 5, n)
    g3 = g2*3 + rng.integers(0, 3, n)
    X['geo_level_1_id'], X['geo_level_2_id'], X['geo_level_3_id'] = g1, g2, g3
    X.loc[[0, 17, 155], 'age'] = 995
    y = pd.Series(np.clip(1 + (g1 % 3) + (rng.random(n) < 0.2), 1, 3).astype(int))
    ids = pd.Series(np.arange(1, n + 1), name='building_id')
    return X, y, ids


def outer_folds(X, level):
    col = {'geo1': 'geo_level_1_id', 'geo2': 'geo_level_2_id'}[level]
    return (X[col].astype(int).to_numpy() % 5).astype(int)


@unittest.skipUnless(HAVE_LGBM, 'lightgbm not installed')
class InnerGroupTests(unittest.TestCase):
    def test_inner_groups_disjoint_and_five_folds(self):
        X, y, _ = data()
        for level in ('geo2', 'geo1'):
            G, yy = X[GEO].reset_index(drop=True), y.to_numpy()
            groups = enc_mod.group_ids(G, level)
            folds = enc_mod.inner_group_fold_ids(yy, groups)
            self.assertEqual(sorted(set(folds)), [0, 1, 2, 3, 4])
            sets = [set(groups[folds == i]) for i in range(5)]
            for i in range(5):
                for j in range(i + 1, 5):
                    self.assertTrue(sets[i].isdisjoint(sets[j]))

    def test_too_few_groups_stops(self):
        yy = np.array([1, 2, 3, 1, 2, 3])
        with self.assertRaises(ValueError):
            enc_mod.inner_group_fold_ids(yy, np.array(['a', 'a', 'b', 'b', 'c', 'c']))

    def test_heldout_group_rows_get_parent_and_zero_count(self):
        X, y, _ = data()
        G, yy = X[GEO].reset_index(drop=True), y.to_numpy()
        for level in ('geo2', 'geo1'):
            enc = enc_mod.GroupInnerTargetEncoder(m=20.0, group_level=level)
            out = enc.fit_transform(G, yy).astype(np.float64)
            self.assertEqual(enc.train_oof_unseen_[level], len(G))   # 100% unseen at the outer group level
            self.assertEqual(enc.train_oof_unseen_['geo3'], len(G))
            self.assertTrue((out[:, 5] == 0).all() and (out[:, 8] == 0).all())  # cnt = 0
            np.testing.assert_array_equal(out[:, 6:8], out[:, 3:5])            # geo3 -> parent (geo2 value)
            if level == 'geo1':
                self.assertTrue((out[:, 2] == 0).all())
                np.testing.assert_array_equal(out[:, 3:5], out[:, 0:2])        # all levels -> inner-training prior
                folds = enc.inner_folds_
                for k in range(5):  # prior from the inner-training part only
                    rows = folds == k
                    np.testing.assert_allclose(out[rows, 0], np.float32(np.mean(yy[~rows] == 1)), rtol=0, atol=1e-7)

    def test_validation_encoding_unchanged_vs_t10(self):
        X, y, _ = data()
        G, yy = X[GEO].reset_index(drop=True), y.to_numpy()
        tr, va = np.arange(450), np.arange(450, 600)
        for level in ('geo2', 'geo1'):
            new = enc_mod.GroupInnerTargetEncoder(m=20.0, group_level=level).fit(G.iloc[tr], yy[tr]).transform(G.iloc[va])
            old = t10.HierGeoTargetEncoder(m=20.0).fit(G.iloc[tr], yy[tr]).transform(G.iloc[va])
            self.assertEqual(new.tobytes(), old.tobytes())
            ref = t10.encode(t10.fit_stats(G.iloc[tr], yy[tr], 20.0), G.iloc[va]).astype(np.float32)
            self.assertEqual(new.tobytes(), ref.tobytes())

    def test_training_encoding_differs_from_t10_random_inner(self):
        X, y, _ = data()
        G, yy = X[GEO].reset_index(drop=True), y.to_numpy()
        new = enc_mod.GroupInnerTargetEncoder(group_level='geo2').fit_transform(G, yy)
        old = t10.HierGeoTargetEncoder().fit_transform(G, yy)
        self.assertFalse(np.array_equal(new, old))

    def test_own_group_labels_never_used(self):
        X, y, _ = data()
        G, yy = X[GEO].reset_index(drop=True), y.to_numpy()
        groups = enc_mod.group_ids(G, 'geo2')
        folds = enc_mod.inner_group_fold_ids(yy, groups)
        base = t10.oof_encode(G, yy, folds, 20.0)
        g = groups[10]
        idx = np.flatnonzero(groups == g)
        y2 = yy.copy(); y2[idx] = (y2[idx] % 3) + 1
        e2 = t10.oof_encode(G, y2, folds, 20.0)
        np.testing.assert_array_equal(e2[idx], base[idx])
        self.assertFalse(np.array_equal(e2[folds != folds[idx[0]]], base[folds != folds[idx[0]]]))

    def test_requires_labels(self):
        X, _, _ = data()
        with self.assertRaises(ValueError):
            enc_mod.GroupInnerTargetEncoder().fit(X[GEO])

    def test_fresh_process_load(self):
        X, y, _ = data()
        pre = t11.make_lgbm_preprocessor(20.0, 'geo2')
        a, b = np.arange(450), np.arange(450, 600)
        pre.fit_transform(X.iloc[a], y.iloc[a])
        expected = pre.transform(X.iloc[b])
        enc = t10.te_encoder(pre)
        self.assertEqual(type(enc).__module__, 'experiments.geo_te_group_inner.encoder')
        ok, text = t11.fresh_process_load(enc, X.iloc[b][GEO].reset_index(drop=True),
                                          dict(matrix_pre=pre, X_query=X.iloc[b], expected_matrix=expected))
        self.assertTrue(ok, text)


@unittest.skipUnless(HAVE_LGBM, 'lightgbm not installed')
class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.X, self.y, self.ids = data(600)
        self.folds = {'geo2': outer_folds(self.X, 'geo2'), 'geo1': outer_folds(self.X, 'geo1')}
        cfg = t11.config()
        cfg['lgbm_params'] = dict(cfg['lgbm_params'], n_estimators=4, verbosity=-1)
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)
        (self.out/'backup').mkdir()
        t11.atomic_write(self.out/'ledger.json', dict(study_id=t11.STUDY_ID, folds={}))
        self.design = dict(config=cfg, study_targets=['geo2/LGBM_TE_G/0', 'geo1/LGBM_TE_G/1', 'geo2/LR_TE_G/0'],
                           max_new_study_fits=20, valid_unseen_geo_keys=t10.unseen_counts(self.X, self.folds))

    def tearDown(self):
        self.tmp.cleanup()

    def test_resume_skips_completed_folds(self):
        for arm in ['LGBM_TE_G', 'LR_TE_G']:
            entry, fitted = t11.fit_fold(self.X, self.y, self.ids, self.folds, self.out, 'geo2', arm, 0, self.design)
            self.assertTrue(fitted)
            self.assertEqual(entry['status'], 'complete')
            with mock.patch.object(t11, 'fit_arm', side_effect=AssertionError('must not refit')):
                again, fitted2 = t11.fit_fold(self.X, self.y, self.ids, self.folds, self.out, 'geo2', arm, 0, self.design)
            self.assertFalse(fitted2)
            self.assertEqual(t11.read_ledger(self.out)['folds'][f'geo2/{arm}/0']['attempts'], 1)
            for name in entry['artifact_sha256']:
                self.assertTrue((self.out/'backup'/'geo2'/arm/name).exists())
        rec = __import__('json').loads((self.out/'geo2'/'LGBM_TE_G'/'fold_0.json').read_text(encoding='utf-8'))
        self.assertEqual(rec['te']['train_oof_unseen_rate']['geo2'], 1.0)
        self.assertTrue(rec['valid_encoding_identical_to_t10'])
        self.assertEqual(rec['encoder_module'], 'experiments.geo_te_group_inner.encoder')

    def test_geo1_scheme_fit_and_unregistered_refused(self):
        entry, fitted = t11.fit_fold(self.X, self.y, self.ids, self.folds, self.out, 'geo1', 'LGBM_TE_G', 1, self.design)
        self.assertTrue(fitted)
        with self.assertRaises(ValueError):
            t11.fit_fold(self.X, self.y, self.ids, self.folds, self.out, 'stratified', 'LGBM_TE_G', 0, self.design)

    def test_tampered_artifact_detected_on_resume(self):
        t11.fit_fold(self.X, self.y, self.ids, self.folds, self.out, 'geo2', 'LGBM_TE_G', 0, self.design)
        path = self.out/'geo2'/'LGBM_TE_G'/'fold_0.json'
        path.write_text(path.read_text() + ' ', encoding='utf-8')
        with self.assertRaises(ValueError):
            t11.fit_fold(self.X, self.y, self.ids, self.folds, self.out, 'geo2', 'LGBM_TE_G', 0, self.design)

    def test_fold_csv_round_trip_exact(self):
        t11.fit_fold(self.X, self.y, self.ids, self.folds, self.out, 'geo2', 'LGBM_TE_G', 0, self.design)
        back = t11.read_pred(self.out/'geo2'/'LGBM_TE_G'/'fold_0_pred.csv.gz')
        self.assertEqual(list(back.columns), ['building_id', 'fold', 'y_true', 'y_pred', *t11.PROBS])
        p = np.random.default_rng(3).dirichlet([1, 1, 1], 50)
        frame = pd.DataFrame({'building_id': np.arange(50), 'fold': 0, 'y_true': 1, 'y_pred': p.argmax(1) + 1,
                              **{n: p[:, i] for i, n in enumerate(t11.PROBS)}})
        path = self.out/'rt.csv.gz'
        t11.write_pred(path, frame)
        np.testing.assert_array_equal(t11.read_pred(path)[t11.PROBS].to_numpy(), p)


@unittest.skipUnless(HAVE_LGBM, 'lightgbm not installed')
class ConfigTests(unittest.TestCase):
    def test_plan_fixed(self):
        c = t11.config()
        self.assertEqual(len(t11.targets()), 20)
        self.assertEqual(len(set(t11.targets())), 20)
        self.assertTrue(all(s in ('geo2', 'geo1') for s, _, _ in t11.targets()))
        self.assertEqual(c['automatic_retries'], 0)
        self.assertFalse(c['holdout_access'])
        self.assertEqual(c['te']['smoothing_m'], 20.0)
        self.assertEqual(c['te']['inner_oof']['splitter'], 'StratifiedGroupKFold')
        self.assertEqual(c['lgbm_params'], t10.config()['lgbm_params'])


if __name__ == '__main__':
    unittest.main()
