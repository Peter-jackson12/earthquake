"""T13 synthetic contracts only: no real development data, no holdout access.

Fits here are synthetic sample checks, never study training runs.
"""
import hashlib
import json
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
    from experiments.geo2_te_confirm import run as t13
    from experiments.geo_hier_te import run as t10
    from experiments.geo_te_group_inner import run as t11
    HAVE_LGBM = True
except ImportError:  # pragma: no cover
    HAVE_LGBM = False


def data(n=900, seed=7):
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


@unittest.skipUnless(HAVE_LGBM, 'lightgbm not installed')
class ManifestTests(unittest.TestCase):
    def test_config_is_preregistered(self):
        c = t13.config()
        self.assertEqual(c['split']['seeds'], [101, 202, 303])
        self.assertEqual(c['max_new_study_fits'], 45)
        self.assertEqual(len(t13.targets()), 45)
        self.assertEqual(len(set(t13.key(*t) for t in t13.targets())), 45)

    def test_generator_deterministic_and_group_disjoint(self):
        X, y, ids = data()
        a, b = t13.make_folds(X, y, 101), t13.make_folds(X, y, 101)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(t13.manifest_bytes(ids, a, X), t13.manifest_bytes(ids, b, X))
        self.assertEqual(sorted(set(a)), [0, 1, 2, 3, 4])
        g, _ = t13.groups(X, t13.GROUP_COLUMNS)
        for f in range(5):
            self.assertTrue(set(g[a == f]).isdisjoint(set(g[a != f])))
        self.assertFalse(np.array_equal(a, t13.make_folds(X, y, 202)))

    def test_seed42_byte_reproduction_logic(self):
        X, y, ids = data()
        # independent T5-style procedure (inline copy of experiments/geo_group_cv prepare for geo2)
        from sklearn.model_selection import StratifiedGroupKFold
        g, _ = t13.groups(X, GEO[:2])
        folds = np.full(len(X), -1)
        for f, (_, b) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(X, y, g)):
            folds[b] = f
        with tempfile.TemporaryDirectory() as tmp:
            ref = Path(tmp)/'manifest.csv'
            pd.DataFrame({'building_id': ids, 'fold': folds, **{c: X[c] for c in GEO[:2]}}).to_csv(
                ref, index=False, lineterminator='\n')
            self.assertTrue(t13.reproduces_reference(X, y, ids, ref))
            self.assertFalse(t13.reproduces_reference(X, y, ids, ref, seed=101))
            sha = t13.write_manifest(Path(tmp)/'m2.csv', ids, folds, X)
            self.assertEqual(sha, hashlib.sha256(ref.read_bytes()).hexdigest())
            self.assertEqual(ref.read_bytes(), (Path(tmp)/'m2.csv').read_bytes())

    def test_similarity_and_gate(self):
        X, y, ids = data()
        a, b = t13.make_folds(X, y, 42), t13.make_folds(X, y, 101)
        self.assertAlmostEqual(t13.fold_similarity(a, a), 1.0)
        self.assertLess(t13.fold_similarity(a, b), 0.5)
        bad = a.copy(); bad[np.flatnonzero(a == 0)[0]] = 1  # splits a group across folds
        with self.assertRaises(ValueError):
            t13.split_gate(X, y, ids, bad)


@unittest.skipUnless(HAVE_LGBM, 'lightgbm not installed')
class RuleTests(unittest.TestCase):
    def setUp(self):
        self.rules = t13.config()['rules']
        self.o = dict(mean_delta=0.012, fold_wins=13, n_folds=15, pooled_grade3_recall_delta=-0.005,
                      repair={101: (0.01, 0.9), 202: (-0.015, 0.85), 303: (0.0, 0.95)})
        self.lr = dict(mean_delta=0.02, fold_wins=14, n_folds=15, te_pooled_grade3_recall=0.52,
                       lr_pooled_grade3_recall=0.525)

    def test_confirmed(self):
        j = t13.judge(self.rules, self.o, self.lr)
        self.assertTrue(j['C1']['holds'] and j['C2']['holds'] and j['confirmed_geo2_candidate'])

    def test_each_part_can_fail(self):
        cases = [('o', 'mean_delta', 0.0099, 'C1'), ('o', 'fold_wins', 11, 'C1'),
                 ('o', 'pooled_grade3_recall_delta', -0.011, 'C1'),
                 ('o', 'repair', {101: (0.03, 0.9), 202: (0.0, 0.9), 303: (0.0, 0.9)}, 'C1'),
                 ('o', 'repair', {101: (0.0, 0.79), 202: (0.0, 0.9), 303: (0.0, 0.9)}, 'C1'),
                 ('lr', 'mean_delta', 0.0, 'C2'), ('lr', 'fold_wins', 11, 'C2'),
                 ('lr', 'te_pooled_grade3_recall', 0.50, 'C2')]
        for which, k, v, rule in cases:
            o, lr = dict(self.o), dict(self.lr)
            (o if which == 'o' else lr)[k] = v
            j = t13.judge(self.rules, o, lr)
            self.assertFalse(j[rule]['holds'], (k, v))
            self.assertFalse(j['confirmed_geo2_candidate'])

    def test_boundary_values_count(self):
        o = dict(self.o, mean_delta=0.010, fold_wins=12, pooled_grade3_recall_delta=-0.01)
        lr = dict(self.lr, fold_wins=12, te_pooled_grade3_recall=0.515, lr_pooled_grade3_recall=0.525)
        j = t13.judge(self.rules, o, lr)
        self.assertTrue(j['C1']['parts']['mean_delta_ok'] and j['C1']['parts']['fold_wins_ok'])
        self.assertTrue(j['C2']['parts']['fold_wins_ok'])


@unittest.skipUnless(HAVE_LGBM, 'lightgbm not installed')
class CheckpointTests(unittest.TestCase):
    def test_csv_round_trip_exact(self):
        rng = np.random.default_rng(3)
        p = rng.dirichlet([0.1, 0.1, 0.1], 200)
        p[0] = [1e-300, 0.5, 0.5 - 1e-300]
        frame = pd.DataFrame({'building_id': np.arange(200), 'fold': 0, 'y_true': 1, 'y_pred': 1,
                              **{n: p[:, i] for i, n in enumerate(t13.PROBS)}})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'p.csv.gz'
            t13.write_pred(path, frame)
            back = t13.read_pred(path)
            np.testing.assert_array_equal(back[t13.PROBS].to_numpy(), p)
            first = path.read_bytes()
            t13.write_pred(path, frame)
            self.assertEqual(first, path.read_bytes())  # deterministic gzip

    def test_resume_skips_completed_and_verifies_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            k = t13.key(101, 'LGBM_O', 0)
            d, bd = out/'s101'/'LGBM_O', out/'backup'/'s101'/'LGBM_O'
            d.mkdir(parents=True); bd.mkdir(parents=True)
            (d/'fold_0.json').write_text('{}'); (bd/'fold_0.json').write_text('{}')
            sha = t13.sha256_file(d/'fold_0.json')
            entry = dict(dir='s101/LGBM_O', seed=101, arm='LGBM_O', fold=0, status='complete', attempts=1,
                         artifact_sha256={'fold_0.json': sha})
            t13.atomic_write(out/'ledger.json', dict(folds={k: entry}))
            (out/'run.log').write_text('')
            design = dict(study_targets=[k], max_new_study_fits=45)
            with mock.patch.object(t13, 'fit_one', side_effect=AssertionError('must not refit')):
                e, fitted = t13.fit_fold(None, None, None, None, out, 101, 'LGBM_O', 0, design)
            self.assertFalse(fitted)
            (d/'fold_0.json').write_text('{"changed": 1}')  # tampered artifact is detected, not silently accepted
            with self.assertRaises(ValueError):
                t13.fit_fold(None, None, None, None, out, 101, 'LGBM_O', 0, design)

    def test_unregistered_and_nonconverged_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            k = t13.key(101, 'LR_C', 0)
            t13.atomic_write(out/'ledger.json', dict(folds={k: dict(status='complete_nonconverged', attempts=1)}))
            (out/'run.log').write_text('')
            design = dict(study_targets=[k], max_new_study_fits=45)
            with self.assertRaises(ValueError):
                t13.fit_fold(None, None, None, None, out, 101, 'LR_C', 0, design)
            with self.assertRaises(ValueError):
                t13.fit_fold(None, None, None, None, out, 999, 'LR_C', 0, design)

    def test_fresh_process_artifact_load(self):
        X, y, _ = data()
        pre = t11.make_lgbm_preprocessor(20.0, 'geo2')
        a, b = np.arange(700), np.arange(700, 900)
        pre.fit_transform(X.iloc[a], y.iloc[a])
        expected = pre.transform(X.iloc[b])
        enc = t10.te_encoder(pre)
        self.assertEqual(type(enc).__module__, 'experiments.geo_te_group_inner.encoder')
        ok, text = t11.fresh_process_load(enc, X.iloc[b][GEO].reset_index(drop=True),
                                          dict(matrix_pre=pre, X_query=X.iloc[b], expected_matrix=expected))
        self.assertTrue(ok, text)

    def test_fit_one_lgbm_arms_small(self):
        X, y, _ = data()
        c = t13.config()
        c['lgbm_params'] = dict(c['lgbm_params'], n_estimators=3, verbosity=-1)
        a, b = np.arange(700), np.arange(700, 900)
        for arm in ('LGBM_O', 'LGBM_TE_G'):
            out = t13.fit_one(arm, X.iloc[a], y.iloc[a], X.iloc[b], c)
            self.assertEqual(out['p'].shape, (200, 3))
            np.testing.assert_allclose(out['p'].sum(1), 1.0, atol=1e-6)
        out = t13.fit_one('LR_C', X.iloc[a], y.iloc[a], X.iloc[b], c)
        self.assertTrue(out['converged'])
        self.assertEqual(out['p'].shape, (200, 3))


if __name__ == '__main__':
    unittest.main()
