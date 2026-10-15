"""T14 synthetic contracts only: no real development data, no holdout access, no real training.

RF fits here use 5-tree models on ~900 synthetic rows; they are contract checks, never study fits.
"""
import builtins
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
from preprocess import GEO, TEXT  # noqa: E402

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedGroupKFold
    from experiments.te_ros_rf_repro import run as t14
    HAVE = True
except ImportError:  # pragma: no cover
    HAVE = False


def data(n=900, seed=7):
    X, _, _, _ = synthetic(n)
    rng = np.random.default_rng(seed)
    g1 = rng.integers(0, 20, n)
    g2 = g1*5 + rng.integers(0, 5, n)
    g3 = g2*3 + rng.integers(0, 3, n)
    X['geo_level_1_id'], X['geo_level_2_id'], X['geo_level_3_id'] = g1.astype(str), g2.astype(str), g3.astype(str)
    X.loc[[0, 17, 155], 'age'] = 995
    y = pd.Series(np.clip(1 + (g1 % 3) + (rng.random(n) < 0.2), 1, 3).astype(int))
    ids = pd.Series(np.arange(1, n + 1), name='building_id')
    strat = np.arange(n) % 5
    groups = g1*100 + g2
    geo = np.full(n, -1)
    for f, (_, b) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=101).split(X, y, groups)):
        geo[b] = f
    return X, y, ids, strat, geo


def small_config():
    c = json.loads(json.dumps(t14.config()))
    c['rf_params'] = dict(n_jobs=1, random_state=0, n_estimators=5)
    c['rf_default_snapshot'] = RandomForestClassifier(**c['rf_params']).get_params()
    c['min_free_disk_gb_per_fold'] = 0
    return c


def design_for(c):
    return dict(config=c, study_targets=[t14.key(*t) for t in t14.targets()], max_new_study_fits=25)


@unittest.skipUnless(HAVE, 'dependencies not installed')
class ConfigTests(unittest.TestCase):
    def test_preregistered_plan(self):
        c = t14.config()
        self.assertEqual(len(t14.targets()), 25)
        self.assertEqual(len({t14.key(*t) for t in t14.targets()}), 25)
        self.assertEqual(sum(t14.STAGE_OF[t[0]] == 'A' for t in t14.targets()), 5)
        self.assertEqual(sum(t14.STAGE_OF[t[0]] == 'B' for t in t14.targets()), 20)
        self.assertEqual(c['expected_model_columns'], 77)
        self.assertEqual(c['rf_params'], dict(n_jobs=4, random_state=0))
        self.assertFalse(c['holdout_access'])
        self.assertEqual(c['comparator_refits'], 0)
        for s in t14.SCHEMES:
            self.assertTrue(set(c['comparators'][s]) >= {'LR_C', 'LGBM_O'})

    def test_original_sources_pinned_and_unmodified(self):
        pins = t14.config_pins()
        for name, digest in pins.items():
            self.assertEqual(t14.sha256_file(ROOT/'experiments/te_ros_rf_exploratory'/name), digest)

    def test_recorded_reference_is_self_consistent(self):
        rec = t14.config()['recorded_reference']
        pooled = sum(np.asarray(f['confusion']) for f in rec['folds'])
        self.assertEqual(pooled.tolist(), rec['pooled_confusion'])
        self.assertEqual(int(pooled[2, 0]), rec['pooled_error31'])
        self.assertEqual(int(pooled.sum()), 208480)


@unittest.skipUnless(HAVE, 'dependencies not installed')
class EncodingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.y, cls.ids, cls.strat, cls.geo = data()
        cls.F = t14.original_frame(cls.X)
        cls.c = t14.config()

    def split(self, folds, f=0):
        return np.flatnonzero(folds != f), np.flatnonzero(folds == f)

    def test_original_frame_matches_exploratory_dtypes(self):
        for g in GEO:
            self.assertEqual(self.F[g].dtype, np.int64)
        for t in TEXT:
            self.assertEqual(str(self.F[t].dtype), 'category')
        self.assertEqual(int((self.F['age'] == 995).sum()), 3)  # raw age 995 untouched

    def test_target_enc_equals_original_function(self):
        a, b = self.split(self.strat)
        y0 = pd.Series(self.y.to_numpy() - 1)
        orig = t14.original_target_enc()
        A0, B0 = orig(self.F.iloc[a], y0.iloc[a], self.F.iloc[b], list(GEO))
        A1, B1 = t14.target_enc(self.F.iloc[a], y0.iloc[a], self.F.iloc[b], list(GEO))
        pd.testing.assert_frame_equal(A0, A1)
        pd.testing.assert_frame_equal(B0, B1)
        A2, B2, state = t14.target_enc(self.F.iloc[a], y0.iloc[a], self.F.iloc[b], list(GEO), return_state=True)
        pd.testing.assert_frame_equal(A0, A2)
        pd.testing.assert_frame_equal(B0, B2)

    def test_stored_state_rebuilds_validation_matrix(self):
        a, b = self.split(self.strat)
        built = t14.build_fold_matrices(self.F.iloc[a], self.X.iloc[a], self.y.iloc[a], self.F.iloc[b], 'strat', self.c
                                        | dict(expected_model_columns=len(self._cols(a, b))))
        Bo2 = t14.apply_state(built['state']['geo'], self.F.iloc[b], built['state']['columns'], built['state']['cats'])
        pd.testing.assert_frame_equal(Bo2, built['Bo'])

    def _cols(self, a, b):
        y0 = pd.Series(self.y.to_numpy() - 1)
        A, _ = t14.target_enc(self.F.iloc[a], y0.iloc[a], self.F.iloc[b], list(GEO))
        return pd.get_dummies(A, columns=[c for c in self.F.columns if c in TEXT], dtype=int).columns

    def test_validation_encoding_uses_training_rows_only(self):
        a, b = self.split(self.strat)
        c = self.c | dict(expected_model_columns=len(self._cols(a, b)))
        base = t14.build_fold_matrices(self.F.iloc[a], self.X.iloc[a], self.y.iloc[a], self.F.iloc[b], 'strat', c)
        altered = self.F.iloc[b].copy()
        altered['geo_level_1_id'] = 999_999  # unseen id; only validation features change
        alt = t14.build_fold_matrices(self.F.iloc[a], self.X.iloc[a], self.y.iloc[a], altered, 'strat', c)
        pd.testing.assert_frame_equal(base['Ao'], alt['Ao'])          # training matrix independent of validation rows
        pd.testing.assert_frame_equal(base['Xos'], alt['Xos'])
        prior = float((self.y.iloc[a] == 1).mean())
        np.testing.assert_allclose(alt['Bo']['geo_level_1_id_te0'].to_numpy(), np.float32(prior), rtol=1e-6)
        self.assertTrue((alt['Bo']['geo_level_1_id_cnt'] == 0).all())
        # hand-computed smoothed rate of one seen validation id from TRAIN rows only
        row = b[0]
        gid = self.F['geo_level_2_id'].iloc[row]
        tr = self.y.iloc[a][self.F['geo_level_2_id'].iloc[a].to_numpy() == gid]
        expected = ((tr == 3).sum() + 20*float((self.y.iloc[a] == 3).mean()))/(len(tr) + 20)
        got = base['Bo']['geo_level_2_id_te2'].iloc[0]
        self.assertAlmostEqual(float(got), float(expected), places=5)

    def test_build_signature_has_no_validation_labels(self):
        import inspect
        params = list(inspect.signature(t14.build_fold_matrices).parameters)
        self.assertEqual(params, ['F_train', 'X_train', 'y_train', 'F_valid', 'scheme', 'c'])

    def test_group_inner_folds_keep_groups_together_and_oof_ignores_own_fold(self):
        a, b = self.split(self.geo)
        c = self.c | dict(expected_model_columns=len(self._cols(a, b)))
        built = t14.build_fold_matrices(self.F.iloc[a], self.X.iloc[a], self.y.iloc[a], self.F.iloc[b], 'geo2_s101', c)
        inner = built['inner_folds']
        comp = (self.F['geo_level_1_id'].iloc[a].astype(str) + '|' + self.F['geo_level_2_id'].iloc[a].astype(str)).to_numpy()
        owner = {}
        for g, f in zip(comp, inner):
            self.assertEqual(owner.setdefault(g, f), f)
        # recompute geo2 class-1 OOF encoding of one training row from rows OUTSIDE its inner fold
        i = 0
        out_rows = np.flatnonzero(inner != inner[i])
        ytr = self.y.iloc[a].to_numpy()
        prior = float((ytr[out_rows] == 1).mean())
        same = out_rows[self.F['geo_level_2_id'].iloc[a].to_numpy()[out_rows] == self.F['geo_level_2_id'].iloc[a].iloc[i]]
        expected = ((ytr[same] == 1).sum() + 20*prior)/(len(same) + 20)
        self.assertAlmostEqual(float(built['Ao']['geo_level_2_id_te0'].iloc[i]), float(expected), places=5)
        self.assertEqual(len(same), 0)  # group-aware: the row's own group never appears in the other inner folds

    def test_ros_sees_training_rows_only(self):
        a, b = self.split(self.strat)
        c = self.c | dict(expected_model_columns=len(self._cols(a, b)))
        seen = {}
        real = t14.RandomOverSampler.fit_resample

        def spy(obj, X, y):
            seen['n_in'] = len(X)
            return real(obj, X, y)
        with mock.patch.object(t14.RandomOverSampler, 'fit_resample', spy):
            built = t14.build_fold_matrices(self.F.iloc[a], self.X.iloc[a], self.y.iloc[a], self.F.iloc[b], 'strat', c)
        self.assertEqual(seen['n_in'], len(a))
        counts = np.bincount(built['yos'])
        self.assertTrue((counts == counts.max()).all())
        self.assertEqual(len(built['Bo']), len(b))                    # validation not resampled
        merged = built['Xos'].drop_duplicates().merge(built['Ao'].drop_duplicates(), how='left', indicator=True)
        self.assertTrue((merged['_merge'] == 'both').all())            # every resampled row is a training row
        valid_only = built['Bo'].merge(built['Ao'], how='inner')
        self.assertLess(len(valid_only), len(b))                       # validation rows are not copied into training


@unittest.skipUnless(HAVE, 'dependencies not installed')
class RuleTests(unittest.TestCase):
    def setUp(self):
        c = t14.config()
        self.rules, self.rec = c['rules'], c['recorded_reference']['folds']

    def obs(self, shift=None):
        out = []
        for r in self.rec:
            cm = np.asarray(r['confusion']).copy()
            if shift:
                cm = shift(cm)
            tp = np.diag(cm); f1 = np.mean(2*tp/(cm.sum(0) + cm.sum(1)))
            out.append(dict(confusion=cm.tolist(), macro_f1=float(f1)))
        return out

    def test_exact(self):
        v = t14.judge_stage_a(self.rules, self.rec, self.obs())
        self.assertEqual(v['status'], 'EXACT')

    def test_small_difference_is_within_tolerance(self):
        def shift(cm):
            cm = cm.copy(); cm[1, 1] -= 5; cm[1, 0] += 5
            return cm
        v = t14.judge_stage_a(self.rules, self.rec, self.obs(shift))
        self.assertEqual(v['status'], 'WITHIN_TOLERANCE', v['parts'])

    def test_large_difference_fails(self):
        def shift(cm):
            cm = cm.copy(); cm[2, 2] -= 800; cm[2, 0] += 800
            return cm
        v = t14.judge_stage_a(self.rules, self.rec, self.obs(shift))
        self.assertEqual(v['status'], 'FAIL')

    def test_stage_b_boundaries(self):
        o = dict(mean_delta=0.01, fold_wins=16, n_folds=20, pooled_grade3_recall_delta=-0.01, splits_with_positive_mean=3)
        lr = dict(mean_delta=0.001, fold_wins=16, n_folds=20, pooled_grade3_recall_delta=-0.01, splits_with_positive_mean=3)
        j = t14.judge_stage_b(self.rules, lr, o)
        self.assertTrue(j['B1_vs_LR_C']['holds'] and j['B2_vs_LGBM_O']['holds'])
        self.assertEqual(j['label'], 'B1_AND_B2')
        for k, v in [('mean_delta', 0.0), ('fold_wins', 15), ('pooled_grade3_recall_delta', -0.011),
                     ('splits_with_positive_mean', 2)]:
            self.assertFalse(t14.judge_stage_b(self.rules, dict(lr, **{k: v}), o)['B1_vs_LR_C']['holds'], k)
        self.assertFalse(t14.judge_stage_b(self.rules, lr, dict(o, mean_delta=0.0099))['B2_vs_LGBM_O']['holds'])
        self.assertEqual(t14.judge_stage_b(self.rules, dict(lr, mean_delta=-1), dict(o, mean_delta=-1))['label'], 'NEITHER')


@unittest.skipUnless(HAVE, 'dependencies not installed')
class CheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.y, cls.ids, cls.strat, cls.geo = data()
        cls.F = t14.original_frame(cls.X)
        cls.c = small_config()
        a, b = np.flatnonzero(cls.strat != 0), np.flatnonzero(cls.strat == 0)
        A, _ = t14.target_enc(cls.F.iloc[a], pd.Series(cls.y.to_numpy() - 1).iloc[a], cls.F.iloc[b], list(GEO))
        cls.c['expected_model_columns'] = len(pd.get_dummies(A, columns=[x for x in cls.F.columns if x in TEXT],
                                                             dtype=int).columns)
        cls.folds = {s: (cls.strat if s == 'strat' else cls.geo) for s in t14.SCHEMES}

    def fresh(self, tmp):
        out = Path(tmp)
        t14.atomic_write(out/'ledger.json', dict(study_id='t', folds={}))
        (out/'backup').mkdir()
        return out

    def run_fold(self, out, scheme='strat', f=0):
        return t14.fit_fold(self.X, self.F, self.y, self.ids, self.folds, out, scheme, f, design_for(self.c))

    def test_fold_checkpoint_reload_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = self.fresh(tmp)
            entry, fitted = self.run_fold(out)
            self.assertTrue(fitted)
            d = out/'strat'/'RF_TE_ROS'
            for name in ['fold_0_model.joblib', 'fold_0_encoder_state.joblib', 'fold_0_pred.csv.gz', 'fold_0.json']:
                self.assertTrue((d/name).exists() and (out/'backup'/'strat'/'RF_TE_ROS'/name).exists())
                self.assertEqual(entry['artifact_sha256'][name], t14.sha256_file(d/name))
            rec = json.loads((d/'fold_0.json').read_text(encoding='utf-8'))
            self.assertLessEqual(rec['reload_probability_max_abs_diff'], 1e-12)
            self.assertEqual(rec['inner_te_folds'], 'stratified_random_rs0')
            self.assertEqual(rec['valid_rows'], int((self.strat == 0).sum()))
            led = t14.read_ledger(out)['folds']['strat/RF_TE_ROS/0']
            self.assertEqual(led['status'], 'complete')
            # completed fold is verified, not refit
            with mock.patch.object(t14, 'fit_rf', side_effect=AssertionError('must not refit')):
                e2, fitted2 = self.run_fold(out)
            self.assertFalse(fitted2)
            self.assertEqual(e2['artifact_sha256'], entry['artifact_sha256'])
            # tampering with a saved artifact or its backup is detected
            (d/'fold_0.json').write_text('{"changed": 1}')
            with self.assertRaises(ValueError):
                self.run_fold(out)

    def test_interrupted_fold_is_resumed_and_stale_files_set_aside(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = self.fresh(tmp)
            d = out/'strat'/'RF_TE_ROS'
            d.mkdir(parents=True)
            (d/'fold_0_model.joblib').write_bytes(b'partial')
            t14.atomic_write(out/'ledger.json', dict(folds={'strat/RF_TE_ROS/0': dict(
                dir='strat/RF_TE_ROS', status='running', attempts=1)}))
            entry, fitted = self.run_fold(out)
            self.assertTrue(fitted)
            self.assertEqual(entry['attempts'], 2)
            self.assertTrue((d/'fold_0_model.joblib.incomplete_attempt1').exists())
            self.assertNotEqual((d/'fold_0_model.joblib').read_bytes(), b'partial')

    def test_unregistered_target_and_fit_budget_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = self.fresh(tmp)
            with self.assertRaises(ValueError):
                t14.fit_fold(self.X, self.F, self.y, self.ids, self.folds, out, 'geo2_s999', 0, design_for(self.c))
            des = design_for(self.c) | dict(max_new_study_fits=0)
            with self.assertRaises(ValueError):
                t14.fit_fold(self.X, self.F, self.y, self.ids, self.folds, out, 'strat', 0, des)

    def test_stage_b_requires_passed_stage_a(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = self.fresh(tmp)
            with self.assertRaises(ValueError):
                self.run_fold(out, 'geo2_s101', 0)
            (out/'stage_a_result.json').write_text(json.dumps(dict(status='FAIL')))
            with self.assertRaises(ValueError):
                self.run_fold(out, 'geo2_s101', 0)
            (out/'stage_a_result.json').write_text(json.dumps(dict(status='WITHIN_TOLERANCE')))
            entry, fitted = self.run_fold(out, 'geo2_s101', 0)
            self.assertTrue(fitted)
            rec = json.loads((out/'geo2_s101'/'RF_TE_ROS'/'fold_0.json').read_text(encoding='utf-8'))
            self.assertEqual(rec['inner_te_folds'], 'stratified_group_geo2_seed42')
            # validation groups are unseen in training for the Group split
            self.assertEqual(rec['geo_diagnostics']['geo_level_2_id']['valid_rows_with_id_seen_in_train'], 0.0)

    def test_holdout_is_never_opened(self):
        opened = []
        real_open = builtins.open

        def spy(file, *a, **k):
            opened.append(str(file))
            return real_open(file, *a, **k)
        with tempfile.TemporaryDirectory() as tmp:
            out = self.fresh(tmp)
            with mock.patch.object(builtins, 'open', spy):
                self.run_fold(out)
        self.assertTrue(opened)
        self.assertFalse([p for p in opened if 'final_holdout' in p.replace('\\', '/').lower()])

    def test_holdout_paths_are_refused_and_source_has_no_holdout_reads(self):
        with self.assertRaises(ValueError):
            t14.require_not_holdout('earthquake/data/final_holdout/holdout.csv')
        with self.assertRaises(ValueError):
            t14.load_inputs('earthquake/data/final_holdout')
        src = (ROOT/'experiments/te_ros_rf_repro/run.py').read_text(encoding='utf-8')
        self.assertEqual(src.count("'final_holdout'"), 1)  # only the guard constant


if __name__ == '__main__':
    unittest.main()
