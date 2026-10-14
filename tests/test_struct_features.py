"""T12 synthetic contracts only: no real development data, no holdout access.

Fits here are synthetic sample checks, never study training runs.
"""
import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'tests'))
from test_modeling import synthetic  # noqa: E402
from preprocess import GEO  # noqa: E402

try:
    import lightgbm  # noqa: F401
    from experiments.struct_features import features as F
    from experiments.struct_features import run as t12
    HAVE_LGBM = True
except ImportError:  # pragma: no cover
    HAVE_LGBM = False


def data(n=400, seed=7):
    X, _, _, _ = synthetic(n)
    X.loc[[0, 17, 55], 'age'] = 995
    X.loc[[3, 4], 'count_floors_pre_eq'] = 1
    rng = np.random.default_rng(seed)
    geo = X.geo_level_1_id.astype(int).to_numpy()
    y = pd.Series(np.clip(1 + (geo % 3) + (rng.random(n) < 0.2), 1, 3).astype(int))
    ids = pd.Series(np.arange(1, n + 1), name='building_id')
    return X, y, ids


@unittest.skipUnless(HAVE_LGBM, 'lightgbm not installed')
class FeatureTests(unittest.TestCase):
    def test_formulas_hand_computed_including_floor_guard(self):
        edge = t12.synthetic_edge_frame()
        got = F.derive(edge, 'SD')
        np.testing.assert_allclose(got, t12.SYNTH_EXPECTED, rtol=0, atol=1e-12)
        np.testing.assert_allclose(got, np.array([t12.py_reference(r) for r in edge.to_dict('records')]), rtol=0, atol=1e-12)
        self.assertEqual(list(got[0, 4:]), [10.0, 0.0, 2.0])  # floors 0 -> divisor 1, product 0
        self.assertEqual(list(got[1, 4:]), [10.0, 7.0, 2.0])  # floors 1

    def test_bundles_are_column_subsets_of_sd(self):
        edge = t12.synthetic_edge_frame()
        sd = F.derive(edge, 'SD')
        np.testing.assert_array_equal(F.derive(edge, 'S'), sd[:, :4])
        np.testing.assert_array_equal(F.derive(edge, 'D'), sd[:, 4:])
        self.assertEqual(F.NAMES['SD'], F.NAMES['S'] + F.NAMES['D'])

    def test_n_use_excludes_aggregate_flag(self):
        self.assertNotIn('has_secondary_use', F.USE)
        self.assertEqual(len(F.USE), 10)
        self.assertEqual(len(F.SUPER), 11)
        self.assertEqual(len(F.WEAK) + len(F.STRONG), 10)  # 'other' belongs to n_super only
        edge = t12.synthetic_edge_frame()
        self.assertEqual(edge.loc[3, 'has_secondary_use'], 1)
        self.assertEqual(F.derive(edge, 'S')[3, 3], 0.0)

    def test_no_geo_age_or_categorical_inputs(self):
        used = set(sum(F.INPUTS.values(), []))
        from preprocess import CATEGORICAL
        self.assertFalse(used & set(GEO))
        self.assertFalse(used & set(CATEGORICAL))
        self.assertNotIn('age', used)
        self.assertFalse({'age_band', 'geo_foundation', 'geo_age', 'foundation_age'} & {n for v in F.NAMES.values() for n in v})

    def test_age_and_geo_do_not_change_features(self):
        X, _, _ = data(120)
        X2 = X.copy()
        X2['age'] = 995
        X2['geo_level_1_id'] = 0
        for bundle in F.BUNDLES:
            np.testing.assert_array_equal(F.derive(X, bundle), F.derive(X2, bundle))

    def test_transformer_schema_and_statelessness(self):
        X, _, _ = data(60)
        step = F.StructFeatures('SD')
        sub = X[F.INPUTS['SD']]
        step.fit(sub)
        np.testing.assert_array_equal(step.transform(sub), F.derive(sub, 'SD'))
        with self.assertRaises(ValueError):
            step.transform(sub[list(reversed(F.INPUTS['SD']))])
        with self.assertRaises(ValueError):
            F.StructFeatures('S').fit(X[F.INPUTS['D']])
        with self.assertRaises(ValueError):
            F.StructFeatures('X').fit(sub)


@unittest.skipUnless(HAVE_LGBM, 'lightgbm not installed')
class MatrixTests(unittest.TestCase):
    def test_column_append_invariance_and_independent_reference(self):
        from experiments.age995.run import make_pipeline
        X, y, _ = data(400)
        a, b = np.arange(320), np.arange(320, 400)
        pre_c = make_pipeline('C')['preprocess'].fit(X.iloc[a])
        mc_a, mc_b = pre_c.transform(X.iloc[a]), pre_c.transform(X.iloc[b])
        for build in (lambda bd: t12.make_lr_pipeline(bd)['preprocess'], t12.make_lgbm_preprocessor):
            for bundle in F.BUNDLES:
                pre = build(bundle)
                ma, mb = pre.fit_transform(X.iloc[a]), pre.transform(X.iloc[b])
                names = t12.check_matrix(pre, mb, bundle)
                k = len(F.NAMES[bundle])
                self.assertTrue(t12.same_prefix_as_C(pre_c, mc_a, ma, names, k)[0])
                self.assertTrue(t12.same_prefix_as_C(pre_c, mc_b, mb, names, k)[0])
                self.assertEqual(ma.shape[1], mc_a.shape[1] + k)
                self.assertFalse(any(n.startswith('struct__') for n in names[:-k]))
                rt, rv, mean, std = t12.reference_scaled(X.iloc[a], X.iloc[b], bundle)
                np.testing.assert_allclose(ma[:, -k:].toarray(), rt, rtol=1e-5, atol=2e-5)
                np.testing.assert_allclose(mb[:, -k:].toarray(), rv, rtol=1e-5, atol=2e-5)
                np.testing.assert_allclose(pre['columns'].named_transformers_['struct']['scale'].mean_, mean, atol=1e-12)
        # a tampered C column is detected
        pre = t12.make_lgbm_preprocessor('S')
        ma = pre.fit_transform(X.iloc[a])
        names = t12.check_matrix(pre, ma, 'S')
        bad = ma.tolil(); bad[0, 1] = bad[0, 1] + 1
        self.assertFalse(t12.same_prefix_as_C(pre_c, mc_a, bad.tocsr(), names, 4)[0])

    def test_scaler_fit_on_training_fold_only(self):
        X, _, _ = data(400)
        a, b = np.arange(320), np.arange(320, 400)
        pre = t12.make_lgbm_preprocessor('SD')
        pre.fit(X.iloc[a])
        scaler = pre['columns'].named_transformers_['struct']['scale']
        np.testing.assert_allclose(scaler.mean_, F.derive(X.iloc[a], 'SD').mean(0), atol=1e-12)
        before = scaler.mean_.copy()
        pre.transform(X.iloc[b])
        np.testing.assert_array_equal(scaler.mean_, before)

    def test_fit_arm_smoke_and_artifacts_in_fresh_process(self):
        X, y, _ = data(400)
        a, b = np.arange(320), np.arange(320, 400)
        c = t12.config()
        c['lgbm_params'] = dict(c['lgbm_params'], n_estimators=4, verbosity=-1)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            paths, probs = {}, {}
            for arm in ('LR_SD', 'LGBM_S'):
                pipe, pre, model, p, p_train, mt, mv, names, info = t12.fit_arm(arm, X.iloc[a], y.iloc[a], X.iloc[b], c)
                self.assertTrue(all(b'__main__' not in pickle.dumps(o) for o in (pipe, pre)))
                if arm == 'LR_SD':
                    self.assertTrue(info['converged'])
                    joblib.dump(pipe, tmp/'lr.joblib')
                    paths['lr'], probs['lr'] = str(tmp/'lr.joblib'), p.tolist()
                else:
                    self.assertEqual(info['rounds_completed'], 4)
                    model.booster_.save_model(str(tmp/'m.txt'))
                    joblib.dump(pre, tmp/'pre.joblib')
                    paths['lgbm'], probs['lgbm'] = str(tmp/'pre.joblib'), p.tolist()
            X.iloc[b].to_pickle(tmp/'x.pkl')
            (tmp/'ref.json').write_text(json.dumps(dict(paths=paths, probs=probs, lgbm_model=str(tmp/'m.txt'))))
            out = t12.fresh_process_check(tmp/'ref.json', tmp/'x.pkl')
            for v in out.values():
                self.assertTrue(v['main_free'])
                self.assertLessEqual(v['max_abs_diff'], 1e-12)
                self.assertNotIn('__main__', v['modules'])
                self.assertIn('experiments.struct_features.features', v['modules'] + ['experiments.struct_features.features'])


@unittest.skipUnless(HAVE_LGBM, 'lightgbm not installed')
class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.X, self.y, self.ids = data(300)
        self.folds = {'stratified': np.arange(300) % 5}
        cfg = t12.config()
        cfg['lgbm_params'] = dict(cfg['lgbm_params'], n_estimators=4, verbosity=-1)
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)
        (self.out/'backup').mkdir()
        t12.atomic_write(self.out/'ledger.json', dict(study_id=t12.STUDY_ID, folds={}))
        self.design = dict(config=cfg, study_targets=['stratified/LGBM_SD/0', 'stratified/LGBM_SD/1', 'stratified/LR_S/0'],
                           max_new_study_fits=90)

    def tearDown(self):
        self.tmp.cleanup()

    def test_resume_skips_completed_folds(self):
        for arm in ['LGBM_SD', 'LR_S']:
            entry, fitted = t12.fit_fold(self.X, self.y, self.ids, self.folds, self.out, 'stratified', arm, 0, self.design)
            self.assertTrue(fitted)
            self.assertEqual(entry['status'], 'complete')
            with mock.patch.object(t12, 'fit_arm', side_effect=AssertionError('must not refit')):
                again, fitted2 = t12.fit_fold(self.X, self.y, self.ids, self.folds, self.out, 'stratified', arm, 0,
                                              self.design)
            self.assertFalse(fitted2)
            self.assertEqual(t12.read_ledger(self.out)['folds'][f'stratified/{arm}/0']['attempts'], 1)
            for name in entry['artifact_sha256']:
                self.assertTrue((self.out/'backup'/'stratified'/arm/name).exists())
            rec = json.loads((self.out/'stratified'/arm/'fold_0.json').read_text())
            self.assertTrue(rec['c_columns_unchanged'])

    def test_tampered_artifact_detected_on_resume(self):
        t12.fit_fold(self.X, self.y, self.ids, self.folds, self.out, 'stratified', 'LGBM_SD', 0, self.design)
        path = self.out/'stratified'/'LGBM_SD'/'fold_0.json'
        path.write_text(path.read_text() + ' ', encoding='utf-8')
        with self.assertRaises(ValueError):
            t12.fit_fold(self.X, self.y, self.ids, self.folds, self.out, 'stratified', 'LGBM_SD', 0, self.design)

    def test_unregistered_target_refused(self):
        with self.assertRaises(ValueError):
            t12.fit_fold(self.X, self.y, self.ids, self.folds, self.out, 'geo2', 'LGBM_SD', 0, self.design)

    def test_fold_csv_round_trip_exact(self):
        t12.fit_fold(self.X, self.y, self.ids, self.folds, self.out, 'stratified', 'LGBM_SD', 1, self.design)
        back = t12.read_pred(self.out/'stratified'/'LGBM_SD'/'fold_1_pred.csv.gz')
        self.assertEqual(list(back.columns), ['building_id', 'fold', 'y_true', 'y_pred', *t12.PROBS])
        rng = np.random.default_rng(3)
        p = rng.dirichlet([1, 1, 1], 50)
        frame = pd.DataFrame({'building_id': np.arange(50), 'fold': 0, 'y_true': 1, 'y_pred': p.argmax(1) + 1,
                              **{n: p[:, i] for i, n in enumerate(t12.PROBS)}})
        path = self.out/'rt.csv.gz'
        t12.write_pred(path, frame)
        np.testing.assert_array_equal(t12.read_pred(path)[t12.PROBS].to_numpy(), p)


@unittest.skipUnless(HAVE_LGBM, 'lightgbm not installed')
class RuleTests(unittest.TestCase):
    def block(self, base_mean, arm_mean, wins, rec_delta=0.0, err_ratio=1.0, base='LR_C', arm='LR_S'):
        def pooled(r, e):
            return dict(recall=[0, 0, r], error32=e)
        comp = {f'{arm}_vs_{base}': dict(mean_delta=arm_mean - base_mean, fold_wins=wins, pooled_grade3_recall_delta=rec_delta)}
        return {base: dict(pooled=pooled(0.5, 1000)), arm: dict(pooled=pooled(0.5 + rec_delta, 1000 * err_ratio)),
                'comparisons': comp}

    def test_joint_candidate_requires_all_conditions(self):
        r = t12.config()['rules']['joint_improvement_candidate']
        ok = self.block(0.6, 0.604, 4)
        self.assertTrue(t12.judge(ok, ok, 'LR_S', r)['joint_improvement_candidate'])
        for st, g2 in [(self.block(0.6, 0.6029, 4), ok),                       # mean too small
                       (ok, self.block(0.6, 0.604, 3)),                        # wins too few
                       (ok, self.block(0.6, 0.604, 5, rec_delta=-0.006)),      # grade-3 recall
                       (self.block(0.6, 0.604, 5, err_ratio=1.02), ok)]:       # 3->2 count
            self.assertFalse(t12.judge(st, g2, 'LR_S', r)['joint_improvement_candidate'])
        lg = self.block(0.6, 0.604, 4, base='LGBM_O', arm='LGBM_SD')
        self.assertTrue(t12.judge(lg, lg, 'LGBM_SD', r)['joint_improvement_candidate'])


@unittest.skipUnless(HAVE_LGBM, 'lightgbm not installed')
class ConfigTests(unittest.TestCase):
    def test_plan_fixed(self):
        from experiments.model_families.run import load_config
        from experiments.age995.run import PARAMS
        c = t12.config()
        self.assertEqual(c['lgbm_params'], load_config()['models']['LGBM'])
        self.assertEqual(c['lr_params'], PARAMS)
        self.assertEqual(len(t12.targets()), 90)
        self.assertEqual(len(set(t12.targets())), 90)
        self.assertEqual(c['automatic_retries'], 0)
        self.assertFalse(c['holdout_access'])
        self.assertEqual({a.split('_')[1] for a in t12.ARMS}, {'S', 'D', 'SD'})
        r = c['rules']['joint_improvement_candidate']
        self.assertEqual((r['stratified_mean_delta_min'], r['geo2_mean_delta_min'], r['stratified_fold_wins_min'],
                          r['pooled_grade3_recall_delta_min'], r['pooled_error32_ratio_max']),
                         (0.003, 0.003, 4, -0.005, 1.01))

    def test_sealed_sources_exist(self):
        for rel in t12.SOURCE:
            if rel.startswith('docs/'):
                continue  # plan is committed together with the code before the seal
            self.assertTrue((ROOT/rel).exists(), rel)


if __name__ == '__main__':
    unittest.main()
