"""T33 synthetic contracts only: no real development data, no holdout access.

Fits here are tiny synthetic checks, never study training runs.  Tests that need the private notebook copy
(outside Git) are skipped when it is absent; pass and skip are reported separately by unittest.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from preprocess import ALLOWED_TEXT, BINARY, FEATURES, normalize_features  # noqa: E402

try:
    import lightgbm  # noqa: F401
    from experiments.ey_pipeline_benchmark import features as F
    from experiments.ey_pipeline_benchmark import original_check as OC
    from experiments.ey_pipeline_benchmark import run as T33
    HAVE = True
except ImportError:  # pragma: no cover
    HAVE = False

CONFIG = ROOT/'experiments'/'ey_pipeline_benchmark'/'config.json'
PRIVATE = ROOT.parents[1]/'local_reproduction'/'T33_20261016_ey_pipeline'/'source_copy'
HAVE_PRIVATE = (PRIVATE/'지진.ipynb').is_file()


def synth(n=1500, seed=3):
    rng = np.random.default_rng(seed)
    g1 = rng.integers(0, 10, n)
    g2 = g1 * 100 + rng.integers(0, 25, n)
    g3 = g2 * 50 + rng.integers(0, 15, n)
    d = {'geo_level_1_id': g1, 'geo_level_2_id': g2, 'geo_level_3_id': g3,
         'count_floors_pre_eq': rng.integers(1, 5, n), 'age': rng.choice([0, 5, 10, 20, 30, 995], n),
         'area_percentage': rng.integers(1, 30, n), 'height_percentage': rng.integers(1, 12, n),
         'count_families': rng.integers(0, 4, n)}
    for c, vals in ALLOWED_TEXT.items():
        d[c] = rng.choice(vals, n)
    for c in BINARY:
        d[c] = (rng.random(n) < 0.12).astype(int)
    X = normalize_features(pd.DataFrame(d)[FEATURES])
    s = X.age.clip(upper=60) / 30 + X.has_superstructure_adobe_mud + rng.normal(0, .8, n) + (g1 % 3) * 0.4
    y = pd.Series(np.digitize(s, np.quantile(s, [0.2, 0.7])) + 1)
    return X, y


def tiny_config(**over):
    c = json.loads(CONFIG.read_text(encoding='utf-8'))
    c['es'].update(max_rounds=40, patience=5)
    c['lgbm_params']['n_jobs'] = 2
    c['budget'] = dict(c['budget'])
    c['budget'].update(over)
    return c


def make_run(td, **budget):
    """A synthetic study folder + context (design built by hand, not by `prepare`)."""
    X, y = synth()
    ids = pd.Series(np.arange(1, len(X) + 1), name='building_id')
    folds = np.arange(len(X)) % 5
    c = tiny_config(**budget)
    design = dict(config=c, max_fits=c['budget']['max_fits'], study_targets=[T33.key(*t) for t in T33.targets()],
                  lightgbm_categorical_effective_defaults={k: float(v) for k, v in
                                                           c['lightgbm_categorical_defaults_expected'].items()})
    out = Path(td)
    (out/'backup').mkdir(parents=True)
    T33.atomic_write(out/'ledger.json', dict(study_id=T33.STUDY_ID, units={}, features={}))
    return T33.Context(X, y, ids, folds, out, design), out


@unittest.skipUnless(HAVE, 'lightgbm not installed')
class FeatureContractTests(unittest.TestCase):
    def test_75_columns_in_design_order(self):
        cols = F.feature_columns()
        self.assertEqual(len(cols), 75)
        self.assertEqual(len(set(cols)), 75)
        self.assertEqual(cols, json.loads(CONFIG.read_text(encoding='utf-8'))['features']['columns'])
        self.assertEqual(cols[:3], ['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id'])
        self.assertEqual(cols[38:42], ['material_group', 'structure_type', 'n_weak_material', 'n_material'])
        self.assertEqual(cols[-5:], ['weak_age', 'g2_material_high_gap', 'g2_material_low_gap',
                                     'g3_material_high_gap', 'g3_material_low_gap'])
        self.assertEqual([cols.index(c) + 1 for c in F.CAT_COLS], [8, 9, 10, 11, 12, 13, 14, 26, 39, 40])
        self.assertNotIn('material_high_rate', cols)

    def test_built_matrices_follow_the_order_and_types(self):
        X, y = synth()
        fr = F.notebook_frame(X, y)
        tr, ev = fr.iloc[:1200], fr.iloc[1200:]
        A, B, tables = F.build_features(tr, ev)
        self.assertEqual(list(A.columns), F.feature_columns())
        self.assertEqual(list(B.columns), F.feature_columns())
        for c in F.CAT_COLS:
            self.assertIsInstance(A[c].dtype, pd.CategoricalDtype)
            self.assertEqual(list(A[c].cat.categories), list(B[c].cat.categories))
        self.assertFalse(A.drop(columns=F.CAT_COLS).isna().any().any())
        self.assertFalse(B.drop(columns=F.CAT_COLS).isna().any().any())

    def test_region_ids_restored_to_integers_with_round_trip(self):
        X, y = synth(200)
        fr = F.notebook_frame(X, y)
        for c in ['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id']:
            self.assertEqual(fr[c].dtype, np.int64)
        for c in BINARY:
            self.assertEqual(fr[c].dtype, np.int64)
        bad = X.copy()
        bad.loc[bad.index[0], 'geo_level_2_id'] = '007'
        with self.assertRaises(ValueError):
            F.notebook_frame(bad, y)

    def test_per_row_derivations_by_hand(self):
        X, y = synth(50)
        fr = F.notebook_frame(X, y)
        for c in BINARY:
            fr[c] = 0
        fr['age'] = 40
        fr['count_floors_pre_eq'] = [1, 2, 3, 4, 1] + [1] * 45
        fr.loc[0, 'has_superstructure_adobe_mud'] = 1                       # mud_stone
        fr.loc[1, ['has_superstructure_adobe_mud', 'has_superstructure_rc_engineered']] = 1  # RC wins
        fr.loc[2, 'has_superstructure_timber'] = 1                          # timber_bamboo
        fr.loc[3, ['has_superstructure_adobe_mud', 'has_superstructure_cement_mortar_brick']] = 1  # cement wins over mud
        out = F.building_features(fr.iloc[:5])
        self.assertEqual(out.material_group.tolist()[:4], ['mud_stone', 'RC', 'timber_bamboo', 'cement'])
        self.assertEqual(out.material_group.iloc[4], 'other')
        self.assertEqual(out.n_weak_material.tolist(), [1, 1, 0, 1, 0])
        self.assertEqual(out.n_material.tolist(), [1, 2, 1, 2, 0])
        self.assertEqual(out.weak_age.tolist(), [40, 0, 0, 0, 0])
        self.assertEqual(out.floor_band.tolist(), ['1F', '2F', '3F+', '3F+', '1F'])
        r = fr.iloc[0]
        self.assertEqual(out.structure_type.iloc[0], f'{r.foundation_type}_{r.roof_type}_{r.ground_floor_type}')
        self.assertEqual(out.key_region_material.iloc[0], f'{r.geo_level_1_id}|mud_stone')

    def test_region_composition_by_hand(self):
        rows = []
        for i in range(30):   # region 7: 30 rows, 12 mud_stone, 3 RC, ages 0..29
            rows.append(dict(level=7, weak=int(i < 12), rc=int(12 <= i < 15), age=i))
        for i in range(5):    # region 9: 5 rows (< 20) -> overall values
            rows.append(dict(level=9, weak=1, rc=0, age=100 + i))
        d = pd.DataFrame(rows)
        train = pd.DataFrame({'geo_level_1_id': d.level, 'age': d.age,
                              'material_group': np.where(d.weak == 1, 'mud_stone', np.where(d.rc == 1, 'RC', 'other'))})
        t = F.region_table(train, 'geo_level_1_id')
        s = t['summary']
        self.assertEqual(int(s.loc[7, 'n_buildings']), 30)
        self.assertAlmostEqual(s.loc[7, 'weak_ratio'], 12 / 30)
        self.assertAlmostEqual(s.loc[7, 'rc_ratio'], 3 / 30)
        self.assertEqual(s.loc[7, 'mean_age'], float(np.median(np.arange(30))))
        overall_weak, overall_age = (17 / 35), float(np.median(d.age))
        self.assertAlmostEqual(s.loc[9, 'weak_ratio'], overall_weak)
        self.assertEqual(s.loc[9, 'mean_age'], overall_age)
        self.assertEqual(int(s.loc[9, 'n_buildings']), 5)
        ev = pd.DataFrame({'geo_level_1_id': [7, 11], 'age': [20, 30]})
        F.attach_region(ev, 'geo_level_1_id', 'g1', t)
        self.assertEqual(ev.g1_n_buildings.tolist(), [30, 0])
        self.assertAlmostEqual(ev.g1_weak_ratio.iloc[1], overall_weak)       # unseen region -> overall
        self.assertAlmostEqual(ev.g1_age_vs_region.iloc[0], 20 - np.median(np.arange(30)))
        self.assertAlmostEqual(ev.g1_age_vs_region.iloc[1], 30 - overall_age)

    def test_target_rate_oof_against_a_slow_independent_loop(self):
        rng = np.random.default_rng(5)
        n = 400
        train = pd.DataFrame({'key_g2': rng.choice(list('abcdefgh'), n), F.GRADE: rng.integers(1, 4, n)})
        out = F.attach_rates_oof(train, ['key_g2'])
        from sklearn.model_selection import KFold
        grade = train[F.GRADE].to_numpy()
        keys = train.key_g2.to_numpy()
        hi_all, lo_all = (grade == 3).mean(), (grade == 1).mean()
        expect_hi = np.zeros(n)
        expect_lo = np.zeros(n)
        for other, this in KFold(5, shuffle=True, random_state=42).split(train):
            for i in this:
                same = [j for j in other if keys[j] == keys[i]]
                if len(same) < F.KEY_MIN['key_g2']:                     # small combination -> other-part overall
                    expect_hi[i] = (grade[other] == 3).mean()
                    expect_lo[i] = (grade[other] == 1).mean()
                else:
                    expect_hi[i] = np.mean(grade[same] == 3)
                    expect_lo[i] = np.mean(grade[same] == 1)
        np.testing.assert_allclose(out.g2_high_rate.to_numpy(), expect_hi, rtol=0, atol=1e-15)
        np.testing.assert_allclose(out.g2_low_rate.to_numpy(), expect_lo, rtol=0, atol=1e-15)
        self.assertGreater(len(set(np.round(expect_hi, 6))), 3)
        self.assertEqual((hi_all, lo_all), ((grade == 3).mean(), (grade == 1).mean()))

    def test_class_weights(self):
        y0 = pd.Series([0] * 60 + [1] * 30 + [2] * 10)
        w = F.class_weights(y0, 0.3)
        self.assertAlmostEqual(w[0], (100 / 180) ** 0.3)
        self.assertAlmostEqual(w[-1], (100 / 30) ** 0.3)


@unittest.skipUnless(HAVE, 'lightgbm not installed')
class FoldInternalTests(unittest.TestCase):
    def setUp(self):
        X, y = synth()
        self.fr = F.notebook_frame(X, y)
        self.train = self.fr.iloc[:1200].reset_index(drop=True)
        self.valid = self.fr.iloc[1200:].reset_index(drop=True)
        self.y0 = self.train[F.GRADE].to_numpy() - 1
        self.tr_pos, self.va_pos = T33.inner_split(self.y0)

    def test_outer_validation_labels_never_touch_any_feature(self):
        A0, V0, _ = F.build_features(self.train, self.valid)
        shuffled = self.valid.copy()
        shuffled[F.GRADE] = np.random.default_rng(1).permutation(shuffled[F.GRADE].to_numpy())
        A1, V1, _ = F.build_features(self.train, shuffled)
        self.assertTrue(OC.frames_identical(A0, A1)[0])
        self.assertTrue(OC.frames_identical(V0, V1)[0])

    def test_training_labels_do_change_features_so_the_check_has_teeth(self):
        A0, V0, _ = F.build_features(self.train, self.valid)
        flipped = self.train.copy()
        flipped[F.GRADE] = (flipped[F.GRADE] % 3) + 1
        A1, V1, _ = F.build_features(flipped, self.valid)
        self.assertFalse(OC.frames_identical(A0, A1)[0])
        self.assertFalse(OC.frames_identical(V0, V1)[0])

    def test_label_probe_matches_the_design(self):
        res = F.label_probe(self.train, self.valid.drop(columns=[F.GRADE]), self.tr_pos, self.va_pos)
        real = F.label_probe(self.train, self.valid, self.tr_pos, self.va_pos)     # outer frame that already carries labels
        self.assertFalse(real['outer_label_column_attached'])
        self.assertTrue(real['passes'])
        self.assertTrue(res['passes'])
        self.assertGreater(res['inner_val_labels_changed_share'], 0.2)
        self.assertTrue(res['outer_label_column_attached'])                          # L3: a real stand-in label is shuffled
        self.assertGreater(res['outer_labels_changed_share'], 0.2)
        self.assertGreater(res['A_order_train_part']['rows_changed_share'], 0.5)   # E1: validation labels leak into tr rows
        self.assertEqual(res['B_order_train_part']['cells_changed_share'], 0)       # E2: no leak
        self.assertEqual(res['B_order_inner_val']['cells_changed_share'], 0)
        self.assertEqual(res['A_order_outer_valid_under_outer_label_shuffle']['cells_changed_share'], 0)

    def test_apply_tables_rebuilds_the_evaluation_matrix(self):
        _, V, tables = F.build_features(self.train, self.valid)
        again = F.apply_tables(self.valid, tables)
        self.assertTrue(OC.frames_identical(V, again)[0])

    def test_features_b_use_only_the_inner_training_rows(self):
        tr = self.train.iloc[self.tr_pos].reset_index(drop=True)
        va = self.train.iloc[self.va_pos].drop(columns=[F.GRADE]).reset_index(drop=True)
        B_tr, B_va, _ = F.features_B(self.train, self.tr_pos, self.va_pos)
        ref_tr, ref_va, _ = F.build_features(tr, va)
        self.assertTrue(OC.frames_identical(B_tr, ref_tr)[0] and OC.frames_identical(B_va, ref_va)[0])

    def test_matrix_fingerprint_changes_with_a_single_cell(self):
        A, _, _ = F.build_features(self.train, self.valid)
        h = F.matrix_sha256(A)
        B = A.copy()
        B.loc[3, 'g3_high_rate'] += 1e-12
        self.assertNotEqual(h, F.matrix_sha256(B))
        self.assertEqual(h, F.matrix_sha256(A.copy()))


@unittest.skipUnless(HAVE, 'lightgbm not installed')
class DecisionRuleTests(unittest.TestCase):
    rules = json.loads(CONFIG.read_text(encoding='utf-8'))['rules']

    def test_d1(self):
        r = self.rules['D1']
        ref = [0.70] * 5
        self.assertEqual(T33.d1_label([0.706] * 5, ref, r)['label'], 'ADVANTAGE_DESCRIPTIVE')
        self.assertEqual(T33.d1_label([0.706] * 3 + [0.69] * 2, ref, r)['label'], 'INDISTINGUISHABLE')   # only 3 wins
        self.assertEqual(T33.d1_label([0.694] * 5, ref, r)['label'], 'DISADVANTAGE_DESCRIPTIVE')
        self.assertEqual(T33.d1_label([0.703] * 5, ref, r)['label'], 'INDISTINGUISHABLE')
        self.assertEqual(T33.d1_label([0.70] * 5, ref, r)['fold_wins'], 0)

    def test_d3(self):
        r = self.rules['D3']
        base = [0.70, 0.70, 0.70, 0.70, 0.70]
        self.assertEqual(T33.d3_label(base, [0.7005] * 5, r)['label'], 'ES_PROTOCOL_EFFECT_SMALL')
        self.assertEqual(T33.d3_label(base, [0.70, 0.70, 0.70, 0.70, 0.7035], r)['label'], 'ES_PROTOCOL_EFFECT_NOTICEABLE')
        self.assertEqual(T33.d3_label(base, [0.702] * 5, r)['label'], 'ES_PROTOCOL_EFFECT_NOTICEABLE')

    def test_d4(self):
        r = self.rules['D4']
        self.assertEqual(T33.d4_label(0.7100, 0.7143, r)['label'], 'NEAR')
        self.assertEqual(T33.d4_label(0.7030, 0.7143, r)['label'], 'MODERATE')
        self.assertEqual(T33.d4_label(0.6900, 0.7143, r)['label'], 'FAR')


@unittest.skipUnless(HAVE, 'lightgbm not installed')
class ConfigAndGuardTests(unittest.TestCase):
    def test_config_is_the_registered_design(self):
        pinned = json.loads(CONFIG.read_text(encoding='utf-8'))['environment']
        if T33.environment() != pinned:
            self.skipTest('not the frozen environment')
        c = T33.config()
        self.assertEqual(c['budget']['planned_fits'], 20)
        self.assertEqual(c['budget']['max_fits'], 24)
        self.assertEqual(c['lgbm_params']['learning_rate'], 0.03)
        self.assertEqual(c['weights']['strength'], 0.3)

    def test_plan_pin_matches_the_plan_file(self):
        c = json.loads(CONFIG.read_text(encoding='utf-8'))
        p = ROOT/c['plan']['path']
        if not p.exists():
            self.skipTest('plan file absent')
        from preprocess import sha256_file
        self.assertEqual(sha256_file(p), c['plan']['sha256'])

    def test_holdout_path_is_refused(self):
        with self.assertRaises(ValueError):
            T33.require_not_holdout('C:/x/earthquake/data/final_holdout/holdout.csv')
        T33.require_not_holdout('C:/x/earthquake/data')

    def test_unplanned_arm_or_fold_is_not_registered(self):
        with tempfile.TemporaryDirectory() as td:
            ctx, _ = make_run(td)
            with self.assertRaises(ValueError):
                T33.fit_unit(ctx, 'E3', 0)
            self.assertEqual(len(ctx.design['study_targets']), 10)
            self.assertEqual(ctx.design['study_targets'][:2], ['E1/0', 'E2/0'])

    def test_lock_blocks_a_second_run(self):
        with tempfile.TemporaryDirectory() as td:
            with T33.run_lock(td):
                with self.assertRaises(ValueError):
                    with T33.run_lock(td):
                        pass
                self.assertTrue((Path(td)/'RUNNING.lock').exists())
            self.assertFalse((Path(td)/'RUNNING.lock').exists())


@unittest.skipUnless(HAVE, 'lightgbm not installed')
class UnitFlowTests(unittest.TestCase):
    def test_both_arms_share_the_refit_matrix_and_differ_only_in_early_stopping(self):
        with tempfile.TemporaryDirectory() as td:
            ctx, out = make_run(td)
            e1, ran1 = T33.fit_unit(ctx, 'E1', 0)
            e2, ran2 = T33.fit_unit(ctx, 'E2', 0)
            self.assertTrue(ran1 and ran2)
            r1 = json.loads((out/'E1'/'fold_0.json').read_text(encoding='utf-8'))
            r2 = json.loads((out/'E2'/'fold_0.json').read_text(encoding='utf-8'))
            s1 = json.loads((out/'E1'/'fold_0_es.json').read_text(encoding='utf-8'))
            s2 = json.loads((out/'E2'/'fold_0_es.json').read_text(encoding='utf-8'))
            self.assertEqual(r1['features_A'], r2['features_A'])                       # same refit matrices
            self.assertEqual(s1['tr_building_id_sha256'], s2['tr_building_id_sha256'])  # same inner split
            self.assertEqual(s1['val_building_id_sha256'], s2['val_building_id_sha256'])
            self.assertNotEqual(s1['es_train_matrix_sha256'], s2['es_train_matrix_sha256'])  # only ES features differ
            self.assertIsNone(s1['features_B_tables'])
            self.assertIsNotNone(s2['features_B_tables'])
            self.assertTrue((out/'E2'/'fold_0_features_B_state.joblib').exists())
            self.assertFalse((out/'E1'/'fold_0_features_B_state.joblib').exists())
            self.assertEqual(r1['effective_parameters']['learning_rate'], r2['effective_parameters']['learning_rate'])
            for r, s in ((r1, s1), (r2, s2)):
                self.assertEqual(r['n_trees'], s['n_trees'])
                self.assertEqual(r['effective_parameters']['n_estimators'], s['n_trees'])
            # same tree count -> identical refit (deterministic), different tree count -> they may differ
            if r1['n_trees'] == r2['n_trees']:
                a = pd.read_csv(out/'E1'/'fold_0_pred.csv.gz', float_precision='round_trip')
                b = pd.read_csv(out/'E2'/'fold_0_pred.csv.gz', float_precision='round_trip')
                np.testing.assert_array_equal(a[T33.PROBS].to_numpy(), b[T33.PROBS].to_numpy())
            led = T33.read_ledger(out)
            self.assertEqual(T33.fits_started(led), 4)
            self.assertEqual(set(led['features']), {'A/0'})
            T33.check_saved(out, led['units']['E1/0'])
            T33.check_saved(out, led['units']['E2/0'])

    def test_complete_units_are_not_refit(self):
        with tempfile.TemporaryDirectory() as td:
            ctx, out = make_run(td)
            T33.fit_unit(ctx, 'E1', 1)
            with mock.patch.object(T33, 'fit_lgbm', side_effect=AssertionError('refit')):
                entry, ran = T33.fit_unit(ctx, 'E1', 1)
            self.assertFalse(ran)
            self.assertEqual(T33.fits_started(T33.read_ledger(out)), 2)

    def test_resume_after_crash_in_the_refit_does_not_repeat_early_stopping(self):
        with tempfile.TemporaryDirectory() as td:
            ctx, out = make_run(td)
            real = T33.fit_lgbm
            calls = {'n': 0}

            def flaky(*a, **k):
                calls['n'] += 1
                if calls['n'] == 2:
                    raise RuntimeError('simulated crash')
                return real(*a, **k)
            with mock.patch.object(T33, 'fit_lgbm', side_effect=flaky):
                with self.assertRaises(RuntimeError):
                    T33.fit_unit(ctx, 'E1', 0)
            led = T33.read_ledger(out)
            self.assertEqual(led['units']['E1/0']['status'], 'final_running')
            es_sha = led['units']['E1/0']['artifact_sha256']['fold_0_es_model.txt']
            self.assertEqual((led['units']['E1/0']['es_attempts'], led['units']['E1/0']['final_attempts']), (1, 1))
            T33.fit_unit(ctx, 'E1', 0)                                   # manual resume: refit only
            led = T33.read_ledger(out)
            e = led['units']['E1/0']
            self.assertEqual((e['status'], e['es_attempts'], e['final_attempts']), ('complete', 1, 2))
            self.assertEqual(e['artifact_sha256']['fold_0_es_model.txt'], es_sha)
            self.assertEqual(T33.recovered_units(led), 1)
            self.assertEqual(T33.fits_started(led), 3)

    def test_crash_during_early_stopping_moves_stale_files_and_counts_the_attempt(self):
        with tempfile.TemporaryDirectory() as td:
            ctx, out = make_run(td)
            real = T33.fit_lgbm
            calls = {'n': 0}

            def flaky(*a, **k):
                calls['n'] += 1
                if calls['n'] == 1:
                    (out/'E2').mkdir(exist_ok=True)
                    (out/'E2'/'fold_0_es_model.txt').write_text('partial')
                    raise RuntimeError('simulated crash')
                return real(*a, **k)
            with mock.patch.object(T33, 'fit_lgbm', side_effect=flaky):
                with self.assertRaises(RuntimeError):
                    T33.fit_unit(ctx, 'E2', 0)
            T33.fit_unit(ctx, 'E2', 0)
            self.assertTrue(list((out/'E2').glob('fold_0_es_model.txt.incomplete_attempt1_*')))
            led = T33.read_ledger(out)
            self.assertEqual((led['units']['E2/0']['es_attempts'], led['units']['E2/0']['final_attempts']), (2, 1))

    def test_fit_budget_cap_is_enforced(self):
        with tempfile.TemporaryDirectory() as td:
            ctx, out = make_run(td, max_fits=3)
            T33.fit_unit(ctx, 'E1', 0)                                    # 2 fits
            with self.assertRaises(ValueError) as cm:
                T33.fit_unit(ctx, 'E2', 0)                                 # 3rd ok (ES), 4th refused
            self.assertIn('budget', str(cm.exception).lower())
            led = T33.read_ledger(out)
            self.assertEqual(T33.fits_started(led), 3)
            self.assertEqual(led['units']['E2/0']['status'], 'es_done')

    def test_at_most_two_units_may_be_recovered(self):
        with tempfile.TemporaryDirectory() as td:
            ctx, out = make_run(td)
            led = T33.read_ledger(out)
            for k in ('E1/0', 'E2/0'):
                led['units'][k] = dict(dir=k[:2], arm=k[:2], fold=0, status='complete', es_attempts=2, final_attempts=1,
                                       artifact_sha256={})
            T33.atomic_write(out/'ledger.json', led)
            with self.assertRaises(ValueError) as cm:
                T33.begin_fit(ctx, 'E1', 1, 'es')
                T33.begin_fit(ctx, 'E1', 1, 'es')                          # a repeat of the same stage = third recovery
            self.assertIn('recovered', str(cm.exception))

    def test_features_a_are_checked_against_the_ledger_when_rebuilt(self):
        with tempfile.TemporaryDirectory() as td:
            ctx, out = make_run(td)
            ctx.get_A(2)
            led = T33.read_ledger(out)
            led['features']['A/2']['train_matrix_sha256'] = '0' * 64
            T33.atomic_write(out/'ledger.json', led)
            ctx._A = {}
            with self.assertRaises(ValueError):
                ctx.get_A(2)

    def test_tree_count_cross_check(self):
        import lightgbm as lgb
        X, y = synth(600)
        fr = F.notebook_frame(X, y)
        train = fr.iloc[:480].reset_index(drop=True)
        valid = fr.iloc[480:].drop(columns=[F.GRADE]).reset_index(drop=True)
        A, V, _ = F.build_features(train, valid)
        c = tiny_config()
        model, _, _ = T33.fit_lgbm(c, A, train[F.GRADE].to_numpy() - 1, None, 12)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/'m.txt'
            model.booster_.save_model(str(path))
            booster = lgb.Booster(model_file=str(path))
        p = np.asarray(booster.predict(V), dtype=np.float64)
        ok = T33.cross_check_trees(booster, 12, V, p)
        self.assertEqual((ok['iterations'], ok['trees']), (12, 36))
        with self.assertRaises(ValueError):
            T33.cross_check_trees(booster, 11, V, p)   # wrong n_trees
        with self.assertRaises(ValueError):
            T33.cross_check_trees(booster, 13, V, p)                                # more than the model holds
        with self.assertRaises(ValueError):
            T33.cross_check_trees(booster, 6, V, p)                                 # prefix prediction differs

    def test_lightgbm_probe_reports_pinned_categorical_defaults(self):
        res = T33.probe_lightgbm(tiny_config())
        self.assertFalse(res['counted_as_study_fit'])
        exp = {k: float(v) for k, v in json.loads(CONFIG.read_text(encoding='utf-8'))['lightgbm_categorical_defaults_expected'].items()}
        if T33.environment() == json.loads(CONFIG.read_text(encoding='utf-8'))['environment']:
            self.assertEqual(res['categorical_effective'], exp)
        self.assertLessEqual(res['reload_max_abs_diff'], 1e-12)


@unittest.skipUnless(HAVE, 'lightgbm not installed')
class EarlyStopAndFinalizeTests(unittest.TestCase):
    def test_early_stopping_below_the_cap_saves_exactly_the_best_trees(self):
        with tempfile.TemporaryDirectory() as td:
            ctx, out = make_run(td)
            ctx.c['lgbm_params']['learning_rate'] = 0.5
            ctx.c['es'].update(max_rounds=300, patience=3)
            T33.fit_unit(ctx, 'E1', 0)
            s = json.loads((out/'E1'/'fold_0_es.json').read_text(encoding='utf-8'))
            r = json.loads((out/'E1'/'fold_0.json').read_text(encoding='utf-8'))
            self.assertFalse(s['cap_hit'])
            self.assertLess(s['n_trees'], 300)
            self.assertEqual(s['curve_length'], len(s['multi_logloss_curve']))
            self.assertEqual(int(np.argmin(s['multi_logloss_curve'])) + 1, s['n_trees'])
            self.assertEqual(s['model_trees_saved'], 3 * s['model_iterations_saved'])
            self.assertEqual(r['effective_parameters']['n_estimators'], s['n_trees'])

    def test_finalize_end_to_end_on_synthetic_comparators(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ctx, out = make_run(root/'run')
            (root/'cmp').mkdir()
            rng = np.random.default_rng(0)
            comps = {}
            from preprocess import sha256_file
            for i, name in enumerate(['T8_N', 'T14', 'LR_C', 'T0', 'LGBM_O']):
                noisy = ctx.y.to_numpy().copy()
                flip = rng.random(len(noisy)) < 0.35
                noisy[flip] = rng.integers(1, 4, flip.sum())
                frame = pd.DataFrame({'building_id': ctx.ids.to_numpy(), 'fold': ctx.folds, 'y_true': ctx.y.to_numpy(),
                                      'y_pred': noisy, 'p_grade_1': 0.3, 'p_grade_2': 0.3, 'p_grade_3': 0.4})
                path = root/'cmp'/f'{name}.csv.gz'
                T33.write_pred(path, frame.sample(frac=1, random_state=1))     # shuffled rows: alignment must be by ID
                comps[name] = dict(label=name, path=f'cmp/{name}.csv.gz', sha256=sha256_file(path), judged=name != 'LGBM_O')
            ctx.c['comparators'] = comps
            ctx.design.update(local_root=str(root), code_commit='0' * 40,
                              preflight=dict(label_probe_fold0={'passes': True}))
            T33.write(out/'design.json', ctx.design)
            for arm, f in T33.targets():
                T33.fit_unit(ctx, arm, f)
            self.assertEqual(T33.status(out)['complete'], 10)
            self.assertEqual(T33.fits_started(T33.read_ledger(out)), 20)
            with mock.patch.object(T33, 'verify', return_value=(ctx.X, ctx.y, ctx.ids, ctx.folds, ctx.design)):
                res = T33.finalize_all('unused', out)
            self.assertEqual(set(res['arms']), {'E1', 'E2'})
            self.assertEqual(len(res['arms']['E1']['fold_macro_f1']), 5)
            self.assertIn(res['judgments']['D3'], ('ES_PROTOCOL_EFFECT_SMALL', 'ES_PROTOCOL_EFFECT_NOTICEABLE'))
            self.assertEqual(res['judgments']['D1']['E1']['LGBM_O'], 'REFERENCE_ONLY')
            self.assertIn(res['judgments']['D4_E1'], ('NEAR', 'MODERATE', 'FAR'))
            self.assertEqual(res['ledger']['extra_fits'], 0)
            self.assertEqual(res['arms']['E1']['pooled']['confusion'], np.array(res['arms']['E1']['pooled']['confusion']).tolist())
            self.assertTrue((out/'analysis.json').exists() and (out/'backup'/'analysis.json').exists())
            self.assertEqual(T33.status(out)['max_fits'], 24)
            with (out/'E1'/'oof.csv.gz').open('ab') as fh:                       # L11: reuse of metrics.json re-hashes the OOF
                fh.write(b'x')
            with self.assertRaises(ValueError):
                T33.assemble(out, ctx.y, ctx.ids, ctx.folds, 'E1')
            json.dumps(res, allow_nan=False)
            with mock.patch.object(T33, 'verify', return_value=(ctx.X, ctx.y, ctx.ids, ctx.folds, ctx.design)):
                with self.assertRaises(ValueError):
                    T33.finalize_all('unused', out)                              # no overwrite


@unittest.skipUnless(HAVE, 'lightgbm not installed')
class ReviewFixTests(unittest.TestCase):
    def test_text_format_state_is_pickle_free_and_recorded_in_the_ledger(self):
        with tempfile.TemporaryDirectory() as td:
            ctx, out = make_run(td)
            T33.fit_unit(ctx, 'E2', 0)
            led = T33.read_ledger(out)
            names = led['features']['A/0']['artifact_sha256']
            want = F.state_text_names('fold_0_features_A')
            self.assertTrue(set(want) <= set(names))
            for n in want:
                self.assertEqual(names[n], T33.sha256_file(out/'shared'/n))
            A = ctx.get_A(0)
            back = F.read_matrix_text(out/'shared', 'fold_0_features_A')
            self.assertEqual(F.matrix_sha256(back), A['valid_sha'])
            again = F.apply_tables(A['valid_frame'], F.read_tables_text(out/'shared', 'fold_0_features_A'))
            self.assertTrue(OC.frames_identical(again, A['valid'])[0])
            state = json.loads((out/'shared'/'fold_0_features_A_state.json').read_text(encoding='utf-8'))
            self.assertEqual(state['columns'], F.feature_columns())
            self.assertEqual(state['matrix_sha256'], A['valid_sha'])
            es_names = led['units']['E2/0']['artifact_sha256']                    # E2: B tables + inner-validation matrix too
            self.assertTrue(set(F.state_text_names('fold_0_features_B')) <= set(es_names))
            self.assertEqual(es_names['fold_0_features_B_matrix.npz'], T33.sha256_file(out/'E2'/'fold_0_features_B_matrix.npz'))
            T33.check_saved(out, led['units']['E2/0'])

    def test_each_stage_may_be_attempted_at_most_twice(self):
        with tempfile.TemporaryDirectory() as td:
            ctx, out = make_run(td)
            T33.begin_fit(ctx, 'E1', 1, 'es')
            T33.begin_fit(ctx, 'E1', 1, 'es')
            with self.assertRaises(ValueError) as cm:
                T33.begin_fit(ctx, 'E1', 1, 'es')
            self.assertIn('attempts per stage', str(cm.exception))

    def test_resume_checks_inner_membership_against_the_checkpoint(self):
        with tempfile.TemporaryDirectory() as td:
            ctx, out = make_run(td)
            real = T33.fit_lgbm
            calls = {'n': 0}

            def flaky(*a, **k):
                calls['n'] += 1
                if calls['n'] == 2:
                    raise RuntimeError('simulated crash')
                return real(*a, **k)
            with mock.patch.object(T33, 'fit_lgbm', side_effect=flaky):
                with self.assertRaises(RuntimeError):
                    T33.fit_unit(ctx, 'E1', 0)
            with mock.patch.object(T33, 'inner_id_shas', return_value=('x', 'y')):
                with self.assertRaises(ValueError) as cm:
                    T33.fit_unit(ctx, 'E1', 0)
            self.assertIn('membership', str(cm.exception))

    def test_leftover_backup_is_moved_aside_not_overwritten(self):
        with tempfile.TemporaryDirectory() as td:
            ctx, out = make_run(td)
            (out/'backup'/'E1').mkdir(parents=True)
            (out/'backup'/'E1'/'fold_0_es.json').write_text('stale partial backup')
            T33.fit_unit(ctx, 'E1', 0)
            self.assertEqual((out/'backup'/'E1'/'fold_0_es.json.incomplete_attempt0').read_text(), 'stale partial backup')
            T33.check_saved(out, T33.read_ledger(out)['units']['E1/0'])

    def test_comparator_recall3_is_checked_as_pooled_only(self):
        from preprocess import sha256_file
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            X, y = synth()
            ids = pd.Series(np.arange(1, len(X) + 1))
            folds = np.arange(len(X)) % 5
            rng = np.random.default_rng(2)
            pred = y.to_numpy().copy()
            flip = rng.random(len(pred)) < 0.4
            pred[flip] = rng.integers(1, 4, flip.sum())
            frame = pd.DataFrame({'building_id': ids, 'fold': folds, 'y_true': y.to_numpy(), 'y_pred': pred,
                                  'p_grade_1': .3, 'p_grade_2': .3, 'p_grade_3': .4})
            T33.write_pred(root/'c.csv.gz', frame)
            v = T33.comparator_values(frame, y.to_numpy())
            self.assertGreater(abs(v['pooled_recall3'] - v['fold_mean_recall3']), 1e-6)   # the two definitions differ here
            c = tiny_config()
            spec = dict(label='x', path='c.csv.gz', sha256=sha256_file(root/'c.csv.gz'), cv_mean=v['cv_mean'],
                        recall3=v['pooled_recall3'], error31=v['error31'], judged=True)
            c['comparators'] = {'x': spec}
            seen = T33.comparator_check(c, root, ids, y, folds)
            self.assertEqual(seen['x']['recall3_definition'], 'pooled')
            c['comparators'] = {'x': dict(spec, recall3=v['fold_mean_recall3'])}
            with self.assertRaises(ValueError):
                T33.comparator_check(c, root, ids, y, folds)
            c['comparators'] = {'x': dict(spec, error31=v['error31'] + 1)}
            with self.assertRaises(ValueError):
                T33.comparator_check(c, root, ids, y, folds)

    def test_registered_split_shas_must_match_the_data_contract(self):
        c = tiny_config()
        contract = dict(files={'split_manifest.csv': '0' * 64, 'development.csv': c['split']['development_sha256']})
        with self.assertRaises(ValueError) as cm:
            T33.input_checks(None, None, None, None, contract, c)
        self.assertIn('data contract', str(cm.exception))


def fake_git(blob_equal=True):
    def g(*args):
        if args[0] == 'rev-parse' and args[1] == 'HEAD':
            return 'a' * 40
        if args[0] == 'hash-object':
            return 'b' * 40
        if args[0] == 'rev-parse':
            return 'b' * 40 if blob_equal else 'c' * 40
        raise AssertionError(args)
    return g


@unittest.skipUnless(HAVE, 'lightgbm not installed')
class SealAndGateTests(unittest.TestCase):
    """prepare / preflight / verify refusal paths with mocked git and temp folders (no real data, no fits)."""

    def setUp(self):
        if T33.environment() != json.loads(CONFIG.read_text(encoding='utf-8'))['environment']:
            self.skipTest('not the frozen environment')
        self.real = T33.config()

    def _pf(self, td, **over):
        pf = dict(status='PASS', study_id=T33.STUDY_ID, classifiers_fitted=0, source_sha256=T33.sources(),
                  plan_sha256=self.real['plan']['sha256'], environment=T33.environment(), data_contract={})
        pf.update(over)
        path = Path(td)/'pf.json'
        path.write_text(json.dumps(pf), encoding='utf-8')
        return path

    def test_prepare_refuses_plan_sha_mismatch(self):
        with tempfile.TemporaryDirectory() as td:
            bad = json.loads(json.dumps(self.real))
            bad['plan']['sha256'] = '0' * 64
            with mock.patch.object(T33, 'git', fake_git()), mock.patch.object(T33, 'config', return_value=bad):
                with self.assertRaises(ValueError) as cm:
                    T33.prepare('data', Path(td)/'run', self._pf(td))
            self.assertIn('Plan document SHA', str(cm.exception))
            self.assertFalse((Path(td)/'run').exists())

    def test_prepare_refuses_sealed_source_that_differs_from_head(self):
        with tempfile.TemporaryDirectory() as td:
            with mock.patch.object(T33, 'git', fake_git(blob_equal=False)):
                with self.assertRaises(ValueError) as cm:
                    T33.prepare('data', Path(td)/'run', self._pf(td))
            self.assertIn('differs from HEAD', str(cm.exception))

    def test_prepare_refuses_wrong_commit_existing_output_and_holdout_path(self):
        with tempfile.TemporaryDirectory() as td:
            with mock.patch.object(T33, 'git', fake_git()):
                with self.assertRaises(ValueError):
                    T33.prepare('data', Path(td)/'run', self._pf(td), code_commit='d' * 40)
                with self.assertRaises(ValueError):
                    T33.prepare('C:/x/final_holdout', Path(td)/'run', self._pf(td))
            (Path(td)/'exists').mkdir()
            with self.assertRaises(ValueError):
                T33.prepare('data', Path(td)/'exists', self._pf(td))

    def test_prepare_refuses_sources_changed_after_the_preflight(self):
        with tempfile.TemporaryDirectory() as td:
            changed = dict(T33.sources(), **{'experiments/ey_pipeline_benchmark/run.py': '0' * 64})
            with mock.patch.object(T33, 'git', fake_git()), mock.patch.object(T33, 'notebook_evidence', return_value=({}, {})):
                with self.assertRaises(ValueError) as cm:
                    T33.prepare('data', Path(td)/'run', self._pf(td, source_sha256=changed))
                self.assertIn('Sources changed', str(cm.exception))
                with self.assertRaises(ValueError) as cm2:
                    T33.prepare('data', Path(td)/'run', self._pf(td, status='FAIL'))
                self.assertIn('Preflight must PASS', str(cm2.exception))
            self.assertFalse((Path(td)/'run').exists())

    def test_preflight_refuses_existing_destination_plan_mismatch_and_holdout(self):
        with tempfile.TemporaryDirectory() as td:
            (Path(td)/'pre').mkdir()
            with self.assertRaises(ValueError):
                T33.preflight('data', Path(td)/'pre')
            bad = json.loads(json.dumps(self.real))
            bad['plan']['sha256'] = '0' * 64
            with mock.patch.object(T33, 'config', return_value=bad):
                with self.assertRaises(ValueError) as cm:
                    T33.preflight('data', Path(td)/'new')
            self.assertIn('Plan document SHA', str(cm.exception))
            with self.assertRaises(ValueError):
                T33.preflight('C:/x/final_holdout', Path(td)/'new')

    def _seal(self, td, design):
        out = Path(td)
        T33.write(out/'design.json', design)
        T33.write(out/'design_seal.json', dict(design_sha256=T33.sha256_file(out/'design.json')))
        return out

    def test_verify_refuses_changed_seal_config_and_sources(self):
        with tempfile.TemporaryDirectory() as td:
            design = dict(config=self.real, source_sha256=T33.sources(), environment=T33.environment(), data_contract={})
            out = self._seal(td, design)
            with (out/'design.json').open('a') as fh:
                fh.write(' ')
            with self.assertRaises(ValueError) as cm:
                T33.verify('data', out)
            self.assertIn('seal', str(cm.exception))
        with tempfile.TemporaryDirectory() as td:
            other = json.loads(json.dumps(self.real))
            other['weights']['strength'] = 0.5
            out = self._seal(td, dict(config=other, source_sha256=T33.sources(), environment=T33.environment(), data_contract={}))
            with self.assertRaises(ValueError) as cm:
                T33.verify('data', out)
            self.assertIn('config', str(cm.exception))
        with tempfile.TemporaryDirectory() as td:
            src = dict(T33.sources(), **{'preprocess.py': '0' * 64})
            out = self._seal(td, dict(config=self.real, source_sha256=src, environment=T33.environment(), data_contract={}))
            with self.assertRaises(ValueError) as cm:
                T33.verify('data', out)
            self.assertIn('source', str(cm.exception))


@unittest.skipUnless(HAVE and HAVE_PRIVATE, 'private notebook copy not present (kept outside Git)')
class OriginalFunctionTests(unittest.TestCase):
    """Bit-identity of our port against the notebook's own function definitions (AST extract of the private copy)."""

    @classmethod
    def setUpClass(cls):
        cls.cfg = json.loads(CONFIG.read_text(encoding='utf-8'))
        from preprocess import sha256_file
        if sha256_file(PRIVATE/'지진.ipynb') != cls.cfg['notebook']['sha256']:
            raise unittest.SkipTest('private notebook copy SHA differs')
        cls.ns, cls.evidence = OC.extract(OC.cell_text(PRIVATE/'지진.ipynb'))

    def test_extract_matches_registered_evidence(self):
        self.assertEqual(self.evidence['extract_text_sha256'], self.cfg['notebook']['extract_text_sha256'])
        self.assertEqual(self.evidence['constants'], self.cfg['notebook']['extract_constants'])

    def test_features_are_bit_identical_in_both_orders(self):
        X, y = synth(1500, seed=9)
        fr = F.notebook_frame(X, y)
        train, valid = fr.iloc[:1200], fr.iloc[1200:]
        res = OC.compare(self.ns, train, valid)
        self.assertTrue(res['passes'], res)
        y0 = train[F.GRADE].to_numpy() - 1
        tr_pos, va_pos = T33.inner_split(y0)
        res_b = OC.compare(self.ns, train.iloc[tr_pos], train.iloc[va_pos].drop(columns=[F.GRADE]))
        self.assertTrue(res_b['passes'], res_b)


if __name__ == '__main__':
    unittest.main()
