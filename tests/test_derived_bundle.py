"""T34 synthetic contracts only: no real development data, no holdout access, no team file, no study training.

Fits here are tiny synthetic checks of the code path (a few boosting rounds on ~1500 rows), never study fits.
Git checks run in a temporary repository only.
"""
import copy
import errno
import inspect
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from preprocess import ALLOWED_TEXT, BINARY, FEATURES, GEO, ID, SECONDARY_DETAIL, SUPERSTRUCTURE, normalize_features, sha256_file  # noqa: E402

try:
    import lightgbm  # noqa: F401
    import psutil  # noqa: F401
    from experiments.derived_bundle import features as F
    from experiments.derived_bundle import run
    HAVE = True
except ImportError:  # pragma: no cover
    HAVE = False

SUP = 'has_superstructure_'


def synth(n=1500, seed=0):
    """Valid normalized development-like frame (all 38 inputs, geo ids as strings), three geo levels, 995 ages."""
    rng = np.random.default_rng(seed)
    g1 = rng.integers(0, 6, n)
    g2 = g1*5 + rng.integers(0, 5, n)
    g3 = g2*6 + rng.integers(0, 6, n)
    d = {'geo_level_1_id': g1, 'geo_level_2_id': g2, 'geo_level_3_id': g3,
         'count_floors_pre_eq': rng.integers(1, 6, n),
         'age': rng.choice([0, 5, 10, 20, 30, 995], n, p=[.05, .2, .25, .2, .2, .1]),
         'area_percentage': rng.integers(1, 40, n), 'height_percentage': rng.integers(1, 20, n),
         'count_families': rng.integers(0, 4, n)}
    for col, values in ALLOWED_TEXT.items():
        d[col] = rng.choice(values, n)
    for col in BINARY:
        d[col] = rng.integers(0, 2, n)
    X = pd.DataFrame(d)
    X.loc[3, 'geo_level_3_id'] = 9999          # regions that exist in one row only
    X.loc[8, 'geo_level_3_id'] = 9998
    X.loc[8, 'geo_level_2_id'] = 7777
    X = normalize_features(X[FEATURES])
    y = pd.Series(np.clip(1 + (X['height_percentage'].to_numpy() > 10) + (rng.random(n) < 0.3), 1, 3).astype(np.uint8))
    ids = pd.Series(np.arange(1, n + 1), name=ID)
    folds = np.arange(n) % 5
    return X, y, ids, folds


def tiny_frame(rows):
    """Hand-built rows for exact arithmetic checks. Each row is a dict that overrides a neutral default."""
    base = {'geo_level_1_id': 1, 'geo_level_2_id': 1, 'geo_level_3_id': 1, 'count_floors_pre_eq': 2, 'age': 10,
            'area_percentage': 10, 'height_percentage': 6, 'count_families': 1, 'land_surface_condition': 'n',
            'foundation_type': 'r', 'roof_type': 'n', 'ground_floor_type': 'f', 'other_floor_type': 'q', 'position': 's',
            'plan_configuration': 'd', 'legal_ownership_status': 'v'}
    for col in BINARY:
        base[col] = 0
    data = [{**base, **r} for r in rows]
    return normalize_features(pd.DataFrame(data)[FEATURES])


def small_config():
    c = copy.deepcopy(run.config())
    c['lgbm_params'].update(n_estimators=5, num_leaves=7, min_child_samples=5, verbosity=-1)
    c['lr']['params'].update(max_iter=300)
    c['train_rows_per_fold'], c['valid_rows_per_fold'] = 1200, 300
    c['resources'].update(min_free_ram_gb=0, min_free_disk_gb_per_fold=0, min_free_disk_gb_at_prepare=0, max_cpu_percent=101)
    return c


def stub_resources(root, c):
    return dict(free_ram_gb=99.0, free_disk_gb=99.0, cpu_percent_mean=1.0)


def new_root(tmp):
    root = Path(tmp)
    (root/'backup').mkdir(parents=True, exist_ok=True)
    run.jatomic(root/'ledger.json', dict(study_id=run.STUDY_ID, folds={}))
    return root


def ctx_for(c, keys, cap=36, before=0, retries_before=0):
    return dict(config=c, code_commit='0'*40, allowed_keys=set(keys), attempt_cap=cap, attempts_before=before,
                retries_before=retries_before, max_retries=c['fit_plan']['max_infrastructure_retries'])


def stub_execute(fail=None, make_partial=False):
    """Writes a tiny fake fold; optionally raises after leaving a partial file."""
    calls = []

    def execute(X, y, ids, folds, d, unit, c):
        calls.append(unit['key'])
        d.mkdir(parents=True, exist_ok=True)
        f = unit['fold']
        pred = d/f'fold_{f}_pred.csv.gz'
        if make_partial:
            pred.write_bytes(b'partial')
        if fail is not None:
            raise fail
        pred.write_bytes(b'prediction-bytes')
        js = d/f'fold_{f}.json'
        record = dict(metrics=dict(macro_f1=0.5), info=dict(fit_seconds=0.1))
        run.jwrite(js, record)
        return record, [pred, js], None
    execute.calls = calls
    return execute


@unittest.skipUnless(HAVE, 'dependencies not installed')
class DerivedColumnTests(unittest.TestCase):
    def test_column_definition_and_order(self):
        self.assertEqual(F.numeric_names('S0'), F.NUMERIC_COLS)
        self.assertEqual(F.derived_names('S0'), F.NUMERIC_COLS + F.COMBO_COLS)
        for arm in ('S1', 'S2'):
            self.assertEqual(F.derived_names(arm), F.NUMERIC_COLS + F.AGE_COLS + F.COMBO_COLS)
        self.assertEqual([F.n_derived(a) for a in F.ARMS], [10, 13, 13])
        self.assertEqual(F.AGE_COLS, ['geo_level_1_id_age_mean', 'geo_level_2_id_age_mean', 'geo_level_3_id_age_mean'])
        self.assertEqual(F.COMBO_COLS, ['foundation_roof', 'ground_other_floor', 'foundation_ground', 'surface_condition'])
        self.assertEqual(F.categorical_positions('S0', 39), [45, 46, 47, 48])
        self.assertEqual(F.categorical_positions('S1', 39), [48, 49, 50, 51])
        self.assertEqual(F.COMBO_SOURCES['surface_condition'], ('land_surface_condition', 'ground_floor_type'))
        self.assertEqual(F.COMBO_SOURCES['foundation_ground'], ('foundation_type', 'ground_floor_type'))
        self.assertEqual(len(F.CONCRETE), 4)
        self.assertEqual(len(F.MUD), 3)
        c = run.config()
        for arm in F.ARMS:
            self.assertEqual(c['derived']['arm_columns'][arm], F.derived_names(arm))

    def test_numeric_hand_computation(self):
        X = tiny_frame([
            dict(height_percentage=6, area_percentage=4, count_floors_pre_eq=3,
                 **{SUP+'adobe_mud': 1, SUP+'mud_mortar_stone': 1, SUP+'rc_engineered': 1, SUP+'other': 1,
                    'has_secondary_use_hotel': 1, 'has_secondary_use_other': 1, 'has_secondary_use': 1}),
            dict(height_percentage=1, area_percentage=100, count_floors_pre_eq=1)])
        raw = F.raw_numeric(X)
        self.assertEqual(raw[0].tolist(), [6/4, 6/3, 4.0, 2.0, 1.0, 2.0])      # aggregate has_secondary_use excluded
        self.assertEqual(raw[1].tolist(), [1/100, 1.0, 0.0, 0.0, 0.0, 0.0])

    def test_boundaries_all_zero_and_all_one(self):
        zero = {c: 0 for c in BINARY}
        one = {c: 1 for c in BINARY}
        X = tiny_frame([zero, one])
        raw = F.raw_numeric(X)
        self.assertEqual(raw[0, 2:].tolist(), [0, 0, 0, 0])
        self.assertEqual(raw[1, 2:].tolist(), [11.0, 10.0, 4.0, 3.0])
        self.assertTrue(np.isfinite(F.to_float32(raw)).all())

    def test_zero_area_is_refused_without_epsilon(self):
        X = tiny_frame([dict(area_percentage=0)])
        with self.assertRaises(F.FeatureError):
            F.raw_numeric(X)

    def test_combination_keys_and_vocabulary(self):
        X = tiny_frame([dict(foundation_type='r', roof_type='n', ground_floor_type='f', other_floor_type='q', land_surface_condition='n'),
                        dict(foundation_type='w', roof_type='x', ground_floor_type='z', other_floor_type='s', land_surface_condition='t')])
        combo = F.combos(X)
        self.assertEqual(combo['foundation_roof'].tolist(), ['r|n', 'w|x'])
        self.assertEqual(combo['ground_other_floor'].tolist(), ['f|q', 'z|s'])
        self.assertEqual(combo['foundation_ground'].tolist(), ['r|f', 'w|z'])
        self.assertEqual(combo['surface_condition'].tolist(), ['n|f', 't|z'])
        bad = X.copy()
        bad.loc[0, 'roof_type'] = 'a|b'
        with self.assertRaises(F.FeatureError):
            F.combos(bad)
        # the plan's theoretical vocabulary bound
        sizes = [len(ALLOWED_TEXT[a])*len(ALLOWED_TEXT[b]) for a, b in F.COMBO_SOURCES.values()]
        self.assertEqual(sum(sizes), F.MAX_COMBINATIONS)

    def test_region_means_s1_s2_with_995_and_zero_age(self):
        # one geo3 region 1: ages 995, 0, 10 -> S1 (995+0+10)/3, S2 (0+10)/2 (age 0 is a real value); region 2 only 995.
        train = tiny_frame([dict(geo_level_3_id=1, age=995), dict(geo_level_3_id=1, age=0), dict(geo_level_3_id=1, age=10),
                            dict(geo_level_3_id=2, age=995), dict(geo_level_3_id=3, age=20)])
        valid = tiny_frame([dict(geo_level_3_id=1, age=7), dict(geo_level_3_id=2, age=7), dict(geo_level_3_id=77, age=7)])
        s1 = F.RegionAgeTable('S1').fit(train)
        s2 = F.RegionAgeTable('S2').fit(train)
        overall1, overall2 = (995 + 0 + 10 + 995 + 20)/5, (0 + 10 + 20)/3
        self.assertAlmostEqual(s1.fallback_, overall1)
        self.assertAlmostEqual(s2.fallback_, overall2)
        m1, st1 = s1.transform(valid, return_stats=True)
        m2, st2 = s2.transform(valid, return_stats=True)
        g3 = 2
        self.assertAlmostEqual(m1[0, g3], 1005/3)
        self.assertAlmostEqual(m2[0, g3], 5.0)
        self.assertAlmostEqual(m1[1, g3], 995.0)                # region 2 in S1: its own 995
        self.assertAlmostEqual(m2[1, g3], overall2)             # all-995 region in S2 -> non-995 overall mean
        self.assertAlmostEqual(m1[2, g3], overall1)             # unseen region -> training overall mean (995 included)
        self.assertAlmostEqual(m2[2, g3], overall2)
        self.assertEqual(st1['geo_level_3_id']['unseen_region_rows'], 1)
        self.assertEqual(st1['geo_level_3_id']['fallback_rows'], 1)
        self.assertEqual(st2['geo_level_3_id']['fallback_rows'], 2)
        self.assertEqual(st2['geo_level_3_id']['known_region_without_usable_age_rows'], 1)
        # training rows include themselves: the single-row region 3 equals its own age, the all-995 training row falls back
        mt2 = s2.transform(train)
        self.assertAlmostEqual(mt2[4, g3], 20.0)
        self.assertAlmostEqual(mt2[3, g3], overall2)
        self.assertAlmostEqual(s1.transform(train)[3, g3], 995.0)

    def test_region_key_is_single_column(self):
        # the same geo3 id under two geo2 parents is ONE region (no hierarchy path in the key)
        train = tiny_frame([dict(geo_level_2_id=1, geo_level_3_id=5, age=10), dict(geo_level_2_id=2, geo_level_3_id=5, age=30)])
        t = F.RegionAgeTable('S1').fit(train)
        m = t.transform(train)
        self.assertEqual(m[:, 2].tolist(), [20.0, 20.0])
        self.assertEqual(sorted(t.tables_['geo_level_3_id'].index), ['5'])

    def test_means_use_raw_age_not_the_c_policy(self):
        X, y, ids, folds = synth()
        tr = X.iloc[folds != 0]
        t = F.RegionAgeTable('S1').fit(tr)
        manual = tr.groupby('geo_level_1_id')['age'].mean().astype('float64')
        np.testing.assert_allclose(t.tables_['geo_level_1_id'].reindex(manual.index).to_numpy(), manual.to_numpy())
        self.assertTrue((tr['age'] == 995).any())
        self.assertGreater(t.tables_['geo_level_3_id'].max(), 100)       # 995s are inside the S1 means (raw age)
        t2 = F.RegionAgeTable('S2').fit(tr)
        self.assertLessEqual(t2.tables_['geo_level_3_id'].max(), 30)


@unittest.skipUnless(HAVE, 'dependencies not installed')
class LeakageTests(unittest.TestCase):
    def setUp(self):
        self.X, self.y, self.ids, self.folds = synth()
        self.tr, self.va = self.X.iloc[self.folds != 0], self.X.iloc[self.folds == 0]

    def test_fit_signature_has_no_label(self):
        self.assertEqual(list(inspect.signature(F.DerivedBundle.fit).parameters), ['self', 'X_train'])
        self.assertEqual(list(inspect.signature(F.RegionAgeTable.fit).parameters), ['self', 'X_train'])
        self.assertNotIn('y_valid', inspect.signature(run.fit_classifier).parameters)
        self.assertEqual(list(inspect.signature(run.fit_classifier).parameters), ['model', 'arm', 'mt', 'mv', 'y_tr', 'c'])
        self.assertNotIn('y', inspect.signature(run.build_inputs).parameters)

    def test_validation_values_do_not_change_the_fitted_state(self):
        for arm in F.ARMS:
            for kind in F.KINDS:
                a = F.DerivedBundle(arm, kind).fit(self.tr)
                changed = self.va.copy()
                changed['age'] = 995
                changed['geo_level_3_id'] = '4242'
                for col in changed.columns:
                    if col.startswith('has_'):
                        changed[col] = 1 - changed[col]
                b = F.DerivedBundle(arm, kind).fit(self.tr)
                a.transform(changed)                                  # lookups must not mutate the learned state
                self.assertEqual(F.state_signature(a), F.state_signature(b))
                # training rows are unaffected by whatever validation contains
                ta, tb = a.transform(self.tr), b.transform(self.tr)
                same = np.array_equal(ta, tb) if kind == 'lgbm' else (ta != tb).nnz == 0
                self.assertTrue(same)

    def test_validation_rows_only_look_up_training_tables(self):
        a = F.RegionAgeTable('S1').fit(self.tr)
        valid_alone = a.transform(self.va)
        shuffled = a.transform(self.va.iloc[::-1])
        np.testing.assert_array_equal(valid_alone[::-1], shuffled)
        # a validation row's mean does not depend on any other validation row
        single = a.transform(self.va.iloc[[5]])
        np.testing.assert_array_equal(single[0], valid_alone[5])

    def test_leaky_fit_on_train_plus_validation_is_detected(self):
        everything = pd.concat([self.tr, self.va])
        for arm in F.ARMS:
            for kind in F.KINDS:
                leaky = F.DerivedBundle(arm, kind).fit(everything)
                with self.assertRaises(F.FeatureError):
                    F.verify_fitted_on(leaky, self.tr)
                with self.assertRaises(F.FeatureError):
                    F.check_against_reference(leaky, self.tr, self.va)
                clean = F.DerivedBundle(arm, kind).fit(self.tr)
                self.assertTrue(F.verify_fitted_on(clean, self.tr))
                F.check_against_reference(clean, self.tr, self.va)

    def test_vocabulary_and_scaler_are_training_only(self):
        # a combination that exists only in validation is unseen: NaN code (LGBM) / all-zero one-hot (LR)
        tr = tiny_frame([dict(foundation_type='r', roof_type='n'), dict(foundation_type='h', roof_type='q')])
        va = tiny_frame([dict(foundation_type='w', roof_type='x')])
        g = F.DerivedBundle('S0', 'lgbm').fit(tr)
        self.assertTrue(np.isnan(g.transform(va)[0, 6]))
        r = F.DerivedBundle('S0', 'lr').fit(tr)
        block = r.transform(va).toarray()
        self.assertEqual(block[0, 6:6 + len(r.ohe_.categories_[0])].sum(), 0)
        self.assertNotIn('w|x', r.vocabulary['foundation_roof'])
        mean = r.scaler_.mean_
        np.testing.assert_allclose(mean, F.to_float32(F.raw_numeric(tr)).astype(np.float64).mean(axis=0), rtol=1e-6)

    def test_serialization_round_trip(self):
        for arm in F.ARMS:
            for kind in F.KINDS:
                b = F.DerivedBundle(arm, kind).fit(self.tr)
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp)/'bundle.joblib'
                    joblib.dump(b, path, compress=3)
                    back = joblib.load(path)
                one, two = b.transform(self.va), back.transform(self.va)
                same = np.array_equal(one, two, equal_nan=True) if kind == 'lgbm' else (one != two).nnz == 0
                self.assertTrue(same)
                self.assertEqual(F.state_signature(b), F.state_signature(back))

    def test_independent_reference_and_arm_relations(self):
        for arm in F.ARMS:
            for kind in F.KINDS:
                res = F.check_against_reference(F.DerivedBundle(arm, kind).fit(self.tr), self.tr, self.va)
                self.assertLess(res['train_numeric_max_relative_diff'], 5e-6)
        rel = F.check_arm_relations(self.tr, self.va)
        self.assertTrue(rel['train']['non_age_columns_identical'] and rel['valid']['non_age_columns_identical'])
        self.assertFalse(rel['train']['s1_s2_age_columns_equal'])

    def test_block_shapes_and_containers(self):
        for arm, width in zip(F.ARMS, (10, 13, 13)):
            lg = F.DerivedBundle(arm, 'lgbm').fit(self.tr).transform(self.va)
            self.assertEqual(lg.shape, (len(self.va), width))
            self.assertEqual(lg.dtype, np.float32)
            lr = F.DerivedBundle(arm, 'lr').fit(self.tr).transform(self.va)
            self.assertTrue(lr.format == 'csr' and lr.dtype == np.float32)
            self.assertEqual(lr.shape[1], F.n_numeric(arm) + sum(len(v) for v in F.DerivedBundle(arm, 'lr').fit(self.tr).vocabulary.values()))

    def test_diagnostics_run_and_are_aggregates(self):
        d = F.self_inclusion_diagnostics(self.tr, self.va, 'S2')
        self.assertIn('train_age995', d['geo_level_3_id'])
        self.assertIn('0 (unseen)', d['geo_level_3_id_by_training_size'])
        m = F.mixed_995_summary(self.tr, self.va)
        self.assertGreater(m['geo_level_3_id']['regions_with_995'], 0)
        masks = F.has_995_region_mask(self.tr, self.va)
        self.assertEqual(masks['geo_level_1_id'].shape, (len(self.va),))


@unittest.skipUnless(HAVE, 'dependencies not installed')
class PlanAndConfigTests(unittest.TestCase):
    def test_pinned_plan_and_fit_counts(self):
        c = run.config()
        plan = ROOT/c['plan']
        self.assertEqual(c['plan_sha256'], '9b9edfed62f70af280b691934da83bfc1d78709627fd429287de1c1b2190255e')
        self.assertEqual(run.PLAN_SHA256, c['plan_sha256'])
        self.assertEqual(sha256_file(plan), c['plan_sha256'])
        self.assertEqual(run.check_plan_sha(c), c['plan_sha256'])
        fp = c['fit_plan']
        self.assertEqual((len(run.study_targets(False)), len(run.study_targets(True))), (30, 35))
        self.assertEqual((fp['completed_base'], fp['completed_with_refit'], fp['attempt_cap'], fp['attempt_cap_with_refit']), (32, 37, 36, 41))
        self.assertEqual(fp['control_fits'], 2)
        t = [run.key(*x) for x in run.study_targets(True)]
        self.assertEqual(t[:5], [f'LR/B_REFIT/{f}' for f in range(5)])
        self.assertEqual([x.rsplit('/', 1)[0] for x in t[5::5]], ['LR/S0', 'LR/S1', 'LR/S2', 'LGBMN/S0', 'LGBMN/S1', 'LGBMN/S2'])
        self.assertEqual(run.study_targets(False)[0], ('LR', 'S0', 0))
        self.assertEqual(len(set(t)), 35)

    def test_config_rejects_other_plan_pin_and_drift(self):
        c = json.loads((run.HERE/'config.json').read_text(encoding='utf-8'))
        with tempfile.TemporaryDirectory() as tmp:
            for mutate in (lambda d: d.update(plan_sha256='0'*64),
                           lambda d: d['lgbm_params'].update(n_estimators=299),
                           lambda d: d['lr']['params'].update(C=2.0),
                           lambda d: d['fit_plan'].update(attempt_cap=40),
                           lambda d: d['rule'].update(mean_delta_min=0.002),
                           lambda d: d.update(holdout_access=True),
                           lambda d: d['derived']['arm_columns']['S0'].reverse()):
                d = copy.deepcopy(c)
                mutate(d)
                path = Path(tmp)/'config.json'
                path.write_text(json.dumps(d), encoding='utf-8')
                with self.assertRaises(ValueError):
                    run.config(path)

    def test_check_plan_sha_rejects_changed_plan_bytes(self):
        c = run.config()
        with tempfile.TemporaryDirectory() as tmp:
            plan = Path(tmp)/c['plan']
            plan.parent.mkdir(parents=True)
            plan.write_bytes((ROOT/c['plan']).read_bytes())
            self.assertEqual(run.check_plan_sha(c, root=tmp), c['plan_sha256'])
            plan.write_bytes(plan.read_bytes() + b'\n')
            with self.assertRaisesRegex(ValueError, 'Plan document SHA'):
                run.check_plan_sha(c, root=tmp)
        wrong = dict(c, plan_sha256='f'*64)
        with self.assertRaisesRegex(ValueError, 'plan_sha256'):
            run.check_plan_sha(wrong)

    def test_prepare_and_verify_refuse_a_different_plan(self):
        c = run.config()
        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp)/'repo'
            plan = fake/c['plan']
            plan.parent.mkdir(parents=True)
            plan.write_bytes((ROOT/c['plan']).read_bytes() + b' ')
            with mock.patch.object(run, 'ROOT', fake):
                with self.assertRaisesRegex(ValueError, 'Plan document SHA'):
                    run.prepare(Path(tmp)/'data', Path(tmp)/'out')
                self.assertFalse((Path(tmp)/'out'/'run').exists())
                out = Path(tmp)/'out2'
                (out/'run').mkdir(parents=True)
                run.jwrite(out/'run'/'design.json', dict(config=c, plan_sha256=c['plan_sha256']))
                run.jwrite(out/'run'/'design_seal.json', dict(design_sha256=sha256_file(out/'run'/'design.json')))
                with self.assertRaisesRegex(ValueError, 'Plan document SHA'):
                    run.verify(out)

    def test_fixed_thread_and_model_settings(self):
        c = run.config()
        self.assertEqual(c['lgbm_params']['n_jobs'], 2)
        self.assertEqual(c['lgbm']['threadpool_limit'], 2)
        self.assertEqual(c['lr']['threadpool_limit'], 1)
        self.assertEqual(c['lgbm_params'], run.t8.config()['lgbm_params'])
        self.assertEqual(c['lr']['params'], run.LR_PARAMS)
        self.assertEqual(c['outer_parallel_jobs'], 1)

    def test_no_whole_matrix_toarray_and_no_forbidden_literals(self):
        run_src = (ROOT/'experiments/derived_bundle/run.py').read_text(encoding='utf-8')
        feat_src = (ROOT/'experiments/derived_bundle/features.py').read_text(encoding='utf-8')
        self.assertNotIn('.toarray(', run_src)
        self.assertEqual(len(re.findall(r'\.toarray\(', feat_src)), 2)         # chunked / small-slice checks only
        for src in (run_src, feat_src):
            self.assertNotIn('earthquake/data/x_train', src)
            self.assertNotIn('data/final_holdout', src)
            self.assertNotIn('team_materials', src)
            self.assertNotIn('read_csv', src.replace('t8.read_pred', ''))     # data only through load_development
        self.assertIn('load_development', (ROOT/'experiments/geo_group_cv/run.py').read_text(encoding='utf-8'))
        self.assertIn('sys.addaudithook', run_src)

    def test_source_blob_check_in_a_temporary_repository(self):
        with tempfile.TemporaryDirectory() as tmp:
            git = lambda *a: subprocess.run(['git', '-c', 'user.name=t', '-c', 'user.email=t@t', *a], cwd=tmp, check=True,   # noqa: E731
                                            capture_output=True, text=True)
            git('init', '-q')
            (Path(tmp)/'a.py').write_text('x = 1\n', encoding='utf-8')
            (Path(tmp)/'b.py').write_text('y = 1\n', encoding='utf-8')
            git('add', 'a.py')
            git('commit', '-q', '-m', 'one')
            self.assertEqual(run.source_mismatches(['a.py'], root=tmp), [])
            (Path(tmp)/'a.py').write_text('x = 2\n', encoding='utf-8')
            self.assertEqual(sorted(p for p, _ in run.source_mismatches(['a.py', 'b.py', 'c.py'], root=tmp)), ['a.py', 'b.py', 'c.py'])
            self.assertEqual(len(run.git_head(tmp)), 40)


@unittest.skipUnless(HAVE, 'dependencies not installed')
class GuardTests(unittest.TestCase):
    def test_guard_path_refuses_holdout_and_team_files(self):
        before = run.guard_summary()['refused']
        for bad in ('C:/p/earthquake/data/final_holdout/holdout.csv', 'C:\\p\\earthquake\\data\\FINAL_HOLDOUT\\x.csv',
                    'data/../final_holdout/a', '/x/final_holdout', 'some/dir/x_train.csv', 'X_TRAIN.CSV',
                    'a/modeling_ver2_raw.ipynb', 'MODELING_VER2_RAW.IPYNB'):
            with self.assertRaises(run.GuardError, msg=bad):
                run.guard_path(bad)
        self.assertEqual(run.guard_summary()['refused'] - before, 8)
        for ok in ('C:/p/earthquake/data', 'C:/p/earthquake/data/development.csv', 'x/split_manifest.csv', 'a/x_train_summary.json'):
            self.assertEqual(run.guard_path(ok), Path(ok))
        self.assertTrue(issubclass(run.GuardError, PermissionError))

    def test_guard_is_applied_to_data_and_reference_entry_points(self):
        c = run.config()
        with self.assertRaises(run.GuardError):
            run.load_data('C:/p/earthquake/data/final_holdout', c)
        with self.assertRaises(run.GuardError):
            run.preflight('C:/p/data', 'C:/p/final_holdout/out')
        spec = dict(name='x', path='local/x_train.csv', sha256='0'*64)
        with self.assertRaises(run.GuardError):
            run.read_reference('C:/p', spec, None, None, None)

    def test_audit_hook_blocks_runtime_opens(self):
        run.install_audit_hook()
        run.install_audit_hook()                                   # idempotent
        tmp = tempfile.mkdtemp()
        holdout = Path(tmp)/'final_holdout'
        os.makedirs(holdout)                                       # mkdir is not an open: allowed
        try:
            for path in (holdout/'a.csv', Path(tmp)/'x_train.csv', Path(tmp)/'sub'/'modeling_ver2_raw.ipynb'):
                for mode in ('r', 'w'):
                    with self.assertRaises(run.GuardError, msg=f'{path} {mode}'):
                        open(path, mode)
                with self.assertRaises(run.GuardError):
                    path.read_text()
                with self.assertRaises(run.GuardError):
                    path.write_text('x')
            with self.assertRaises(run.GuardError):
                os.listdir(holdout)
            with self.assertRaises(run.GuardError):
                list(os.scandir(holdout))
            ok = Path(tmp)/'development.csv'
            ok.write_text('fine', encoding='utf-8')
            self.assertEqual(ok.read_text(encoding='utf-8'), 'fine')
            self.assertEqual(sorted(os.listdir(tmp)), ['development.csv', 'final_holdout'])
            self.assertFalse((Path(tmp)/'x_train.csv').exists())      # refused writes never created the file
        finally:
            os.rmdir(holdout)
            shutil.rmtree(tmp)
        self.assertTrue(run.guard_summary()['audit_hook_installed'])
        self.assertGreaterEqual(run.guard_summary()['refused'], 12)

    def test_pandas_cannot_read_a_guarded_path_through_the_hook(self):
        run.install_audit_hook()
        tmp = tempfile.mkdtemp()
        try:
            with self.assertRaises(run.GuardError):
                pd.read_csv(Path(tmp)/'final_holdout'/'a.csv')
            with self.assertRaises(run.GuardError):
                pd.read_csv(Path(tmp)/'x_train.csv')
            with self.assertRaises(run.GuardError):
                joblib.load(Path(tmp)/'final_holdout'/'m.joblib')
        finally:
            shutil.rmtree(tmp)


@unittest.skipUnless(HAVE, 'dependencies not installed')
class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = new_root(self.tmp.name)
        self.c = small_config()
        self.unit = run.make_unit('LR', 'S0', 0, 'study')
        self.other = run.make_unit('LR', 'S0', 1, 'study')
        self.ctx = ctx_for(self.c, [self.unit['key'], self.other['key']], cap=4)

    def tearDown(self):
        self.tmp.cleanup()

    def go(self, execute, unit=None, **kw):
        return run.run_unit(None, None, None, None, self.root, unit or self.unit, self.ctx, execute_fn=execute,
                            resource_fn=stub_resources, **kw)

    def events(self):
        return [json.loads(x) for x in (self.root/'attempts.jsonl').read_text(encoding='utf-8').splitlines()]

    def test_complete_unit_is_hashed_backed_up_and_skipped_on_rerun(self):
        ex = stub_execute()
        entry, fitted = self.go(ex)
        self.assertTrue(fitted)
        self.assertEqual(entry['status'], 'complete')
        for name, digest in entry['artifact_sha256'].items():
            self.assertEqual(sha256_file(self.root/entry['dir']/name), digest)
            self.assertEqual(sha256_file(self.root/'backup'/entry['dir']/name), digest)
        ev = self.events()
        self.assertEqual([(e['status'], e['attempt'], e['code_commit']) for e in ev], [('running', 1, '0'*40), ('complete', 1, '0'*40)])
        entry2, fitted2 = self.go(ex)
        self.assertFalse(fitted2)
        self.assertEqual(ex.calls, [self.unit['key']])                 # not refit
        (self.root/entry['dir']/'fold_0_pred.csv.gz').write_bytes(b'tampered')
        with self.assertRaises(ValueError):
            self.go(ex)

    def test_running_is_recorded_before_the_fit_starts(self):
        seen = {}

        def execute(X, y, ids, folds, d, unit, c):
            seen['entry'] = run.read_ledger(self.root)['folds'][unit['key']]['status']
            seen['events'] = [e['status'] for e in self.events()]
            return stub_execute()(X, y, ids, folds, d, unit, c)
        self.go(execute)
        self.assertEqual(seen['entry'], 'running')
        self.assertEqual(seen['events'], ['running'])

    def test_defect_stops_and_blocks_every_later_unit(self):
        with self.assertRaises(RuntimeError):
            self.go(stub_execute(fail=RuntimeError('boom'), make_partial=True))
        self.assertEqual(run.read_ledger(self.root)['folds'][self.unit['key']]['status'], 'failed')
        self.assertEqual(self.events()[-1]['status'], 'failed')
        with self.assertRaisesRegex(ValueError, 'Failed unit'):
            self.go(stub_execute(), unit=self.other)
        with self.assertRaisesRegex(ValueError, 'Failed unit'):
            self.go(stub_execute(), retry_interrupted=True, retry_reason='try again')
        self.assertTrue((self.root/'LR'/'S0'/'fold_0_pred.csv.gz').exists())       # the partial file is not deleted

    def test_infrastructure_errors_are_interrupted_and_retry_once(self):
        for exc in (MemoryError('oom'), OSError(errno.ENOSPC, 'disk full'), KeyboardInterrupt()):
            with tempfile.TemporaryDirectory() as tmp:
                self.root = new_root(tmp)
                with self.assertRaises(type(exc)):
                    self.go(stub_execute(fail=exc, make_partial=True))
                self.assertEqual(run.read_ledger(self.root)['folds'][self.unit['key']]['status'], 'interrupted')
                with self.assertRaisesRegex(ValueError, 'retry'):
                    self.go(stub_execute())                                        # no flags: refused
                with self.assertRaisesRegex(ValueError, 'retry'):
                    self.go(stub_execute(), retry_interrupted=True, retry_reason='  ')
                entry, fitted = self.go(stub_execute(), retry_interrupted=True, retry_reason='disk was full')
                self.assertTrue(fitted)
                self.assertEqual((entry['attempts'], entry['retry_reasons']), (2, ['disk was full']))
                preserved = list((self.root/'LR'/'S0').glob('*.incomplete_attempt1'))
                self.assertEqual([p.name for p in preserved], ['fold_0_pred.csv.gz.incomplete_attempt1'])
                self.assertEqual(preserved[0].read_bytes(), b'partial')
                self.assertEqual([e['status'] for e in self.events()], ['running', 'interrupted', 'running', 'complete'])

    def test_second_interruption_of_the_same_unit_is_not_retried(self):
        with self.assertRaises(MemoryError):
            self.go(stub_execute(fail=MemoryError()))
        with self.assertRaises(MemoryError):
            self.go(stub_execute(fail=MemoryError()), retry_interrupted=True, retry_reason='first')
        with self.assertRaisesRegex(ValueError, 'one infrastructure retry'):
            self.go(stub_execute(), retry_interrupted=True, retry_reason='second')

    def test_vanished_process_leaves_running_and_needs_the_retry_flags(self):
        ledger = run.read_ledger(self.root)
        ledger['folds'][self.unit['key']] = dict(dir='LR/S0', model='LR', arm='S0', fold=0, kind='study', status='running', attempts=1,
                                                 retry_reasons=[])
        run.jatomic(self.root/'ledger.json', ledger)
        with self.assertRaisesRegex(ValueError, 'did not finish'):
            self.go(stub_execute())
        entry, fitted = self.go(stub_execute(), retry_interrupted=True, retry_reason='machine was powered off')
        self.assertTrue(fitted and entry['attempts'] == 2)

    def test_attempt_cap_includes_earlier_control_attempts_and_retry_reserve(self):
        self.ctx.update(attempt_cap=3, attempts_before=2)
        self.go(stub_execute())                                               # 2 + 1 = 3 reached
        with self.assertRaisesRegex(ValueError, 'Attempt cap'):
            self.go(stub_execute(), unit=self.other)
        self.ctx.update(attempt_cap=36, attempts_before=0, retries_before=4)
        with self.assertRaises(MemoryError):
            self.go(stub_execute(fail=MemoryError()), unit=self.other)
        with self.assertRaisesRegex(ValueError, 'retry reserve'):
            self.go(stub_execute(), unit=self.other, retry_interrupted=True, retry_reason='x')

    def test_unregistered_unit_and_untracked_files_are_refused(self):
        with self.assertRaisesRegex(ValueError, 'not a registered fit'):
            self.go(stub_execute(), unit=run.make_unit('LR', 'S1', 0, 'study'))
        d = self.root/'LR'/'S0'
        d.mkdir(parents=True)
        (d/'fold_0_pred.csv.gz').write_bytes(b'x')
        with self.assertRaisesRegex(ValueError, 'Untracked'):
            self.go(stub_execute())

    def test_resource_floor_blocks_before_an_attempt_is_recorded(self):
        def busy(root, c):
            raise run.ResourceBusy('busy')
        with self.assertRaises(run.ResourceBusy):
            run.run_unit(None, None, None, None, self.root, self.unit, self.ctx, execute_fn=stub_execute(), resource_fn=busy)
        self.assertEqual(run.read_ledger(self.root)['folds'], {})
        self.assertFalse((self.root/'attempts.jsonl').exists())

    def test_resource_check_uses_floors(self):
        c = small_config()
        rec = run.resource_check(self.root, c, cpu_samples=1, interval=0.01)
        self.assertIn('free_ram_gb', rec)
        c['resources']['min_free_ram_gb'] = 10**6
        with self.assertRaises(run.ResourceBusy):
            run.resource_check(self.root, c, cpu_samples=1, interval=0.01)

    def test_exclusive_lock_and_stale_failed_guard(self):
        out = Path(self.tmp.name)/'study'
        (out/'run').mkdir(parents=True)
        (out/'run'/'design_seal.json').write_text('{}', encoding='utf-8')
        (out/'run'/'RUNNING.lock').write_text('pid=1', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Another run holds'):
            run.fit_all(out)
        self.assertTrue((out/'run'/'RUNNING.lock').exists())                # a foreign lock is not removed
        (out/'run'/'RUNNING.lock').unlink()

        def broken_verify(root, data):
            raise ValueError('verify failed')
        with self.assertRaisesRegex(ValueError, 'verify failed'):
            run.fit_all(out, verify_fn=broken_verify)
        self.assertFalse((out/'run'/'RUNNING.lock').exists())              # released after failure
        with self.assertRaisesRegex(ValueError, 'Run prepare first'):
            run.fit_all(Path(self.tmp.name)/'nowhere')

    def test_fit_all_runs_the_fixed_order_with_stubbed_fits(self):
        out = Path(self.tmp.name)/'study2'
        run_dir = out/'run'
        (run_dir/'backup').mkdir(parents=True)
        (run_dir/'design_seal.json').write_text('{}', encoding='utf-8')
        run.jatomic(run_dir/'ledger.json', dict(study_id=run.STUDY_ID, folds={}))
        keys = [run.key(*t) for t in run.study_targets(False)]
        design = dict(study_targets=keys, config=self.c, code_commit='1'*40, attempt_cap=36, lr_b_refit=False,
                      control=dict(attempts=2, retries=0))
        ex = stub_execute()
        X = pd.DataFrame({'a': [1, 2]})
        run.fit_all(out, execute_fn=ex, resource_fn=stub_resources, verify_fn=lambda r, d: (X, None, None, None, design))
        self.assertEqual(ex.calls, keys)
        self.assertFalse((run_dir/'RUNNING.lock').exists())
        self.assertEqual(len(list(run_dir.glob('guard_log_*.json'))), 1)
        ex2 = stub_execute()
        run.fit_all(out, execute_fn=ex2, resource_fn=stub_resources, verify_fn=lambda r, d: (X, None, None, None, design))
        self.assertEqual(ex2.calls, [])                                       # completed fits are only hash-checked
        st = run.status(out) if (run_dir/'design.json').exists() else None
        self.assertIsNone(st)


@unittest.skipUnless(HAVE, 'dependencies not installed')
class FitPathTests(unittest.TestCase):
    """Real (tiny) fits through execute_fit: matrices, prefix invariance, reload, ledger."""

    @classmethod
    def setUpClass(cls):
        cls.X, cls.y, cls.ids, cls.folds = synth()
        cls.c = small_config()
        cls.tmp = tempfile.TemporaryDirectory()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def fit(self, model, arm, fold=0, root=None):
        root = new_root(root or tempfile.mkdtemp(dir=self.tmp.name))
        unit = run.make_unit(model, arm, fold, 'control' if arm == 'BASE' else 'study')
        ctx = ctx_for(self.c, [unit['key']])
        entry, fitted = run.run_unit(self.X, self.y, self.ids, self.folds, root, unit, ctx, resource_fn=stub_resources)
        self.assertTrue(fitted)
        return root, entry, json.loads((root/entry['dir']/f'fold_{fold}.json').read_text(encoding='utf-8'))

    def test_lgbm_arms_have_the_planned_widths_and_native_categoricals(self):
        widths = {'BASE': 39, 'S0': 49, 'S1': 52, 'S2': 52}
        for arm, width in widths.items():
            root, entry, rec = self.fit('LGBMN', arm)
            self.assertEqual(rec['input_columns'], width)
            self.assertEqual(rec['base_columns'], 39)
            self.assertEqual(rec['info']['rounds_completed'], 5)
            cat = rec['info']['categorical_feature']
            self.assertEqual(cat[:11], list(range(28, 39)))
            if arm != 'BASE':
                self.assertEqual(cat[11:], F.categorical_positions(arm, 39))
                self.assertTrue(rec['checks']['prefix_equals_independent_base_train'] and rec['checks']['prefix_equals_independent_base_valid'])
                self.assertEqual(len(rec['derived_gain']), F.n_derived(arm))
                self.assertIn('max_cat_to_onehot', rec['info']['categorical_effective_parameters'])
            self.assertTrue(rec['csv_round_trip_exact'])
            self.assertLessEqual(rec['reload_probability_max_abs_diff'], 1e-12)
            names = set(entry['artifact_sha256'])
            self.assertIn('fold_0_model.txt', names)
            self.assertEqual('fold_0_bundle.joblib' in names, arm != 'BASE')
            self.assertEqual(rec['metrics']['n'], 300)

    def test_lr_arms_append_scaled_numeric_and_onehot_csr(self):
        base_root, base_entry, base = self.fit('LR', 'BASE')
        self.assertEqual(base['info']['n_iter'][0] > 0, True)
        for arm in F.ARMS:
            root, entry, rec = self.fit('LR', arm)
            self.assertEqual(rec['base_columns'], base['input_columns'])
            extra = rec['input_columns'] - rec['base_columns']
            self.assertEqual(extra, F.n_numeric(arm) + rec['derived']['total_combinations'])
            self.assertLessEqual(rec['derived']['total_combinations'], 75)
            self.assertTrue(rec['checks']['independent_reference']['valid_numeric_max_relative_diff'] < 5e-6)
            self.assertTrue(rec['checks']['arm_relations']['train']['non_age_columns_identical'])
            self.assertEqual(rec['info']['max_iter'], 300)
            self.assertIn('converged', rec['info'])
            self.assertEqual(('region_age' in rec['derived']), arm != 'S0')
            self.assertEqual(sorted(entry['artifact_sha256']), sorted(
                ['fold_0_pred.csv.gz', 'fold_0_model.joblib', 'fold_0_base_pre.joblib', 'fold_0_bundle.joblib', 'fold_0.json']))

    def test_control_fit_equals_a_second_fit_bit_for_bit(self):
        a_root, a_entry, a = self.fit('LGBMN', 'BASE')
        b_root, b_entry, b = self.fit('LGBMN', 'BASE')
        pa = run.read_pred(a_root/'LGBMN'/'BASE'/'fold_0_pred.csv.gz')
        pb = run.read_pred(b_root/'LGBMN'/'BASE'/'fold_0_pred.csv.gz')
        cmp = run.tie_aware_compare(pa, pb, 1e-9)
        self.assertEqual(cmp['status'], 'PASS')
        self.assertTrue(cmp['bit_identical'] and cmp['labels_identical'])
        pb2 = pb.copy()
        pb2['p_grade_1'] += 1e-6
        self.assertEqual(run.tie_aware_compare(pa, pb2, 1e-9)['status'], 'FAIL')
        pb3 = pb.copy()
        pb3['y_pred'] = (pb3['y_pred'] % 3) + 1
        self.assertEqual(run.tie_aware_compare(pa, pb3, 1e-9)['status'], 'FAIL')
        pb4 = pb.copy()
        pb4['y_true'] = (pb4['y_true'] % 3) + 1
        self.assertFalse(run.tie_aware_compare(pa, pb4, 1e-9)['aligned'])
        # a label difference on an exactly tied reference row is recorded and tolerated, any other is not
        tie = pb.copy()
        tie.loc[0, ['p_grade_1', 'p_grade_2', 'p_grade_3']] = [0.5, 0.5, 0.0]
        new = tie.copy()
        new.loc[0, 'y_pred'] = 1 if tie.loc[0, 'y_pred'] != 1 else 2
        got = run.tie_aware_compare(new, tie, 1e-9)
        self.assertEqual((got['status'], got['label_mismatch_on_exact_top2_ties']), ('PASS', 1))

    def test_validation_labels_and_ages_cannot_change_training_matrices(self):
        a = np.flatnonzero(self.folds != 0)
        b = np.flatnonzero(self.folds == 0)
        X_tr, X_va = self.X.iloc[a], self.X.iloc[b]
        one = run.build_inputs('LGBMN', 'S1', X_tr, X_va, self.c)
        X_va2 = X_va.copy()
        X_va2['age'] = 995
        X_va2['geo_level_3_id'] = '31415'
        two = run.build_inputs('LGBMN', 'S1', X_tr, X_va2, self.c)
        self.assertTrue(np.array_equal(one['mt'], two['mt'], equal_nan=True))
        self.assertFalse(np.array_equal(one['mv'], two['mv'], equal_nan=True))
        self.assertEqual(F.state_signature(one['bundle']), F.state_signature(two['bundle']))
        one = run.build_inputs('LR', 'S2', X_tr, X_va, self.c)
        two = run.build_inputs('LR', 'S2', X_tr, X_va2, self.c)
        self.assertEqual((one['mt'] != two['mt']).nnz, 0)

    def test_prefix_check_detects_a_changed_base_column(self):
        a = np.flatnonzero(self.folds != 0)
        b = np.flatnonzero(self.folds == 0)
        X_tr, X_va = self.X.iloc[a], self.X.iloc[b]
        for model in run.MODELS:
            built = run.build_inputs(model, 'S1', X_tr, X_va, self.c)
            base = built['base']['mt']
            self.assertTrue(run.prefix_equal(model, built['mt'], base))
            broken = base.copy()
            if model == 'LGBMN':
                broken[0, 0] += 1
            else:
                broken = broken.tolil()
                broken[0, 0] += 1
                broken = broken.tocsr()
            self.assertFalse(run.prefix_equal(model, built['mt'], broken))

    def test_wrong_input_shapes_are_refused(self):
        a = np.flatnonzero(self.folds != 0)
        b = np.flatnonzero(self.folds == 0)
        X_tr, X_va = self.X.iloc[a], self.X.iloc[b]
        built = run.build_inputs('LGBMN', 'S0', X_tr, X_va, self.c, verify=False)
        with self.assertRaises(ValueError):
            run.check_input_matrix('LGBMN', 'S1', built['mt'], self.c)        # 49 columns are not an S1 matrix
        bad = built['mt'].copy()
        bad[0, 45] = 1.5
        with self.assertRaises(ValueError):
            run.check_input_matrix('LGBMN', 'S0', bad, self.c)
        bad = built['mt'].copy()
        bad[0, 40] = np.inf
        with self.assertRaises(ValueError):
            run.check_input_matrix('LGBMN', 'S0', bad, self.c)

    def test_lr_nonconvergence_is_recorded_not_stopped(self):
        c = copy.deepcopy(self.c)
        c['lr']['params']['max_iter'] = 2
        root = new_root(tempfile.mkdtemp(dir=self.tmp.name))
        unit = run.make_unit('LR', 'S1', 0, 'study')
        entry, fitted = run.run_unit(self.X, self.y, self.ids, self.folds, root, unit, ctx_for(c, [unit['key']]), resource_fn=stub_resources)
        rec = json.loads((root/'LR'/'S1'/'fold_0.json').read_text(encoding='utf-8'))
        self.assertTrue(fitted)
        self.assertFalse(rec['info']['converged'])
        self.assertTrue(rec['info']['convergence_warning'])
        self.assertEqual(rec['info']['n_iter'], [2])
        self.assertEqual(entry['status'], 'complete')


def fake_oof(y, folds, ids, seed, noise):
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    p = rng.dirichlet(np.ones(3), size=len(y))
    p[np.arange(len(y)), y - 1] += noise
    p /= p.sum(1, keepdims=True)
    return pd.DataFrame({ID: np.asarray(ids), 'fold': folds, 'y_true': y, 'y_pred': p.argmax(1) + 1,
                         **{n: p[:, i] for i, n in enumerate(run.PROBS)}})


@unittest.skipUnless(HAVE, 'dependencies not installed')
class JudgingTests(unittest.TestCase):
    rule = None

    @classmethod
    def setUpClass(cls):
        cls.rule = run.config()['rule']

    def row(self, deltas, **over):
        d = list(deltas)
        base = dict(fold_delta=d, mean_delta=float(np.mean(d)), delta_sd=float(np.std(d, ddof=1)), wins=int(sum(x > 0 for x in d)),
                    worst_fold_delta=min(d), delta_correct3=5, delta_error31=-1)
        base.update(over)
        return base

    def test_all_met_requires_every_condition(self):
        ok = run.judge(self.row([0.004, 0.0035, 0.0041, 0.0032, 0.005]), self.rule, caveat=False)
        self.assertEqual((ok['verdict'], ok['unmet_conditions'], ok['direction_tag'], ok['candidate']), ('ALL_MET', [], 'HIGHER', True))
        cases = {1: dict(deltas=[0.002]*5), 2: dict(deltas=[0.01, 0.01, 0.01, -0.0001, -0.0001]),
                 3: dict(deltas=[0.004]*5, delta_correct3=-1), 4: dict(deltas=[0.004]*5, delta_error31=1),
                 5: dict(deltas=[0.02, 0.02, 0.02, 0.02, -0.003])}
        for k, spec in cases.items():
            deltas = spec.pop('deltas')
            got = run.judge(self.row(deltas, **spec), self.rule, caveat=False)
            self.assertIn(k, got['unmet_conditions'], msg=str(k))
            self.assertEqual(got['verdict'], 'NOT_ALL_MET')
            self.assertFalse(got['candidate'])

    def test_boundaries_are_as_registered(self):
        exact = run.judge(self.row([0.003]*5), self.rule, False)
        self.assertTrue(exact['conditions'][1])                                   # >= +0.003 with the float epsilon
        self.assertFalse(run.judge(self.row([0.00299]*5), self.rule, False)['conditions'][1])
        self.assertFalse(run.judge(self.row([0.01, 0.01, 0.01, 0.0, 0.0]), self.rule, False)['conditions'][2])   # strict wins
        self.assertTrue(run.judge(self.row([0.01]*4 + [0.0]), self.rule, False)['conditions'][2])
        self.assertTrue(run.judge(self.row([0.004]*5, delta_error31=0), self.rule, False)['conditions'][4])
        self.assertFalse(run.judge(self.row([0.004]*5, delta_error31=1), self.rule, False)['conditions'][4])    # +1 fails
        self.assertTrue(run.judge(self.row([0.02]*4 + [-0.0029]), self.rule, False)['conditions'][5])
        self.assertFalse(run.judge(self.row([0.02]*4 + [-0.003]), self.rule, False)['conditions'][5])

    def test_direction_tags(self):
        self.assertEqual(run.judge(self.row([0.004]*5), self.rule, False)['direction_tag'], 'HIGHER')
        self.assertEqual(run.judge(self.row([-0.004]*5), self.rule, False)['direction_tag'], 'LOWER')
        self.assertEqual(run.judge(self.row([-0.02, -0.02, -0.02, -0.02, 0.04]), self.rule, False)['direction_tag'], 'LOWER')   # 1 win
        self.assertEqual(run.judge(self.row([-0.003]*5), self.rule, False)['direction_tag'], 'LOWER')           # boundary
        two_wins = self.row([-0.006, -0.006, -0.006, 0.0005, 0.0005], mean_delta=-0.0035)
        self.assertEqual(run.judge(two_wins, self.rule, False)['direction_tag'], 'NEUTRAL')              # wins > 1
        self.assertEqual(run.judge(self.row([0.001, -0.001, 0.0, 0.0, 0.0]), self.rule, False)['direction_tag'], 'NEUTRAL')

    def test_convergence_caveat_removes_candidate_status(self):
        got = run.judge(self.row([0.004]*5), self.rule, caveat=True)
        self.assertEqual(got['verdict'], 'ALL_MET')
        self.assertTrue(got['convergence_caveat'])
        self.assertFalse(got['candidate'])

    def test_double_flag_for_3_to_1(self):
        r = self.rule
        self.assertTrue(run.safety_flag(600, 530, r))        # +13.2% and +70
        self.assertFalse(run.safety_flag(560, 530, r))       # +30 but only +5.7%
        self.assertFalse(run.safety_flag(590, 560, r))       # +30 and +5.4% -> not above 10%
        self.assertFalse(run.safety_flag(70, 60, r))         # +16.7% but only +10 cases
        self.assertTrue(run.safety_flag(300, 200, r))
        self.assertFalse(run.safety_flag(500, 520, r))

    def test_difference_and_turnover(self):
        y = np.array([1, 2, 3, 3, 3, 1, 2, 3, 1, 2])
        fo = np.arange(10) % 5
        a = pd.DataFrame({'y_true': y, 'fold': fo, 'y_pred': [1, 2, 3, 3, 1, 1, 2, 2, 1, 2]})
        b = pd.DataFrame({'y_true': y, 'fold': fo, 'y_pred': [1, 2, 3, 1, 2, 1, 2, 3, 1, 1]})
        sa, sb = run.summarize_oof(a, 4), run.summarize_oof(b, 4)
        d = run.difference(sa, sb, a, b)
        self.assertEqual(d['delta_error31'], sa['error31'] - sb['error31'])
        self.assertEqual(sa['error31'], 1)
        self.assertEqual(sa['error32'], 1)
        self.assertEqual(len(d['fold_delta']), 5)
        self.assertEqual(d['turnover']['corrected_predictions'], int(((b.y_pred != y) & (a.y_pred == y)).sum()))
        self.assertEqual(d['turnover']['new_errors'], int(((b.y_pred == y) & (a.y_pred != y)).sum()))
        # the result of `difference(a, b)` reports a - b with a as candidate
        self.assertAlmostEqual(d['mean_delta'], sa['cv_mean'] - sb['cv_mean'])

    def test_family_notes_use_candidates_not_caveated_all_met(self):
        def judged(met):
            out = {}
            for m in run.MODELS:
                for a in run.ARMS:
                    ok = (m, a) in met
                    caveat = ok and m == 'LR'
                    out[f'{m}/{a}'] = dict(verdict='ALL_MET' if ok else 'NOT_ALL_MET', candidate=bool(ok and not caveat))
            return out
        # only a caveated LR comparison is ALL_MET: no candidate exists, so no model dependence
        fam = run.family_summary(judged({('LR', 'S1')}))
        self.assertEqual(fam['all_met_by_model'], {'LR': ['S1'], 'LGBMN': []})
        self.assertEqual(fam['candidates_by_model'], {'LR': [], 'LGBMN': []})
        self.assertFalse(fam['model_dependent'])
        self.assertIsNone(fam['partial_arms_note'])
        # exactly one model has a candidate -> model dependent; a partial set of arms is flagged
        fam = run.family_summary(judged({('LGBMN', 'S0'), ('LR', 'S2')}))
        self.assertEqual(fam['candidates_by_model'], {'LR': [], 'LGBMN': ['S0']})
        self.assertTrue(fam['model_dependent'])
        self.assertIsNotNone(fam['partial_arms_note'])
        fam = run.family_summary(judged({('LGBMN', 'S0'), ('LGBMN', 'S1'), ('LGBMN', 'S2')}))
        self.assertTrue(fam['model_dependent'])
        self.assertIsNone(fam['partial_arms_note'])
        fam = run.family_summary(judged(set()))
        self.assertFalse(fam['model_dependent'])

    def test_no_row_level_arrays_allowed_in_aggregates(self):
        self.assertTrue(run.no_row_level_data({'a': [1, 2, 3], 'b': {'c': [[1, 2], [3, 4]]}}))
        self.assertFalse(run.no_row_level_data({'a': list(range(1000))}))
        self.assertFalse(run.no_row_level_data({'a': {'b': [list(range(100))]}}))

    def test_subgroup_counts_only_below_the_row_floor(self):
        oof = fake_oof(np.array([1, 2, 3]*100), np.arange(300) % 5, np.arange(300), 0, 0.5)
        small = run.subgroup_summary(oof, np.arange(300) < 150, self.rule)
        self.assertIsNone(small['macro_f1'])
        self.assertEqual(small['rows'], 150)
        big = run.subgroup_summary(oof, np.ones(300, bool), self.rule)
        self.assertIsNotNone(big['macro_f1'])


@unittest.skipUnless(HAVE, 'dependencies not installed')
class EndToEndAnalysisTests(unittest.TestCase):
    """Whole tiny study on synthetic data: 35 tiny fits (with LR_B_REFIT), assembly, judging, aggregate-only output."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls.tmp.name)
        cls.X, cls.y, cls.ids, cls.folds = synth(1000, seed=3)
        cls.c = c = small_config()
        c['train_rows_per_fold'], c['valid_rows_per_fold'] = 800, 200
        c['grade3_support'] = int((cls.y == 3).sum())
        cls.root = root = tmp/'T34'
        run_dir = root/'run'
        (run_dir/'backup').mkdir(parents=True)
        run.jatomic(run_dir/'ledger.json', dict(study_id=run.STUDY_ID, scope='study', folds={}))
        # reused-B stand-ins (different file per model) on a synthetic local_root
        local = tmp/'local'
        local.mkdir()
        refs = {}
        for name, seed in (('LGBMN_B', 1), ('LR_B', 2), ('LR_ORIGINAL', 3)):
            path = local/f'{name}.csv.gz'
            run.write_pred(path, fake_oof(cls.y, cls.folds, cls.ids, seed, 0.6))
            refs[name] = dict(name=name, role='stand-in', path=f'{name}.csv.gz', sha256=sha256_file(path), cv_mean=0.0)
        c['references'] = refs
        keys = [run.key(*t) for t in run.study_targets(True)]
        ctx = ctx_for(c, keys, cap=41, before=2)
        for t in run.study_targets(True):
            run.run_unit(cls.X, cls.y, cls.ids, cls.folds, run_dir, run.make_unit(*t, 'refit' if t[1] == 'B_REFIT' else 'study'),
                         ctx, resource_fn=stub_resources)
        cls.design = dict(
            study_id=run.STUDY_ID, config=c, code_commit='2'*40, study_targets=keys, planned_completed_fits=35,
            completed_fits_including_control=37, attempt_cap=41, lr_b_refit=True, local_root=str(local),
            references=dict(LR_B=dict(n_iter=[[1]]*5, converged=[True]*5)),
            control=dict(path='control_check/control_check.json', attempts=2, retries=0))
        run.jwrite(run_dir/'design.json', cls.design)
        run.jwrite(root/'control_check'/'control_check.json', dict(lgbm=dict(status='PASS'), lr=dict(status='CONTROL_FAILED_LR')))
        cls.result = run.analyze(root, cls.X, cls.y, cls.ids, cls.folds, cls.design)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_ledger_counts_and_paths(self):
        led = run.read_ledger(self.root/'run')['folds']
        self.assertEqual(len(led), 35)
        self.assertTrue(all(e['status'] == 'complete' and e['attempts'] == 1 for e in led.values()))
        self.assertEqual(self.result['fits'], 35)
        self.assertEqual(self.result['attempts_including_control'], 37)
        self.assertEqual(self.result['retries'], 0)
        self.assertTrue(self.result['lr_b_refit'])
        self.assertFalse(self.result['holdout_accessed'])

    def test_six_judged_comparisons_and_six_descriptive_ones(self):
        self.assertEqual(sorted(self.result['judged']), sorted(f'{m}/{a}' for m in run.MODELS for a in run.ARMS))
        for row in self.result['judged'].values():
            self.assertEqual(len(row['fold_delta']), 5)
            self.assertIn(row['verdict'], ('ALL_MET', 'NOT_ALL_MET'))
            self.assertIn(row['direction_tag'], ('HIGHER', 'LOWER', 'NEUTRAL'))
            self.assertEqual(set(row['conditions']), {'1', '2', '3', '4', '5'})
            self.assertIn('safety_3to1_double_flag', row)
            self.assertIn('delta_error32', row)
            self.assertIn('delta_error23', row)
        for m in run.MODELS:
            self.assertEqual(sorted(self.result['descriptive'][m]), ['S1-S0', 'S2-S0', 'S2-S1'])
            for d in self.result['descriptive'][m].values():
                self.assertNotIn('verdict', d)
                self.assertNotIn('direction_tag', d)
                self.assertIn('turnover', d)

    def test_lr_b_is_the_refit_when_the_control_failed(self):
        led = run.read_ledger(self.root/'run')['folds']
        b = run.read_pred(self.root/'run'/'LR'/'B_REFIT'/'oof.csv.gz')
        self.assertEqual(len(b), len(self.ids))
        f1 = run.fold_f1(b)
        self.assertEqual(f1, self.result['baseline']['LR']['fold_macro_f1'])
        self.assertEqual(self.result['judged']['LR/S1']['n_iter'] is not None, True)
        self.assertEqual(self.result['diagnostics']['convergence']['B']['source'], 'LR_B_REFIT (new fits)')
        self.assertEqual(len(led), 35)

    def test_aggregate_only_and_json_safe(self):
        self.assertTrue(run.no_row_level_data(self.result))
        text = json.dumps(self.result, allow_nan=False)
        self.assertNotIn('building_id', text)
        self.assertGreater(len(text), 1000)
        run.jwrite(Path(self.tmp.name)/'a.json', self.result)

    def test_family_notes_and_caveat_logic(self):
        fam = self.result['family']
        self.assertEqual(set(fam['all_met_by_model']), {'LR', 'LGBMN'})
        self.assertEqual(set(fam['candidates_by_model']), {'LR', 'LGBMN'})
        for key_, row in self.result['judged'].items():
            if row['verdict'] == 'ALL_MET':
                self.assertEqual(row['candidate'], not row['convergence_caveat'])
            else:
                self.assertFalse(row['candidate'])
        self.assertTrue(any('ALL_MET' in n for n in self.result['notes']))

    def test_diagnostics_cover_the_requested_items(self):
        d = self.result['diagnostics']
        self.assertEqual(sorted(d['folds']), sorted(f'{m}/{a}' for m in run.MODELS for a in run.ARMS))
        for arm in ('S1', 'S2'):
            self.assertEqual(sorted(d['self_inclusion_by_fold'][arm]), [str(f) for f in range(5)])
        self.assertIn('geo_level_3_id_region_has_995_in_training', d['subgroups']['LGBMN/S1'])
        self.assertIn('age995', d['subgroups']['LR/B'])
        self.assertIn('geo1_26', d['subgroups']['LR/S0'])
        self.assertEqual(d['subgroups']['LR/B']['age995']['macro_f1'], None)     # < 200 rows -> counts only
        self.assertTrue(all(g is not None for g in d['folds']['LGBMN/S1']['gain_share_derived']))
        self.assertEqual(sorted(self.result['baseline']), ['LGBMN', 'LR'])
        self.assertEqual(self.result['cited_lr_original']['support3'], self.c['grade3_support'])

    def test_finalize_writes_once_and_backs_up(self):
        root = self.root
        analysis = root/'run'/'analysis.json'
        self.assertFalse(analysis.exists())
        with mock.patch.object(run, 'analyze', return_value=dict(small=[1, 2])):
            run.finalize(root, verify_fn=lambda r, d: (None, None, None, None, self.design))
        self.assertTrue(analysis.exists() and (root/'run'/'backup'/'analysis.json').exists())
        with self.assertRaisesRegex(ValueError, 'Analysis exists'):
            run.finalize(root, verify_fn=lambda r, d: (None, None, None, None, self.design))
        analysis.unlink()
        (root/'run'/'backup'/'analysis.json').unlink()


@unittest.skipUnless(HAVE, 'dependencies not installed')
class ControlAndCommandTests(unittest.TestCase):
    def test_control_requires_preflight_and_committed_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'Sealed sources|preflight'):
                run.control_check(Path(tmp)/'data', Path(tmp)/'out')
            with mock.patch.object(run, 'source_mismatches', return_value=[]):
                with self.assertRaisesRegex(ValueError, 'preflight'):
                    run.control_check(Path(tmp)/'data', Path(tmp)/'out')

    def test_control_check_flow_with_stubbed_fits_and_failures(self):
        c = small_config()
        X, y, ids, folds = synth(1000, seed=5)
        c['train_rows_per_fold'], c['valid_rows_per_fold'] = 800, 200
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            local = tmp/'local'
            local.mkdir()
            refs = {}
            for name, seed in (('LGBMN_B', 1), ('LR_B', 2), ('LR_ORIGINAL', 3)):
                path = local/f'{name}.csv.gz'
                run.write_pred(path, fake_oof(y, folds, ids, seed, 0.6))
                refs[name] = dict(name=name, role='stand-in', path=f'{name}.csv.gz', sha256=sha256_file(path), cv_mean=0.0)
            c['references'] = refs
            head = '3'*40

            def make_execute(match):
                """Fake fits whose probabilities equal the reference rows (match) or differ slightly."""
                def execute(X_, y_, ids_, folds_, d, unit, c_):
                    ref = run.read_reference(local, c['references']['LGBMN_B' if unit['model'] == 'LGBMN' else 'LR_B'], ids, y, folds)
                    frame = ref[ref.fold == 0].copy()
                    if not match.get(unit['model'], True):
                        frame['p_grade_1'] += 1e-6
                    d.mkdir(parents=True, exist_ok=True)
                    pred = d/'fold_0_pred.csv.gz'
                    run.write_pred(pred, frame)
                    js = d/'fold_0.json'
                    rec = dict(metrics=dict(macro_f1=0.5), info=dict(fit_seconds=0.1))
                    run.jwrite(js, rec)
                    return rec, [pred, js], None
                return execute

            def go(out, match, **kw):
                patches = [mock.patch.object(run, 'config', return_value=c), mock.patch.object(run, 'check_plan_sha'),
                           mock.patch.object(run, 'source_mismatches', return_value=[]), mock.patch.object(run, 'require_environment'),
                           mock.patch.object(run, 'git_head', return_value=head), mock.patch.object(run, 'read_preflight'),
                           mock.patch.object(run, 'load_data', return_value=(X, y, ids, folds, {'files': {}})),
                           mock.patch.object(run, 'verify_references', return_value={})]
                for p in patches:
                    p.start()
                try:
                    return run.control_check(tmp/'data', out, local_root=local, execute_fn=make_execute(match), resource_fn=stub_resources, **kw)
                finally:
                    for p in reversed(patches):
                        p.stop()

            ok = go(tmp/'ok', {})
            self.assertEqual((ok['lgbm']['status'], ok['lr']['status'], ok['attempts'], ok['lr_b_refit_required']), ('PASS', 'PASS', 2, False))
            self.assertEqual(ok['control_fits_completed'], 2)
            self.assertEqual(ok['code_commit'], head)
            self.assertEqual(run.control_check.__defaults__ is not None, True)
            again = go(tmp/'ok', {})                                                 # idempotent: verified, not refit
            self.assertEqual(again['attempts'], 2)
            lr_fail = go(tmp/'lrfail', {'LR': False})
            self.assertEqual((lr_fail['lgbm']['status'], lr_fail['lr']['status'], lr_fail['lr_b_refit_required']),
                             ('PASS', 'CONTROL_FAILED_LR', True))
            with self.assertRaisesRegex(ValueError, 'CONTROL_FAILED_LGBM'):
                go(tmp/'lgbmfail', {'LGBMN': False})
            res = json.loads((tmp/'lgbmfail'/'control_check'/'control_check.json').read_text(encoding='utf-8'))
            self.assertEqual((res['lgbm']['status'], res['lr']['status']), ('CONTROL_FAILED_LGBM', 'NOT_RUN'))
            self.assertTrue((tmp/'lgbmfail'/'control_check'/'LGBMN'/'BASE'/'fold_0_pred.csv.gz').exists())   # kept, not discarded
            with self.assertRaisesRegex(ValueError, 'CONTROL_FAILED_LGBM'):
                go(tmp/'lgbmfail', {})                                                  # a failed control stays failed

    def test_cli_requires_a_reason_for_retries_and_known_commands(self):
        with self.assertRaises(ValueError):
            with mock.patch.object(run, 'status', return_value={}):
                run.main(['status', '--out', 'x', '--retry-interrupted'])
        with self.assertRaises(SystemExit):
            run.main(['unknown', '--out', 'x'])
        for name in ('preflight', 'control-check', 'prepare', 'fit-all', 'status', 'finalize'):
            with mock.patch.object(run, name.replace('-', '_'), return_value={}):
                if name in ('preflight', 'control-check', 'prepare'):
                    run.main([name, '--out', 'x', '--data', 'd'])
                else:
                    run.main([name, '--out', 'x'])

    def test_preflight_and_prepare_stop_on_zero_area_and_bad_inputs(self):
        X, y, ids, folds = synth(200)
        X.loc[0, 'area_percentage'] = 0
        c = small_config()
        with mock.patch.object(run, 'load_data', return_value=(X, y, ids, folds, {})), \
                mock.patch.object(run, 'verify_references', return_value={}):
            with self.assertRaisesRegex(ValueError, 'area_percentage == 0'):
                run.preflight_checks('d', c, 'l')


def write_reference_files(tmp, y, ids, folds, seeds=(1, 2, 3)):
    """Synthetic stand-ins of the three pinned reference sets with consistent metrics files."""
    tmp = Path(tmp)
    specs = {}
    for (name, seed) in zip(('LGBMN_B', 'LR_B', 'LR_ORIGINAL'), seeds):
        frame = fake_oof(y, folds, ids, seed, 0.6)
        path = tmp/f'{name}.csv.gz'
        run.write_pred(path, frame)
        vals = run.fold_f1(frame)
        spec = dict(name=name, role='stand-in', path=f'{name}.csv.gz', sha256=sha256_file(path), cv_mean=float(np.mean(vals)))
        if name == 'LGBMN_B':
            run.jwrite(tmp/'m_lgbm.json', dict(cv_mean=float(np.mean(vals)), fold_macro_f1=vals))
            spec.update(metrics_path='m_lgbm.json', metrics_sha256=sha256_file(tmp/'m_lgbm.json'))
        if name == 'LR_B':
            run.jwrite(tmp/'m_lr.json', dict(cv_macro_f1_mean=float(np.mean(vals))))
            run.jwrite(tmp/'fm_lr.json', [dict(validation=dict(macro_f1=v), n_iter=[500 + i], convergence_ok=True) for i, v in enumerate(vals)])
            spec.update(metrics_path='m_lr.json', metrics_sha256=sha256_file(tmp/'m_lr.json'),
                        fold_metrics_path='fm_lr.json', fold_metrics_sha256=sha256_file(tmp/'fm_lr.json'),
                        expected_n_iter=[[500 + i] for i in range(5)])
        specs[name] = spec
    return specs


@unittest.skipUnless(HAVE, 'dependencies not installed')
class ReferenceTests(unittest.TestCase):
    def test_verify_references_accepts_consistent_and_rejects_drift(self):
        X, y, ids, folds = synth(1000, seed=11)
        c = small_config()
        with tempfile.TemporaryDirectory() as tmp:
            c['references'] = write_reference_files(tmp, y, ids, folds)
            out = run.verify_references(c, tmp, ids, y, folds)
            self.assertEqual(sorted(out), ['LGBMN_B', 'LR_B', 'LR_ORIGINAL'])
            self.assertEqual(out['LR_B']['n_iter'], [[500 + i] for i in range(5)])
            self.assertTrue(all(out['LR_B']['converged']))
            self.assertLess(out['LGBMN_B']['fold_macro_f1_max_diff_vs_metrics'], 1e-9)
            self.assertEqual(sum(sum(r) for r in out['LGBMN_B']['pooled_confusion']), 1000)
            self.assertEqual(run.verify_references(c, tmp, ids, y, folds), out)             # deterministic
            for mutate, message in (
                    (lambda d: d['LGBMN_B'].update(sha256='0'*64), 'SHA mismatch'),
                    (lambda d: d['LR_B'].update(cv_mean=d['LR_B']['cv_mean'] + 1e-3), 'recomputed mean'),
                    (lambda d: d['LR_B'].update(metrics_sha256='0'*64), 'metrics.json SHA'),
                    (lambda d: d['LR_B'].update(expected_n_iter=[[1]]*5), 'n_iter'),
                    (lambda d: d['LGBMN_B'].update(path='missing.csv.gz'), 'SHA mismatch')):
                bad = copy.deepcopy(c)
                mutate(bad['references'])
                with self.assertRaisesRegex(ValueError, message):
                    run.verify_references(bad, tmp, ids, y, folds)
            # a metrics.json whose fold values disagree with the recomputed ones is refused
            original = (Path(tmp)/'m_lgbm.json').read_bytes()
            m = json.loads(original.decode('utf-8'))
            m['fold_macro_f1'][0] += 1e-6
            run.jwrite(Path(tmp)/'m_lgbm.json', m)
            bad = copy.deepcopy(c)
            bad['references']['LGBMN_B']['metrics_sha256'] = sha256_file(Path(tmp)/'m_lgbm.json')
            with self.assertRaisesRegex(ValueError, 'Macro F1 differs'):
                run.verify_references(bad, tmp, ids, y, folds)
            (Path(tmp)/'m_lgbm.json').write_bytes(original)
            # misaligned reference (shifted fold labels) is refused
            frame = run.read_pred(Path(tmp)/'LR_ORIGINAL.csv.gz')
            frame['fold'] = (frame['fold'] + 1) % 5
            run.write_pred(Path(tmp)/'LR_ORIGINAL.csv.gz', frame)
            bad = copy.deepcopy(c)
            bad['references']['LR_ORIGINAL']['sha256'] = sha256_file(Path(tmp)/'LR_ORIGINAL.csv.gz')
            with self.assertRaises(AssertionError):
                run.verify_references(bad, tmp, ids, y, folds)

    def test_preflight_binding_checks(self):
        c = run.config()
        head = '4'*40
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, 'Run preflight first'):
                run.read_preflight(root, c, head)
            good = dict(status='PASS', classifiers_fitted=0, code_commit=head, plan_sha256=c['plan_sha256'],
                        config_sha256=sha256_file(run.HERE/'config.json'), source_sha256=run.sources())
            run.jwrite(root/'preflight'/'preflight.json', good)
            self.assertEqual(run.read_preflight(root, c, head)['status'], 'PASS')
            for key_, value in (('code_commit', '5'*40), ('status', 'FAIL'), ('config_sha256', '0'*64), ('plan_sha256', '0'*64),
                                ('classifiers_fitted', 1)):
                run.jwrite(root/'preflight'/'preflight.json', dict(good, **{key_: value}))
                with self.assertRaisesRegex(ValueError, 'Preflight'):
                    run.read_preflight(root, c, head)

    def test_preflight_checks_on_synthetic_data(self):
        X, y, ids, folds = synth(1000, seed=21)
        c = small_config()
        c['train_rows_per_fold'], c['valid_rows_per_fold'] = 800, 200
        with mock.patch.object(run, 'load_data', return_value=(X, y, ids, folds, {'files': {}})), \
                mock.patch.object(run, 'verify_references', return_value={'stand': 'in'}):
            result = run.preflight_checks('d', c, 'l')
        self.assertEqual((result['status'], result['classifiers_fitted'], result['holdout_accessed']), ('PASS', 0, False))
        self.assertEqual(result['area_percentage_zero_rows'], 0)
        self.assertEqual(sorted(result['folds']), ['0', '1', '2', '3', '4'])
        for g in GEO:
            self.assertTrue(result['geo_keys'][g]['all_digit_strings'])
        m = result['fold0_matrices']
        self.assertEqual([m[f'LGBMN/{a}']['train_shape'][1] for a in ('BASE', 'S0', 'S1', 'S2')], [39, 49, 52, 52])
        self.assertEqual(m['LR/S1']['container'], 'csr')
        self.assertEqual(m['LGBMN/S1']['container'], 'dense')
        self.assertGreaterEqual(m['LR/S0']['derived_columns'], F.n_numeric('S0'))
        self.assertIn('valid_unseen_combination_rows', result['folds']['0']['arms']['S1'])
        self.assertIn('region_stats_valid', result['folds']['0']['arms']['S2'])
        self.assertIn('mixed_995', result['folds']['3'])
        json.dumps(run.py(result), allow_nan=False)


@unittest.skipUnless(HAVE, 'dependencies not installed')
class SealFlowTests(unittest.TestCase):
    """control-check -> prepare -> verify -> fit-all (stub fits) -> status with all external inputs stubbed."""

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.TemporaryDirectory()
        tmp = cls.tmp = Path(cls.tmpdir.name)
        cls.X, cls.y, cls.ids, cls.folds = synth(1000, seed=5)
        cls.c = c = small_config()
        c['train_rows_per_fold'], c['valid_rows_per_fold'] = 800, 200
        cls.local = tmp/'local'
        cls.local.mkdir()
        c['references'] = write_reference_files(cls.local, cls.y, cls.ids, cls.folds)
        cls.head = '3'*40

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    def patches(self, head=None):
        c = self.c
        pf = dict(references={}, data_contract={'files': {}})
        return [mock.patch.object(run, 'config', return_value=c), mock.patch.object(run, 'source_mismatches', return_value=[]),
                mock.patch.object(run, 'git_head', return_value=head or self.head),
                mock.patch.object(run, 'read_preflight', return_value=pf),
                mock.patch.object(run, 'load_data', return_value=(self.X, self.y, self.ids, self.folds, {'files': {}})),
                mock.patch.object(run, 'verify_references', return_value={})]

    def call(self, fn, *a, head=None, **kw):
        ps = self.patches(head)
        for p in ps:
            p.start()
        try:
            return fn(*a, **kw)
        finally:
            for p in reversed(ps):
                p.stop()

    def execute(self, match=True, lr_shift=False):
        def ex(X, y, ids, folds, d, unit, c):
            ref = run.read_reference(self.local, self.c['references']['LGBMN_B' if unit['model'] == 'LGBMN' else 'LR_B'],
                                     self.ids, self.y, self.folds)
            frame = ref[ref.fold == 0].copy()
            if not match or (lr_shift and unit['model'] == 'LR'):
                frame['p_grade_1'] += 1e-6
            d.mkdir(parents=True, exist_ok=True)
            pred, js = d/f'fold_{unit["fold"]}_pred.csv.gz', d/f'fold_{unit["fold"]}.json'
            run.write_pred(pred, frame)
            rec = dict(metrics=dict(macro_f1=0.5), info=dict(fit_seconds=0.1))
            run.jwrite(js, rec)
            return rec, [pred, js], None
        return ex

    def controlled(self, name, **kw):
        out = self.tmp/name
        run.jwrite(out/'preflight'/'preflight.json', dict(status='PASS', stand_in=True))
        self.call(run.control_check, self.tmp/'data', out, local_root=self.local, execute_fn=self.execute(**kw),
                  resource_fn=stub_resources)
        return out

    def test_prepare_binds_control_and_seals(self):
        out = self.controlled('ok')
        design = self.call(run.prepare, self.tmp/'data', out, local_root=self.local)
        self.assertEqual(design['code_commit'], self.head)
        self.assertEqual(design['plan_sha256'], run.PLAN_SHA256)
        self.assertFalse(design['lr_b_refit'])
        self.assertEqual((len(design['study_targets']), design['attempt_cap'], design['planned_completed_fits']), (30, 36, 30))
        self.assertEqual(design['completed_fits_including_control'], 32)
        self.assertEqual((design['control']['lgbm_status'], design['control']['lr_status'], design['control']['attempts']), ('PASS', 'PASS', 2))
        self.assertEqual(design['control']['sha256'], sha256_file(out/'control_check'/'control_check.json'))
        self.assertEqual(design['control']['code_commit'], self.head)
        self.assertEqual(design['study_targets'][0], 'LR/S0/0')
        self.assertEqual(design['study_targets'][-1], 'LGBMN/S2/4')
        run_dir = out/'run'
        seal = json.loads((run_dir/'design_seal.json').read_text(encoding='utf-8'))
        self.assertEqual(seal['design_sha256'], sha256_file(run_dir/'design.json'))
        self.assertEqual(seal['classifiers_fitted'], 0)
        self.assertTrue((run_dir/'backup'/'design.json').exists() and (run_dir/'ledger.json').exists())
        with self.assertRaisesRegex(ValueError, 'Study output exists'):
            self.call(run.prepare, self.tmp/'data', out, local_root=self.local)
        X, y, ids, folds, d = self.call(run.verify, out)
        self.assertEqual(d['study_targets'], design['study_targets'])
        # the seal detects changes of the design and of the control result
        original = (out/'control_check'/'control_check.json').read_bytes()
        (out/'control_check'/'control_check.json').write_bytes(original + b' ')
        with self.assertRaisesRegex(ValueError, 'Control check changed'):
            self.call(run.verify, out)
        (out/'control_check'/'control_check.json').write_bytes(original)
        text = (run_dir/'design.json').read_bytes()
        (run_dir/'design.json').write_bytes(text + b' ')
        with self.assertRaisesRegex(ValueError, 'Design seal changed'):
            self.call(run.verify, out)
        (run_dir/'design.json').write_bytes(text)
        self.call(run.verify, out)
        # the preflight record is bound by the seal too
        pre = out/'preflight'/'preflight.json'
        original_pre = pre.read_bytes()
        pre.write_bytes(original_pre + b' ')
        with self.assertRaisesRegex(ValueError, 'Preflight record changed'):
            self.call(run.verify, out)
        pre.write_bytes(original_pre)
        self.call(run.verify, out)
        # psutil version is recorded and re-checked
        self.assertEqual(d['environment_extra'], run.environment_extra())
        with mock.patch.object(run, 'environment_extra', return_value=dict(psutil='0.0.0')):
            with self.assertRaisesRegex(ValueError, 'psutil'):
                self.call(run.verify, out)
        cc = json.loads((out/'control_check'/'control_check.json').read_text(encoding='utf-8'))
        self.assertEqual(cc['environment_extra'], run.environment_extra())

    def test_fit_all_status_and_resume_through_the_sealed_design(self):
        out = self.controlled('flow')
        self.call(run.prepare, self.tmp/'data', out, local_root=self.local)
        st = run.status(out)
        self.assertEqual((st['prepared'], st['complete'], st['total'], st['attempts_including_control']), (True, 0, 30, 2))
        ex = stub_execute()
        self.call(run.fit_all, out, execute_fn=ex, resource_fn=stub_resources)
        self.assertEqual(len(ex.calls), 30)
        st = run.status(out)
        self.assertEqual((st['complete'], st['attempts_study'], st['attempts_including_control'], st['failed'], st['lock_present']),
                         (30, 30, 32, [], False))
        self.assertEqual(st['per_arm']['LR/S1'], dict(complete=5, of=5))
        self.assertLessEqual(st['attempts_including_control'], st['attempt_cap'])
        ex2 = stub_execute()
        self.call(run.fit_all, out, execute_fn=ex2, resource_fn=stub_resources)
        self.assertEqual(ex2.calls, [])
        guard = json.loads(sorted((out/'run').glob('guard_log_*.json'))[0].read_text(encoding='utf-8'))
        self.assertIn('audit_hook_installed', guard)
        # a foreign lock blocks a second executor; the lock is not removed by the blocked process
        (out/'run'/'RUNNING.lock').write_text('pid=1', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Another run holds'):
            self.call(run.fit_all, out, execute_fn=stub_execute(), resource_fn=stub_resources)
        self.assertTrue((out/'run'/'RUNNING.lock').exists())
        (out/'run'/'RUNNING.lock').unlink()

    def test_lr_control_failure_adds_the_refit_arm_and_budget(self):
        out = self.controlled('lrfail', lr_shift=True)
        design = self.call(run.prepare, self.tmp/'data', out, local_root=self.local)
        self.assertTrue(design['lr_b_refit'])
        self.assertEqual((len(design['study_targets']), design['attempt_cap'], design['completed_fits_including_control']), (35, 41, 37))
        self.assertEqual(design['study_targets'][:5], [f'LR/B_REFIT/{f}' for f in range(5)])
        self.assertEqual(design['control']['lr_status'], 'CONTROL_FAILED_LR')
        ex = stub_execute()
        self.call(run.fit_all, out, execute_fn=ex, resource_fn=stub_resources)
        self.assertEqual(ex.calls[:5], [f'LR/B_REFIT/{f}' for f in range(5)])           # the refit arm runs first
        self.assertEqual(run.status(out)['attempts_including_control'], 37)

    def test_control_alignment_defect_stops_instead_of_adding_the_refit_arm(self):
        base_ex = self.execute()

        def ex(X, y, ids, folds, d, unit, c):
            rec, files, p = base_ex(X, y, ids, folds, d, unit, c)
            if unit['model'] == 'LR':
                frame = run.read_pred(files[0])
                frame['y_true'] = (frame['y_true'] % 3) + 1                    # labels no longer match the reference rows
                run.write_pred(files[0], frame)
            return rec, files, p
        out = self.tmp/'misaligned'
        run.jwrite(out/'preflight'/'preflight.json', dict(status='PASS', stand_in=True))
        with self.assertRaisesRegex(ValueError, 'alignment'):
            self.call(run.control_check, self.tmp/'data', out, local_root=self.local, execute_fn=ex, resource_fn=stub_resources)
        self.assertFalse((out/'control_check'/'control_check.json').exists())      # no CONTROL_FAILED_LR record is written
        with self.assertRaisesRegex(ValueError, 'Run control-check first'):
            self.call(run.prepare, self.tmp/'data', out, local_root=self.local)
        self.assertFalse((out/'run').exists())

    def test_prepare_refuses_wrong_commit_failed_lgbm_and_missing_control(self):
        out = self.controlled('commit')
        with self.assertRaisesRegex(ValueError, 'another commit'):
            self.call(run.prepare, self.tmp/'data', out, local_root=self.local, head='9'*40)
        self.assertFalse((out/'run').exists())
        with self.assertRaisesRegex(ValueError, 'code_commit differs'):
            self.call(run.prepare, self.tmp/'data', out, local_root=self.local, code_commit='8'*40)
        with self.assertRaisesRegex(ValueError, 'Run control-check first'):
            self.call(run.prepare, self.tmp/'data', self.tmp/'nocontrol', local_root=self.local)
        bad = self.tmp/'lgbmbad'
        run.jwrite(bad/'preflight'/'preflight.json', dict(status='PASS', stand_in=True))
        with self.assertRaisesRegex(ValueError, 'CONTROL_FAILED_LGBM'):
            self.call(run.control_check, self.tmp/'data', bad, local_root=self.local, execute_fn=self.execute(match=False),
                      resource_fn=stub_resources)
        with self.assertRaisesRegex(ValueError, 'CONTROL_FAILED_LGBM'):
            self.call(run.prepare, self.tmp/'data', bad, local_root=self.local)
        # an unfinished control ledger entry blocks sealing
        part = self.controlled('partial')
        led = run.read_ledger(part/'control_check')
        led['folds']['LR/BASE/0']['status'] = 'running'
        run.jatomic(part/'control_check'/'ledger.json', led)
        with self.assertRaisesRegex(ValueError, 'unfinished'):
            self.call(run.prepare, self.tmp/'data', part, local_root=self.local)


def tearDownModule():
    if HAVE:
        run.disarm_audit_hook_for_tests()                      # the process-wide hook must not leak into other test modules


if __name__ == '__main__':
    unittest.main()
