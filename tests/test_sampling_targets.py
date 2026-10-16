"""T32 synthetic contracts only: no real development data, no holdout access, no study training.

Fits here are tiny synthetic checks of the code path, never study fits.
"""
import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'tests'))
from test_modeling import synthetic  # noqa: E402
from experiments.sampling_targets import targets as tg  # noqa: E402

try:
    import imblearn  # noqa: F401
    import lightgbm  # noqa: F401
    from experiments.sampling_targets import run as t32
    HAVE = True
except ImportError:  # pragma: no cover
    HAVE = False

FOLD0 = (16080, 94885, 55819)
FOLD2 = (16079, 94886, 55819)


def labels(n1=300, n2=1700, n3=1000, seed=0):
    y = np.concatenate([np.full(n1, 1), np.full(n2, 2), np.full(n3, 3)]).astype(np.uint8)
    return np.random.default_rng(seed).permutation(y)


class TargetFormulaTests(unittest.TestCase):
    def test_fold0_table_is_reproduced(self):
        expected = {
            ('UP', 'P25'): (35781, 94885, 65586), ('UP', 'P50'): (55483, 94885, 75352),
            ('UP', 'P75'): (75184, 94885, 85119), ('UP', 'EQ'): (94885, 94885, 94885),
            ('UP', 'ONE1'): (94885, 94885, 55819), ('UP', 'ONE3'): (16080, 94885, 94885),
            ('DOWN', 'P25'): (16080, 75184, 45884), ('DOWN', 'P50'): (16080, 55482, 35949),
            ('DOWN', 'P75'): (16080, 35781, 26015), ('DOWN', 'EQ'): (16080, 16080, 16080)}
        for (direction, scenario), want in expected.items():
            self.assertEqual(tg.scenario_targets(FOLD0, direction, scenario), want, (direction, scenario))
        self.assertEqual(sum(tg.scenario_targets(FOLD0, 'UP', 'P50')), 225720)
        self.assertEqual(sum(tg.scenario_targets(FOLD0, 'DOWN', 'P50')), 107511)
        self.assertEqual(tg.scenario_targets(FOLD0, None, 'ORI'), FOLD0)

    def test_deltas_of_fold0_examples(self):
        d = tg.describe_targets(FOLD0, tg.scenario_targets(FOLD0, 'UP', 'P25'))
        self.assertEqual(d['delta'], [19701, 0, 9767])
        d = tg.describe_targets(FOLD0, tg.scenario_targets(FOLD0, 'DOWN', 'P75'))
        self.assertEqual(d['delta'], [0, -59104, -29804])

    def test_fold2_has_the_extra_majority_row(self):
        self.assertEqual(tg.scenario_targets(FOLD2, 'UP', 'EQ'), (94886, 94886, 94886))
        self.assertEqual(tg.scenario_targets(FOLD2, 'DOWN', 'EQ'), (16079, 16079, 16079))

    def test_half_rounds_up_and_integers_only(self):
        # gap 11, half -> 5.5 -> 6 ; gap 10, quarter -> 2.5 -> 3 ; gap 9 quarter -> 2.25 -> 2
        self.assertEqual(tg.half_up_share(11, 1, 2), 6)
        self.assertEqual(tg.half_up_share(10, 1, 4), 3)
        self.assertEqual(tg.half_up_share(9, 1, 4), 2)
        self.assertEqual(tg.half_up_share(0, 1, 2), 0)
        self.assertEqual(tg.half_up_share(7, 1, 1), 7)
        for bad in [(-1, 1, 2), (3, 0, 2), (3, 3, 2)]:
            with self.assertRaises(tg.TargetError):
                tg.half_up_share(*bad)

    def test_order_violation_stops(self):
        for n in [(100, 50, 70), (100, 200, 100), (0, 20, 10), (10, 20, 20), (30, 20, 25)]:
            with self.assertRaises(tg.TargetError):
                tg.scenario_targets(n, 'UP', 'P50')
        with self.assertRaises(tg.TargetError):
            tg.count_vector(np.array([1, 2, 3, 4]))

    def test_special_scenarios_are_up_only(self):
        with self.assertRaises(tg.TargetError):
            tg.scenario_targets(FOLD0, 'DOWN', 'ONE1')
        with self.assertRaises(tg.TargetError):
            tg.scenario_targets(FOLD0, 'UP', 'NOPE')


@unittest.skipUnless(HAVE, 'imbalanced-learn/lightgbm missing')
class SamplerContractTests(unittest.TestCase):
    def test_strategy_keys_exclude_the_fixed_grade_and_are_ascending(self):
        n = (300, 1700, 1000)
        up = tg.sampler_strategy(n, tg.scenario_targets(n, 'UP', 'P50'), 'UP')
        down = tg.sampler_strategy(n, tg.scenario_targets(n, 'DOWN', 'P50'), 'DOWN')
        self.assertEqual(list(up), [1, 3])
        self.assertEqual(list(down), [2, 3])
        self.assertNotIn(2, up)
        self.assertNotIn(1, down)
        self.assertEqual(list(tg.sampler_strategy(n, tg.scenario_targets(n, 'UP', 'ONE1'), 'UP')), [1])
        self.assertEqual(list(tg.sampler_strategy(n, tg.scenario_targets(n, 'UP', 'ONE3'), 'UP')), [3])
        with self.assertRaises(tg.TargetError):                  # fixed grade must not move
            tg.sampler_strategy(n, (300, 1600, 1000), 'UP')
        with self.assertRaises(tg.TargetError):                  # over-sampling cannot shrink
            tg.sampler_strategy(n, (200, 1700, 1000), 'UP')

    def test_gidx_passes_on_synthetic_labels_for_five_folds(self):
        y = labels()
        folds = np.arange(len(y)) % 5
        result = tg.gidx(y, folds, 42)
        self.assertTrue(result['all_pass'])
        for f in range(5):
            r = result['folds'][str(f)]
            self.assertTrue(r['UP']['identical'] and r['DOWN']['identical'])
            self.assertTrue(r['UP']['after_equals_target'] and r['DOWN']['after_equals_target'])

    def test_dict_path_equals_auto_path_only_without_the_fixed_key(self):
        y = labels()
        n = tg.count_vector(y)
        auto = tg.auto_sample_indices(y, 'DOWN', 42)
        good = tg.sample_indices(y, {2: n[0], 3: n[0]}, 'DOWN', 42)
        self.assertTrue(np.array_equal(good, auto))
        # including the fixed minority key consumes the random stream and breaks the equality
        from imblearn.under_sampling import RandomUnderSampler
        bad = RandomUnderSampler(sampling_strategy={1: n[0], 2: n[0], 3: n[0]}, random_state=42)
        bad.fit_resample(np.arange(len(y)).reshape(-1, 1), y)
        self.assertFalse(np.array_equal(bad.sample_indices_, auto))

    def test_gidx_detects_a_mismatch(self):
        y = labels()
        with mock.patch.object(tg, 'auto_sample_indices', side_effect=lambda yy, d, s: np.arange(5)):
            r = tg.gidx_fold(y)
        self.assertFalse(r['pass'])

    def test_partial_up_adds_exactly_the_target(self):
        y = labels()
        n = tg.count_vector(y)
        for scenario in ['P25', 'P50', 'P75', 'EQ', 'ONE1', 'ONE3']:
            t = tg.scenario_targets(n, 'UP', scenario)
            idx = tg.sample_indices(y, tg.sampler_strategy(n, t, 'UP'), 'UP', 42)
            self.assertEqual(np.bincount(y[idx], minlength=4)[1:].tolist(), list(t))
            self.assertTrue(np.array_equal(idx[:len(y)], np.arange(len(y))))

    def test_partial_down_keeps_a_subset_without_duplicates(self):
        y = labels()
        n = tg.count_vector(y)
        for scenario in ['P25', 'P50', 'P75', 'EQ']:
            t = tg.scenario_targets(n, 'DOWN', scenario)
            idx = tg.sample_indices(y, tg.sampler_strategy(n, t, 'DOWN'), 'DOWN', 42)
            self.assertEqual(len(np.unique(idx)), len(idx))
            self.assertEqual(np.bincount(y[idx], minlength=4)[1:].tolist(), list(t))

    def test_same_seed_is_reproducible_and_other_seed_differs(self):
        y = labels()
        n = tg.count_vector(y)
        s = tg.sampler_strategy(n, tg.scenario_targets(n, 'UP', 'P50'), 'UP')
        a = tg.sample_indices(y, s, 'UP', 42)
        self.assertTrue(np.array_equal(a, tg.sample_indices(y, s, 'UP', 42)))
        self.assertFalse(np.array_equal(a, tg.sample_indices(y, s, 'UP', 101)))


class WeightTests(unittest.TestCase):
    def test_w_obj_and_w_norm_values_for_fold0(self):
        t = tg.scenario_targets(FOLD0, 'UP', 'P25')
        np.testing.assert_allclose(tg.class_weights(FOLD0, t, 'obj'), [2.2252, 1.0, 1.1750], atol=5e-5)
        np.testing.assert_allclose(tg.class_weights(FOLD0, t, 'norm'), [1.8911, 0.8498, 0.9985], atol=5e-5)
        t = tg.scenario_targets(FOLD0, 'UP', 'EQ')
        np.testing.assert_allclose(tg.class_weights(FOLD0, t, 'obj'), [5.9008, 1.0, 1.6999], atol=5e-5)

    def test_w_norm_equal_targets_match_balanced(self):
        y = labels()
        n = tg.count_vector(y)
        w = tg.class_weights(n, tg.scenario_targets(n, 'UP', 'EQ'), 'norm')[y.astype(int) - 1]
        np.testing.assert_allclose(w, compute_sample_weight('balanced', y), rtol=1e-12, atol=0)
        self.assertAlmostEqual(float(w.sum()), len(y), places=8)

    def test_effective_mass_equals_target_for_w_obj(self):
        y = labels()
        n = tg.count_vector(y)
        t = tg.scenario_targets(n, 'UP', 'P75')
        mass = tg.effective_mass(y, tg.class_weights(n, t, 'obj'))
        np.testing.assert_allclose(mass, t, rtol=1e-12)

    def test_integer_weights_equal_row_replication_for_lr(self):
        rng = np.random.default_rng(3)
        X = rng.normal(size=(240, 4))
        y = rng.integers(1, 4, 240)
        w_class = np.array([3, 1, 2])
        w = w_class[y - 1].astype(float)
        kw = dict(solver='lbfgs', C=1.0, max_iter=5000, tol=1e-12)
        a = LogisticRegression(**kw).fit(X, y, sample_weight=w)
        idx = np.repeat(np.arange(len(y)), w.astype(int))
        b = LogisticRegression(**kw).fit(X[idx], y[idx])
        np.testing.assert_allclose(a.coef_, b.coef_, atol=2e-5)
        np.testing.assert_allclose(a.intercept_, b.intercept_, atol=2e-5)


def make_data(n=1500, seed=11):
    X, _, _, _ = synthetic(n)
    X.loc[[0, 17, 155], 'age'] = 995
    y = pd.Series(labels(int(n * .10), int(n * .55), n - int(n * .10) - int(n * .55), seed))
    ids = pd.Series(np.arange(1, n + 1), name='building_id')
    folds = np.zeros(n, dtype=int)
    for f, (_, v) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y)):
        folds[v] = f
    return X, y, ids, folds


def make_design(y, folds, n_estimators=3, **overrides):
    c = t32.config()
    c['lgbm_params'] = dict(c['lgbm_params'], n_estimators=n_estimators, verbosity=-1)
    c.update(overrides)
    counts = {str(f): list(tg.count_vector(y.to_numpy()[folds != f])) for f in range(5)}
    items = t32.items_from_config(c)
    return dict(config=c, fold_train_counts=counts, items=items, local_root='synthetic-root',
                fit_keys={t32.key(it['id'], f): it['kind'] for it in items for f in it['folds']})


def item_of(design, item_id):
    return next(i for i in design['items'] if i['id'] == item_id)


@unittest.skipUnless(HAVE, 'imbalanced-learn/lightgbm missing')
class ConfigTests(unittest.TestCase):
    def test_registered_plan(self):
        c = t32.config()
        items = t32.items_from_config(c)
        study = [i for i in items if i['kind'] == 'study']
        control = [i for i in items if i['kind'] == 'control']
        self.assertEqual(len(study), 22)
        self.assertEqual(sum(len(i['folds']) for i in study), 110)
        self.assertEqual(sum(len(i['folds']) for i in control), 8)
        for s, fits in {1: 30, 2: 25, 3: 55}.items():
            self.assertEqual(sum(len(i['folds']) for i in study if i['stage'] == s), fits)
        self.assertEqual({i['id'] for i in control if i['stage'] == 1}, {'c1', 'c4'})
        self.assertEqual([i['id'] for i in control if i['stage'] == 2], ['c2'])
        self.assertEqual([i['id'] for i in control if i['stage'] == 3], ['c3'])
        self.assertEqual(c['automatic_retries'], 0)
        self.assertFalse(c['scope']['rf_extension'])
        self.assertEqual(sorted({i['sampler_seed'] for i in study}), [42, 101, 202])

    def test_fit_all_needs_an_explicit_stage(self):
        with self.assertRaises(TypeError):
            t32.fit_all(Path('x'))
        with mock.patch.object(t32, 'fit_all') as fa:
            with self.assertRaises(ValueError):
                t32.main(['fit-all', '--out', 'x'])
            fa.assert_not_called()
            t32.main(['fit-all', '--out', 'x', '--stage', '2'])
            fa.assert_called_once_with('x', '2')

    def test_lr_c_error31_pin_and_plan_pin(self):
        c = t32.config()
        lrc = next(r for r in c['references'] if r['name'] == 'LR_C')
        self.assertEqual(lrc['error31'], 455)
        self.assertTrue(all(r['error31'] is not None for r in c['references']))

    def test_sealed_sources_exist(self):
        for p in t32.SOURCE:
            self.assertTrue((ROOT/p).is_file(), p)
        self.assertIn('tests/test_sampling_targets.py', t32.SOURCE)

    def test_holdout_guard(self):
        for bad in ['data/final_holdout/x.csv', 'C:\\a\\FINAL_HOLDOUT', Path('x')/'final_holdout']:
            with self.assertRaises(PermissionError):
                t32.guard_path(bad)
        with self.assertRaises(PermissionError):
            t32.gidx_command('data/final_holdout', Path(tempfile.gettempdir())/'nope_gidx.json')
        with self.assertRaises(PermissionError):
            t32.prepare('data/final_holdout', Path(tempfile.gettempdir())/'nope_prepare')

    def test_leak_proof_signatures(self):
        self.assertEqual(list(inspect.signature(t32.build_plan).parameters), ['item', 'y_train'])
        self.assertEqual(list(inspect.signature(tg.gidx_fold).parameters), ['y_train', 'seed'])
        self.assertEqual(list(inspect.signature(tg.sample_indices).parameters), ['y', 'strategy', 'direction', 'seed'])


@unittest.skipUnless(HAVE, 'imbalanced-learn/lightgbm missing')
class PlanTests(unittest.TestCase):
    def test_after_equals_target_for_every_registered_cell(self):
        y = labels()
        for it in t32.items_from_config(t32.config()):
            plan = t32.build_plan(it, y)
            r = plan['record']
            if r['kind'] == 'sample':
                self.assertEqual(r['after'], r['target'], it['id'])
                self.assertEqual(r['fit_rows'], sum(r['target']))
            elif r['kind'] == 'weight':
                self.assertEqual(r['after'], r['before'])
                np.testing.assert_allclose(r['effective_mass'], r['effective_mass_target'], rtol=1e-9)
                self.assertEqual(len(plan['row_weight']), len(y))
            else:
                self.assertEqual(r['after'], r['before'])
                self.assertIsNone(plan['idx'])

    def test_wobj_mass_is_the_target_and_wnorm_sums_to_n(self):
        y = labels()
        a = t32.build_plan(dict(arm='WOBJ', scenario='P50', sampler_seed=42), y)['record']
        np.testing.assert_allclose(a['effective_mass'], a['target'], rtol=1e-12)
        b = t32.build_plan(dict(arm='WNORM', scenario='P50', sampler_seed=42), y)
        self.assertAlmostEqual(float(b['row_weight'].sum()), len(y), places=7)

    def test_bad_arm_scenario_rejected(self):
        y = labels()
        for arm, scen in [('RDN', 'ONE1'), ('WOBJ', 'ONE3'), ('ORI', 'P50'), ('XX', 'EQ')]:
            with self.assertRaises((ValueError, KeyError)):
                t32.build_plan(dict(arm=arm, scenario=scen, sampler_seed=42), y)

    def test_noise_seeds_change_the_draw_not_the_counts(self):
        y = labels()
        a = t32.build_plan(dict(arm='RUP', scenario='P50', sampler_seed=42), y)['record']
        b = t32.build_plan(dict(arm='RUP', scenario='P50', sampler_seed=101), y)['record']
        self.assertEqual(a['after'], b['after'])
        self.assertNotEqual(a['indices_sha256'], b['indices_sha256'])


@unittest.skipUnless(HAVE, 'imbalanced-learn/lightgbm missing')
class GidxGateTests(unittest.TestCase):
    def test_gidx_command_refuses_on_failure_and_writes_on_pass(self):
        X, y, ids, folds = make_data()
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(t32, 'load', return_value=(X, y, ids, folds, {})):
                good = Path(tmp)/'good.json'
                result = t32.gidx_command('some-data', good)
                self.assertTrue(result['all_pass'] and good.is_file())
                self.assertEqual(result['classifiers_fitted'], 0)
                with mock.patch.object(t32.tg, 'gidx', return_value=dict(all_pass=False, folds={'0': dict(**{'pass': False})}, seed=42)):
                    with self.assertRaises(ValueError):
                        t32.gidx_command('some-data', Path(tmp)/'bad.json')


@unittest.skipUnless(HAVE, 'imbalanced-learn/lightgbm missing')
class ControlCompareTests(unittest.TestCase):
    def frame(self):
        rng = np.random.default_rng(1)
        p = rng.dirichlet([1, 1, 1], 40)
        df = pd.DataFrame({'building_id': np.arange(40), 'fold': 2, 'y_true': rng.integers(1, 4, 40),
                           'y_pred': p.argmax(1) + 1, **{n: p[:, i] for i, n in enumerate(t32.PROBS)}})
        return df

    def test_pass_fail_cases(self):
        new = self.frame()
        ref = new.set_index('building_id')
        self.assertEqual(t32.compare_control(new, ref, 2, 1e-9)['status'], 'PASS')
        shifted = new.copy(); shifted['p_grade_1'] += 1e-6
        self.assertEqual(t32.compare_control(shifted, ref, 2, 1e-9)['status'], 'FAIL')
        flipped = new.copy(); flipped.loc[0, 'y_pred'] = 1 + flipped.loc[0, 'y_pred'] % 3
        self.assertEqual(t32.compare_control(flipped, ref, 2, 1e-9)['status'], 'FAIL')
        self.assertEqual(t32.compare_control(new, ref, 3, 1e-9)['status'], 'FAIL')           # wrong fold label
        missing = ref.drop(index=[0])
        r = t32.compare_control(new, missing, 2, 1e-9)
        self.assertEqual(r['status'], 'FAIL')
        self.assertIsNone(r['probability_max_abs_diff'])


def fake_execute(fail=None, interrupt=None):
    def run(X, y, ids, folds, item, f, design, d):
        if interrupt and interrupt(item, f):
            raise KeyboardInterrupt()
        if fail and fail(item, f):
            raise RuntimeError('boom')
        d.mkdir(parents=True, exist_ok=True)
        (d/f'fold_{f}_pred.csv.gz').write_bytes(b'x' * (f + 1))
        record = dict(metrics=dict(macro_f1=0.5), info=dict(fit_seconds=0.1), sampling=dict(after=[1, 2, 3], target=[1, 2, 3]),
                      reload_probability_max_abs_diff=0.0)
        control = None
        if item['kind'] == 'control':
            control = dict(status='PASS', probability_max_abs_diff=0.0, predicted_labels_identical=True)
        return record, [d/f'fold_{f}_pred.csv.gz'], control
    return run


@unittest.skipUnless(HAVE, 'imbalanced-learn/lightgbm missing')
class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)/'run'
        self.out.mkdir()
        (self.out/'backup').mkdir()
        self.design = make_design(pd.Series(labels(150, 825, 525)), np.arange(1500) % 5)
        t32.atomic_write(self.out/'ledger.json', dict(study_id='x', folds={}, recoveries=[]))
        self.item = item_of(self.design, 'LR.RUP.P25')

    def tearDown(self):
        self.tmp.cleanup()

    def call(self, f, fn=None, item=None):
        return t32.fit_unit(None, None, None, None, self.out, item or self.item, f, self.design, fn or fake_execute())

    def test_completed_fold_is_not_refit(self):
        self.assertTrue(self.call(0)[1])
        called = []
        def spy(*a):
            called.append(1); return fake_execute()(*a)
        entry, fitted = self.call(0, spy)
        self.assertFalse(fitted)
        self.assertEqual(called, [])
        self.assertEqual(entry['attempts'], 1)
        ledger = t32.read_ledger(self.out)
        self.assertEqual(ledger['folds']['LR.RUP.P25/0']['status'], 'complete')
        self.assertTrue((self.out/'backup'/'cells'/'LR.RUP.P25'/'fold_0_pred.csv.gz').is_file())

    def test_tampered_artifact_is_detected_on_resume(self):
        self.call(1)
        (self.out/'cells'/'LR.RUP.P25'/'fold_1_pred.csv.gz').write_bytes(b'changed')
        with self.assertRaises(ValueError):
            self.call(1)

    def test_unregistered_fold_refused(self):
        with self.assertRaises(ValueError):
            self.call(7)

    def test_failure_blocks_without_automatic_retry(self):
        with self.assertRaises(RuntimeError):
            self.call(0, fake_execute(fail=lambda i, f: True))
        e = t32.read_ledger(self.out)['folds']['LR.RUP.P25/0']
        self.assertEqual(e['status'], 'failed')
        with self.assertRaises(ValueError):
            self.call(0)                                    # no automatic retry
        with self.assertRaises(ValueError):
            t32.preflight_ledger(t32.read_ledger(self.out))

    def test_budget_cap_is_enforced(self):
        self.design['config']['max_new_fits'] = 2
        self.call(0); self.call(1)
        with self.assertRaises(ValueError):
            self.call(2)

    def test_manual_recovery_once(self):
        with self.assertRaises(KeyboardInterrupt):
            self.call(0, fake_execute(interrupt=lambda i, f: True))
        self.assertEqual(t32.read_ledger(self.out)['folds']['LR.RUP.P25/0']['status'], 'running')
        entry, fitted = self.call(0)                       # first manual recovery is allowed
        self.assertTrue(fitted)
        self.assertEqual(entry['attempts'], 2)
        self.assertEqual(t32.read_ledger(self.out)['recoveries'], ['LR.RUP.P25/0'])
        with self.assertRaises(KeyboardInterrupt):
            self.call(1, fake_execute(interrupt=lambda i, f: True))
        with self.assertRaises(ValueError):                # second recovery is refused
            self.call(1)

    def test_recovery_extends_only_its_own_budget(self):
        self.design['config']['max_new_fits'] = 1
        with self.assertRaises(KeyboardInterrupt):
            self.call(0, fake_execute(interrupt=lambda i, f: True))
        self.call(0)                                        # attempts=2 allowed with cap 1 + 1 recovery
        with self.assertRaises(ValueError):
            self.call(1)

    def test_control_budget_and_status_recorded(self):
        ctl = item_of(self.design, 'c1')
        entry, _ = self.call(0, item=ctl)
        self.assertEqual(entry['control_status'], 'PASS')
        self.assertEqual(entry['kind'], 'control')
        self.design['config']['max_control_fits'] = 1
        with self.assertRaises(ValueError):
            self.call(2, item=ctl)

    def test_control_failure_stops_everything(self):
        ctl = item_of(self.design, 'c1')
        def failing(*a):
            r, files, c = fake_execute()(*a)
            return r, files, dict(c, status='FAIL', probability_max_abs_diff=1.0)
        with self.assertRaises(t32.ControlFailure):
            self.call(0, failing, ctl)
        with self.assertRaises(ValueError):
            t32.preflight_ledger(t32.read_ledger(self.out))
        with self.assertRaises(t32.ControlFailure):          # recorded FAIL re-raises on resume
            self.call(0, item=ctl)


@unittest.skipUnless(HAVE, 'imbalanced-learn/lightgbm missing')
class StageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)/'run'
        self.out.mkdir()
        (self.out/'backup').mkdir()
        self.design = make_design(pd.Series(labels(150, 825, 525)), np.arange(1500) % 5)
        t32.atomic_write(self.out/'ledger.json', dict(study_id='x', folds={}, recoveries=[]))
        X = pd.DataFrame({'a': [1, 2, 3]})
        self.verify = lambda out: (X, None, None, None, self.design)
        self.calls = []

    def tearDown(self):
        self.tmp.cleanup()

    def unit(self, stop_at=None):
        def run(X, y, ids, folds, out, item, f, design):
            if stop_at and (item['id'], f) == stop_at:
                raise RuntimeError('stop')
            self.calls.append((item['stage'], item['kind'], item['id'], f))
            ledger = t32.read_ledger(out)
            e = dict(kind=item['kind'], id=item['id'], dir=t32.item_dir(item), fold=f, stage=item['stage'], status='complete',
                     attempts=1, artifact_sha256={}, fit_seconds=0.0)
            if item['kind'] == 'control':
                e['control_status'] = 'PASS'
            ledger['folds'][t32.key(item['id'], f)] = e
            t32.atomic_write(out/'ledger.json', ledger)
            return e, True
        return run

    def test_stages_run_in_order_with_controls_first(self):
        t32.fit_all(self.out, 'all', unit_fn=self.unit(), verify_fn=self.verify)
        stages = [c[0] for c in self.calls]
        self.assertEqual(stages, sorted(stages))
        self.assertEqual(len(self.calls), 110 + 8)
        for s in (1, 2, 3):
            chunk = [c for c in self.calls if c[0] == s]
            kinds = [c[1] for c in chunk]
            self.assertEqual(kinds, sorted(kinds, key=lambda k: k != 'control'))
        self.assertEqual({c[2] for c in self.calls if c[0] == 1 and c[1] == 'control'}, {'c1', 'c4'})
        self.assertEqual(sum(1 for c in self.calls if c[1] == 'study' and c[0] == 1), 30)
        self.assertEqual(sum(1 for c in self.calls if c[1] == 'study' and c[0] == 2), 25)
        self.assertEqual(sum(1 for c in self.calls if c[1] == 'study' and c[0] == 3), 55)
        self.assertFalse((self.out/'RUNNING.lock').exists())

    def test_later_stage_needs_earlier_stage(self):
        with self.assertRaises(ValueError):
            t32.fit_all(self.out, '2', unit_fn=self.unit(), verify_fn=self.verify)
        self.assertEqual(self.calls, [])
        t32.fit_all(self.out, '1', unit_fn=self.unit(), verify_fn=self.verify)
        with self.assertRaises(ValueError):
            t32.fit_all(self.out, '3', unit_fn=self.unit(), verify_fn=self.verify)
        t32.fit_all(self.out, '2', unit_fn=self.unit(), verify_fn=self.verify)
        t32.fit_all(self.out, '3', unit_fn=self.unit(), verify_fn=self.verify)

    def test_stop_in_stage_one_blocks_the_rest(self):
        with self.assertRaises(RuntimeError):
            t32.fit_all(self.out, 'all', unit_fn=self.unit(stop_at=('LR.RUP.P50', 2)), verify_fn=self.verify)
        self.assertTrue(all(c[0] == 1 for c in self.calls))
        self.assertFalse((self.out/'RUNNING.lock').exists())
        with self.assertRaises(ValueError):
            t32.fit_all(self.out, '3', unit_fn=self.unit(), verify_fn=self.verify)

    def test_missing_control_status_is_incomplete(self):
        t32.fit_all(self.out, '1', unit_fn=self.unit(), verify_fn=self.verify)
        ledger = t32.read_ledger(self.out)
        self.assertTrue(t32.stage_complete(ledger, self.design, 1))
        del ledger['folds']['c1/0']['control_status']
        self.assertFalse(t32.stage_complete(ledger, self.design, 1))
        ledger['folds']['c1/0']['control_status'] = 'PASS'
        self.assertTrue(t32.stage_complete(ledger, self.design, 1))

    def test_lock_is_exclusive(self):
        (self.out/'RUNNING.lock').write_text('pid=1')
        with self.assertRaises(ValueError):
            t32.fit_all(self.out, '1', unit_fn=self.unit(), verify_fn=self.verify)
        self.assertTrue((self.out/'RUNNING.lock').exists())


@unittest.skipUnless(HAVE, 'imbalanced-learn/lightgbm missing')
class RealFitTests(unittest.TestCase):
    """Real LR/LGBM code paths on a tiny synthetic table (not study fits)."""
    @classmethod
    def setUpClass(cls):
        cls.X, cls.y, cls.ids, cls.folds = make_data()
        cls.design = make_design(cls.y, cls.folds)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)/'run'
        (self.out/'backup').mkdir(parents=True)
        t32.atomic_write(self.out/'ledger.json', dict(study_id='x', folds={}, recoveries=[]))
        t32._REF_CACHE.clear()

    def tearDown(self):
        self.tmp.cleanup()
        t32._REF_CACHE.clear()

    def run_unit(self, item_id, f, design=None):
        d = design or self.design
        return t32.fit_unit(self.X, self.y, self.ids, self.folds, self.out, item_of(d, item_id), f, d)

    def record(self, item_id, f):
        item = item_of(self.design, item_id)
        return json.loads((self.out/t32.item_dir(item)/f'fold_{f}.json').read_text(encoding='utf-8'))

    def test_lr_partial_up_checkpoint(self):
        entry, fitted = self.run_unit('LR.RUP.P50', 1)
        self.assertTrue(fitted and entry['status'] == 'complete')
        r = self.record('LR.RUP.P50', 1)
        s = r['sampling']
        self.assertEqual(s['after'], s['target'])
        self.assertEqual(s['before'], self.design['fold_train_counts']['1'])
        self.assertGreater(s['fit_rows'], r['train_rows'])
        self.assertEqual(r['valid_class_counts'], [int((self.y[self.folds == 1] == v).sum()) for v in (1, 2, 3)])
        self.assertEqual(r['valid_rows'], int((self.folds == 1).sum()))
        d = self.out/'cells'/'LR.RUP.P50'
        for name in ['fold_1_model.joblib', 'fold_1_preprocessor.joblib', 'fold_1_pred.csv.gz', 'fold_1_sample_indices.npy', 'fold_1.json']:
            self.assertTrue((d/name).is_file(), name)
            self.assertTrue((self.out/'backup'/'cells'/'LR.RUP.P50'/name).is_file(), name)
        pred = t32.read_pred(d/'fold_1_pred.csv.gz')
        self.assertEqual(len(pred), int((self.folds == 1).sum()))
        np.testing.assert_array_equal(pred.y_pred.to_numpy(), pred[t32.PROBS].to_numpy().argmax(1) + 1)
        self.assertFalse(self.run_unit('LR.RUP.P50', 1)[1])                  # resume: verified, not refit

    def test_validation_fold_is_untouched_and_encoder_sees_training_only(self):
        self.run_unit('LR.RDN.P25', 0)
        r = self.record('LR.RDN.P25', 0)
        self.assertEqual(r['sampling']['after'], r['sampling']['target'])
        self.assertLess(r['sampling']['fit_rows'], r['train_rows'])
        self.assertEqual(r['valid_rows'], int((self.folds == 0).sum()))
        import joblib
        pre = joblib.load(self.out/'cells'/'LR.RDN.P25'/'fold_0_preprocessor.joblib')
        train = self.X[self.folds != 0]
        from preprocess import CATEGORICAL
        for col, learned in zip(CATEGORICAL, pre['columns'].named_transformers_['categorical'].categories_):
            self.assertEqual(set(learned), set(train[col].astype(str)))

    def test_lr_weight_arm_keeps_rows(self):
        self.run_unit('LR.WOBJ.P75', 2)
        r = self.record('LR.WOBJ.P75', 2)
        self.assertEqual(r['sampling']['kind'], 'weight')
        self.assertEqual(r['sampling']['fit_rows'], r['train_rows'])
        np.testing.assert_allclose(r['sampling']['effective_mass'], r['sampling']['target'], rtol=1e-9)
        self.assertFalse((self.out/'cells'/'LR.WOBJ.P75'/'fold_2_sample_indices.npy').exists())

    def test_lgbm_weight_and_sampling_arms(self):
        self.run_unit('LGBMN.WNORM.P25', 0)
        self.run_unit('LGBMN.RUP.P50', 0)
        w, s = self.record('LGBMN.WNORM.P25', 0), self.record('LGBMN.RUP.P50', 0)
        self.assertEqual(w['sampling']['fit_rows'], w['train_rows'])
        self.assertAlmostEqual(sum(w['sampling']['effective_mass']), w['train_rows'], places=6)
        self.assertEqual(s['sampling']['after'], s['sampling']['target'])
        self.assertEqual(s['info']['rounds_completed'], 3)
        self.assertTrue((self.out/'cells'/'LGBMN.RUP.P50'/'fold_0_model.txt').is_file())

    def test_nonconvergence_stops_and_is_not_retried(self):
        design = make_design(self.y, self.folds)
        design['config']['lr'] = dict(design['config']['lr'], max_iter=2)
        with self.assertRaises(t32.NonConvergence):
            self.run_unit('LR.RUP.P25', 0, design)
        e = t32.read_ledger(self.out)['folds']['LR.RUP.P25/0']
        self.assertEqual((e['status'], e['nonconvergence']), ('failed', True))
        with self.assertRaises(ValueError):
            self.run_unit('LR.RUP.P25', 0, design)
        with self.assertRaises(ValueError):
            t32.preflight_ledger(t32.read_ledger(self.out))

    def make_reference(self, folds=(0, 2)):
        """Reference = an earlier identical-path run (no-sampling LR), as the reused OOF would be."""
        parts = []
        with tempfile.TemporaryDirectory() as tmp:
            for f in folds:
                item = dict(kind='study', id='ref', model='LR', arm='ORI', scenario='ORI', sampler_seed=42, stage=1, folds=[f])
                t32.execute(self.X, self.y, self.ids, self.folds, item, f, self.design, Path(tmp)/'ref')
                parts.append(t32.read_pred(Path(tmp)/'ref'/f'fold_{f}_pred.csv.gz'))
        return pd.concat(parts).set_index('building_id')

    def test_control_passes_against_an_identical_run_and_fails_on_a_changed_one(self):
        ref = self.make_reference()
        t32._REF_CACHE[('synthetic-root', 'LR.ORI')] = ref
        entry, _ = self.run_unit('c1', 0)
        self.assertEqual(entry['control_status'], 'PASS')
        self.assertEqual(entry['control_max_abs_diff'], 0.0)
        bad = ref.copy(); bad['p_grade_2'] += 1e-6
        t32._REF_CACHE[('synthetic-root', 'LR.ORI')] = bad
        with self.assertRaises(t32.ControlFailure):
            self.run_unit('c1', 2)
        self.assertEqual(t32.read_ledger(self.out)['folds']['c1/2']['control_status'], 'FAIL')

    def test_c2_c3_plans_equal_the_stock_auto_sampler_and_equal_counts(self):
        y = self.y[self.folds != 0].to_numpy()
        up = t32.build_plan(item_of(self.design, 'c2'), y)
        n2 = up['record']['before'][1]
        self.assertEqual(up['record']['target'], [n2, n2, n2])
        self.assertTrue(np.array_equal(up['idx'], tg.auto_sample_indices(y, 'UP', 42)))      # = stock ROS used by the reused cell
        down = t32.build_plan(item_of(self.design, 'c3'), y)
        n1 = down['record']['before'][0]
        self.assertEqual(down['record']['target'], [n1, n1, n1])
        self.assertTrue(np.array_equal(down['idx'], tg.auto_sample_indices(y, 'DOWN', 42)))  # = stock RUS used by the reused cell

    @staticmethod
    def expected_scaler(X_train):
        from preprocess import NUMERIC
        v = X_train[NUMERIC].to_numpy(dtype=np.float32).astype(np.float64)
        return v.mean(0), v.std(0)

    def scaler_of(self, item_id, f):
        import joblib
        pre = joblib.load(self.out/'cells'/item_id/f'fold_{f}_preprocessor.joblib')
        return pre['columns'].named_transformers_['numeric']['scale']

    def test_scaler_and_vocabulary_come_from_the_original_training_fold(self):
        for item_id, f in [('LR.RUP.P75', 0), ('LR.RDN.P25', 1), ('LR.WOBJ.P50', 2)]:
            self.run_unit(item_id, f)
            train = self.X[self.folds != f]
            mean, scale = self.expected_scaler(train)
            sc = self.scaler_of(item_id, f)
            np.testing.assert_allclose(sc.mean_, mean, rtol=1e-5, atol=1e-6)
            np.testing.assert_allclose(sc.scale_, scale, rtol=1e-5, atol=1e-6)
            self.assertEqual(int(sc.n_samples_seen_), int((self.folds != f).sum()))   # original rows, not duplicated/dropped ones
            import joblib
            from preprocess import CATEGORICAL
            pre = joblib.load(self.out/'cells'/item_id/f'fold_{f}_preprocessor.joblib')
            for col, learned in zip(CATEGORICAL, pre['columns'].named_transformers_['categorical'].categories_):
                self.assertEqual(set(learned), set(train[col].astype(str)))

    def test_fitting_after_sampling_would_fail_the_scaler_check(self):
        f = 0
        a = np.flatnonzero(self.folds != f)
        y_tr = self.y.iloc[a].to_numpy()
        plan = t32.build_plan(item_of(self.design, 'LR.RUP.P75'), y_tr)
        wrong = t32.make_lr_preprocessor(True).fit(self.X.iloc[a].iloc[plan['idx']], y_tr[plan['idx']])   # sampler first (forbidden order)
        sc = wrong['columns'].named_transformers_['numeric']['scale']
        mean, scale = self.expected_scaler(self.X.iloc[a])
        self.assertGreater(int(sc.n_samples_seen_), len(a))
        self.assertFalse(np.allclose(sc.mean_, mean, rtol=1e-5, atol=1e-6) and np.allclose(sc.scale_, scale, rtol=1e-5, atol=1e-6))

    def test_lgbm_vocabulary_is_training_fold_only(self):
        self.run_unit('LGBMN.RUP.P50', 3)
        import joblib
        from preprocess import CATEGORICAL
        train = self.X[self.folds != 3]
        pre = joblib.load(self.out/'cells'/'LGBMN.RUP.P50'/'fold_3_preprocessor.joblib')
        for col, learned in zip(CATEGORICAL, pre['columns'].named_transformers_['categorical'].categories_):
            self.assertEqual(set(learned), set(train[col].astype(str)))
        sc = pre['columns'].named_transformers_['numeric']['scale']
        self.assertEqual(int(sc.n_samples_seen_), int((self.folds != 3).sum()))


@unittest.skipUnless(HAVE, 'imbalanced-learn/lightgbm missing')
class EndToEndSyntheticTests(unittest.TestCase):
    """fit-all (all three stages, controls first) -> finalize on a tiny synthetic table.

    Sealed-design creation (`prepare`) needs the real data contract and is not exercised here;
    the design dict is built by hand. These are code-path checks, not study fits.
    """
    @classmethod
    def setUpClass(cls):
        cls.X, cls.y, cls.ids, cls.folds = make_data()
        cls.design = make_design(cls.y, cls.folds, grade3_support=int((cls.y == 3).sum()))

    def reference(self, model, arm, scenario):
        parts = []
        with tempfile.TemporaryDirectory() as tmp:
            for f in range(5):
                item = dict(kind='study', id='ref', model=model, arm=arm, scenario=scenario, sampler_seed=42, stage=1, folds=[f])
                t32.execute(self.X, self.y, self.ids, self.folds, item, f, self.design, Path(tmp)/'ref')
                parts.append(t32.read_pred(Path(tmp)/'ref'/f'fold_{f}_pred.csv.gz'))
        return pd.concat(parts).set_index('building_id')

    def test_all_stages_then_finalize(self):
        d = dict(self.design)
        counts = d['fold_train_counts']
        d['target_table'] = {str(f): {dr: {sc: tg.describe_targets(counts[str(f)], tg.scenario_targets(counts[str(f)], dr, sc))
                                           for sc in (tg.ARM_SCENARIOS['RUP'] if dr == 'UP' else tg.ARM_SCENARIOS['RDN'])}
                                      for dr in ('UP', 'DOWN')} for f in range(5)}
        d.update(code_commit='0' * 40, plan_sha256='p', gidx=dict(all_pass=True), references={})
        refs = {'LR.ORI': self.reference('LR', 'ORI', 'ORI'), 'LR.RUP.EQ': self.reference('LR', 'RUP', 'EQ'),
                'LR.RDN.EQ': self.reference('LR', 'RDN', 'EQ'), 'LGBMN.ORI': self.reference('LGBMN', 'ORI', 'ORI')}
        refs.update({'LR.BAL': self.reference('LR', 'WOBJ', 'EQ'), 'LR.SMOTE.EQ': refs['LR.RUP.EQ'], 'LR_C': refs['LR.ORI']})
        t32._REF_CACHE.clear()
        for name, frame in refs.items():
            t32._REF_CACHE[(d['local_root'], name)] = frame
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'run'
            for sub in ('cells', 'controls', 'backup'):
                (out/sub).mkdir(parents=True)
            t32.jwrite(out/'design.json', d)
            t32.atomic_write(out/'ledger.json', dict(study_id='x', folds={}, recoveries=[]))
            verify = lambda _o: (self.X, self.y, self.ids, self.folds, d)
            t32.fit_all(out, '1', verify_fn=verify)
            with self.assertRaises(ValueError):                 # finalize is refused while stages are open
                with mock.patch.object(t32, 'verify', verify):
                    t32.finalize_all(out)
            t32.fit_all(out, 'all', verify_fn=verify)           # stage 1 resumes (verified, not refit)
            ledger = t32.read_ledger(out)
            self.assertEqual(sum(1 for e in ledger['folds'].values() if e['kind'] == 'study' and e['status'] == 'complete'), 110)
            self.assertEqual(sum(1 for e in ledger['folds'].values() if e['kind'] == 'control' and e['control_status'] == 'PASS'), 8)
            self.assertTrue(all(e['attempts'] == 1 for e in ledger['folds'].values()))
            with mock.patch.object(t32, 'verify', verify):
                res = t32.finalize_all(out)
                with self.assertRaises(ValueError):             # no overwrite
                    t32.finalize_all(out)
            self.assertTrue((out/'analysis.json').is_file() and (out/'aggregate.json').is_file())
            self.assertTrue((out/'backup'/'analysis.json').is_file())
            for name, table in res['tables'].items():
                if 'delta_macro_f1' in table['columns']:
                    self.assertIn('delta_error31', table['columns'], name)
                    self.assertIn('delta_rate31', table['columns'], name)
            self.assertEqual(set(res['verdicts']) >= {u['id'] for u in d['config']['units']}, True)
            self.assertEqual(len(res['realised_counts_table']), 110 + 8)
            self.assertTrue(all(r['after_equals_target'] for r in res['realised_counts_table']))
            self.assertEqual(res['ledger']['study_fits_complete'], 110)
            self.assertEqual(res['ledger']['automatic_retries'], 0)
            self.assertEqual(set(res['noise']), {'RUP', 'RDN'})
            self.assertFalse(res['holdout_accessed'])
            agg = json.loads((out/'aggregate.json').read_text(encoding='utf-8'))
            self.assertNotIn('references_used', agg)
            with mock.patch('builtins.print'):
                t32.status(out)
        t32._REF_CACHE.clear()


@unittest.skipUnless(HAVE, 'imbalanced-learn/lightgbm missing')
class ReferenceAndGitTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        oof = synthetic_oof(21, 1.0, n=600)
        self.oof = oof
        self.ids = oof.building_id
        self.y = oof.y_true
        self.folds = oof.fold.to_numpy()
        path = self.root/'runs'/'a'/'oof.csv.gz'
        path.parent.mkdir(parents=True)
        oof.to_csv(path, index=False, compression={'method': 'gzip', 'mtime': 0}, float_format='%.17g')
        sc = t32.summarize_oof(oof, int((oof.y_true == 3).sum()))
        self.ref = dict(name='A', kind='oof', role='reference_uncontrolled', path='runs/a/oof.csv.gz',
                        sha256=t32.sha256_file(path), cv_mean=round(sc['cv_mean'], 6), error31=sc['error31'])
        self.c = dict(references=[self.ref], expected_rows=600, reference_cv_mean_tolerance=5e-7)

    def tearDown(self):
        self.tmp.cleanup()

    def verify(self, **change):
        c = dict(self.c, references=[dict(self.ref, **change)])
        return t32.verify_references(c, self.root, self.ids, self.y, self.folds)

    def test_good_reference(self):
        out = self.verify()
        self.assertAlmostEqual(out['A']['cv_mean'], self.ref['cv_mean'], places=5)
        self.assertEqual(out['A']['error31'], self.ref['error31'])

    def test_wrong_pins_are_refused(self):
        for change in [dict(sha256='0' * 64), dict(cv_mean=self.ref['cv_mean'] + 1e-4), dict(error31=self.ref['error31'] + 1)]:
            with self.assertRaises(ValueError):
                self.verify(**change)

    def test_misaligned_fold_or_label_is_refused(self):
        with self.assertRaises(AssertionError):
            t32.verify_references(self.c, self.root, self.ids, self.y, (self.folds + 1) % 5)

    def test_holdout_reference_path_is_refused(self):
        with self.assertRaises(PermissionError):
            self.verify(path='final_holdout/x.csv.gz')

    def test_git_blob_check(self):
        try:
            import subprocess
            subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, check=True)
        except Exception:                                   # pragma: no cover
            self.skipTest('no git checkout')
        self.assertEqual(len(t32.git_blob_check(['preprocess.py'])), 40)
        with self.assertRaises(ValueError):
            t32.git_blob_check(['preprocess.py', 'no_such_file_for_t32.py'])
        with self.assertRaises(ValueError):
            t32.git_blob_check(['preprocess.py'], code_commit='f' * 40)


def synthetic_oof(seed, quality, n=1000, shift=0.0):
    rng = np.random.default_rng(seed)
    y = rng.choice([1, 2, 3], n, p=[.1, .55, .35])
    logits = rng.normal(size=(n, 3)); logits[np.arange(n), y - 1] += quality; logits[:, 0] += shift
    p = np.exp(logits); p /= p.sum(1, keepdims=True)
    return pd.DataFrame({'building_id': np.arange(n), 'fold': np.arange(n) % 5, 'y_true': y, 'y_pred': p.argmax(1) + 1,
                         **{c: p[:, i] for i, c in enumerate(t32.PROBS)}})


class AnalysisTests(unittest.TestCase):
    TH = dict(T_F1=0.003, T_win=4, T_31=0.003, T_R3=0.02, C_keep=-0.003, C_small=-0.01, eps=1e-12)

    @unittest.skipUnless(HAVE, 'imbalanced-learn/lightgbm missing')
    def test_classification_rules(self):
        c = lambda f1, wins, r31, r3: t32.classify(f1, wins, r31, r3, self.TH)
        self.assertEqual(c(0.004, 5, 0.001, 0.0), 'A')
        self.assertEqual(c(0.003, 4, 0.003, 0.0), 'A')                 # inclusive boundaries
        self.assertEqual(c(0.004, 4, 0.0031, 0.0), 'B')
        self.assertEqual(c(0.004, 3, 0.0, 0.03), 'C1')                 # wins too few for A/B, recall gain, F1 kept
        self.assertEqual(c(-0.003, 1, 0.01, 0.02), 'C1')
        self.assertEqual(c(-0.004, 1, 0.01, 0.02), 'C2')
        self.assertEqual(c(-0.01, 0, 0.01, 0.05), 'C2')
        self.assertEqual(c(-0.0101, 0, 0.01, 0.05), 'C3')
        self.assertEqual(c(-0.02, 0, 0.03, 0.0199), 'D')
        self.assertEqual(c(0.001, 5, 0.0, 0.0), 'D')

    @unittest.skipUnless(HAVE, 'imbalanced-learn/lightgbm missing')
    def test_every_delta_f1_table_carries_delta_31(self):
        with self.assertRaises(ValueError):
            t32.make_table('bad', [dict(delta_macro_f1=0.1)], ['cell', 'delta_macro_f1'])
        with self.assertRaises(ValueError):
            t32.make_table('bad', [dict(delta_macro_f1=0.1)], ['delta_macro_f1', 'delta_error31', 'delta_rate31'])
        ok = t32.make_table('ok', [dict(delta_macro_f1=0.1, delta_error31=3, delta_rate31=0.01)],
                            ['delta_macro_f1', 'delta_error31', 'delta_rate31'])
        self.assertEqual(ok['rows'][0]['delta_error31'], 3)
        t32.make_table('no_delta', [dict(a=1)], ['a'])

    @unittest.skipUnless(HAVE, 'imbalanced-learn/lightgbm missing')
    def test_sign_changes(self):
        self.assertEqual(t32.sign_changes([0, 1, 2, 3]), 0)
        self.assertEqual(t32.sign_changes([0, 1, 0, 1]), 2)
        self.assertEqual(t32.sign_changes([0, 1, 1, 0]), 1)

    @unittest.skipUnless(HAVE, 'imbalanced-learn/lightgbm missing')
    def test_build_analysis_on_synthetic_oof(self):
        c = t32.config()
        oofs = {'LR.ORI': synthetic_oof(1, 1.0), 'LGBMN.ORI': synthetic_oof(2, 1.5),
                'LR.RUP.P25': synthetic_oof(3, 1.0, shift=0.4), 'LR.RUP.P50': synthetic_oof(4, 1.0, shift=0.8),
                'LR.RUP.P50.s101': synthetic_oof(5, 1.0, shift=0.8), 'LR.RUP.P50.s202': synthetic_oof(6, 1.0, shift=0.8),
                'LR.RUP.EQ': synthetic_oof(7, 1.0, shift=1.5), 'LR.RUP.ONE1': synthetic_oof(8, 1.0, shift=1.0),
                'LR.RUP.ONE3': synthetic_oof(9, 1.0, shift=0.2), 'LR.WOBJ.P25': synthetic_oof(10, 1.0, shift=0.4),
                'LR.WOBJ.P50': synthetic_oof(11, 1.0, shift=0.8), 'LR.WOBJ.EQ': synthetic_oof(12, 1.0, shift=1.5),
                'LR.BAL': synthetic_oof(13, 1.0, shift=1.5), 'LR_C': synthetic_oof(14, 1.2),
                'LGBMN.WNORM.P50': synthetic_oof(15, 1.5, shift=0.5), 'LGBMN.RUP.P50': synthetic_oof(16, 1.5, shift=0.5)}
        design = dict(target_table={str(f): {d: {s: tg.describe_targets(FOLD0, tg.scenario_targets(FOLD0, d, s))
                                                 for s in (tg.ARM_SCENARIOS['RUP'] if d == 'UP' else tg.ARM_SCENARIOS['RDN'])}
                                             for d in ('UP', 'DOWN')} for f in range(5)})
        c = dict(c, grade3_support=int((oofs['LR.ORI'].y_true == 3).sum()))
        res = t32.build_analysis(c, design, None, None, None, oofs, {}, {}, dict(note='synthetic'))
        for name, table in res['tables'].items():
            if 'delta_macro_f1' in table['columns']:
                self.assertIn('delta_error31', table['columns'], name)
                self.assertIn('delta_rate31', table['columns'], name)
        rows = {r['cell']: r for r in res['tables']['cells_vs_ori']['rows']}
        self.assertIn('LR.RUP.P50', rows)
        self.assertNotIn('LR.ORI', rows)
        self.assertIn(rows['LR.RUP.P50']['class'], ['A', 'B', 'C1', 'C2', 'C3', 'D'])
        base = t32.summarize_oof(oofs['LR.ORI'], c['grade3_support'])
        cell = t32.summarize_oof(oofs['LR.RUP.P50'], c['grade3_support'])
        self.assertAlmostEqual(rows['LR.RUP.P50']['delta_macro_f1'], cell['cv_mean'] - base['cv_mean'], places=12)
        self.assertEqual(rows['LR.RUP.P50']['delta_error31'], cell['error31'] - base['error31'])
        self.assertAlmostEqual(rows['LR.RUP.P50']['delta_rate31'], (cell['error31'] - base['error31']) / c['grade3_support'])
        self.assertIn('RUP', res['noise'])
        self.assertEqual(len(res['noise']['RUP']['cells']), 3)
        self.assertIn('LR.RUP.P25 minus LR.WOBJ.P25', {r['comparison'] for r in res['tables']['sampling_vs_weight']['rows']})
        self.assertTrue(any('BAL' in r['comparison'] for r in res['tables']['sampling_vs_weight']['rows']))
        self.assertEqual(len(res['tables']['a3_offset_emulation']['rows']), 4)
        self.assertEqual(len(res['target_counts_table']), 5 * (6 + 4))
        json.dumps(t32.py(res), allow_nan=False)                       # fully JSON-safe


if __name__ == '__main__':
    unittest.main()
