"""T31 synthetic contracts only: no real development data, no holdout access, no real study fit.

Fits here are tiny synthetic checks of the experiment code, never study training runs.
"""
import gzip
import json
import os
import subprocess
import sys
import tempfile
import time
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
    import psutil  # noqa: F401
    from experiments.geo_code_encoding import arms, measure, run, verdict
    HAVE = True
except ImportError:  # pragma: no cover
    HAVE = False

IDLE_OK = lambda: dict(noisy=False, samples=[0.0], waited_seconds=0.0)  # noqa: E731


def make_data(n=300, seed=3):
    X, _, _, _ = synthetic(n)
    rng = np.random.default_rng(seed)
    X = X.copy()
    X['geo_level_1_id'] = rng.integers(0, 30, n).astype(str)
    X['geo_level_2_id'] = rng.integers(0, 300, n).astype(str)
    X['geo_level_3_id'] = rng.integers(0, 3000, n).astype(str)
    X.loc[[0, 17, 155], 'age'] = 995
    folds = np.arange(n) % 5
    y = pd.Series(1 + (X.geo_level_1_id.astype(int).to_numpy() % 3) + (rng.random(n) < 0.3), dtype='int64').clip(1, 3)
    ids = pd.Series(np.arange(1, n + 1), name='building_id')
    return X.reset_index(drop=True), y, ids, folds


class FakeProc:
    """In-process stand-in for the child: runs the job inside wait() so the sampler observes it."""
    def __init__(self, out, args, env, loader):
        self.out, self.args, self.env, self.loader = out, args, env, loader
        self.pid = os.getpid()

    def wait(self):
        a = self.args
        with mock.patch.dict(os.environ, self.env):
            try:
                run.child_main(self.out, a['arm'], a['fold'], a['role'], a['tag'], self.loader)
            except BaseException:
                return 1
        return 0


def make_study(tmp, data, references=None):
    X, y, ids, folds = data
    out = Path(tmp)/'run'
    out.mkdir()
    (out/'backup').mkdir()
    c = run.config()
    contract = {'synthetic': True}
    design = dict(study_id=run.STUDY_ID, config=c, code_commit='0' * 40, source_sha256=run.sources(),
                  environment=run.environment(), data_dir='synthetic', data_contract=contract,
                  manifest_sha256=c['manifest_sha256'], references=references or {},
                  study_targets=run.study_targets(c), execution_plan=run.execution_plan(c),
                  max_attempts=c['fit_plan']['max_attempts'], holdout_accessed=False, classifiers_fitted=0)
    run.write_json(out/'design.json', design)
    run.write_json(out/'design_seal.json', dict(design_sha256=run.sha256_file(out/'design.json'), classifiers_fitted=0))
    run.atomic_write(out/'ledger.json', dict(study_id=run.STUDY_ID, folds={}, controls={}, measure_only={}))
    loader = lambda data_dir: (X, y, ids, folds, contract)  # noqa: E731
    return out, loader


def runner_for(out, loader, retry_reason=None):
    spawn = lambda args, logfile, env: FakeProc(out, args, env, loader)  # noqa: E731
    return run.Runner(out, loader=loader, spawn=spawn, idle=IDLE_OK, retry_reason=retry_reason)


@unittest.skipUnless(HAVE, 'lightgbm/psutil not installed')
class ArmRepresentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.y, cls.ids, cls.folds = make_data()
        cls.c = run.config()
        cls.tr, cls.va = cls.X.iloc[:240].copy(), cls.X.iloc[240:].copy()
        cls.va.loc[cls.va.index[:5], 'geo_level_3_id'] = '99999'     # unseen fine region
        cls.va.loc[cls.va.index[5:7], 'geo_level_2_id'] = '88888'

    def test_num_changes_only_the_geo_representation(self):
        for model in arms.MODELS:
            oh = arms.arm_id(model, 'OH')
            num = arms.arm_id(model, 'NUM')
            pre, mt, mv = run.build_matrices(num, self.tr, self.va)
            eq = run.equivalence_report(num, pre, mt, mv, self.tr, self.va, self.c, include_train=True)
            self.assertTrue(eq['ok'], (num, eq))
            self.assertTrue(eq[f'valid_vs_{oh}']['ok'] and eq[f'train_vs_{oh}']['ok'])
            self.assertTrue(eq['geo_values_valid']['ok'] and eq['geo_values_train']['ok'])
            self.assertTrue(eq['train_only']['ok'])

    def test_equivalence_detects_a_changed_non_geo_column(self):
        oh_pre, oh_t, oh_v = run.build_matrices('DT_OH', self.tr, self.va)
        num_pre, num_t, num_v = run.build_matrices('DT_NUM', self.tr, self.va)
        names = arms.feature_names(num_pre)
        col = [i for i, n in enumerate(names) if arms.suffix(n) == 'age'][0]
        bad = num_v.tolil(copy=True)
        bad[0, col] = bad[0, col] + 1.0
        res = arms.column_equivalence(oh_v, arms.feature_names(oh_pre), bad.tocsr(), names)
        self.assertFalse(res['ok'])
        self.assertTrue(arms.column_equivalence(oh_v, arms.feature_names(oh_pre), num_v, names)['ok'])

    def test_geo_columns_are_original_integers_not_ordinal_codes(self):
        tr = self.tr.copy()
        tr['geo_level_1_id'] = ['10', '2'] * 120                        # string order '10' < '2'
        va = self.va.copy()
        va['geo_level_1_id'] = ['2', '10'] * (len(va) // 2)
        pre, mt, mv = run.build_matrices('DT_NUM', tr, va)
        names = arms.feature_names(pre)
        i = [k for k, n in enumerate(names) if arms.suffix(n) == 'geo_level_1_id'][0]
        self.assertEqual(sorted(set(mv[:, i].toarray().ravel())), [2.0, 10.0])
        # LGBM NUM geo value equals the integer; NAT holds the string-order code
        _, _, nv = run.build_matrices('LGBM_NUM', tr, va)
        _, _, av = run.build_matrices('LGBM_NAT', tr, va)
        self.assertEqual(sorted(set(nv[:, 28].toarray().ravel())), [2.0, 10.0])
        self.assertEqual(sorted(set(av[:, 28])), [0.0, 1.0])

    def test_unseen_geo_value_is_kept_for_num_and_zero_or_nan_for_categorical(self):
        _, _, nv = run.build_matrices('LGBM_NUM', self.tr, self.va)
        self.assertTrue((nv[:5, 30].toarray().ravel() == 99999).all())
        _, _, av = run.build_matrices('LGBM_NAT', self.tr, self.va)
        self.assertTrue(np.isnan(av[:5, 30]).all())
        pre, ot, ov = run.build_matrices('DT_OH', self.tr, self.va)
        names = arms.feature_names(pre)
        geo3 = [i for i, n in enumerate(names) if arms.suffix(n).startswith('geo_level_3_id')]
        self.assertEqual(float(ov[:5, geo3].sum()), 0.0)

    def test_container_types_are_fixed_and_never_densified(self):
        calls = []
        real = sparse.csr_matrix.toarray

        def guarded(self_, *a, **k):
            calls.append(self_.shape[1])
            return real(self_, *a, **k)
        with mock.patch.object(sparse.csr_matrix, 'toarray', guarded):
            for model in arms.MODELS:
                for rep in self.c['arms_by_model'][model]:
                    arm = arms.arm_id(model, rep)
                    _, mt, mv = run.build_matrices(arm, self.tr, self.va)
                    want = self.c['containers'][arm]
                    self.assertEqual(arms.container_kind(mt), want, arm)
                    self.assertEqual(arms.container_kind(mv), want, arm)
                    self.assertEqual(mt.dtype, np.float32)
                    arms.check_container(arm, mt, self.c)
        self.assertTrue(all(w <= 8 for w in calls), calls)          # only tiny column subsets, never the matrix
        self.assertEqual(self.c['containers']['LGBM_NAT'], 'dense')
        self.assertEqual([a for a, k in self.c['containers'].items() if k == 'dense'], ['LGBM_NAT'])

    def test_container_check_rejects_wrong_format(self):
        _, mt, _ = run.build_matrices('DT_NUM', self.tr, self.va)
        with self.assertRaises(ValueError):
            arms.check_container('LGBM_NAT', mt, self.c)
        with self.assertRaises(ValueError):
            arms.check_container('DT_NUM', mt.toarray(), self.c)       # 240 x 68 test-size dense only

    def test_lgbm_layouts_and_categorical_declarations(self):
        pre = arms.make_preprocessor('LGBM_NUM').fit(self.tr)
        names = [arms.suffix(n) for n in arms.feature_names(pre)]
        self.assertEqual(names[28:31], list(GEO))
        self.assertEqual(names[31:], list(TEXT))
        self.assertEqual(self.c['lightgbm_categorical_feature']['LGBM_NUM'], list(range(31, 39)))
        self.assertEqual(self.c['lightgbm_categorical_feature']['LGBM_NAT'], list(range(28, 39)))
        self.assertEqual(self.c['lightgbm_categorical_feature']['LGBM_OH'], list(range(28, 36)))
        names_p = [arms.suffix(n) for n in arms.feature_names(arms.make_preprocessor('LGBM_OH').fit(self.tr))]
        self.assertEqual(names_p[28:36], list(TEXT))

    def test_geo_float32_exactness_limit(self):
        self.assertEqual(arms.geo_float32(pd.DataFrame({'a': ['16777215']}))[0, 0], 16777215.0)
        for bad in (['16777216'], ['-1'], ['1.5'], ['x']):
            with self.assertRaises((ValueError, TypeError)):
                arms.geo_float32(pd.DataFrame({'a': bad}))

    def test_lr_num_scales_geo_with_train_fold_statistics(self):
        pre = arms.make_preprocessor('LR_NUM')
        pre.fit(self.tr)
        scaler = pre['columns'].named_transformers_['numeric']['scale']
        g = self.tr[GEO].astype('int64').to_numpy().astype(np.float32).astype(np.float64)
        np.testing.assert_allclose(scaler.mean_[5:], g.mean(0), rtol=1e-6)
        self.assertEqual(len(scaler.mean_), len(NUMERIC) + 3)

    def test_classifier_parameters_are_the_registered_ones(self):
        self.assertEqual(arms.make_classifier('RF_NUM', self.c).get_params()['max_features'], 0.1)
        self.assertEqual(arms.make_classifier('GBC_OH', self.c).get_params()['max_features'], None)
        self.assertEqual(arms.make_classifier('LGBM_NUM', self.c).get_params()['n_jobs'], 1)
        self.assertEqual(arms.make_classifier('DT_OH', self.c).get_params()['max_depth'], 12)
        self.assertEqual(arms.make_classifier('RF_OH', self.c).get_params()['n_jobs'], 1)


@unittest.skipUnless(HAVE, 'lightgbm/psutil not installed')
class MeasurementLogicTests(unittest.TestCase):
    def test_interval_increment_semantics(self):
        t = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        rss = np.array([100, 100, 150, 300, 200, 180]) * measure.MIB
        r = measure.interval_increment(t, rss, 0.15, 0.45)
        self.assertEqual(r['baseline_bytes'], 100 * measure.MIB)      # sample at/before the interval start
        self.assertEqual(r['peak_bytes'], 300 * measure.MIB)
        self.assertEqual(r['increment_bytes'], 200 * measure.MIB)
        self.assertEqual(r['n_samples'], 3)
        down = measure.interval_increment(t, np.array([300, 250, 200, 150, 100, 90]) * measure.MIB, 0.25, 0.45)
        self.assertEqual(down['increment_bytes'], 0)                  # never negative
        self.assertEqual(measure.interval_increment([], [], 0, 1)['n_samples'], 0)

    def test_memory_intervals_and_sampling_limited_flag(self):
        t = np.linspace(0, 1, 101)
        rss = np.full(101, 100 * measure.MIB)
        rss[50:80] = 400 * measure.MIB
        marks = dict(enc_start=0.1, fit_start=0.4, fit_end=0.9, pred_end=0.95)
        m = measure.memory_intervals(t, rss, marks, sampling_limited_below=50)
        self.assertEqual(m['encode_train']['increment_mib'], 300.0)
        self.assertEqual(m['encode']['increment_mib'], 0.0)
        self.assertEqual(m['train']['increment_mib'], 300.0)
        self.assertFalse(m['encode_train']['sampling_limited'])
        self.assertTrue(m['predict']['sampling_limited'])             # only 5 samples in the interval

    def test_wset_floor_flag(self):
        self.assertTrue(measure.wset_floor(500 * measure.MIB, 500 * measure.MIB)['floor_limited'])
        self.assertTrue(measure.wset_floor(500 * measure.MIB, 400 * measure.MIB)['floor_limited'])
        r = measure.wset_floor(500 * measure.MIB, 900 * measure.MIB)
        self.assertFalse(r['floor_limited'])
        self.assertEqual(r['increment_mib'], 400.0)

    def test_memory_pair_eligibility(self):
        self.assertTrue(verdict.memory_pair_eligible(20, 40, False, False, 16))
        self.assertFalse(verdict.memory_pair_eligible(15.9, 40, False, False, 16))
        self.assertFalse(verdict.memory_pair_eligible(20, 40, True, False, 16))
        self.assertFalse(verdict.memory_pair_eligible(20, 40, False, True, 16))

    def test_idle_check_waits_then_marks_noisy(self):
        clock = iter(range(0, 1000, 5))
        values = iter([40.0, 30.0, 8.0])
        ok = measure.idle_check(10, 5, 60, cpu_percent=lambda w: next(values), clock=lambda: float(next(clock)))
        self.assertFalse(ok['noisy'])
        self.assertEqual(ok['samples'], [40.0, 30.0, 8.0])
        clock2 = iter(range(0, 1000, 20))
        noisy = measure.idle_check(10, 5, 60, cpu_percent=lambda w: 50.0, clock=lambda: float(next(clock2)))
        self.assertTrue(noisy['noisy'])
        self.assertGreaterEqual(noisy['waited_seconds'], 60)

    def test_real_sampler_sees_a_child_allocation(self):
        code = ("import time,numpy as np\n"
                "time.sleep(0.5)\n"
                "s=time.perf_counter();a=np.ones(40_000_000);a+=1;time.sleep(1.0);e=time.perf_counter();del a\n"
                "time.sleep(0.3);import os;print(repr(s),repr(e),os.getpid())")
        proc = subprocess.Popen([sys.executable, '-c', code], stdout=subprocess.PIPE, text=True)
        sampler = measure.RssSampler(proc.pid, 0.02).start()
        out = proc.communicate()[0].split()
        t, rss = sampler.stop()
        s, e = float(out[0]), float(out[1])
        self.assertIn(int(out[2]), sampler.sampled_pids)              # the interpreter itself, even behind a venv launcher
        r = measure.interval_increment(t, rss, s, e)
        self.assertGreater(len(t), 50)
        self.assertGreater(r['increment_bytes'], 200 * measure.MIB)    # 320 MB allocated
        self.assertGreater(r['n_samples'], 20)

    def test_clock_is_shared_between_processes(self):
        self.assertTrue(measure.clock_common_check()['common'])


class VerdictRuleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.v = json.loads((ROOT/'experiments/geo_code_encoding/config.json').read_text(encoding='utf-8'))['verdict']

    def test_performance_tags(self):
        v = self.v
        self.assertEqual(verdict.performance_tag(0.004, 4, 0.0, v), 'NUM_HIGHER')
        self.assertEqual(verdict.performance_tag(0.003, 5, -0.01, v), 'NUM_HIGHER')
        self.assertEqual(verdict.performance_tag(0.004, 3, 0.0, v), 'MIXED')
        self.assertEqual(verdict.performance_tag(0.004, 5, -0.02, v), 'MIXED')
        self.assertEqual(verdict.performance_tag(-0.004, 1, 0.0, v), 'NUM_LOWER')
        self.assertEqual(verdict.performance_tag(-0.004, 2, 0.0, v), 'MIXED')
        self.assertEqual(verdict.performance_tag(0.0029, 5, 0.0, v), 'NO_MATERIAL_DIFFERENCE')
        self.assertEqual(verdict.performance_tag(-0.0029, 0, 0.0, v), 'NO_MATERIAL_DIFFERENCE')
        self.assertEqual(verdict.performance_tag(0.1, 5, -0.5, v), 'MIXED')

    def test_safety_3to1_is_a_double_condition_and_bidirectional(self):
        v = self.v
        self.assertTrue(verdict.safety_3to1_exceeds(200, 165, v))       # +21%, +35
        self.assertFalse(verdict.safety_3to1_exceeds(190, 165, v))      # +15%, +25 (abs fails)
        self.assertFalse(verdict.safety_3to1_exceeds(1040, 1000, v))    # +40 abs but only +4%
        self.assertEqual(verdict.safety_flags('NUM', 100, 'OH', 150, v), ['OH'])
        self.assertEqual(verdict.safety_flags('NUM', 150, 'OH', 100, v), ['NUM'])
        self.assertEqual(verdict.safety_flags('NUM', 150, 'OH', 150, v), [])

    def test_ratio_tags_and_boundary(self):
        v = self.v
        s = verdict.ratio_summary([0.3, 0.4, 0.5, 0.35, 0.45], v)
        self.assertEqual((s['tag'], s['boundary']), ('LOWER', False))
        s = verdict.ratio_summary([0.4, 0.6, 0.3, 0.45, 0.55], v)
        self.assertEqual((s['tag'], s['boundary']), ('LOWER', True))
        self.assertEqual(verdict.ratio_summary([1.0, 1.1, 0.9], v)['tag'], 'SIMILAR')
        self.assertEqual(verdict.ratio_summary([2.0, 2.5, 3.0], v)['tag'], 'HIGHER')
        self.assertIsNone(verdict.ratio_summary([None, None], v)['tag'])
        self.assertIsNone(verdict.safe_ratio(1.0, 0.0))


@unittest.skipUnless(HAVE, 'lightgbm/psutil not installed')
class ControlToleranceTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)
        p = self.rng.dirichlet([1, 1, 1], 200)
        self.p = p
        self.pred = p.argmax(1) + 1

    def test_tree_probability_tolerance(self):
        ok = run.compare_probabilities(self.p + 5e-10, self.p, self.pred, 1e-9, 2e-9)
        self.assertEqual(ok['status'], 'PASS')
        bad = run.compare_probabilities(self.p + 2e-9, self.p, self.pred, 1e-9, 2e-9)
        self.assertEqual(bad['status'], 'FAIL')

    def test_tie_rows_may_flip_the_label_only_within_2e9(self):
        p = np.array([[0.5, 0.5 - 1.5e-9, 0.0 + 1.5e-9], [0.2, 0.7, 0.1]])
        ref = np.array([1, 2])
        # new probabilities that move the first row's argmax by less than 1e-9 per entry
        new = p.copy(); new[0, 0] -= 9e-10; new[0, 1] += 9e-10
        r = run.compare_probabilities(new, p, ref, 1e-9, 2e-9)
        self.assertEqual(r['status'], 'PASS')
        self.assertEqual((r['label_mismatch_rows'], r['tie_allowed_rows']), (1, 1))
        # a label flip on a row whose reference top-two gap is large is not allowed
        flip = np.array([[0.2, 0.8, 0.0]])
        self.assertEqual(run.compare_probabilities(flip, np.array([[0.2 + 5e-10, 0.8 - 5e-10, 0.0]]), np.array([1]),
                                                   1e-9, 2e-9)['status'], 'FAIL')

    def test_lr_fold_f1_tolerance_continues_when_diverged(self):
        y = np.tile([1, 2, 3], 200)
        ref = y.copy(); ref[:150] = 1
        new_close = ref.copy()
        r = run.compare_lr_control(y, new_close, ref, 5e-4)
        self.assertEqual(r['status'], 'PASS')
        new_far = ref.copy(); new_far[:40] = 3
        r = run.compare_lr_control(y, new_far, ref, 5e-4)
        self.assertEqual(r['status'], 'DIVERGED_KNOWN_ENV')

    def test_config_tolerances_are_the_registered_ones(self):
        c = run.config()['control']
        self.assertEqual((c['tree_probability_atol'], c['tree_tie_gap_atol'], c['lr_fold_f1_atol']), (1e-9, 2e-9, 5e-4))
        self.assertEqual(c['lr_secondary_probability_atol'], 1e-6)


@unittest.skipUnless(HAVE, 'lightgbm/psutil not installed')
class PlanAndConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = run.config()
        cls.plan = run.execution_plan(cls.c)

    def test_counts_and_limits(self):
        fits = [s for s in self.plan if s['kind'] == 'fit']
        self.assertEqual(len(fits), 55)
        self.assertEqual(len({(s['arm'], s['fold']) for s in fits}), 55)
        self.assertEqual(len(run.study_targets(self.c)), 55)
        self.assertEqual(self.c['fit_plan']['max_attempts'], 59)
        self.assertEqual(self.c['fit_plan']['max_retries_total'], 4)
        measures = [s for s in self.plan if s['kind'] == 'measure']
        self.assertEqual([(m['role'], m['arm']) for m in measures],
                         [('warmup', 'DT_OH'), ('repeat', 'DT_OH'), ('repeat', 'DT_NUM'), ('repeat', 'DT_OH'), ('repeat', 'DT_NUM')])
        self.assertEqual(self.plan[0]['kind'], 'measure')
        self.assertNotIn('LGBM_NUM_DENSE', arms.all_arms(self.c))
        self.assertFalse([a for a in arms.all_arms(self.c) if 'SQRT' in a])

    def test_models_run_in_registered_order_and_fold0_is_control_gated(self):
        models = []
        for s in self.plan:
            m = s.get('model') or (s['arm'].split('_')[0] if s['kind'] == 'fit' else None)
            if m and (not models or models[-1] != m):
                models.append(m)
        self.assertEqual(models, ['LR', 'DT', 'GBC', 'RF', 'LGBM'])
        for model in self.c['model_order']:
            seq = [(s['kind'], s.get('arm')) for s in self.plan
                   if (s['kind'] == 'control' and s['model'] == model)
                   or (s['kind'] == 'fit' and s['fold'] == 0 and s['arm'].startswith(model + '_'))]
            reps = self.c['arms_by_model'][model]
            expected = [('fit', f'{model}_{r}') for r in reps[:-1]] + [('control', None), ('fit', f'{model}_NUM')]
            self.assertEqual(seq, expected)
            self.assertEqual(reps[-1], 'NUM')

    def test_alternation_starts_with_fold_1(self):
        c = self.c
        self.assertEqual(run.arm_order('LR', 1, c), ['LR_NUM', 'LR_OH'])
        self.assertEqual(run.arm_order('LR', 2, c), ['LR_OH', 'LR_NUM'])
        self.assertEqual(run.arm_order('RF', 3, c), ['RF_NUM', 'RF_OH'])
        self.assertEqual(run.arm_order('RF', 4, c), ['RF_OH', 'RF_NUM'])
        self.assertEqual(run.arm_order('LGBM', 0, c), ['OH', 'NAT', 'NUM'])
        orders = [tuple(run.arm_order('LGBM', f, c)) for f in range(1, 5)]
        self.assertEqual(orders[0], ('LGBM_NAT', 'LGBM_NUM', 'LGBM_OH'))
        for o in orders:
            self.assertEqual(sorted(o), ['LGBM_NAT', 'LGBM_NUM', 'LGBM_OH'])

    def test_every_model_num_fit_comes_after_its_fold0_control(self):
        seen_control = set()
        for s in self.plan:
            if s['kind'] == 'control':
                seen_control.add(s['model'])
            if s['kind'] == 'fit' and s['arm'].endswith('_NUM'):
                self.assertIn(s['arm'].split('_')[0], seen_control, s)

    def test_parameters_match_registered_sources(self):
        p = self.c['classifier_params']
        self.assertEqual(p['RF']['max_features'], 0.1)
        self.assertIsNone(p['GBC']['max_features'])
        self.assertEqual(p['LGBM']['n_jobs'], 1)
        self.assertEqual(p['LGBM']['deterministic'], True)
        self.assertEqual(self.c['age_policy'], {'LR': 'C', 'DT': 'A', 'RF': 'A', 'GBC': 'A', 'LGBM': 'C'})

    def test_blob_check_and_head_with_stub_runner(self):
        class R:
            def __init__(self, out, rc=0): self.stdout, self.returncode = out, rc
        calls = []

        def good(args):
            calls.append(args)
            return R('abc\n')
        self.assertEqual(run.git_blob_mismatches(['a.py', 'b.py'], good), [])

        def one_bad(args):
            if args[0] == 'hash-object' and args[1] == 'b.py':
                return R('zzz\n')
            return R('abc\n')
        self.assertEqual(run.git_blob_mismatches(['a.py', 'b.py'], one_bad), ['b.py'])
        self.assertEqual(run.git_blob_mismatches(['x.py'], lambda args: R('', 128)), ['x.py'])
        self.assertEqual(run.git_head(lambda args: R('1234\n')), '1234')

    def test_sealed_sources_exist_and_no_holdout_reference(self):
        for rel in run.SOURCE:
            self.assertTrue((ROOT/rel).is_file(), rel)
        for p in (ROOT/'experiments/geo_code_encoding').glob('*'):
            if p.is_file() and p.suffix in ('.py', '.json'):
                self.assertNotIn('final_holdout', p.read_text(encoding='utf-8'), p.name)

    def test_prepare_refuses_existing_output_and_blob_mismatch(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                run.prepare('unused', d)
        with tempfile.TemporaryDirectory() as d:
            class R:
                def __init__(self, out, rc=0): self.stdout, self.returncode = out, rc

            def stub(args):
                if args[0] == 'hash-object':
                    return R('different\n')
                return R('abc\n')
            with self.assertRaises(ValueError):
                run.prepare('unused', Path(d)/'new', runner=stub)


@unittest.skipUnless(HAVE, 'lightgbm/psutil not installed')
class PreflightTests(unittest.TestCase):
    def test_preflight_matrices_cover_five_folds_without_a_classifier(self):
        X, y, ids, folds = make_data(300)
        c = run.config()
        res = run.preflight_matrices(X, folds, c, log_fn=lambda m: None)
        self.assertEqual(len(res['folds']), 5)
        self.assertEqual(res['classifiers_fitted'], 0)
        f0 = res['folds'][0]
        self.assertEqual(f0['groups']['LGBM']['LGBM_NAT']['train']['format'], 'dense')
        self.assertEqual(f0['groups']['LGBM']['LGBM_NUM']['train']['format'], 'csr')
        self.assertEqual(f0['groups']['TREE']['DT_NUM']['train']['format'], 'csr')
        self.assertTrue(f0['groups']['LR']['LR_NUM']['equivalence']['ok'])
        self.assertLess(res['geo_max_value'], 2 ** 24)
        self.assertIn('any', f0['unseen_valid_geo'])
        json.dumps(run.jsonable(res), allow_nan=False)


@unittest.skipUnless(HAVE, 'lightgbm/psutil not installed')
class FitJobTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = make_data(300)
        cls.c = run.config()

    def test_every_arm_fits_saves_reloads_and_measures(self):
        X, y, ids, folds = self.data
        env = {k: v for k, v in self.c['threading']['env'].items()}
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, env):
            for arm in arms.all_arms(self.c):
                rec = run.fit_arm_job(arm, X, y, ids, folds, 1, Path(d)/arm, 'fold_1', self.c, save=True)
                self.assertEqual(rec['containers']['train']['format'], self.c['containers'][arm])
                self.assertTrue(rec['reload_probability_max_abs_diff'] <= 1e-12)
                self.assertTrue(rec['csv_round_trip_exact'])
                m = rec['marks']
                self.assertTrue(m['enc_start'] < m['fit_start'] < m['fit_end'] < m['pred_end'])
                t = rec['timings']
                self.assertAlmostEqual(t['total_seconds'], t['encode_seconds'] + t['fit_seconds'] + t['predict_seconds'])
                self.assertTrue(all(v >= 0 for v in [t['encode_seconds'], t['fit_seconds'], t['predict_seconds'],
                                                     t['normalization_seconds'], t['representation_seconds']]))
                self.assertGreater(rec['sizes']['classifier_bytes'], 0)
                self.assertGreater(rec['sizes']['preprocessor_bytes'], 0)
                self.assertEqual(rec['n_encoded_columns'], rec['containers']['train']['columns'])
                self.assertTrue(rec['thread_pools'] is not None)
                if arm.endswith('_NUM'):
                    self.assertTrue(rec['equivalence']['ok'])
                json.dumps(run.jsonable(rec), allow_nan=False)
                if arm.startswith('LGBM'):
                    text = (Path(d)/arm/'fold_1_model.txt').read_text(encoding='utf-8')
                    want = ','.join(map(str, self.c['lightgbm_categorical_feature'][arm]))
                    self.assertIn(f'[categorical_feature: {want}]', text)
                    self.assertEqual(rec['internals']['rounds_completed'], 300)
                    self.assertGreater(rec['internals']['trees'], 0)

    def test_measurement_only_job_saves_nothing(self):
        X, y, ids, folds = self.data
        env = {k: v for k, v in self.c['threading']['env'].items()}
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, env):
            rec = run.fit_arm_job('DT_OH', X, y, ids, folds, 0, d, 'warmup_x', self.c, save=False)
            self.assertEqual(rec['files'], {})
            self.assertEqual(list(Path(d).iterdir()), [])

    def test_lgbm_text_stats_counts_geo_splits(self):
        X, y, ids, folds = self.data
        pre, mt, mv = run.build_matrices('LGBM_NUM', X.iloc[:240], X.iloc[240:])
        from lightgbm import LGBMClassifier
        clf = LGBMClassifier(**dict(self.c['classifier_params']['LGBM'], n_estimators=3, verbosity=-1))
        clf.fit(mt, y.iloc[:240].to_numpy() - 1, categorical_feature=self.c['lightgbm_categorical_feature']['LGBM_NUM'])
        stats = run.lgbm_text_stats(clf.booster_.model_to_string(), arms.geo_indices(arms.feature_names(pre)))
        self.assertEqual(stats['trees'], 9)
        self.assertGreaterEqual(stats['splits'], stats['geo_splits'])


@unittest.skipUnless(HAVE, 'lightgbm/psutil not installed')
class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.data = make_data(300)
        self.out, self.loader = make_study(self.tmp.name, self.data)

    def tearDown(self):
        self.tmp.cleanup()

    def test_fit_step_checkpoints_hashes_backs_up_and_resumes_without_refit(self):
        r = runner_for(self.out, self.loader)
        entry, fitted = r.fit_step('DT_OH', 0)
        self.assertTrue(fitted)
        self.assertEqual(entry['status'], 'complete')
        self.assertEqual(entry['attempts'], 1)
        for name, digest in entry['artifact_sha256'].items():
            self.assertEqual(run.sha256_file(self.out/entry['dir']/name), digest)
            self.assertEqual(run.sha256_file(self.out/'backup'/entry['dir']/name), digest)
        self.assertTrue({'fold_0_model.joblib', 'fold_0_pred.csv.gz', 'fold_0_preprocessor.joblib', 'fold_0.json',
                         'fold_0_rss.csv.gz'} <= set(entry['artifact_sha256']))
        rec = json.loads((self.out/'arms/DT_OH/fold_0.json').read_text(encoding='utf-8'))
        self.assertFalse(rec['noisy'])
        self.assertIn(rec['pid'], rec['rss_sampled_pids'])
        self.assertIn('encode_train', rec['memory'])
        self.assertGreater(rec['rss_sample_count'], 0)
        calls = []
        r2 = run.Runner(self.out, loader=self.loader, spawn=lambda *a: calls.append(a), idle=IDLE_OK)
        entry2, fitted2 = r2.fit_step('DT_OH', 0)
        self.assertFalse(fitted2)
        self.assertEqual(calls, [])                                    # completed fits are never re-run
        self.assertEqual(run.status(self.out)['complete'], 1)

    def test_resume_detects_a_changed_artifact(self):
        r = runner_for(self.out, self.loader)
        r.fit_step('DT_OH', 0)
        path = self.out/'arms/DT_OH/fold_0_pred.csv.gz'
        path.write_bytes(path.read_bytes() + b'x')
        with self.assertRaises(ValueError):
            runner_for(self.out, self.loader).fit_step('DT_OH', 0)

    def test_only_registered_fits_and_attempt_budget_are_enforced(self):
        r = runner_for(self.out, self.loader)
        with self.assertRaises(ValueError):
            r.fit_step('DT_OH', 5)
        with self.assertRaises(ValueError):
            r.fit_step('LGBM_NUM_DENSE', 0)
        ledger = run.read_ledger(self.out)
        for k in run.study_targets(r.c):
            if k != 'DT_OH/1':
                ledger['folds'][k] = dict(dir='x', status='complete', attempts=1, artifact_sha256={})
        ledger['folds']['LR_OH/0']['attempts'] = 6          # 53 + 6 = 59 attempts already used
        run.atomic_write(self.out/'ledger.json', ledger)
        with self.assertRaises(ValueError) as ctx:
            runner_for(self.out, self.loader).fit_step('DT_OH', 1)
        self.assertIn('budget', str(ctx.exception))

    def test_interrupted_fit_needs_manual_retry_reason_and_is_limited(self):
        run.atomic_write(self.out/'ledger.json', dict(study_id=run.STUDY_ID, controls={}, measure_only={}, folds={
            'DT_OH/0': dict(dir='arms/DT_OH', arm='DT_OH', fold=0, status='running', attempts=1)}))
        with self.assertRaises(ValueError):
            runner_for(self.out, self.loader).fit_step('DT_OH', 0)      # no reason: automatic retry is forbidden
        e, fitted = runner_for(self.out, self.loader, 'power loss').fit_step('DT_OH', 0)
        self.assertTrue(fitted)
        self.assertEqual((e['attempts'], e['retry_reasons']), (2, ['power loss']))
        # per-fit retry cap
        ledger = run.read_ledger(self.out)
        ledger['folds']['DT_NUM/0'] = dict(dir='arms/DT_NUM', arm='DT_NUM', fold=0, status='interrupted', attempts=2)
        run.atomic_write(self.out/'ledger.json', ledger)
        with self.assertRaises(ValueError):
            runner_for(self.out, self.loader, 'again').fit_step('DT_NUM', 0)

    def test_total_retry_cap_of_four(self):
        folds = {}
        for i in range(4):
            folds[f'LR_OH/{i}'] = dict(dir='x', status='complete', attempts=2, artifact_sha256={})
        folds['DT_OH/0'] = dict(dir='arms/DT_OH', arm='DT_OH', fold=0, status='interrupted', attempts=1)
        run.atomic_write(self.out/'ledger.json', dict(study_id=run.STUDY_ID, controls={}, measure_only={}, folds=folds))
        with self.assertRaises(ValueError) as ctx:
            runner_for(self.out, self.loader, 'x').fit_step('DT_OH', 0)
        self.assertIn('Retry budget', str(ctx.exception))

    def test_code_or_data_defect_is_not_retried(self):
        run.atomic_write(self.out/'ledger.json', dict(study_id=run.STUDY_ID, controls={}, measure_only={}, folds={
            'DT_OH/0': dict(dir='arms/DT_OH', arm='DT_OH', fold=0, status='failed', attempts=1)}))
        with self.assertRaises(ValueError):
            runner_for(self.out, self.loader, 'please').fit_step('DT_OH', 0)

    def test_child_failure_is_recorded_as_failed_or_interrupted(self):
        bad_loader = lambda data_dir: (_ for _ in ()).throw(RuntimeError('boom'))  # noqa: E731
        r = runner_for(self.out, self.loader)
        r.spawn = lambda args, logfile, env: FakeProc(self.out, args, env, bad_loader)
        with self.assertRaises(RuntimeError):
            r.fit_step('DT_OH', 0)
        self.assertEqual(run.read_ledger(self.out)['folds']['DT_OH/0']['status'], 'failed')   # error.json was written
        (self.out/'arms/DT_OH/fold_0_error.json').unlink()
        run.atomic_write(self.out/'ledger.json', dict(study_id=run.STUDY_ID, controls={}, measure_only={}, folds={}))
        r2 = runner_for(self.out, self.loader)
        r2.spawn = lambda args, logfile, env: type('P', (), dict(pid=os.getpid(), wait=lambda s: 137))()
        with self.assertRaises(RuntimeError):
            r2.fit_step('DT_OH', 0)
        self.assertEqual(run.read_ledger(self.out)['folds']['DT_OH/0']['status'], 'interrupted')

    def test_fit_one_cli_in_a_real_child_process_is_sampled_through_the_launcher(self):
        code = chr(10).join([
            'import sys',
            f"sys.path.insert(0, r'{ROOT}'); sys.path.insert(0, r'{ROOT}/tests')",
            'from unittest import mock',
            'import test_geo_code_encoding as T',
            'from experiments.geo_code_encoding import run',
            'X, y, ids, folds = T.make_data(300)',
            "with mock.patch.object(run, 'load', lambda d: (X, y, ids, folds, {'synthetic': True})):",
            f"    sys.exit(run.main(['fit-one', '--out', r'{self.out}', '--arm', 'DT_NUM', '--fold', '1', "
            "'--role', 'main', '--tag', 'main']))",
        ])
        r = runner_for(self.out, self.loader)
        r.spawn = lambda args, logfile, env: subprocess.Popen([sys.executable, '-c', code], cwd=ROOT, env=env,
                                                              stdout=logfile, stderr=subprocess.STDOUT)
        entry, fitted = r.fit_step('DT_NUM', 1)
        self.assertTrue(fitted)
        rec = json.loads((self.out/'arms/DT_NUM/fold_1.json').read_text(encoding='utf-8'))
        self.assertNotEqual(rec['pid'], os.getpid())
        self.assertIn(rec['pid'], rec['rss_sampled_pids'])
        self.assertEqual(rec['env_threads'], r.c['threading']['env'])
        self.assertGreater(rec['rss_sample_count'], 10)
        self.assertGreater(rec['memory']['encode_train']['n_samples'], 0)

    def test_child_requires_single_thread_environment(self):
        r = runner_for(self.out, self.loader)
        r.spawn = lambda args, logfile, env: FakeProc(self.out, args, dict(env, OMP_NUM_THREADS='4'), self.loader)
        with self.assertRaises(RuntimeError):
            r.fit_step('DT_OH', 0)

    def test_tampered_sources_or_config_stop_the_run(self):
        design = json.loads((self.out/'design.json').read_text(encoding='utf-8'))
        design['source_sha256'][run.SOURCE[0]] = '0' * 64
        run.write_json(self.out/'design.json', design)
        run.write_json(self.out/'design_seal.json', dict(design_sha256=run.sha256_file(self.out/'design.json')))
        with self.assertRaises(ValueError):
            run.verify(self.out, self.loader)

    def test_control_check_against_a_complete_reference_file(self):
        X, y, ids, folds = self.data
        r = runner_for(self.out, self.loader)
        r.fit_step('DT_OH', 0)
        new = run.read_pred(self.out/'arms/DT_OH/fold_0_pred.csv.gz')
        ref = pd.DataFrame({'building_id': ids, 'fold': folds, 'y_true': y.to_numpy(), 'y_pred': 1,
                            'p_grade_1': 0.8, 'p_grade_2': 0.1, 'p_grade_3': 0.1})
        ref = ref.set_index('building_id')
        new_i = new.set_index('building_id')
        ref.loc[new_i.index, ['y_pred', 'p_grade_1', 'p_grade_2', 'p_grade_3']] = new_i[['y_pred', 'p_grade_1', 'p_grade_2', 'p_grade_3']]
        ref = ref.reset_index()
        refpath = Path(self.tmp.name)/'ref.csv.gz'
        run.write_pred(refpath, ref)
        sha = run.sha256_file(refpath)
        design = dict(r.design, references={'DT_OH': dict(path=str(refpath), sha256=sha)})
        rec = run.control_check(self.out, 'DT_OH', r.ids, r.y, r.folds, design)
        self.assertEqual(rec['status'], 'PASS')
        self.assertTrue((self.out/'arms/DT_OH/control_check.json').exists())
        # a perturbed reference (1e-6) fails
        ref2 = ref.copy()
        m = ref2.fold == 0
        ref2.loc[m, 'p_grade_1'] += 1e-6
        ref2.loc[m, 'p_grade_2'] -= 1e-6
        refpath2 = Path(self.tmp.name)/'ref2.csv.gz'
        run.write_pred(refpath2, ref2)
        design2 = dict(r.design, references={'DT_OH': dict(path=str(refpath2), sha256=run.sha256_file(refpath2))})
        self.assertEqual(run.control_check(self.out, 'DT_OH', r.ids, r.y, r.folds, design2)['status'], 'FAIL')
        # a reference with a wrong SHA is refused
        design3 = dict(r.design, references={'DT_OH': dict(path=str(refpath2), sha256='0' * 64)})
        with self.assertRaises(ValueError):
            run.control_check(self.out, 'DT_OH', r.ids, r.y, r.folds, design3)

    def test_failed_control_blocks_num_and_is_remembered(self):
        r = runner_for(self.out, self.loader)
        r.fit_step('DT_OH', 0)
        failed = dict(status='FAIL', arm='DT_OH')
        with mock.patch.object(run, 'control_check', lambda *a, **k: (
                run.write_json(self.out/'arms/DT_OH/control_check.json', failed) or failed)):
            with self.assertRaises(run.ControlFailed):
                r.control_step('DT')
        ledger = run.read_ledger(self.out)
        self.assertEqual(ledger['controls']['DT_OH']['status'], 'CONTROL_FAILED')
        with self.assertRaises(ValueError):
            runner_for(self.out, self.loader).control_step('DT')        # remembered: new experiment ID required

    def test_measurement_only_run_is_recorded_separately_and_not_retried(self):
        r = runner_for(self.out, self.loader)
        e = r.measure_step('warmup', 'DT_OH', 0, 1)
        self.assertEqual(e['status'], 'complete')
        self.assertFalse(e['counted_in_study_fits'])
        self.assertEqual(run.status(self.out)['complete'], 0)
        self.assertEqual(run.status(self.out)['attempts'], 0)
        files = {p.name for p in (self.out/'measure_only').iterdir()}
        self.assertFalse([f for f in files if 'model' in f or 'pred' in f])
        calls = []
        r2 = run.Runner(self.out, loader=self.loader, spawn=lambda *a: calls.append(a), idle=IDLE_OK)
        r2.measure_step('warmup', 'DT_OH', 0, 1)
        self.assertEqual(calls, [])

    def test_noisy_flag_is_recorded_not_discarded(self):
        r = runner_for(self.out, self.loader)
        r.idle = lambda: dict(noisy=True, samples=[50.0], waited_seconds=60.0)
        e, _ = r.fit_step('DT_OH', 0)
        self.assertTrue(e['noisy'])
        self.assertEqual(run.status(self.out)['noisy'], ['DT_OH/0'])


@unittest.skipUnless(HAVE, 'lightgbm/psutil not installed')
class ReviewFixTests(unittest.TestCase):
    def test_plan_sha_is_enforced(self):
        c = run.config()
        self.assertEqual(c['plan_sha256'], 'bb792e626ba3015044d6470f22d5d2b58c488d03d07266a6727fe18be4c3b22c')
        self.assertEqual(c['plan_sha256_stated_at_assignment'][:8], '5bf1600e')
        self.assertEqual(run.check_plan_sha(c), c['plan_sha256'])
        with tempfile.TemporaryDirectory() as d:
            (Path(d)/'docs').mkdir()
            (Path(d)/c['plan_doc']).write_text('a different plan', encoding='utf-8')
            with self.assertRaises(ValueError):
                run.check_plan_sha(c, root=d)

    def test_prepare_and_verify_reject_a_changed_plan(self):
        bad = dict(run.config(), plan_sha256='0' * 64)
        with tempfile.TemporaryDirectory() as d:
            with mock.patch.object(run, 'config', lambda *a, **k: bad):
                with self.assertRaises(ValueError) as ctx:
                    run.prepare('unused', Path(d)/'new')
            self.assertIn('Plan document SHA', str(ctx.exception))
        with tempfile.TemporaryDirectory() as d:
            out, loader = make_study(d, make_data(200))
            design = json.loads((out/'design.json').read_text(encoding='utf-8'))
            design['config'] = bad
            run.write_json(out/'design.json', design)
            run.write_json(out/'design_seal.json', dict(design_sha256=run.sha256_file(out/'design.json')))
            with mock.patch.object(run, 'config', lambda *a, **k: bad):
                with self.assertRaises(ValueError) as ctx:
                    run.verify(out, loader)
            self.assertIn('Plan document SHA', str(ctx.exception))

    def test_memory_and_disk_errors_are_infrastructure_interruptions(self):
        import errno
        self.assertTrue(run.is_infrastructure_error(MemoryError()))
        self.assertTrue(run.is_infrastructure_error(OSError(errno.ENOSPC, 'No space left on device')))
        self.assertFalse(run.is_infrastructure_error(OSError(errno.ENOENT, 'missing')))
        self.assertFalse(run.is_infrastructure_error(ValueError('defect')))

    def test_memory_error_in_child_is_recorded_as_interrupted_and_retryable(self):
        with tempfile.TemporaryDirectory() as d:
            out, loader = make_study(d, make_data(200))
            r = runner_for(out, loader)
            boom = lambda data_dir: (_ for _ in ()).throw(MemoryError())  # noqa: E731
            r.spawn = lambda args, logfile, env: FakeProc(out, args, env, boom)
            with self.assertRaises(RuntimeError):
                r.fit_step('DT_OH', 0)
            self.assertEqual(run.read_ledger(out)['folds']['DT_OH/0']['status'], 'interrupted')
            self.assertFalse((out/'arms/DT_OH/fold_0_error.json').exists())
            e, fitted = runner_for(out, loader, 'out of memory').fit_step('DT_OH', 0)      # allowed manual retry
            self.assertTrue(fitted and e['attempts'] == 2)

    def test_partial_backup_does_not_deadlock_the_retry(self):
        with tempfile.TemporaryDirectory() as d:
            out, loader = make_study(d, make_data(200))
            run.atomic_write(out/'ledger.json', dict(study_id=run.STUDY_ID, controls={}, measure_only={}, folds={
                'DT_OH/0': dict(dir='arms/DT_OH', arm='DT_OH', fold=0, status='running', attempts=1)}))
            bd = out/'backup/arms/DT_OH'
            bd.mkdir(parents=True)
            (bd/'fold_0_model.joblib').write_bytes(b'partial')
            e, fitted = runner_for(out, loader, 'power loss').fit_step('DT_OH', 0)
            self.assertTrue(fitted)
            self.assertTrue((bd/'fold_0_model.joblib.incomplete_attempt1').exists())      # kept, not deleted
            for name, digest in e['artifact_sha256'].items():
                self.assertEqual(run.sha256_file(bd/name), digest)

    def test_train_only_checks_detect_leaked_statistics(self):
        X, y, ids, folds = make_data(300)
        tr, va = X.iloc[:240].copy(), X.iloc[240:].copy()
        va.loc[va.index[:3], 'geo_level_3_id'] = '99999'
        va.loc[va.index[3:5], 'land_surface_condition'] = 'ZZ'
        for arm in ['DT_OH', 'DT_NUM', 'LR_OH', 'LR_NUM', 'LGBM_OH', 'LGBM_NAT', 'LGBM_NUM']:
            pre = arms.make_preprocessor(arm).fit(tr)
            clean = arms.train_only_checks(pre, arm, pre['schema'].transform(tr))
            self.assertTrue(clean['ok'], (arm, clean))
            both = pd.concat([tr, va])                    # as if the encoder/scaler had seen validation rows
            leak = arms.train_only_checks(pre, arm, pre['schema'].transform(both))
            self.assertFalse(leak['ok'], (arm, leak))
        pre = arms.make_preprocessor('LR_NUM').fit(pd.concat([tr, va]))       # scaler fitted on train + validation
        self.assertFalse(arms.train_only_checks(pre, 'LR_NUM', pre['schema'].transform(tr))['ok'])


@unittest.skipUnless(HAVE, 'lightgbm/psutil not installed')
class EndToEndTests(unittest.TestCase):
    """Full registered plan (55 fits + 5 measurement-only runs + controls + finalize) on tiny synthetic data."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.data = make_data(250)
        cls.out, cls.loader = make_study(cls.tmp.name, cls.data, references={
            n: dict(path='fake', sha256='0' * 64) for n in run.config()['references']})
        cls.loader = staticmethod(cls.loader)
        X, y, ids, folds = cls.data

        cache = {}

        def same_as_new(design, name, ids_, y_, folds_):
            # independent reference: a separate refit of fold 0 in its own directory (never the orchestrator's output)
            arm = 'LR_OH' if name == 'LR_OH_LOCAL' else name
            if arm not in cache:
                d = Path(cls.tmp.name)/'independent'/arm
                run.fit_arm_job(arm, X, y, ids, folds, 0, d, 'fold_0', run.config(), save=True)
                cache[arm] = run.read_pred(d/'fold_0_pred.csv.gz')
            return cache[arm]
        cls.patch = mock.patch.object(run, 'load_control_reference', same_as_new)
        cls.patch.start()
        cls.runner = runner_for(cls.out, cls.loader)
        cls.runner.run_plan()
        cls.result = run.finalize_all(cls.out, cls.loader)

    @classmethod
    def tearDownClass(cls):
        cls.patch.stop()
        cls.tmp.cleanup()

    def test_ledger_counts(self):
        st = run.status(self.out)
        self.assertEqual((st['complete'], st['attempts'], st['retries']), (55, 55, 0))
        self.assertEqual(len(st['measure_only']), 5)
        self.assertTrue(all(v == 'complete' for v in st['measure_only'].values()))
        self.assertEqual(set(st['controls']), {'LR_OH', 'DT_OH', 'GBC_OH', 'RF_OH', 'LGBM_OH', 'LGBM_NAT'})
        self.assertTrue(all(v in ('PASS', 'DIVERGED_KNOWN_ENV') for v in st['controls'].values()))

    def test_fits_are_strictly_serial(self):
        ledger = run.read_ledger(self.out)
        spans = sorted((e['process_perf_counter_start'], e['process_perf_counter_end'])
                       for e in list(ledger['folds'].values()) + list(ledger['measure_only'].values()))
        self.assertEqual(len(spans), 60)
        for (a0, a1), (b0, b1) in zip(spans, spans[1:]):
            self.assertLessEqual(a1, b0)

    def test_control_precedes_each_models_num_fit(self):
        events = [json.loads(line) for line in (self.out/'attempts.jsonl').read_text(encoding='utf-8').splitlines()]
        order = [(e['action'], e.get('fold')) for e in events]
        for model in ['LR', 'DT', 'GBC', 'RF', 'LGBM']:
            ctl = max(i for i, (a, f) in enumerate(order) if a == 'control_check' and f.startswith(model + '_'))
            first_num = min(i for i, (a, f) in enumerate(order) if a == 'fit_start' and f == f'{model}_NUM/0')
            self.assertLess(ctl, first_num)

    def test_analysis_structure_and_verdicts(self):
        res = self.result
        json.dumps(res, allow_nan=False)
        pairs = [(c['candidate'], c['baseline']) for c in res['comparisons']]
        self.assertEqual(pairs, [('LR_NUM', 'LR_OH'), ('DT_NUM', 'DT_OH'), ('GBC_NUM', 'GBC_OH'), ('RF_NUM', 'RF_OH'),
                                 ('LGBM_NUM', 'LGBM_OH'), ('LGBM_NUM', 'LGBM_NAT'), ('LGBM_NAT', 'LGBM_OH')])
        tags = {'NUM_HIGHER', 'NUM_LOWER', 'NO_MATERIAL_DIFFERENCE', 'MIXED'}
        for cp in res['comparisons']:
            if cp['candidate'].endswith('_NUM'):
                self.assertIn(cp['performance_tag'], tags)
            else:
                self.assertIsNone(cp['performance_tag'])
            self.assertEqual(len(cp['fold_delta']), 5)
            self.assertIn('train_seconds', cp['compute'])
            self.assertIn('turnover', cp)
        confounded = [c for c in res['comparisons'] if c['container_confounded']]
        self.assertEqual([(c['candidate'], c['baseline']) for c in confounded],
                         [('LGBM_NUM', 'LGBM_NAT'), ('LGBM_NAT', 'LGBM_OH')])       # every comparison involving dense NAT
        for cp in confounded:
            for m in ['train_seconds', 'predict_seconds', 'total_seconds', 'model_bytes', 'matrix_storage_bytes', 'memory_mib']:
                self.assertIsNone(cp['compute'][m]['tag'])
        tagged = [(c['candidate'], c['baseline']) for c in res['comparisons'] if c['compute']['train_seconds']['tag']]
        self.assertEqual(tagged, [('LR_NUM', 'LR_OH'), ('DT_NUM', 'DT_OH'), ('GBC_NUM', 'GBC_OH'), ('RF_NUM', 'RF_OH'),
                                  ('LGBM_NUM', 'LGBM_OH')])
        self.assertIn('n_eligible_folds', res['comparisons'][0]['compute']['memory_mib'])
        eq = res['arms']['LR_NUM']['num_equivalence_per_fold']
        self.assertEqual(len(eq), 5)
        self.assertIn('bitwise_equal', eq[0]['valid_vs_LR_OH'])
        self.assertEqual(set(res['arms']), set(arms.all_arms(run.config())))
        self.assertEqual(res['ledger']['complete'], 55)
        self.assertFalse(res['holdout_accessed'])
        self.assertIn('DT_OH/fit_seconds', res['repeat_cv'])
        self.assertEqual(len(res['measurement_only']), 5)

    def test_oof_covers_every_row_once_and_metrics_recompute(self):
        X, y, ids, folds = self.data
        for arm in arms.all_arms(run.config()):
            oof = run.read_pred(self.out/'arms'/arm/'oof.csv.gz')
            self.assertEqual(sorted(oof.building_id), sorted(ids))
            self.assertTrue(oof.building_id.is_unique)
            fold_f1 = [run.score(oof.y_true[oof.fold == f], oof.y_pred[oof.fold == f])['macro_f1'] for f in range(5)]
            self.assertAlmostEqual(np.mean(fold_f1), self.result['arms'][arm]['cv_mean'])
            self.assertEqual(self.result['arms'][arm]['pooled']['n'], len(ids))

    def test_analysis_and_backup_exist_with_matching_hashes(self):
        self.assertEqual(run.sha256_file(self.out/'analysis.json'), run.sha256_file(self.out/'backup/analysis.json'))
        with self.assertRaises(ValueError):
            run.finalize_all(self.out, self.loader)                     # never overwrite


if __name__ == '__main__':
    unittest.main()
