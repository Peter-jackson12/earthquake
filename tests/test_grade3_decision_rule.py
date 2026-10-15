"""T15 synthetic contracts only: no real development data, no holdout access.

Fits here are tiny synthetic sample checks, never study training runs.
"""
import copy
import inspect
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

try:
    import lightgbm  # noqa: F401
    from experiments.grade3_decision_rule import run as t15
    HAVE_LGBM = True
except ImportError:  # pragma: no cover
    HAVE_LGBM = False

ARMS = ['LR_C', 'LGBM_O', 'LGBM_TE_G']


def data(n=1500, seed=7):
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


def fake_fit(calls=None, fail_at=None, converged=True):
    """Label-free fake of fit_inner_arm with the same checkpoint interface."""
    def fit(arm, kind, Xa, ya, Xb, c):
        if calls is not None:
            calls.append((arm, kind, Xa.index.to_numpy(), Xb.index.to_numpy()))
            if fail_at is not None and len(calls) == fail_at:
                raise RuntimeError('simulated crash')
        rng = np.random.default_rng(int(Xb.index[0]))
        p = rng.dirichlet([2.0, 2.0, 1.0], len(Xb))

        def save(prefix):
            path = Path(f'{prefix}_model.json')
            path.write_text(json.dumps({'arm': arm}))
            return [path]
        return dict(p=p, converged=converged, fit_seconds=0.01, warnings=[], save=save,
                    verify=lambda paths, Xv, ps: 0.0)
    return fit


class Env:
    """Synthetic outer split + reference OOFs + design, with outer labels optionally scrambled in one fold."""
    def __init__(self, tmp, kind='group2', scramble_fold=None):
        self.tmp = Path(tmp)
        self.X, y, self.ids = data()
        self.kind = kind
        from experiments.geo2_te_confirm import run as t13
        self.folds = t13.make_folds(self.X, y, 101)
        y = y.copy()
        if scramble_fold is not None:  # change ONLY outer-validation labels of that fold
            rows = np.flatnonzero(self.folds == scramble_fold)
            y.iloc[rows] = np.random.default_rng(5).permutation(y.iloc[rows].to_numpy())
        self.y = y
        c = copy.deepcopy(t15.config())
        c['schemes'] = {'toy': {'kind': kind, 'seed': 101, 'refs': {}}}
        rng = np.random.default_rng(11)
        for arm in ARMS:
            p = rng.dirichlet([2.0, 2.0, 1.0], len(self.X))
            frame = pd.DataFrame({'building_id': self.ids, 'fold': self.folds, 'y_true': self.y.to_numpy(),
                                  'y_pred': p.argmax(1) + 1, **{n: p[:, i] for i, n in enumerate(t15.PROBS)}})
            path = self.tmp/f'ref_{arm}.csv.gz'
            t15.write_pred(path, frame)
            c['schemes']['toy']['refs'][arm] = {'path': str(path), 'sha256': t15.sha256_file(path)}
        self.c = c
        self.inner_sha = {}
        for f in range(5):
            a = np.flatnonzero(self.folds != f)
            inner = t15.inner_folds(self.X.iloc[a], self.y.iloc[a], kind, c)
            self.inner_sha[t15.unit_scheme('toy', f)] = __import__('hashlib').sha256(inner.astype(np.int8).tobytes()).hexdigest()
        self.design = dict(config=c, max_new_study_fits=75, inner_fold_sha256=self.inner_sha,
                           study_targets=[t15.inner_key('toy', a, f, j) for a in ARMS for f in range(5) for j in range(5)])
        self.out = self.tmp/'run'
        self.out.mkdir()
        t15.atomic_write(self.out/'ledger.json', dict(folds={}, decisions={}))
        (self.out/'run.log').write_text('')
        self.fo = {'toy': self.folds}

    def fit(self, arm, f, fit_fn, js=range(5), cache=None):
        return [t15.fit_inner_fold(self.X, self.y, self.ids, self.fo, self.out, 'toy', arm, f, j, self.design, fit_fn, cache)
                for j in js]

    def decide(self, arm, f):
        return t15.decide_unit(self.X, self.y, self.ids, self.fo, self.out, 'toy', arm, f, self.design)


@unittest.skipUnless(HAVE_LGBM, 'lightgbm not installed')
class RuleTests(unittest.TestCase):
    def test_zero_offset_is_plain_argmax_and_offset_adds_grade3(self):
        p = np.random.default_rng(1).dirichlet([1, 1, 1], 500)
        np.testing.assert_array_equal(t15.adjust_predict(p, 0.0), p.argmax(1) + 1)
        shares = [np.mean(t15.adjust_predict(p, b) == 3) for b in (0.0, 0.5, 1.0, 2.0)]
        self.assertTrue(all(a <= b for a, b in zip(shares, shares[1:])) and shares[-1] > shares[0])
        with self.assertRaises(ValueError):
            t15.adjust_predict(p, -0.1)

    def test_select_offset_picks_positive_b_when_grade3_underpredicted_and_zero_when_calibrated(self):
        rng = np.random.default_rng(2)
        y = rng.choice([1, 2, 3], 6000, p=[0.2, 0.4, 0.4])
        # grade-3 probability is systematically shrunk by 0.5 before renormalisation
        raw = np.eye(3)[y - 1]*0.55 + rng.dirichlet([2, 2, 2], len(y))*0.45
        shrunk = raw*np.array([1, 1, 0.5]); shrunk /= shrunk.sum(1, keepdims=True)
        grid = t15.config()['rule']['grid']
        b_shrunk, _ = t15.select_offset(y, shrunk, grid)
        self.assertGreater(b_shrunk, 0.2)
        y2 = rng.choice([1, 2, 3], 6000, p=[1/3, 1/3, 1/3])
        flat = np.tile([1/3, 1/3, 1/3], (6000, 1)) + 1e-9*rng.random((6000, 3))
        flat /= flat.sum(1, keepdims=True)
        self.assertEqual(t15.select_offset(y2, np.eye(3)[y2 - 1]*0.9 + 0.033, grid)[0], 0.0)  # perfect probs: no change
        self.assertEqual(t15.select_offset(y, shrunk, grid), t15.select_offset(y, shrunk, grid))  # deterministic

    def test_ties_go_to_smallest_b(self):
        p = np.tile([0.5, 0.3, 0.2], (50, 1)); y = np.ones(50, dtype=int)
        self.assertEqual(t15.select_offset(y, p, [0.0, 0.1, 0.2])[0], 0.0)

    def test_decide_has_no_outer_label_argument_and_b_ignores_outer_probabilities(self):
        self.assertEqual(list(inspect.signature(t15.decide).parameters), ['y_inner', 'p_inner', 'p_outer', 'grid'])
        rng = np.random.default_rng(3)
        y = rng.integers(1, 4, 800); p = rng.dirichlet([1, 1, 1], 800)
        grid = t15.config()['rule']['grid']
        b1, a1, _ = t15.decide(y, p, rng.dirichlet([1, 1, 1], 100), grid)
        b2, a2, _ = t15.decide(y, p, rng.dirichlet([1, 1, 1], 300), grid)
        self.assertEqual(b1, b2)

    def test_config_is_preregistered(self):
        c = t15.config()
        self.assertEqual(len(t15.targets(c)), 450)
        self.assertEqual(len({t15.inner_key(*t) for t in t15.targets(c)}), 450)
        self.assertEqual(c['rule']['grid'][0], 0.0)
        self.assertEqual(c['rule']['grid'][-1], 2.0)
        self.assertEqual(c['primary_schemes'], ['geo2_s42', 'geo2_s101', 'geo2_s202', 'geo2_s303'])
        self.assertEqual(c['outer_model_refits'], 0)
        self.assertEqual({s['manifest']['sha256'][:8] for s in c['schemes'].values()},
                         {'36a40ddf', '7877b21b', '9b3a4805', '26f42c0b', '7673dbeb', '1b4616ff'})


@unittest.skipUnless(HAVE_LGBM, 'lightgbm not installed')
class InnerFoldTests(unittest.TestCase):
    def test_group_inner_folds_do_not_split_groups_and_pass_gate(self):
        X, y, _ = data()
        c = t15.config()
        for kind, cols in (('group2', ['geo_level_1_id', 'geo_level_2_id']), ('group1', ['geo_level_1_id'])):
            folds = t15.inner_folds(X, y, kind, c)
            t15.inner_gate(X, y, folds, kind, c)
            g, _ = t15.groups(X, cols)
            for k in range(5):
                self.assertTrue(set(g[folds == k]).isdisjoint(set(g[folds != k])), kind)
            self.assertEqual(sorted(set(folds)), [0, 1, 2, 3, 4])
            np.testing.assert_array_equal(folds, t15.inner_folds(X, y, kind, c))  # deterministic

    def test_strat_inner_and_gate_failure(self):
        X, y, _ = data()
        c = t15.config()
        folds = t15.inner_folds(X, y, 'strat', c)
        t15.inner_gate(X, y, folds, 'strat', c)
        y_bad = y.copy(); y_bad[y_bad == 3] = 2; y_bad.iloc[:10] = 3  # fewer than 30 grade-3 rows
        with self.assertRaises(ValueError):
            t15.inner_gate(X, y_bad, t15.inner_folds(X, y_bad, 'strat', c), 'strat', c)
        bad = t15.inner_folds(X, y, 'group2', c).copy()
        g, _ = t15.groups(X, ['geo_level_1_id', 'geo_level_2_id'])
        row = np.flatnonzero(g == g[0])[0]
        bad[row] = (bad[row] + 1) % 5  # splits a group across inner folds
        with self.assertRaises(ValueError):
            t15.inner_gate(X, y, bad, 'group2', c)

    def test_inner_folds_depend_only_on_outer_train_rows(self):
        X, y, _ = data()
        c = t15.config()
        a = np.arange(0, 1200)
        base = t15.inner_folds(X.iloc[a], y.iloc[a], 'group2', c)
        y2 = y.copy(); y2.iloc[1200:] = 1  # change only rows outside the outer-training set
        np.testing.assert_array_equal(base, t15.inner_folds(X.iloc[a], y2.iloc[a], 'group2', c))


@unittest.skipUnless(HAVE_LGBM, 'lightgbm not installed')
class LeakFreeProtocolTests(unittest.TestCase):
    def test_inner_fits_never_touch_outer_validation_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = Env(tmp)
            calls = []
            env.fit('LR_C', 0, fake_fit(calls))
            outer_valid = set(np.flatnonzero(env.folds == 0).tolist())
            outer_train = set(np.flatnonzero(env.folds != 0).tolist())
            union_val = set()
            for _, _, a, b in calls:
                self.assertFalse((set(a.tolist()) | set(b.tolist())) & outer_valid)
                self.assertTrue(set(a.tolist()).isdisjoint(set(b.tolist())))
                union_val |= set(b.tolist())
            self.assertEqual(union_val, outer_train)  # inner OOF covers exactly the outer-training rows
            g, _ = t15.groups(env.X, ['geo_level_1_id', 'geo_level_2_id'])
            for _, _, a, b in calls:  # group-aware inner split: no group on both sides
                self.assertTrue(set(g[a]).isdisjoint(set(g[b])))

    def test_selected_b_is_invariant_to_outer_validation_labels(self):
        sel = []
        for scramble in (None, 0):
            with tempfile.TemporaryDirectory() as tmp:
                env = Env(tmp, scramble_fold=scramble)
                env.fit('LGBM_O', 0, fake_fit())
                entry, fresh = env.decide('LGBM_O', 0)
                rec = json.loads((env.out/'LGBM_O'.join(['toy/', ''])/'decision_f0.json').read_text())
                sel.append((rec['selected_b'], rec['inner_curve']))
                self.assertTrue(fresh)
                # outer labels only enter evaluation: the metrics differ, the tuning does not
                if scramble is not None:
                    self.assertNotEqual(rec['outer_adjusted']['macro_f1'], sel_f1)
                else:
                    sel_f1 = rec['outer_adjusted']['macro_f1']
        self.assertEqual(sel[0], sel[1])

    def test_decision_requires_all_five_inner_fits_and_exact_row_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = Env(tmp)
            env.fit('LR_C', 1, fake_fit(), js=range(4))
            with self.assertRaises(ValueError):
                env.decide('LR_C', 1)

    def test_outer_validation_rows_in_inner_oof_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = Env(tmp)
            env.fit('LR_C', 0, fake_fit())
            ledger_path = env.out/'toy'/'LR_C'/'f0_i0_pred.csv.gz'
            frame = t15.read_pred(ledger_path)
            frame.loc[0, 'building_id'] = int(env.ids[np.flatnonzero(env.folds == 0)[0]])  # an outer-validation id
            t15.write_pred(ledger_path, frame)  # also breaks the SHA: either guard must stop the study
            with self.assertRaises(ValueError):
                env.decide('LR_C', 0)

    def test_decision_artifacts_and_unadjusted_reproduces_saved_prediction(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = Env(tmp)
            env.fit('LR_C', 2, fake_fit())
            env.decide('LR_C', 2)
            d = env.out/'toy'/'LR_C'
            adj = t15.read_pred(d/'adjusted_f2_pred.csv.gz')
            ref = t15.read_pred(env.c['schemes']['toy']['refs']['LR_C']['path'])
            ref = ref[ref.fold == 2].reset_index(drop=True)
            np.testing.assert_array_equal(adj.y_pred_unadjusted.to_numpy(), ref.y_pred.to_numpy())
            rec = json.loads((d/'decision_f2.json').read_text())
            self.assertIn(rec['selected_b'], env.c['rule']['grid'])
            self.assertEqual(len(rec['inner_curve']), 21)
            self.assertTrue((env.out/'backup'/'toy'/'LR_C'/'decision_f2.json').exists())


@unittest.skipUnless(HAVE_LGBM, 'lightgbm not installed')
class CheckpointTests(unittest.TestCase):
    def test_resume_skips_completed_and_ledger_has_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = Env(tmp)
            calls = []
            with self.assertRaises(RuntimeError):
                env.fit('LGBM_O', 0, fake_fit(calls, fail_at=3))
            ledger = t15.read_ledger(env.out)
            status = {k: e['status'] for k, e in ledger['folds'].items()}
            self.assertEqual(sum(v == 'complete' for v in status.values()), 2)
            self.assertEqual(sum(v == 'running' for v in status.values()), 1)
            calls2 = []
            res = env.fit('LGBM_O', 0, fake_fit(calls2))
            self.assertEqual(len(calls2), 3)  # only the 3 unfinished inner fits ran
            self.assertEqual([fresh for _, fresh in res], [False, False, True, True, True])
            ledger = t15.read_ledger(env.out)
            self.assertTrue(all(e['status'] == 'complete' and e['artifact_sha256'] for e in ledger['folds'].values()))
            self.assertEqual(ledger['folds'][t15.inner_key('toy', 'LGBM_O', 0, 2)]['attempts'], 2)
            for e in ledger['folds'].values():  # every artifact has a byte-identical backup
                t15.check_saved(env.out, e)
            calls3 = []
            env.fit('LGBM_O', 0, fake_fit(calls3))
            self.assertEqual(calls3, [])

    def test_tampered_artifact_unregistered_key_and_nonconverged_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = Env(tmp)
            env.fit('LR_C', 0, fake_fit(), js=[0])
            (env.out/'toy'/'LR_C'/'f0_i0.json').write_text('{"changed": 1}')
            with self.assertRaises(ValueError):
                env.fit('LR_C', 0, fake_fit(), js=[0])
        with tempfile.TemporaryDirectory() as tmp:
            env = Env(tmp)
            with self.assertRaises(ValueError):
                t15.fit_inner_fold(env.X, env.y, env.ids, env.fo, env.out, 'toy', 'LR_C', 0, 9, env.design, fake_fit())
            with self.assertRaises(ValueError):
                env.fit('LR_C', 0, fake_fit(converged=False), js=[0])
            ledger = t15.read_ledger(env.out)
            self.assertEqual(ledger['folds'][t15.inner_key('toy', 'LR_C', 0, 0)]['status'], 'complete_nonconverged')
            with self.assertRaises(ValueError):  # study stays stopped, no retry
                env.fit('LR_C', 0, fake_fit(), js=[0])

    def test_fit_budget_is_enforced(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = Env(tmp)
            env.design['max_new_study_fits'] = 2
            env.fit('LR_C', 0, fake_fit(), js=[0, 1])
            with self.assertRaises(ValueError):
                env.fit('LR_C', 0, fake_fit(), js=[2])

    def test_sealed_inner_folds_are_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = Env(tmp)
            env.design['inner_fold_sha256'][t15.unit_scheme('toy', 0)] = '0'*64
            with self.assertRaises(ValueError):
                env.fit('LR_C', 0, fake_fit(), js=[0])

    def test_csv_round_trip_exact(self):
        p = np.random.default_rng(3).dirichlet([0.1, 0.1, 0.1], 200)
        frame = pd.DataFrame({'building_id': np.arange(200), 'y_true': 1, **{n: p[:, i] for i, n in enumerate(t15.PROBS)}})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'p.csv.gz'
            t15.write_pred(path, frame)
            np.testing.assert_array_equal(t15.read_pred(path)[t15.PROBS].to_numpy(), p)


@unittest.skipUnless(HAVE_LGBM, 'lightgbm not installed')
class RealArmSmallTests(unittest.TestCase):
    def test_fit_inner_arm_saves_and_reloads_each_arm(self):
        X, y, _ = data()
        c = t15.config()
        c['lgbm_params'] = dict(c['lgbm_params'], n_estimators=3, verbosity=-1)
        a, b = np.arange(1000), np.arange(1000, 1200)
        for arm, kind in (('LR_C', 'group2'), ('LGBM_O', 'group2'), ('LGBM_TE_G', 'group2'), ('LGBM_TE_G', 'strat'),
                          ('LGBM_TE_G', 'group1')):
            with tempfile.TemporaryDirectory() as tmp:
                r = t15.fit_inner_arm(arm, kind, X.iloc[a], y.iloc[a], X.iloc[b], c)
                self.assertEqual(r['p'].shape, (200, 3))
                np.testing.assert_allclose(r['p'].sum(1), 1.0, atol=1e-6)
                paths = r['save'](Path(tmp)/'m')
                self.assertTrue(all(p.exists() for p in paths))
                self.assertLessEqual(r['verify'](paths, X.iloc[b], r['p']), 1e-12, (arm, kind))


@unittest.skipUnless(HAVE_LGBM, 'lightgbm not installed')
class JudgeAndHoldoutTests(unittest.TestCase):
    def setUp(self):
        self.rules = t15.config()['rules']
        self.geo2 = dict(fold_deltas=[0.01]*20, seed_mean_deltas={'42': .01, '101': .01, '202': .01, '303': .01},
                         grade3_recall=(0.46, 0.52), grade1_recall=(0.6, 0.58), error31=(500, 550))
        self.strat = dict(mean_delta=-0.001)

    def test_candidate_and_each_part_can_fail(self):
        self.assertTrue(t15.judge_arm(self.rules, self.geo2, self.strat)['candidate'])
        cases = [('fold_deltas', [0.0029]*20, 'P1_mean_delta_ok'),
                 ('fold_deltas', [0.02]*13 + [-0.001]*7, 'P2_fold_wins_ok'),
                 ('seed_mean_deltas', {'42': .01, '101': -.01, '202': -.01, '303': .01}, 'P3_positive_seeds_ok'),
                 ('grade3_recall', (0.46, 0.485), 'G1_grade3_recall_gain_ok'),
                 ('error31', (500, 626), 'G2_error31_ratio_ok'),
                 ('grade1_recall', (0.6, 0.56), 'G3_grade1_recall_drop_ok')]
        for k, v, part in cases:
            g = dict(self.geo2, **{k: v})
            j = t15.judge_arm(self.rules, g, self.strat)
            self.assertFalse(j['parts'][part], (k, part))
            self.assertFalse(j['candidate'])
        j = t15.judge_arm(self.rules, self.geo2, dict(mean_delta=-0.0031))
        self.assertFalse(j['parts']['G4_stratified_not_worse_ok'] or j['candidate'])

    def test_boundaries_count(self):
        g = dict(self.geo2, fold_deltas=[0.005]*14 + [-0.0001]*6, grade3_recall=(0.46, 0.49), error31=(500, 625))
        j = t15.judge_arm(self.rules, g, dict(mean_delta=-0.003))
        self.assertTrue(j['parts']['P2_fold_wins_ok'] and j['parts']['G2_error31_ratio_ok']
                        and j['parts']['G4_stratified_not_worse_ok'])

    def test_holdout_guard(self):
        with self.assertRaises(ValueError):
            t15.guard('C:/x/earthquake/data/final_holdout/holdout.csv')
        t15.guard('C:/x/earthquake/data/development.csv')
        text = json.dumps(t15.config())
        self.assertNotIn('final_holdout/', text)
        c = t15.config()
        c['schemes']['geo1_s42']['refs']['LR_C']['path'] = 'C:/a/final_holdout/oof.csv.gz'
        p = ROOT/'tmp_cfg_should_not_exist.json'
        try:
            p.write_text(json.dumps(c))
            with self.assertRaises(ValueError):
                t15.config(p)
        finally:
            p.unlink(missing_ok=True)

    def test_no_file_under_final_holdout_is_opened_during_a_synthetic_run(self):
        opened = []
        real_open = open

        def spy(file, *a, **k):
            opened.append(str(file))
            return real_open(file, *a, **k)
        with tempfile.TemporaryDirectory() as tmp, mock.patch('builtins.open', spy):
            env = Env(tmp)
            env.fit('LR_C', 0, fake_fit())
            env.decide('LR_C', 0)
        self.assertTrue(opened)
        self.assertFalse([p for p in opened if 'final_holdout' in p.replace('\\', '/').lower()])

    def test_source_mentions_holdout_only_as_guard(self):
        src = (ROOT/'experiments/grade3_decision_rule/run.py').read_text(encoding='utf-8')
        lines = [ln for ln in src.splitlines() if 'final_holdout' in ln]
        self.assertTrue(all(('FORBIDDEN' in ln or 'earthquake/data' in ln) for ln in lines), lines)
        self.assertNotIn('.toarray()', src)


if __name__ == '__main__':
    unittest.main()
