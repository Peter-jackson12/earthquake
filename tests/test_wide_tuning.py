"""T38 synthetic contracts for `experiments/wide_tuning/run.py`.

Synthetic frames only: no real development data, no holdout access.  Small fits are test fixtures outside the budget.
"""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiments.wide_tuning import run as R  # noqa: E402
from experiments.required_tuning import run as T37  # noqa: E402
from test_required_tuning import raw_frame  # noqa: E402

CONFIG = R.HERE/'config.json'


def small_config():
    c = copy.deepcopy(json.loads(CONFIG.read_text(encoding='utf-8')))
    c['fixed_params']['RF'] = dict(c['fixed_params']['RF'], n_estimators=5, n_jobs=1)
    c['threads'] = dict(c['threads'], rf_n_jobs=1)
    for cand, v in c['candidates']['GBC'].items():
        c['candidates']['GBC'][cand] = dict(v, n_estimators=v['n_estimators'] // 60)   # 300 -> 5, 600 -> 10
    return c


class ConfigTests(unittest.TestCase):
    def test_config_is_the_registered_plan(self):
        c = R.config()
        self.assertEqual(c['plan']['sha256'], R.sha256_file(ROOT/c['plan']['path']))
        self.assertEqual({m: len(c['candidates'][m]) for m in R.MODELS}, dict(DT=20, RF=8, GBC=4))
        self.assertEqual(len(R.inner_targets(c)), 450)
        self.assertEqual(c['budget']['planned_fits'], 450 + 15)

    def test_anchors_are_the_t37_most_selected_settings(self):
        c, t37 = R.config(), T37.config()
        self.assertEqual(R.model_params(c, 'DT', 'DT0'), T37.model_params(t37, 'DT', 'DT8'))
        self.assertEqual({k: v for k, v in R.model_params(c, 'RF', 'RF0').items()},
                         T37.model_params(t37, 'RF', 'RF7'))
        gbc = dict(R.model_params(c, 'GBC', 'GBC0'))
        self.assertEqual(gbc.pop('subsample'), 1.0)                 # sklearn default, made explicit
        self.assertEqual(gbc, T37.model_params(t37, 'GBC', 'GBC3'))

    def test_grid_contains_the_wider_values(self):
        c = R.config()
        dt = {(v['max_depth'], v['min_samples_leaf']) for v in c['candidates']['DT'].values()}
        self.assertEqual(dt, {(d, l) for d in (16, 20, 24, 32, None) for l in (5, 20, 50, 100)})
        rf = {(v['max_features'], v['max_depth']) for v in c['candidates']['RF'].values()}
        self.assertEqual(rf, {(m, d) for m in (0.3, 0.45, 0.6, 0.8) for d in (20, 30)})
        self.assertEqual(R.fit_groups(c, 'GBC'), ['d5', 'd7'])
        self.assertEqual(R.group_candidates(c, 'GBC', 'd7'), ['GBC2', 'GBC3'])

    def test_changed_config_is_refused(self):
        base = json.loads(CONFIG.read_text(encoding='utf-8'))
        for mutate in (lambda c: c['candidates']['DT']['DT19'].update(max_depth=64),
                       lambda c: c['fixed_params']['GBC'].update(subsample=0.8),
                       lambda c: c['gbc_groups'].update(d5=['GBC1', 'GBC0']),
                       lambda c: c.update(class_weighting='balanced'),
                       lambda c: c['budget'].update(max_fits=999),
                       lambda c: c['rules']['J1'].update(mean_delta=0.001),
                       lambda c: c['control']['fits']['RF'].update(fold=0),
                       lambda c: c.update(holdout_access=True)):
            c = copy.deepcopy(base)
            mutate(c)
            with tempfile.TemporaryDirectory() as td:
                p = Path(td)/'config.json'
                p.write_text(json.dumps(c), encoding='utf-8')
                with self.assertRaises(ValueError):
                    R.config(p)


class StagedTests(unittest.TestCase):
    def test_first_stages_equal_a_separate_shorter_fit(self):
        X, y = raw_frame(400, seed=4)
        res = R.staged_identity(R.config(), X, y)
        self.assertTrue(res['passes'])
        self.assertEqual(set(res['depths']), {'d5', 'd7'})

    def test_group_probabilities_give_both_candidates(self):
        c = small_config()
        X, y = raw_frame(400, seed=6)
        pipe, _, _, _ = R.fit(c, 'GBC', 'GBC1', X.iloc[:300], y.iloc[:300])
        probs = R.probabilities(c, 'GBC', 'd5', pipe, X.iloc[300:])
        self.assertEqual(set(probs), {'GBC0', 'GBC1'})
        np.testing.assert_array_equal(probs['GBC1'], pipe.predict_proba(X.iloc[300:]))
        short, _, _, _ = R.fit(c, 'GBC', 'GBC0', X.iloc[:300], y.iloc[:300])
        np.testing.assert_array_equal(probs['GBC0'], short.predict_proba(X.iloc[300:]))


def fake_ctx(out, c, n=420):
    X, y = raw_frame(n)
    ids = pd.Series(np.arange(n) * 7 + 3)
    folds = np.arange(n) % 5
    R.atomic_write(out/'ledger.json', dict(study_id=R.STUDY_ID, units={}, selections={}))
    return R.Context(X, y, ids, folds, out, dict(config=c))


class UnitTests(unittest.TestCase):
    def test_inner_units_save_reload_and_pool(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            c = small_config()
            ctx = fake_ctx(out, c)
            for k in range(3):
                for g in R.fit_groups(c, 'GBC'):
                    e = R.inner_unit(ctx, 'GBC', 0, k, g)
                    self.assertEqual(e['candidates'], R.group_candidates(c, 'GBC', g))
                    R.check_saved(out, e)
            self.assertEqual(R.fits_started(R.read_ledger(out)), 6)       # 2 depth groups x 3 inner folds
            scores, n = R.pooled_inner_scores(out, c, 'GBC', 0)
            self.assertEqual(set(scores), {'GBC0', 'GBC1', 'GBC2', 'GBC3'})
            self.assertEqual(n, len(ctx.positions(0)[0]))
            sel = R.selection(ctx, 'GBC', 0)
            self.assertEqual(sel['selected'], R.select(scores, R.candidates(c, 'GBC')))
            e = R.outer_unit(ctx, 'GBC', 0)
            self.assertEqual(e['fitted_candidate'], sel['selected'])
            self.assertEqual(R.fits_started(R.read_ledger(out)), 7)
            again = R.inner_unit(ctx, 'GBC', 0, 0, 'd5')                    # complete: verified, not refit
            self.assertEqual(again['status'], 'complete')
            self.assertEqual(R.fits_started(R.read_ledger(out)), 7)
            p = out/'GBC'/'inner'/'f0'/R.pred_name(0, 1, 'GBC2')
            p.write_bytes(p.read_bytes() + b'x')
            with self.assertRaises(ValueError):
                R.pooled_inner_scores(out, c, 'GBC', 0)

    def test_dt_and_rf_units(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            c = small_config()
            ctx = fake_ctx(out, c)
            for m, g in (('DT', 'DT19'), ('RF', 'RF7')):
                e = R.inner_unit(ctx, m, 1, 2, g)
                rec = json.loads((out/m/'inner'/'f1'/f'f1_k2_{g}.json').read_text(encoding='utf-8'))
                self.assertEqual(rec['fitted_params'], R.model_params(c, m, g))
                self.assertLessEqual(rec['reload_probability_max_abs_diff'], 1e-12)
                R.check_saved(out, e)


class ProbeTests(unittest.TestCase):
    def test_probe_all_candidates(self):
        X, y = raw_frame(300, seed=9)
        res = R.probe(R.config(), X, y)
        self.assertEqual(len(res['candidates']), 32)
        self.assertTrue(all(v['passes'] for v in res['candidates'].values()))


if __name__ == '__main__':
    unittest.main()
