"""T37 synthetic contracts for `experiments/required_tuning/run.py`.

Synthetic frames only: no real development data, no holdout access.  Small fits on a few hundred synthetic rows are test
fixtures outside the study budget.
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
from preprocess import CATEGORICAL, FEATURES  # noqa: E402
from experiments.required_tuning import run as R  # noqa: E402
from experiments.ey_pipeline_benchmark import run as T33  # noqa: E402
from test_ey_pipeline_v2 import synth  # noqa: E402

CONFIG = R.HERE/'config.json'


def raw_frame(n, seed=5):
    """Synthetic development-like frame: 38 raw inputs (categoricals as text) and labels 1/2/3."""
    fr = synth(n, seed=seed)
    y = fr.pop('damage_grade')
    X = fr[FEATURES].copy()
    for col in CATEGORICAL:
        X[col] = X[col].astype(str)
    return X.reset_index(drop=True), y.reset_index(drop=True)


def small_config():
    c = copy.deepcopy(json.loads(CONFIG.read_text(encoding='utf-8')))
    c['fixed_params']['RF'] = dict(c['fixed_params']['RF'], n_estimators=5, n_jobs=1)
    for cand in c['candidates']['GBC']:
        c['candidates']['GBC'][cand] = dict(c['candidates']['GBC'][cand], n_estimators=5)
    c['threads'] = dict(c['threads'], rf_n_jobs=1)
    return c


class ConfigTests(unittest.TestCase):
    def test_config_is_the_registered_plan(self):
        c = R.config()
        self.assertEqual(c['plan']['sha256'], R.sha256_file(ROOT/c['plan']['path']))
        self.assertEqual(c['models'], ['LR', 'DT', 'GBC', 'RF'])
        self.assertEqual({m: len(c['candidates'][m]) for m in R.MODELS}, dict(LR=6, DT=9, GBC=4, RF=9))
        self.assertIsNone(c['class_weighting'])
        self.assertIsNone(c['sampling'])

    def test_baseline_candidates_are_the_a2_settings(self):
        c = R.config()
        base = json.loads((ROOT/'configs/baseline.json').read_text(encoding='utf-8'))['models']
        fb = json.loads((ROOT/'configs/feature_budget.json').read_text(encoding='utf-8'))['models']
        self.assertEqual(R.model_params(c, 'LR', 'LR0'), base['lr'])
        self.assertEqual(R.model_params(c, 'DT', 'DT0'), base['dt'])
        self.assertEqual(R.model_params(c, 'GBC', 'GBC0'), fb['gbc'])
        rf = R.model_params(c, 'RF', 'RF0')
        self.assertEqual({k: v for k, v in rf.items() if k != 'n_jobs'}, fb['rf'])   # n_jobs does not change RF results
        for m in R.MODELS:
            self.assertEqual(R.baseline_candidate(c, m), f'{m}0')

    def test_changed_config_is_refused(self):
        base = json.loads(CONFIG.read_text(encoding='utf-8'))
        for mutate in (lambda c: c['candidates']['LR']['LR5'].update(C=100.0),
                       lambda c: c['fixed_params']['GBC'].update(learning_rate=0.05),
                       lambda c: c.update(class_weighting='balanced'),
                       lambda c: c['budget'].update(max_fits=999),
                       lambda c: c['inner'].update(random_state=0),
                       lambda c: c['rules']['J1'].update(fold_wins_min=3),
                       lambda c: c['control'].update(lr_probability_atol=1e-3),
                       lambda c: c.update(holdout_access=True)):
            c = copy.deepcopy(base)
            mutate(c)
            with tempfile.TemporaryDirectory() as td:
                p = Path(td)/'config.json'
                p.write_text(json.dumps(c), encoding='utf-8')
                with self.assertRaises(ValueError):
                    R.config(p)

    def test_targets_and_budget(self):
        c = R.config()
        t = R.inner_targets(c)
        self.assertEqual(len(t), 420)
        self.assertEqual(len({R.inner_key(*x) for x in t}), 420)
        self.assertEqual([x[0] for x in t[:1]], ['LR'])
        self.assertEqual(c['budget']['planned_fits_max'], 420 + 20)

    def test_holdout_path_refused(self):
        with self.assertRaises(ValueError):
            R.require_not_holdout('C:/x/earthquake/data/final_holdout/a.csv')


class RuleTests(unittest.TestCase):
    def test_selection_and_ties(self):
        order = ['DT0', 'DT1', 'DT2']
        self.assertEqual(R.select(dict(DT0=.6, DT1=.62, DT2=.61), order), 'DT1')
        self.assertEqual(R.select(dict(DT0=.62, DT1=.62, DT2=.61), order), 'DT0')
        self.assertEqual(R.select(dict(DT0=.6, DT1=.62, DT2=.62), order), 'DT1')
        with self.assertRaises(ValueError):
            R.select(dict(DT0=.6), order)

    def test_j1_labels(self):
        rule = R.config()['rules']['J1']
        base = [0.6] * 5
        lab = lambda t: T33.d1_label(t, base, rule)['label']  # noqa: E731
        self.assertEqual(lab([0.604, 0.604, 0.604, 0.6, 0.604]), 'IMPROVED_DESCRIPTIVE')
        self.assertEqual(lab(base), 'NOT_MET')
        self.assertEqual(lab([0.6, 0.6, 0.6, 0.62, 0.6]), 'NOT_MET')
        self.assertEqual(lab([0.596, 0.596, 0.596, 0.596, 0.6]), 'WORSENED_DESCRIPTIVE')

    def test_inner_split_partition(self):
        y = np.random.default_rng(1).choice([1, 2, 3], 3000, p=[.1, .57, .33])
        parts = R.inner_split(y, R.config())
        allv = np.concatenate([va for _, va in parts])
        self.assertEqual(sorted(allv.tolist()), list(range(3000)))
        for tr, va in parts:
            self.assertFalse(set(tr) & set(va))


def fake_ctx(out, c, n=420):
    X, y = raw_frame(n)
    ids = pd.Series(np.arange(n) * 7 + 3)
    folds = np.arange(n) % 5
    R.atomic_write(out/'ledger.json', dict(study_id=R.STUDY_ID, units={}, selections={}))
    return SimpleNamespace(out=out, c=c, X=X, y=y, ids=ids, folds=folds)


class FitAndSaveTests(unittest.TestCase):
    def test_each_model_saves_reloads_and_is_backed_up(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            c = small_config()
            ctx = fake_ctx(out, c)
            a, b = np.flatnonzero(ctx.folds != 0), np.flatnonzero(ctx.folds == 0)
            for m, cand in (('LR', 'LR3'), ('DT', 'DT4'), ('GBC', 'GBC2'), ('RF', 'RF6')):
                e = R.fit_and_save(ctx, R.inner_key(m, 0, 0, cand), m, cand, a, b, out/m/'inner'/'f0', f'f0_k0_{cand}',
                                   dict(outer_fold=0, inner_fold=0, pred_columns={'inner_fold': 0, 'outer_fold': 0}))
                self.assertEqual(e['status'], 'complete')
                R.check_saved(out, e)
                rec = json.loads((out/m/'inner'/'f0'/f'f0_k0_{cand}.json').read_text(encoding='utf-8'))
                self.assertEqual(rec['params'], R.model_params(c, m, cand))
                self.assertLessEqual(rec['reload_probability_max_abs_diff'], 1e-12)
                self.assertEqual(rec['valid_rows'], len(b))
            self.assertEqual(R.fits_started(R.read_ledger(out)), 4)
            p = out/'RF'/'inner'/'f0'/'f0_k0_RF6_pred.csv.gz'
            p.write_bytes(p.read_bytes() + b'x')
            with self.assertRaises(ValueError):
                R.check_saved(out, R.read_ledger(out)['units'][R.inner_key('RF', 0, 0, 'RF6')])

    def test_preprocessing_is_fitted_on_training_rows_only(self):
        c = small_config()
        X, y = raw_frame(400)
        X.loc[300:, 'foundation_type'] = 'zz_unseen'     # category present only in the "validation" rows
        pipe, _, _, _ = R.fit_pipeline(c, 'LR', 'LR0', X.iloc[:300], y.iloc[:300])
        enc = pipe['preprocess']['columns'].named_transformers_['categorical']
        vocab = dict(zip(CATEGORICAL, enc.categories_))
        self.assertNotIn('zz_unseen', vocab['foundation_type'])
        scaler = pipe['preprocess']['columns'].named_transformers_['numeric']['scale']
        np.testing.assert_allclose(scaler.mean_[1], X.iloc[:300]['age'].astype(np.float32).mean(), rtol=1e-6)

    def test_budget_and_attempt_caps(self):
        with tempfile.TemporaryDirectory() as td:
            c = small_config()
            c['budget'] = dict(c['budget'], max_fits=1)
            ctx = fake_ctx(Path(td), c)
            R.begin_fit(ctx, 'LR/I/0/0/LR0')
            with self.assertRaises(ValueError):
                R.begin_fit(ctx, 'LR/I/0/0/LR1')
        with tempfile.TemporaryDirectory() as td:
            ctx = fake_ctx(Path(td), small_config())
            R.begin_fit(ctx, 'LR/I/0/0/LR0')
            R.begin_fit(ctx, 'LR/I/0/0/LR0')
            with self.assertRaises(ValueError):
                R.begin_fit(ctx, 'LR/I/0/0/LR0')


class PooledSelectionTests(unittest.TestCase):
    def test_pooled_scores_come_from_saved_inner_predictions(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            c = R.config()
            rng = np.random.default_rng(3)
            y = rng.choice([1, 2, 3], 90)
            ledger = dict(study_id=R.STUDY_ID, units={}, selections={})
            truth = {}
            for i, cand in enumerate(R.candidates(c, 'GBC')):
                preds = y.copy()
                wrong = rng.random(90) < 0.3 - 0.05 * i       # GBC3 has the fewest errors by construction
                preds[wrong] = (preds[wrong] % 3) + 1
                truth[cand] = R.score(y, preds)['macro_f1']
                d = out/'GBC'/'inner'/'f1'
                d.mkdir(parents=True, exist_ok=True)
                for k in range(3):
                    sl = slice(30 * k, 30 * (k + 1))
                    fr = pd.DataFrame(dict(building_id=np.arange(90)[sl], outer_fold=1, inner_fold=k, y_true=y[sl],
                                           y_pred=preds[sl], p_grade_1=0.3, p_grade_2=0.4, p_grade_3=0.3))
                    name = f'f1_k{k}_{cand}_pred.csv.gz'
                    R.write_pred(d/name, fr)
                    R.backup_files(out, 'GBC/inner/f1', [name])
                    ledger['units'][R.inner_key('GBC', 1, k, cand)] = dict(
                        dir='GBC/inner/f1', status='complete', attempts=1, artifact_sha256={name: R.sha256_file(d/name)})
            R.atomic_write(out/'ledger.json', ledger)
            scores, n = R.pooled_inner_scores(out, c, 'GBC', 1)
            self.assertEqual(n, 90)
            for cand in truth:
                self.assertAlmostEqual(scores[cand], truth[cand], places=12)
            order = R.candidates(c, 'GBC')
            self.assertEqual(R.select(scores, order), max(order, key=lambda k2: (truth[k2], -order.index(k2))))


class ProbeTests(unittest.TestCase):
    def test_probe_all_candidates(self):
        X, y = raw_frame(300, seed=9)
        res = R.probe(R.config(), X, y)
        self.assertEqual(len(res['candidates']), 28)
        for v in res['candidates'].values():
            self.assertTrue(v['repeat_within_tolerance'])
            self.assertTrue(v['reload_within_tolerance'])


if __name__ == '__main__':
    unittest.main()
