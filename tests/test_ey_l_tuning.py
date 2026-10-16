"""T36 synthetic contracts for `experiments/ey_l_tuning/run.py`.

Synthetic frames only: no real development data, no holdout access.  Small LightGBM fits on a few hundred synthetic rows
are test fixtures outside the study budget.
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
from experiments.ey_l_tuning import run as R  # noqa: E402
from experiments.ey_pipeline_benchmark import run as T33  # noqa: E402
from experiments.ey_pipeline_v2 import features as F  # noqa: E402
from test_ey_pipeline_v2 import synth  # noqa: E402

CONFIG = R.HERE/'config.json'


def small_config():
    c = copy.deepcopy(json.loads(CONFIG.read_text(encoding='utf-8')))
    for cand in c['candidates']:
        c['candidates'][cand] = dict(c['candidates'][cand], n_estimators=5 if cand in ('C0', 'C1', 'C2') else 10)
    c['fixed_lgbm_params'] = dict(c['fixed_lgbm_params'], min_child_samples=5, n_jobs=1)
    c['threads'] = dict(c['threads'], model_n_jobs=1)
    return c


class ConfigTests(unittest.TestCase):
    def test_config_is_the_registered_plan(self):
        c = R.config()
        self.assertEqual(c['plan']['sha256'], R.sha256_file(ROOT/c['plan']['path']))
        self.assertEqual(R.CANDIDATES, ['C0', 'C1', 'C2', 'C3', 'C4', 'C5'])
        self.assertEqual(R.model_params(c, 'C0'), dict(random_state=42, n_jobs=4, deterministic=True, verbose=-1,
                                                       learning_rate=0.05, min_child_samples=50, subsample=0.8,
                                                       subsample_freq=1, colsample_bytree=0.5, num_leaves=127,
                                                       n_estimators=800))
        # C0 must be exactly the sealed T35 L setting
        t35 = json.loads((ROOT/'experiments/ey_pipeline_v2/config.json').read_text(encoding='utf-8'))
        self.assertEqual(R.model_params(c, 'C0'), t35['lgbm_params'])
        self.assertEqual(c['weights'], t35['weights'])

    def test_changed_config_is_refused(self):
        base = json.loads(CONFIG.read_text(encoding='utf-8'))
        for mutate in (lambda c: c['candidates']['C5'].update(num_leaves=511),
                       lambda c: c['candidate_order'].reverse(),
                       lambda c: c['budget'].update(max_fits=200),
                       lambda c: c['inner'].update(n_splits=5),
                       lambda c: c['rules']['J1'].update(mean_delta=0.001),
                       lambda c: c['fixed_lgbm_params'].update(learning_rate=0.1),
                       lambda c: c.update(holdout_access=True)):
            c = copy.deepcopy(base)
            mutate(c)
            with tempfile.TemporaryDirectory() as td:
                p = Path(td)/'config.json'
                p.write_text(json.dumps(c), encoding='utf-8')
                with self.assertRaises(ValueError):
                    R.config(p)

    def test_targets_and_budget(self):
        t = R.inner_targets()
        self.assertEqual(len(t), 90)
        self.assertEqual(len({R.inner_key(*x) for x in t}), 90)
        self.assertEqual(t[:6], [(0, 0, cand) for cand in R.CANDIDATES])
        self.assertEqual(R.config()['budget']['planned_fits_max'], 90 + 5)

    def test_holdout_path_refused(self):
        with self.assertRaises(ValueError):
            R.require_not_holdout('C:/x/earthquake/data/final_holdout/a.csv')


class InnerSplitTests(unittest.TestCase):
    def test_partition_stratified_and_deterministic(self):
        rng = np.random.default_rng(1)
        y = rng.choice([1, 2, 3], 3000, p=[.1, .57, .33])
        c = R.config()
        parts = R.inner_split(y, c)
        self.assertEqual(len(parts), 3)
        allv = np.concatenate([va for _, va in parts])
        self.assertEqual(sorted(allv.tolist()), list(range(3000)))
        for tr, va in parts:
            self.assertFalse(set(tr) & set(va))
            self.assertEqual(len(tr) + len(va), 3000)
            share = np.bincount(y[va] - 1, minlength=3) / len(va)
            np.testing.assert_allclose(share, np.bincount(y - 1, minlength=3) / 3000, atol=0.005)
        again = R.inner_split(y, c)
        for (a1, b1), (a2, b2) in zip(parts, again):
            np.testing.assert_array_equal(a1, a2)
            np.testing.assert_array_equal(b1, b2)


class SelectionRuleTests(unittest.TestCase):
    def test_highest_wins(self):
        s = dict(C0=.70, C1=.69, C2=.71, C3=.705, C4=.709, C5=.7099)
        self.assertEqual(R.select(s), 'C2')

    def test_exact_tie_goes_to_registered_order(self):
        self.assertEqual(R.select(dict(C0=.71, C1=.70, C2=.70, C3=.71, C4=.70, C5=.71)), 'C0')
        self.assertEqual(R.select(dict(C0=.69, C1=.70, C2=.70, C3=.71, C4=.71, C5=.70)), 'C3')

    def test_missing_candidate_refused(self):
        with self.assertRaises(ValueError):
            R.select(dict(C0=.7, C1=.7))


class J1RuleTests(unittest.TestCase):
    rule = R.config()['rules']['J1']

    def label(self, tuned, base):
        return T33.d1_label(tuned, base, self.rule)['label']

    def test_labels(self):
        base = [0.70] * 5
        self.assertEqual(self.label([0.704, 0.704, 0.704, 0.70, 0.704], base), 'IMPROVED_DESCRIPTIVE')   # mean .0032, 4 wins
        self.assertEqual(self.label([0.70, 0.70, 0.70, 0.72, 0.70], base), 'NOT_MET')                   # mean .004, 1 win
        self.assertEqual(self.label(base, base), 'NOT_MET')                                             # all C0: zero deltas
        self.assertEqual(self.label([0.696, 0.696, 0.696, 0.696, 0.70], base), 'WORSENED_DESCRIPTIVE')
        self.assertEqual(self.label([0.7029, 0.7029, 0.7029, 0.7029, 0.7029], base), 'NOT_MET')         # just below


class InnerFeatureLabelTests(unittest.TestCase):
    def test_inner_validation_labels_do_not_reach_the_inner_matrix(self):
        fr = synth(900, seed=7)
        tr, va = R.inner_split(fr[F.GRADE].to_numpy(), R.config())[0]
        probe = F.label_probe(fr.iloc[tr].reset_index(drop=True), fr.iloc[va].reset_index(drop=True))
        self.assertTrue(probe['passes'])
        self.assertGreater(probe['outer_labels_changed_share'], 0.3)


def fake_ctx(out, c, n_train=300, n_valid=120, seed=5):
    fr = synth(n_train + n_valid, seed=seed)
    ids = pd.Series(np.arange(10, 10 + len(fr)) * 3)
    y = fr[F.GRADE]
    a, b = np.arange(n_train), np.arange(n_train, n_train + n_valid)
    X_tr, X_va, _, _, diag = F.fold_matrices(fr.iloc[a].reset_index(drop=True),
                                             fr.iloc[b].drop(columns=[F.GRADE]).reset_index(drop=True))
    (out/'shared').mkdir(parents=True)
    R.write_matrix(out/'shared', 'inner_0_0', X_va, ids.to_numpy()[b], F.matrix_sha256(X_tr), diag, {})
    m = dict(a=a, b=b, train=X_tr, valid=X_va, train_sha=F.matrix_sha256(X_tr), valid_sha=F.matrix_sha256(X_va),
             y0=y.to_numpy()[a] - 1, diagnostics=diag, stem='inner_0_0')
    R.atomic_write(out/'ledger.json', dict(study_id=R.STUDY_ID, units={}, features={}, selections={}))
    return SimpleNamespace(out=out, c=c, y=y, ids=ids), m


class FitAndSaveTests(unittest.TestCase):
    def test_saved_unit_reloads_and_is_backed_up(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            ctx, m = fake_ctx(out, small_config())
            e = R.fit_and_save(ctx, R.inner_key(0, 0, 'C4'), m, 'C4', out/'inner'/'f0', 'f0_k0_C4',
                               dict(outer_fold=0, inner_fold=0, pred_columns={'inner_fold': 0, 'outer_fold': 0}))
            self.assertEqual(e['status'], 'complete')
            self.assertEqual(e['attempts'], 1)
            R.check_saved(out, e)
            rec = json.loads((out/'inner'/'f0'/'f0_k0_C4.json').read_text(encoding='utf-8'))
            self.assertEqual(rec['params']['n_estimators'], 10)
            self.assertLessEqual(rec['reload_probability_max_abs_diff'], 1e-12)
            pred = R.read_pred(out/'inner'/'f0'/'f0_k0_C4_pred.csv.gz')
            self.assertEqual(list(pred.columns[:3]), ['building_id', 'outer_fold', 'inner_fold'])
            self.assertEqual(R.fits_started(R.read_ledger(out)), 1)
            # a tampered artifact is caught
            p = out/'inner'/'f0'/'f0_k0_C4_pred.csv.gz'
            p.write_bytes(p.read_bytes() + b'x')
            with self.assertRaises(ValueError):
                R.check_saved(out, e)

    def test_budget_and_attempt_caps(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            c = small_config()
            c['budget'] = dict(c['budget'], max_fits=1)
            ctx, m = fake_ctx(out, c)
            R.begin_fit(ctx, 'I/0/0/C0')
            with self.assertRaises(ValueError):
                R.begin_fit(ctx, 'I/0/0/C1')        # budget of 1 fit exhausted
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            ctx, m = fake_ctx(out, small_config())
            R.begin_fit(ctx, 'I/0/0/C0')
            R.begin_fit(ctx, 'I/0/0/C0')            # manual resume: attempt 2
            with self.assertRaises(ValueError):
                R.begin_fit(ctx, 'I/0/0/C0')        # at most 2 attempts per unit


class PooledSelectionTests(unittest.TestCase):
    def test_pooled_scores_come_from_saved_inner_predictions(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            rng = np.random.default_rng(3)
            y = rng.choice([1, 2, 3], 90)
            ledger = dict(study_id=R.STUDY_ID, units={}, features={}, selections={})
            truth = {}
            for cand_i, cand in enumerate(R.CANDIDATES):
                preds = y.copy()
                wrong = rng.random(90) < 0.1 + 0.05 * cand_i    # C0 has the fewest errors by construction
                preds[wrong] = (preds[wrong] % 3) + 1
                truth[cand] = R.score(y, preds)['macro_f1']
                d = out/'inner'/'f0'
                d.mkdir(parents=True, exist_ok=True)
                (out/'backup'/'inner'/'f0').mkdir(parents=True, exist_ok=True)
                for k in range(3):
                    sl = slice(30 * k, 30 * (k + 1))
                    fr = pd.DataFrame(dict(building_id=np.arange(90)[sl], outer_fold=0, inner_fold=k, y_true=y[sl],
                                           y_pred=preds[sl], p_grade_1=0.3, p_grade_2=0.4, p_grade_3=0.3))
                    name = f'f0_k{k}_{cand}_pred.csv.gz'
                    R.write_pred(d/name, fr)
                    R.backup_files(out, 'inner/f0', [name])
                    ledger['units'][R.inner_key(0, k, cand)] = dict(
                        dir='inner/f0', status='complete', attempts=1, artifact_sha256={name: R.sha256_file(d/name)})
            R.atomic_write(out/'ledger.json', ledger)
            scores, n = R.pooled_inner_scores(out, 0)
            self.assertEqual(n, 90)
            for cand in R.CANDIDATES:
                self.assertAlmostEqual(scores[cand], truth[cand], places=12)
            self.assertEqual(R.select(scores), max(R.CANDIDATES, key=lambda k2: (truth[k2], -R.CANDIDATES.index(k2))))


class ProbeTests(unittest.TestCase):
    def test_model_probe_all_candidates(self):
        res = R.probe_models(R.config())
        self.assertEqual(set(res['candidates']), set(R.CANDIDATES))
        for v in res['candidates'].values():
            self.assertTrue(v['repeat_identical'])
            self.assertTrue(v['reload_within_tolerance'])


if __name__ == '__main__':
    unittest.main()
