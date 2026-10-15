"""T20 전용 합성 검사. 개발 자료와 홀드아웃을 읽지 않는다."""
import copy
import inspect
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'tests'))
from test_modeling import synthetic
from experiments.lgbm_grade3_weight import run as t20


def fake_fit(calls=None, fail=False, invalid=False, reload_diff=0):
    def fit(Xa, ya, Xb, c):
        if calls is not None:
            calls.append((Xa.index.to_numpy(), np.asarray(ya), Xb.index.to_numpy()))
        if fail:
            raise RuntimeError('합성 중단')
        p = np.full((len(Xb), 3), 0.05)
        p[np.arange(len(Xb)), Xb.index.to_numpy() % 3] = 0.9
        if invalid:
            p[0, 0] = np.nan

        def save(prefix):
            path = Path(f'{prefix}_model.txt')
            with path.open('x', encoding='utf-8') as f:
                f.write('합성 모델')
            return [path]

        return dict(p=p, fit_seconds=0.01, save=save, verify=lambda *args: reload_diff)
    return fit


class Env:
    def __init__(self, root):
        self.out = Path(root)/'run'
        self.out.mkdir()
        (self.out/'backup').mkdir()
        t20.atomic_write(self.out/'ledger.json', dict(fits={}))
        self.X, self.y, self.ids, _ = synthetic(450)
        self.fo = np.arange(len(self.X)) % 5
        self.c = copy.deepcopy(t20.config())
        p = np.full((len(self.X), 3), 0.05)
        p[np.arange(len(self.X)), (self.y.to_numpy()+1) % 3] = 0.9
        self.ref = pd.DataFrame({t20.ID: self.ids, 'fold': self.fo, 'y_true': self.y, 'y_pred': p.argmax(1)+1,
                                 **{n: p[:, j] for j, n in enumerate(t20.PROBS)}})
        self.design = dict(config=self.c, code_commit='a'*40)
        self.inputs = (self.X, self.y, self.ids, {s: self.fo for s in self.c['schemes']},
                       {s: self.ref for s in self.c['schemes']}, self.design)

    def fit(self, scheme='geo2_s42', f=0, fit_fn=None):
        a, b = np.flatnonzero(self.fo != f), np.flatnonzero(self.fo == f)
        return t20.checkpoint_fit(self.out, scheme, f, self.X.iloc[a], self.y.iloc[a], self.X.iloc[b],
                                  self.y.iloc[b], self.ids.iloc[b], self.c, fit_fn or fake_fit())


class WeightTests(unittest.TestCase):
    def test_frozen_config_and_exact_reference_paths_hashes(self):
        c, b = t20.config(), t20.t15.config()
        self.assertEqual([s['seed'] for s in c['schemes'].values()], [42, 101, 202, 303])
        self.assertEqual(c['lgbm_params'], b['lgbm_params'])
        self.assertEqual(c['lgbm_params']['n_estimators'], 300)
        self.assertEqual(c['lgbm_params']['num_leaves'], 31)
        self.assertEqual(c['max_new_study_fits'], 20)
        for s in c['schemes']:
            self.assertEqual(c['schemes'][s]['manifest'], b['schemes'][s]['manifest'])
            self.assertEqual(c['schemes'][s]['refs']['LGBM_O'], b['schemes'][s]['refs']['LGBM_O'])

    def test_config_rejects_weight_model_rule_and_seed_drift(self):
        mutations = [('sample_weight_by_actual_grade', {'1': 1., '2': 1., '3': 2.}),
                     ('class_weight', 'balanced'), ('max_new_study_fits', 21),
                     ('decision_rule', 'grade3_offset'), ('early_stopping', True)]
        for name, value in mutations:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as d:
                c = copy.deepcopy(t20.config())
                c[name] = value
                path = Path(d)/'config.json'
                t20.write(path, c)
                with self.assertRaises(ValueError):
                    t20.config(path)

    def test_actual_grade_mapping_not_zero_based_mapping(self):
        np.testing.assert_array_equal(t20.weights([3, 2, 1, 3]), [1.5, 1., 1., 1.5])
        for invalid in [[0, 1, 2], [1, 2, 4], [1, 2, np.nan], [[1], [2]]]:
            with self.assertRaises(ValueError):
                t20.weights(invalid)

    def test_validation_labels_not_a_fit_argument_and_matrix_is_sparse(self):
        self.assertEqual(list(inspect.signature(t20.fit_weighted).parameters), ['Xa', 'ya', 'Xb', 'c'])
        from scipy import sparse
        mat = sparse.csr_matrix(np.eye(3, dtype=np.float32))
        evidence = t20.matrix_evidence(mat)
        self.assertEqual(evidence['shape'], [3, 3])
        self.assertEqual(evidence['dtype'], 'float32')
        self.assertEqual(set(evidence['arrays']), {'data', 'indices', 'indptr'})
        with self.assertRaises(ValueError):
            t20.matrix_evidence(np.eye(3))

    def test_source_import_closure_includes_custom_preprocessor_and_tests(self):
        sources = t20.sources()
        for path in ['experiments/age995/run.py', 'experiments/lgbm_native_cat/run.py',
                     'experiments/grade3_decision_rule/run.py', 'tests/test_modeling.py', *t20.OWN_SOURCE]:
            self.assertIn(path, sources)
        self.assertTrue(all(len(s) == 64 for s in sources.values()))

    def test_guard_blocks_raw_and_resolved_holdout(self):
        for p in ['../../earthquake/data/final_holdout', 'FINAL_HOLDOUT/data.csv']:
            with self.assertRaises(ValueError):
                t20.guard(p)

    def test_checkpoint_skip_backup_and_probability_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            e, calls = Env(d), []
            result, fresh = e.fit(fit_fn=fake_fit(calls))
            self.assertTrue(fresh)
            same, fresh = e.fit(fit_fn=fake_fit(calls))
            self.assertFalse(fresh)
            self.assertEqual(len(calls), 1)
            self.assertEqual(result, same)
            t20.check_saved(e.out, same)
            pred = t20.read_pred(e.out/same['dir']/'pred.csv.gz')
            self.assertTrue((pred.y_pred.to_numpy() == pred[t20.PROBS].to_numpy().argmax(1)+1).all())

    def test_crash_records_failure_never_retries_and_preserves_original(self):
        with tempfile.TemporaryDirectory() as d:
            e = Env(d)
            with self.assertRaises(RuntimeError):
                e.fit(fit_fn=fake_fit(fail=True))
            before = (e.out/'geo2_s42/f0/attempt1/failure.txt').read_bytes()
            with self.assertRaises(ValueError):
                e.fit()
            ledger = t20.read_ledger(e.out)
            self.assertEqual(ledger['fits']['geo2_s42/f0']['status'], 'failed')
            self.assertEqual(ledger['fits']['geo2_s42/f0']['attempts'], 1)
            self.assertEqual((e.out/'geo2_s42/f0/attempt1/failure.txt').read_bytes(), before)

    def test_invalid_probability_reload_and_budget_fail_closed(self):
        for fit in [fake_fit(invalid=True), fake_fit(reload_diff=1e-6)]:
            with self.subTest(fit=fit), tempfile.TemporaryDirectory() as d:
                e = Env(d)
                with self.assertRaises(ValueError):
                    e.fit(fit_fn=fit)
                self.assertEqual(t20.read_ledger(e.out)['fits']['geo2_s42/f0']['status'], 'failed')
        with tempfile.TemporaryDirectory() as d:
            e = Env(d)
            e.c['max_new_study_fits'] = 0
            with self.assertRaises(ValueError):
                e.fit()
            self.assertEqual(t20.read_ledger(e.out)['fits'], {})

    def test_backup_tamper_blocks_completed_skip(self):
        with tempfile.TemporaryDirectory() as d:
            e = Env(d)
            record, _ = e.fit()
            (e.out/'backup'/record['dir']/'metrics.json').write_text('{}', encoding='utf-8')
            with self.assertRaises(ValueError):
                e.fit()

    def test_existing_lock_is_preserved_and_verify_not_called(self):
        with tempfile.TemporaryDirectory() as d:
            e = Env(d)
            (e.out/'RUNNING.lock').write_text('다른 실행', encoding='utf-8')
            with mock.patch.object(t20, 'verify') as verify:
                with self.assertRaises(FileExistsError):
                    t20.fit_all('synthetic', e.out)
                verify.assert_not_called()
            self.assertEqual((e.out/'RUNNING.lock').read_text(encoding='utf-8'), '다른 실행')

    def test_filters_one_fold_then_remaining_no_duplicate_and_finalize(self):
        with tempfile.TemporaryDirectory() as d:
            e, calls = Env(d), []
            with mock.patch.object(t20, 'verify', return_value=e.inputs):
                t20.fit_all('synthetic', e.out, ['geo2_s42'], [0], fake_fit(calls))
                self.assertEqual(len(calls), 1)
                t20.fit_all('synthetic', e.out, fit_fn=fake_fit(calls))
                self.assertEqual(len(calls), 20)
                t20.write(e.out/'design.json', e.design)
                result = t20.finalize_all('synthetic', e.out)
                self.assertEqual(result['fit_attempts'], 20)
                self.assertEqual(result['pooled_denominator']['predictions'], 1800)
                self.assertEqual(result['pooled_denominator']['unique_development_rows'], 450)
                self.assertEqual(len(result['fold_records']), 20)
                self.assertEqual(set(result['judgment']['gates']), {'mean_delta_ok', 'fold_wins_ok',
                                  'positive_seeds_ok', 'pooled_recall3_ok', 'pooled_error31_rate_ok'})
                self.assertEqual((e.out/'analysis.json').read_bytes(), (e.out/'backup/analysis.json').read_bytes())
                with self.assertRaises(ValueError):
                    t20.finalize_all('synthetic', e.out)
            for a, _, b in calls:
                self.assertFalse(set(a) & set(b))

    def test_finalize_incomplete_or_empty_filter_stops(self):
        with tempfile.TemporaryDirectory() as d:
            e = Env(d)
            with mock.patch.object(t20, 'verify', return_value=e.inputs):
                with self.assertRaises(ValueError):
                    t20.finalize_all('synthetic', e.out)
                for schemes, folds in [([], None), (None, []), (['invalid'], None), (None, [5])]:
                    with self.assertRaises(ValueError):
                        t20.fit_all('synthetic', e.out, schemes, folds)

    def test_bad_seal_stops_before_reading_any_data(self):
        with tempfile.TemporaryDirectory() as d:
            t20.write(Path(d)/'design.json', {})
            t20.write(Path(d)/'design_seal.json', {'design_sha256': 'bad'})
            with mock.patch.object(t20, 'load') as load:
                with self.assertRaises(ValueError):
                    t20.verify('synthetic', d)
                load.assert_not_called()

    def test_judgment_strict_wins_seed_support_and_safety_gates(self):
        c = t20.config()
        cm = [[10, 0, 0], [0, 10, 0], [1, 3, 6]]
        baseline = dict(macro_f1=.7, confusion=cm)
        records = [dict(scheme=s, fold=f, baseline=baseline,
                        weighted=dict(macro_f1=.704, confusion=[[10,0,0],[0,10,0],[0,3,7]]))
                   for s in c['schemes'] for f in range(5)]
        self.assertTrue(t20.judge(records, c)['supports_fixed_hypothesis'])
        for record in records[:3]:
            record['weighted']['confusion'] = [[10,0,0],[0,10,0],[10,0,0]]
        verdict = t20.judge(records, c)
        self.assertFalse(verdict['gates']['pooled_error31_rate_ok'])
        self.assertFalse(verdict['supports_fixed_hypothesis'])
        for record in records:
            record['weighted'] = copy.deepcopy(baseline)
        self.assertEqual(t20.judge(records, c)['fold_wins'], 0)

    def test_real_synthetic_training_uses_exact_weights_and_sparse_reload(self):
        from lightgbm import LGBMClassifier
        from scipy import sparse
        X, y, _, _ = synthetic(900)
        X.loc[:2, 'age'] = 995
        X.loc[700:, 'geo_level_2_id'] = 999999
        c = copy.deepcopy(t20.config())
        c['lgbm_params'].update(n_estimators=5, verbosity=-1)
        original_fit = LGBMClassifier.fit
        calls = []

        def spy(model, matrix, labels, **kw):
            self.assertTrue(sparse.issparse(matrix))
            np.testing.assert_array_equal(labels, y.iloc[:700].to_numpy()-1)
            np.testing.assert_array_equal(kw['sample_weight'], np.where(y.iloc[:700] == 3, 1.5, 1.))
            self.assertEqual(set(kw), {'sample_weight'})
            calls.append(1)
            return original_fit(model, matrix, labels, **kw)

        with tempfile.TemporaryDirectory() as d, mock.patch.object(LGBMClassifier, 'fit', new=spy):
            result = t20.fit_weighted(X.iloc[:700], y.iloc[:700], X.iloc[700:], c)
            paths = result['save'](Path(d)/'fit')
            self.assertLessEqual(result['verify'](paths, X.iloc[700:], result['p']), 1e-12)
            self.assertTrue(sparse.issparse(t20.joblib.load(paths[1]).transform(X.iloc[700:])))
            self.assertEqual(result['info']['internal_classes'], [0, 1, 2])
            self.assertEqual(result['p'].shape, (200, 3))
            self.assertEqual(calls, [1])


if __name__ == '__main__':
    unittest.main()
