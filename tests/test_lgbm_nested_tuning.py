"""T18의 합성 데이터 검증. 실데이터/홀드아웃을 읽지 않는다."""
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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'tests'))
from test_modeling import synthetic
from experiments.lgbm_nested_tuning import run as t18


def fake_fit(calls=None, fail=False):
    def fit(candidate, Xa, ya, Xb, c):
        if calls is not None:
            calls.append((candidate, Xa.index.to_numpy(), Xb.index.to_numpy()))
        if fail:
            raise RuntimeError('합성 중단')
        # 합성 행 인덱스만 사용한다. 검증 정답 인자는 받지 않는다.
        pred = Xb.index.to_numpy() % 3
        if candidate == 'B':
            pred = (pred + 1) % 3
        elif candidate == 'L63':
            pred = (pred + (Xb.index.to_numpy() % 2)) % 3
        p = np.full((len(Xb), 3), 0.05)
        p[np.arange(len(Xb)), pred] = 0.9

        def save(prefix):
            path = Path(f'{prefix}_model.json')
            path.write_text(json.dumps(dict(candidate=candidate)), encoding='utf-8')
            return [path]
        return dict(p=p, save=save, verify=lambda *args: 0.0, fit_seconds=0.01, warnings=[])
    return fit


class Env:
    def __init__(self, root):
        self.root = Path(root)
        self.X, self.y, self.ids, _ = synthetic(1800)
        rng = np.random.default_rng(8)
        self.X['geo_level_1_id'] = rng.integers(0, 15, len(self.X))
        self.X['geo_level_2_id'] = rng.integers(0, 20, len(self.X))
        self.fo = t18.t15.t13.make_folds(self.X, self.y, 101)
        self.c = copy.deepcopy(t18.config())
        self.c['schemes'] = {'toy': dict(kind='group2', seed=101)}
        self.c['max_new_study_fits'] = 40
        self.out = self.root/'run'
        self.out.mkdir()
        (self.out/'backup').mkdir()
        t18.atomic_write(self.out/'ledger.json', dict(fits={}, selections={}, evaluations={}))
        self.design = dict(config=self.c, inner_fold_sha256={})
        for f in range(5):
            tr = np.flatnonzero(self.fo != f)
            inner = t18.inner_folds(self.X.iloc[tr], self.y.iloc[tr], self.c)
            self.design['inner_fold_sha256'][f'toy/f{f}'] = t18.inner_digest(inner)
        p = np.full((len(self.X), 3), 0.05)
        p[np.arange(len(self.X)), (self.y.to_numpy() % 3)] = 0.9
        self.ref = pd.DataFrame(dict(building_id=self.ids, fold=self.fo, y_true=self.y, y_pred=p.argmax(1)+1,
                                     **{n: p[:, j] for j, n in enumerate(t18.PROBS)}))

    def select(self, calls=None, f=0):
        return t18.selection_unit(self.X, self.y, self.ids, self.fo, self.out, 'toy', f, self.design, fake_fit(calls))

    def evaluate(self, selection, calls=None, f=0):
        return t18.evaluate_unit(self.X, self.y, self.ids, self.fo, self.ref, self.out, 'toy', f,
                                 self.design, selection, fake_fit(calls))


class DesignTests(unittest.TestCase):
    def test_preregistered_single_factors_and_fit_budget(self):
        c = t18.config()
        self.assertEqual(list(c['schemes']), ['geo2_s42', 'geo2_s101', 'geo2_s202', 'geo2_s303'])
        for k, v in c['candidates'].items():
            self.assertLessEqual(sum(v[n] != c['candidates']['B'][n] for n in v), 1)
        self.assertEqual(4*5*3*4, c['max_inner_fits'])
        self.assertEqual(c['max_new_study_fits'], 280)
        self.assertEqual(c['lgbm_params'], t18.t15.t13.config()['lgbm_params'])
        self.assertEqual(t18.environment(), c['environment'])

    def test_config_rejects_candidate_or_policy_changes(self):
        for name, change in [('candidates', {'B': dict(n_estimators=600, num_leaves=63)}),
                             ('decision_rule', 'grade3_offset'), ('early_stopping', True)]:
            with tempfile.TemporaryDirectory() as d:
                c = copy.deepcopy(t18.config())
                c[name] = change
                path = Path(d)/'config.json'
                t18.write(path, c)
                with self.assertRaises(ValueError):
                    t18.config(path)

    def test_only_internal_labels_enter_selection_and_ties_prefer_baseline(self):
        y = np.array([1, 2, 3]*20)
        p = np.eye(3)[y-1]
        chosen, scores = t18.select_candidates(y, {k: p.copy() for k in t18.CANDIDATES}, t18.config())
        self.assertEqual(chosen, dict(ROUNDS='B', LEAVES='B'))
        self.assertEqual(set(scores.values()), {1.0})
        self.assertEqual(list(inspect.signature(t18.select_candidates).parameters), ['y_inner', 'probabilities', 'c'])

    def test_policy_search_spaces_do_not_mix_factors(self):
        y = np.array([1, 2, 3]*20)
        good = np.eye(3)[y-1]
        bad = np.roll(good, 1, axis=1)
        chosen, _ = t18.select_candidates(y, dict(B=bad, R600=good, L15=good, L63=bad), t18.config())
        self.assertEqual(chosen, dict(ROUNDS='R600', LEAVES='L15'))

    def test_candidate_fit_reuses_exact_baseline_and_only_overrides_two_registered_keys(self):
        c = t18.config()
        X, y, _, _ = synthetic(150)
        original = copy.deepcopy(c)
        for candidate in c['candidates']:
            with mock.patch.object(t18.t15, 'fit_inner_arm', return_value={}) as fit:
                t18.fit_candidate(candidate, X.iloc[:100], y.iloc[:100], X.iloc[100:], c)
                args = fit.call_args.args
                self.assertEqual(args[:2], ('LGBM_O', 'group2'))
                self.assertEqual(args[-1]['lgbm_params'], dict(c['lgbm_params'], **c['candidates'][candidate]))
        self.assertEqual(c, original)

    def test_holdout_path_guard_including_normalized_parent(self):
        for path in ('earthquake/data/final_holdout/file.csv', 'FINAL_HOLDOUT', 'final_holdout/../file.csv'):
            with self.assertRaises(ValueError):
                t18.guard(path)

    def test_all_declared_sources_exist(self):
        self.assertEqual(set(t18.sources()), set(t18.SOURCE))

    def test_pooled_confusion_metrics_match_reference_score(self):
        y = np.array([1, 1, 1, 2, 2, 3, 3, 3, 3])
        pred = np.array([1, 1, 3, 2, 3, 3, 2, 1, 3])
        reference = t18.score(y, pred)
        pooled = t18.pooled_metrics([reference['confusion'], reference['confusion']])
        for metric in ('precision', 'recall', 'f1', 'macro_f1'):
            np.testing.assert_allclose(pooled[metric], reference[metric])
        self.assertEqual((pooled['error31'], pooled['error32'], pooled['error23']), (2, 2, 2))
        self.assertEqual(pooled['rows'], 2*len(y))
        self.assertEqual(pooled['error31_rate'], 0.25)


class LifecycleTests(unittest.TestCase):
    def test_full_synthetic_lifecycle_finalize_and_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(t18, 'log'):
            e = Env(d)
            e.c['max_new_study_fits'] = 100
            e.c['max_inner_fits'] = 60
            for f in range(5):
                selection = e.select(f=f)
                e.evaluate(selection, f=f)
            t18.write(e.out/'design.json', {'synthetic': True})
            verification = (e.X, e.y, e.ids, {'toy': e.fo}, {'toy': e.ref}, dict(e.design, code_commit='a'*40))
            with mock.patch.object(t18, 'verify', return_value=verification):
                result = t18.finalize_all('synthetic', e.out)
                self.assertEqual(result['actual_inner_fits'], 60)
                self.assertEqual(result['actual_outer_refits'], 10)
                self.assertEqual(result['pooled_denominator']['unique_development_rows'], 1800)
                self.assertEqual(result['pooled_baseline']['rows'], 1800)
                self.assertEqual(result['policies']['ROUNDS']['pooled']['macro_f1'], 1.0)
                with self.assertRaises(ValueError):
                    t18.finalize_all('synthetic', e.out)
            self.assertEqual(t18.sha256_file(e.out/'analysis.json'), t18.sha256_file(e.out/'backup/analysis.json'))

    def test_selection_backup_before_ledger_crash_recovers_without_refit(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(t18, 'log'):
            e = Env(d)
            selection = e.select()
            ledger = t18.read_ledger(e.out)
            ledger['selections'] = {}
            t18.atomic_write(e.out/'ledger.json', ledger)
            calls = []
            recovered = e.select(calls)
            self.assertEqual(calls, [])
            self.assertEqual(recovered, selection)

    def test_inner_groups_and_coverage_and_outer_exclusion(self):
        with tempfile.TemporaryDirectory() as d:
            e = Env(d)
            calls = []
            selection = e.select(calls)
            self.assertEqual(selection['chosen'], dict(ROUNDS='R600', LEAVES='L15'))
            self.assertEqual(len(calls), 12)
            outer = set(np.flatnonzero(e.fo == 0))
            for candidate in t18.CANDIDATES:
                valid = []
                for cand, a, b in calls:
                    if cand != candidate:
                        continue
                    self.assertFalse((set(a) | set(b)) & outer)
                    ga = set(map(tuple, e.X.iloc[a][['geo_level_1_id', 'geo_level_2_id']].to_numpy()))
                    gb = set(map(tuple, e.X.iloc[b][['geo_level_1_id', 'geo_level_2_id']].to_numpy()))
                    self.assertFalse(ga & gb)
                    valid.extend(b.tolist())
                self.assertEqual(sorted(valid), np.flatnonzero(e.fo != 0).tolist())

    def test_completed_inner_and_outer_checkpoints_are_not_refit(self):
        with tempfile.TemporaryDirectory() as d:
            e = Env(d)
            calls = []
            selection = e.select(calls)
            e.evaluate(selection, calls)
            self.assertEqual(len(calls), 14)
            self.assertEqual(e.select(calls), selection)
            e.evaluate(selection, calls)
            self.assertEqual(len(calls), 14)
            ledger = t18.read_ledger(e.out)
            for section in ('fits', 'selections', 'evaluations'):
                for record in ledger[section].values():
                    t18.check_saved(e.out, record)

    def test_outer_label_permutation_cannot_change_selection(self):
        with tempfile.TemporaryDirectory() as d:
            e = Env(d)
            before = e.select()
            rows = np.flatnonzero(e.fo == 0)
            e.y.iloc[rows] = np.roll(e.y.iloc[rows].to_numpy(), 17)
            # 새 출력 경로에서 같은 내부 OOF를 다시 만드는 합성 검증.
            e.out = e.root/'second'
            e.out.mkdir()
            (e.out/'backup').mkdir()
            t18.atomic_write(e.out/'ledger.json', dict(fits={}, selections={}, evaluations={}))
            after = e.select()
            self.assertEqual(before['chosen'], after['chosen'])
            self.assertEqual(before['inner_scores'], after['inner_scores'])

    def test_baseline_selected_requires_no_outer_fit(self):
        with tempfile.TemporaryDirectory() as d:
            e = Env(d)
            selection = e.select()
            # 동률 합성 상황의 저장 기록으로 정책을 고정한다.
            selection['chosen'] = dict(ROUNDS='B', LEAVES='B')
            ledger = t18.read_ledger(e.out)
            record = ledger['selections']['toy/f0']
            for prefix in (e.out, e.out/'backup'):
                t18.write(prefix/record['dir']/'selection.json', selection)
            record['artifact_sha256']['selection.json'] = t18.sha256_file(e.out/record['dir']/'selection.json')
            t18.atomic_write(e.out/'ledger.json', ledger)
            calls = []
            e.evaluate(selection, calls)
            self.assertEqual(calls, [])
            evaluation = json.loads((e.out/'toy/f0/evaluation/evaluation.json').read_text(encoding='utf-8'))
            self.assertTrue(all(v['delta_macro_f1'] == 0 for v in evaluation['policies'].values()))

    def test_backup_tampering_prevents_resume(self):
        with tempfile.TemporaryDirectory() as d:
            e = Env(d)
            selection = e.select()
            ledger = t18.read_ledger(e.out)
            entry = ledger['selections']['toy/f0']
            (e.out/'backup'/entry['dir']/'selection.json').write_text('{}', encoding='utf-8')
            with self.assertRaises(ValueError):
                e.evaluate(selection)

    def test_crash_retry_is_explicit_preserves_attempt_and_counts_budget(self):
        with tempfile.TemporaryDirectory() as d:
            e = Env(d)
            a, b = np.arange(100), np.arange(100, 150)
            args = (e.out, 'toy/f0/inner/B/i0', 'B', 'inner', e.X.iloc[a], e.y.iloc[a],
                    e.X.iloc[b], e.y.iloc[b], e.ids.iloc[b], e.c)
            with self.assertRaises(RuntimeError):
                t18.checkpoint_fit(*args, fit_fn=fake_fit(fail=True))
            marker = e.out/'toy/f0/inner/B/i0/attempt1/preserve.txt'
            marker.write_text('유지', encoding='utf-8')
            with self.assertRaises(ValueError):
                t18.checkpoint_fit(*args, fit_fn=fake_fit())
            result, fresh = t18.checkpoint_fit(*args, fit_fn=fake_fit(), resume_incomplete=True)
            self.assertTrue(fresh)
            self.assertEqual(result['attempts'], 2)
            self.assertEqual(marker.read_text(encoding='utf-8'), '유지')
            self.assertTrue((e.out/'toy/f0/inner/B/i0/attempt2').exists())

    def test_fit_budget_and_invalid_probability_stop(self):
        with tempfile.TemporaryDirectory() as d:
            e = Env(d)
            args = (e.out, 'toy/f0/inner/B/i0', 'B', 'inner', e.X.iloc[:100], e.y.iloc[:100],
                    e.X.iloc[100:150], e.y.iloc[100:150], e.ids.iloc[100:150])
            c = copy.deepcopy(e.c)
            c['max_new_study_fits'] = 0
            with self.assertRaises(ValueError):
                t18.checkpoint_fit(*args, c, fit_fn=fake_fit())
            def invalid(*values):
                result = fake_fit()(*values)
                result['p'][0] = np.nan
                return result
            with self.assertRaises(ValueError):
                t18.checkpoint_fit(*args, e.c, fit_fn=invalid)

    def test_source_or_seal_tampering_stops_before_load(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            t18.write(out/'design.json', {})
            t18.write(out/'design_seal.json', dict(design_sha256='bad'))
            with mock.patch.object(t18, 'load') as load:
                with self.assertRaises(ValueError):
                    t18.verify('synthetic', out)
                load.assert_not_called()

    def test_lock_is_exclusive(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            (out/'RUNNING.lock').write_text('원래 실행', encoding='utf-8')
            with self.assertRaises(FileExistsError):
                t18.fit_all('synthetic', out)
            self.assertEqual((out/'RUNNING.lock').read_text(encoding='utf-8'), '원래 실행')

    def test_real_synthetic_model_sparse_reload_and_unseen_categories(self):
        from scipy import sparse
        import joblib
        X, y, _, _ = synthetic(450)
        X.loc[:2, 'age'] = 995
        X.loc[350:, 'geo_level_2_id'] = 999999
        c = copy.deepcopy(t18.config())
        c['lgbm_params']['verbosity'] = -1
        c['candidates']['B']['n_estimators'] = 5
        with tempfile.TemporaryDirectory() as d:
            result = t18.fit_candidate('B', X.iloc[:350], y.iloc[:350], X.iloc[350:], c)
            paths = result['save'](Path(d)/'fit')
            self.assertLessEqual(result['verify'](paths, X.iloc[350:], result['p']), 1e-12)
            pre = joblib.load(paths[1])
            self.assertTrue(sparse.issparse(pre.transform(X.iloc[350:])))
            self.assertEqual(result['p'].shape, (100, 3))


if __name__ == '__main__':
    unittest.main()
