"""T17 합성 계약 검증. 실데이터/모델 학습/홀드아웃 없음."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from experiments.grade3_rule_freeze import run as t


class RuleTests(unittest.TestCase):
    def test_config_fixes_only_two_arms_four_seeds_and_zero_fits(self):
        c = t.config()
        self.assertEqual(c['arms'], ['LGBM_O', 'LGBM_TE_G'])
        self.assertEqual([v['seed'] for v in c['schemes'].values()], [42, 101, 202, 303])
        self.assertEqual(c['new_model_fits'], 0)
        self.assertIsNone(c['class_weight'])
        self.assertFalse(c['independent_performance_estimate'])

    def test_ties_and_mean_seed_objective(self):
        self.assertEqual(t.choose([[.2, .3, .3], [.4, .3, .3]], [0., .1, .2])[0], 0.)
        b, mean = t.choose([[.2, .8, .7], [.2, .1, .9]], [0., .1, .2])
        self.assertEqual(b, .2)
        np.testing.assert_allclose(mean, [.2, .45, .8])

    def test_curve_matches_independent_sklearn_and_positive_offset(self):
        y = np.tile([1, 2, 3], 100)
        p = np.tile([[.8, .1, .1], [.1, .7, .2], [.1, .5, .4]], (100, 1))
        got = t.curve(y, p)
        expected = [f1_score(y, (p * [1, 1, np.exp(b)]).argmax(1)+1,
                             labels=[1, 2, 3], average='macro', zero_division=0) for b in t.GRID]
        np.testing.assert_array_equal(got, expected)
        self.assertGreater(t.choose([got]*4)[0], 0.)
        np.testing.assert_array_equal(t.adjust_predict(p, 0.), p.argmax(1)+1)

    def test_probabilities_and_grid_are_checked(self):
        for p in ([[.1, .2, .1]], [[-.1, .5, .6]], [[float('nan'), .2, .8]], [[.1, .9]]):
            with self.assertRaises(ValueError):
                t.adjust_predict(p, 0.)
        with self.assertRaises(ValueError):
            t.adjust_predict([[.1, .2, .7]], -.1)
        with self.assertRaises(ValueError):
            t.adjust_predict([[.1, .2, .7]], .15)

    def test_guard_and_private_output_boundary(self):
        with self.assertRaises(ValueError):
            t.guard('X/FINAL_HOLDOUT/not_opened.csv')
        with self.assertRaises(ValueError):
            t.check_output(t.ROOT/'reports')
        self.assertEqual(t.check_output(t.ROOT.parents[1]/'local_reproduction/T17_20261015_rule_freeze/run').name, 'run')


class Env:
    def __init__(self, root):
        self.root = Path(root)
        self.c = copy.deepcopy(t.config())
        n = 1500
        group = np.arange(n)//3
        self.X = pd.DataFrame({'geo_level_1_id': np.zeros(n, dtype=int), 'geo_level_2_id': group})
        self.y = pd.Series(np.tile([1, 2, 3], n//3))
        self.ids = pd.Series(np.arange(n)+1, name=t.ID)
        self.folds = group % 5
        self.contract = {'synthetic': True}
        rng = np.random.default_rng(17)
        for s, sc in self.c['schemes'].items():
            pth = self.root/f'{s}_manifest.csv'
            pd.DataFrame({t.ID: self.ids, 'fold': self.folds}).to_csv(pth, index=False)
            sc['manifest'] = dict(path=str(pth), sha256=t.sha256_file(pth))
            for a in t.ARMS:
                p = rng.dirichlet([1., 2., 1.], n)
                r = pd.DataFrame({t.ID: self.ids, 'fold': self.folds, 'y_true': self.y,
                                  'y_pred': p.argmax(1)+1, **{v: p[:, j] for j, v in enumerate(t.t5.PROBS)}})
                path = self.root/f'{s}_{a}.csv.gz'
                t.io.write_pred(path, r)
                sc['refs'][a] = dict(path=str(path), sha256=t.sha256_file(path))
        self.out = self.root/'run'

    def load(self, _):
        return self.X, self.y, self.ids, self.folds, self.contract

    def patches(self):
        return [mock.patch.object(t, 'config', return_value=self.c),
                mock.patch.object(t, 'check_output', side_effect=lambda p: Path(p)),
                mock.patch.object(t.t5, 'load', side_effect=self.load),
                mock.patch.object(t.subprocess, 'check_output', side_effect=lambda args, **_: 'a'*40 if 'rev-parse' in args else '')]


class CheckpointTests(unittest.TestCase):
    def test_reference_exact_coverage_and_alignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            e = Env(tmp)
            ref = e.c['schemes'][t.SCHEMES[0]]['refs'][t.ARMS[0]]
            got = t.validate_reference(ref['path'], ref['sha256'], e.ids, e.y, e.folds)
            self.assertEqual(len(got), 1500)
            original = t.io.read_pred(ref['path'])
            for bad in (original.iloc[:-1], pd.concat([original, original.iloc[:1]]),
                        original.assign(y_true=1), original.assign(fold=0), original.assign(y_pred=1)):
                t.io.write_pred(ref['path'], bad)
                with self.assertRaises((ValueError, AssertionError)):
                    t.validate_reference(ref['path'], t.sha256_file(ref['path']), e.ids, e.y, e.folds)

    def test_group_overlap_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            e = Env(tmp)
            e.folds[0] = 1  # 같은 geo2 그룹의 첫 행만 다른 Fold로 이동
            with self.assertRaises(ValueError):
                t.t5.validate_assignment(e.X, e.y, e.ids, e.folds, e.c['group_columns'], t.t5.cfg())

    def test_end_to_end_seal_resume_finalize_and_tamper(self):
        from contextlib import ExitStack
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            e = Env(tmp)
            for p in e.patches():
                stack.enter_context(p)
            t.prepare('synthetic', e.out, 'a'*40)
            with self.assertRaises(ValueError):
                t.prepare('synthetic', e.out, 'a'*40)
            t.freeze_all('synthetic', e.out)
            ledger = t.io.read_ledger(e.out)
            self.assertEqual(len(ledger['units']), 10)
            self.assertEqual(ledger['new_model_fits'], 0)
            with mock.patch.object(t, 'metrics_curve', side_effect=AssertionError('완료 체크포인트 재계산 금지')):
                t.freeze_all('synthetic', e.out)
            result = t.finalize('synthetic', e.out)
            self.assertFalse(result['independent_performance_estimate'])
            self.assertFalse(result['adopted'])
            self.assertFalse(result['model_selection'])
            with self.assertRaises(ValueError):
                t.finalize('synthetic', e.out)
            key = next(iter(ledger['units']))
            (e.out/'backup'/'units'/f'{key}.json').write_text('{}')
            with self.assertRaises(ValueError):
                t.freeze_all('synthetic', e.out)
            self.assertFalse((e.out/'RUNNING.lock').exists())

    def test_incomplete_preserved_backup_and_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out/'units').mkdir()
            (out/'units'/'toy.json').write_text('partial')
            ledger = {'units': {}}
            rec = {'rows': 12, 'new_model_fits': 0}
            t.persist(out, 'toy', rec, ledger)
            self.assertEqual(t.saved(out, 'toy', ledger), rec)
            self.assertEqual(len(list((out/'units').glob('*.incomplete_*'))), 1)
            self.assertEqual(json.loads((out/'ledger.json').read_text()), ledger)

    def test_lock_prevents_parallel_freezing_and_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out/'RUNNING.lock').write_text('owner')
            with mock.patch.object(t, 'check_output', return_value=out):
                with self.assertRaises(FileExistsError):
                    t.freeze_all('synthetic', out)
            self.assertEqual((out/'RUNNING.lock').read_text(), 'owner')

    def test_sealed_source_change_refuses_run(self):
        from contextlib import ExitStack
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            e = Env(tmp)
            for p in e.patches():
                stack.enter_context(p)
            t.prepare('synthetic', e.out, 'a'*40)
            with mock.patch.object(t, 'sources', return_value={}):
                with self.assertRaises(ValueError):
                    t.verify('synthetic', e.out)


if __name__ == '__main__':
    unittest.main()
