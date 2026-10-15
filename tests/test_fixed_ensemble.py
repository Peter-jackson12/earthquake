"""T19 합성 입력으로 고정 평균, 계약 실패, 오류 분모와 재개를 확인한다."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from experiments.fixed_ensemble import run as t19


class FixedEnsembleTest(unittest.TestCase):
    def test_fixed_average_tie_first_label(self):
        a = np.array([[.8, .1, .1], [.1, .2, .7]])
        b = np.array([[.1, .8, .1], [.1, .7, .2]])
        p, pred = t19.ensemble(a, b)
        np.testing.assert_array_equal(p, .5*a + .5*b)
        np.testing.assert_array_equal(pred, [1, 2])

    def test_probability_bad_shape_nonfinite_bounds_sum(self):
        for p in [np.array([.2, .3, .5]), [[np.nan, 0, 1]], [[-.1, .2, .9]],
                  [[1.1, 0, -.1]], [[.1, .2, .6]]]:
            with self.subTest(p=p), self.assertRaises(ValueError):
                t19.probabilities(p)
        with self.assertRaises(ValueError):
            t19.ensemble([[1, 0, 0]], [[1, 0, 0], [0, 1, 0]])

    def frame(self):
        return pd.DataFrame(dict(building_id=[20, 10, 30], fold=[1, 0, 2],
            y_true=[2, 1, 3], y_pred=[2, 1, 3],
            p_grade_1=[.1, .8, .1], p_grade_2=[.8, .1, .1], p_grade_3=[.1, .1, .8]))

    def test_id_alignment_not_row_position(self):
        got = t19.align(self.frame(), [10, 20, 30], [1, 2, 3], [0, 1, 2])
        self.assertEqual(got.building_id.tolist(), [10, 20, 30])

    def test_reject_duplicate_missing_ids_fractional_fold_wrong_y_pred(self):
        cases = [('building_id', [10, 10, 30]), ('building_id', [10, 20, 40]),
                 ('fold', [.5, 0, 2]), ('fold', [1, 1, 2]),
                 ('y_true', [1, 1, 3]), ('y_pred', [1, 1, 3])]
        for col, values in cases:
            f = self.frame()
            f[col] = values
            with self.subTest(col=col, values=values), self.assertRaises(ValueError):
                t19.align(f, [10, 20, 30], [1, 2, 3], [0, 1, 2])

    def test_class_metrics_error_denominators(self):
        got = t19.score([1, 2, 2, 3, 3, 3], [1, 3, 2, 1, 2, 3])
        self.assertEqual(got['confusion'], [[1, 0, 0], [0, 1, 1], [1, 1, 1]])
        self.assertEqual(got['errors']['3_to_1'], dict(count=1, denominator_actual_class=3, rate=1/3))
        self.assertEqual(got['errors']['2_to_3'], dict(count=1, denominator_actual_class=2, rate=.5))
        self.assertEqual(got['support'], [1, 2, 3])

    def test_overlap_partitions_and_rescues(self):
        got = t19.overlap([1, 2, 3, 2], [1, 2, 1, 1], [1, 1, 3, 3], [1, 2, 3, 1])
        self.assertEqual(got['error_intersection'], 1)
        self.assertEqual(got['error_union'], 3)
        self.assertTrue(all(v['count'] == 1 for v in got['categories'].values()))
        self.assertEqual(got['categories']['O_rescues_TE']['ensemble_correct'], 1)
        self.assertEqual(got['categories']['TE_rescues_O']['ensemble_correct'], 1)
        self.assertEqual(got['categories']['both_wrong']['ensemble_wrong'], 1)

    def test_grade3_event_overlap_and_removed_is_not_always_correct(self):
        got = t19.overlap([3, 3, 3, 3, 2], [1, 1, 2, 3, 3],
                          [1, 3, 1, 3, 2], [2, 3, 1, 3, 2])
        event = got['error_events']['3_to_1']
        self.assertEqual(event['denominator_actual_class'], 4)
        self.assertTrue(all(v['count'] == 1 for v in event['categories'].values()))
        o = event['by_baseline']['LGBM_O']
        self.assertEqual((o['event_removed'], o['event_introduced']), (2, 1))
        self.assertEqual((o['removed_to_actual_correct'], o['removed_to_other_wrong']), (1, 1))
        self.assertEqual(got['by_actual_class']['3']['n'], 4)
        self.assertEqual(got['error_events']['2_to_3']['by_baseline']['LGBM_O']['event_removed'], 1)

    def records(self, delta=.004, recall=0):
        d = {a: dict(macro_f1=delta, grade3_recall=recall) for a in t19.ARMS}
        return [dict(pooled=dict(deltas=copy.deepcopy(d)),
                     folds=[dict(deltas=copy.deepcopy(d)) for _ in range(5)]) for _ in range(4)]

    def test_technical_gates_and_baseline_separation(self):
        c = t19.config()['criteria']
        self.assertTrue(t19.summarize(self.records(), c)['technical_criteria_pass'])
        self.assertFalse(t19.summarize(self.records(delta=.0029), c)['technical_criteria_pass'])
        self.assertFalse(t19.summarize(self.records(recall=-.01001), c)['technical_criteria_pass'])
        r = self.records()
        r[0]['pooled']['deltas']['LGBM_O']['macro_f1'] = 0
        got = t19.summarize(r, c)
        self.assertFalse(got['technical_criteria_pass'])
        self.assertFalse(got['by_baseline']['LGBM_O']['technical_criteria_pass'])
        self.assertTrue(got['by_baseline']['LGBM_TE_G']['technical_criteria_pass'])

    def test_15_wins_and_full_20_folds_required(self):
        r = self.records(delta=.02)
        for unit in r:
            unit['folds'][0]['deltas']['LGBM_O']['macro_f1'] = -.001
        self.assertTrue(t19.summarize(r, t19.config()['criteria'])['technical_criteria_pass'])
        r[0]['folds'][1]['deltas']['LGBM_O']['macro_f1'] = -.001
        r[0]['folds'][2]['deltas']['LGBM_O']['macro_f1'] = -.001
        self.assertFalse(t19.summarize(r, t19.config()['criteria'])['technical_criteria_pass'])
        with self.assertRaises(ValueError):
            t19.summarize(r[:3], t19.config()['criteria'])

    def test_saved_hash_backup_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            p = out/'units'/'x.json'
            t19.write_new(p, dict(a=1))
            t19.backup(out, p)
            rp = out/'receipts'/'x.json'
            t19.write_new(rp, dict(status='complete', key='x', sha256=t19.digest(p)))
            t19.backup(out, rp)
            self.assertEqual(t19.saved(out, 'x'), dict(a=1))
            with self.assertRaises(FileExistsError):
                t19.write_new(p, dict(a=2))
            p.write_text(json.dumps(dict(a=2)), encoding='utf-8')
            with self.assertRaises(ValueError):
                t19.saved(out, 'x')

    def test_orphan_preserved_for_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            p = out/'units'/'x.json'
            t19.write_new(p, dict(a=1))
            t19.backup(out, p)
            self.assertIsNone(t19.saved(out, 'x'))
            self.assertFalse(p.exists())
            self.assertEqual(len(list((out/'units').glob('x.json.incomplete_*'))), 1)
            self.assertEqual(len(list((out/'backup'/'units').glob('x.json.incomplete_*'))), 1)

    def test_checkpoint_resume_skips_and_pending_receipt_recovery(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            t19.persist_unit(out, 'complete', dict(a=1))
            before = t19.digest(out/'units'/'complete.json')
            self.assertEqual(t19.saved(out, 'complete'), dict(a=1))
            self.assertEqual(before, t19.digest(out/'units'/'complete.json'))
            p = out/'units'/'interrupted.json'
            t19.write_new(p, dict(a=2))
            t19.backup(out, p)
            t19.write_new(out/'receipts'/'interrupted.pending.json', dict(status='complete'))
            t19.write_new(out/'backup'/'receipts'/'interrupted.json', dict(status='complete'))
            self.assertIsNone(t19.saved(out, 'interrupted'))
            t19.persist_unit(out, 'interrupted', dict(a=3))
            self.assertEqual(t19.saved(out, 'interrupted'), dict(a=3))

    def test_holdout_and_public_output_guards(self):
        with self.assertRaises(ValueError):
            t19.guard('anything/final_holdout/file')
        with self.assertRaises(ValueError):
            t19.output_path(t19.ROOT/'reports'/'bad')


if __name__ == '__main__':
    unittest.main()
