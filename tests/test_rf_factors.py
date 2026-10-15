"""T16 합성 계약: 실제 개발 데이터/홀드아웃/기존 모델을 읽지 않는다."""
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
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'tests'))
from test_te_ros_rf_repro import data
from experiments.rf_factors import run as t16
from experiments.te_ros_rf_repro import run as t14
from preprocess import GEO


def small_config():
    c = copy.deepcopy(t16.config())
    c['rf_params'] = dict(n_jobs=1, random_state=0, n_estimators=5)
    c['min_free_disk_gb_per_fold'] = 0
    return c


class EncodingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.y, cls.ids, _, cls.fold = data(n=450)
        cls.F = t14.original_frame(cls.X)
        cls.c = small_config()
        cls.a, cls.b = np.flatnonzero(cls.fold != 0), np.flatnonzero(cls.fold == 0)

    def build(self, arm, valid=None):
        return t16.build_fold_matrices(self.F.iloc[self.a], self.X.iloc[self.a], self.y.iloc[self.a],
                                      self.F.iloc[self.b] if valid is None else valid, 'geo2_s101', self.c, arm)

    def test_plan_and_pins(self):
        c = t16.config()
        self.assertEqual(len(t16.targets()), 80)
        self.assertEqual(len(set(t16.key(*t) for t in t16.targets())), 80)
        self.assertEqual(c['max_new_study_fits'], 84)
        self.assertEqual(c['max_infrastructure_retries'], 4)
        self.assertEqual(c['baseline_refits'], 0)
        self.assertEqual(c['automatic_retries'], 0)
        self.assertEqual(c['rf_params'], dict(n_jobs=4, random_state=0))
        self.assertFalse(c['holdout_access'])

    def test_baseline_reconstruction_is_t14_bit_exact(self):
        c14 = copy.deepcopy(t14.config())
        base = self.build(t16.BASE)
        c14['expected_model_columns'] = len(base['selected'])
        ref = t14.build_fold_matrices(self.F.iloc[self.a], self.X.iloc[self.a], self.y.iloc[self.a],
                                      self.F.iloc[self.b], 'geo2_s101', c14)
        for name in ['Ao', 'Bo', 'Xos']:
            self.assertTrue(t16.matrix_equal(base[name], ref[name]))
        np.testing.assert_array_equal(base['yos'], ref['yos'])

    def test_no_ros_changes_only_training_multiplicity(self):
        base, new = self.build(t16.BASE), self.build('NO_ROS')
        pd.testing.assert_frame_equal(base['Ao'], new['Ao'])
        pd.testing.assert_frame_equal(base['Bo'], new['Bo'])
        pd.testing.assert_frame_equal(new['Ao'], new['Xos'])
        np.testing.assert_array_equal(new['yos'], self.y.iloc[self.a].to_numpy()-1)
        self.assertGreater(len(base['yos']), len(new['yos']))

    def test_drop_removes_only_four_te_features(self):
        base, new = self.build(t16.BASE), self.build('DROP_GEO23_TE')
        removed = {f'{g}_te{k}' for g in GEO[1:] for k in (0, 2)}
        self.assertEqual(set(base['selected'])-set(new['selected']), removed)
        pd.testing.assert_frame_equal(base['Ao'][new['selected']], new['Ao'])
        pd.testing.assert_frame_equal(base['Bo'][new['selected']], new['Bo'])
        for g in GEO:
            self.assertIn(g, new['selected'])
            self.assertIn(g+'_cnt', new['selected'])

    def test_onehot_replaces_te_only_and_remains_sparse(self):
        base, new = self.build(t16.BASE), self.build('ONEHOT')
        self.assertTrue(sparse.isspmatrix_csr(new['Ao']))
        self.assertTrue(sparse.isspmatrix_csr(new['Bo']))
        self.assertEqual(new['Ao'].dtype, np.float32)
        cols = new['state']['columns']
        removed = {f'{g}_te{k}' for g in GEO for k in (0, 2)}
        self.assertEqual(set(base['selected'])-set(cols), removed)
        expected = sparse.csr_matrix(base['Ao'][cols].to_numpy())
        self.assertEqual((new['Ao'][:, :len(cols)] != expected).nnz, 0)
        enc = new['state']['encoder']
        for g, vocab in zip(GEO, enc.categories_):
            np.testing.assert_array_equal(vocab, np.unique(self.F.iloc[self.a][g]))
        unseen = self.F.iloc[self.b].copy()
        for g in GEO:
            unseen[g] = 999999
        V = self.build('ONEHOT', unseen)['Bo']
        self.assertEqual(V[:, len(cols):].nnz, 0)

    def test_age_c_is_train_median_including_zero_and_indicator(self):
        base, new = self.build(t16.BASE), self.build('AGE_C')
        regular = self.F.iloc[self.a].age
        expected = float(regular[regular.ne(995)].median())
        self.assertEqual(new['state']['age'].median_, expected)
        self.assertEqual(set(new['selected'])-set(base['selected']), {'age_is_995'})
        unchanged = [c for c in base['selected'] if c != 'age']
        pd.testing.assert_frame_equal(base['Ao'][unchanged], new['Ao'][unchanged])
        flags = self.F.iloc[self.a].age.eq(995)
        np.testing.assert_array_equal(new['Ao'].age_is_995, flags.astype(float))
        self.assertTrue((new['Ao'].loc[flags, 'age'] == expected).all())

    def test_validation_changes_never_change_training_statistics(self):
        V = self.F.iloc[self.b].copy()
        V['age'] = 995
        for g in GEO:
            V[g] = 999999
        for arm in t16.ARMS:
            base, new = self.build(arm), self.build(arm, V)
            self.assertTrue(t16.matrix_equal(base['Ao'], new['Ao']), arm)
            self.assertTrue(t16.matrix_equal(base['Xos'], new['Xos']), arm)
            np.testing.assert_array_equal(base['yos'], new['yos'])

    def test_every_arm_reload_matrix_exact(self):
        for arm in t16.ARMS:
            built = self.build(arm)
            self.assertTrue(t16.matrix_equal(built['Bo'], t16.apply_state(built['state'], self.F.iloc[self.b])), arm)

    def test_group_inner_no_overlap(self):
        built = self.build(t16.BASE)
        G = self.F.iloc[self.a]
        groups = G[GEO[0]].astype(str)+'|'+G[GEO[1]].astype(str)
        owners = {}
        for g, f in zip(groups, built['inner_folds']):
            self.assertEqual(owners.setdefault(g, f), f)

    def test_no_validation_label_argument_or_dense_conversion(self):
        self.assertNotIn('y_valid', inspect.signature(t16.build_fold_matrices).parameters)
        text = (ROOT/'experiments/rf_factors/run.py').read_text(encoding='utf-8')
        self.assertNotIn('.toarray(', text)


class CheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        EncodingTests.setUpClass.__func__(cls)

    def fresh(self, tmp):
        out = Path(tmp)
        (out/'backup').mkdir()
        t16.atomic_write(out/'ledger.json', dict(study_id=t16.STUDY_ID, folds={}))
        design = dict(config=self.c, study_targets=[t16.key(*t) for t in t16.targets()], max_new_study_fits=84)
        return out, design

    def fit(self, out, design, arm='NO_ROS', **kwargs):
        folds = {s: self.fold for s in t16.SCHEMES}
        return t16.fit_fold(self.X, self.F, self.y, self.ids, folds, out, 'geo2_s101', arm, 0, design, **kwargs)

    def test_all_arms_checkpoint_reload_backup_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, design = self.fresh(tmp)
            for arm in t16.ARMS:
                entry, fitted = self.fit(out, design, arm)
                self.assertTrue(fitted)
                t16.check_saved(out, entry)
                rec = json.loads((out/entry['dir']/'fold_0.json').read_text(encoding='utf-8'))
                self.assertEqual(rec['reload_probability_max_abs_diff'], 0)
                self.assertEqual(rec['effective_parameters'], self.c['rf_params'] | {
                    k: v for k, v in t14.RandomForestClassifier(**self.c['rf_params']).get_params().items()
                    if k not in self.c['rf_params']})
                with mock.patch.object(t14, 'fit_rf', side_effect=AssertionError('No refit')):
                    saved, fitted = self.fit(out, design, arm)
                self.assertFalse(fitted)
                self.assertEqual(entry, saved)

    def test_backup_tampering_fails_before_refit(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, design = self.fresh(tmp)
            entry, _ = self.fit(out, design)
            (out/'backup'/entry['dir']/'fold_0.json').write_text('{}')
            with mock.patch.object(t14, 'fit_rf', side_effect=AssertionError('No refit')):
                with self.assertRaises(ValueError):
                    self.fit(out, design)

    def test_manual_infrastructure_retry_preserves_partial_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, design = self.fresh(tmp)
            d = out/'geo2_s101'/'NO_ROS'
            d.mkdir(parents=True)
            (d/'fold_0_model.joblib').write_bytes(b'partial')
            t16.atomic_write(out/'ledger.json', dict(folds={'geo2_s101/NO_ROS/0': dict(
                dir='geo2_s101/NO_ROS', status='running', attempts=1)}))
            with self.assertRaises(ValueError):
                self.fit(out, design)
            with self.assertRaises(ValueError):
                self.fit(out, design, retry_interrupted=True)
            entry, fitted = self.fit(out, design, retry_interrupted=True, retry_reason='OS interruption')
            self.assertTrue(fitted)
            self.assertEqual(entry['attempts'], 2)
            self.assertEqual((d/'fold_0_model.joblib.incomplete_attempt1').read_bytes(), b'partial')

    def test_retry_reserve_and_fit_budget_are_enforced(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, design = self.fresh(tmp)
            for des in [dict(design, study_targets=[]), dict(design, max_new_study_fits=0)]:
                with self.assertRaises(ValueError):
                    self.fit(out, des)
            entries = {'geo2_s101/NO_ROS/0': dict(dir='geo2_s101/NO_ROS', status='running', attempts=1)}
            entries.update({f'failed/{i}': dict(attempts=2) for i in range(4)})
            t16.atomic_write(out/'ledger.json', dict(folds=entries))
            with self.assertRaises(ValueError):
                self.fit(out, design, retry_interrupted=True, retry_reason='OS interruption')

    def test_exclusive_lock_prevents_second_executor(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, _ = self.fresh(tmp)
            (out/'RUNNING.lock').write_text('other process')
            with mock.patch.object(t16, 'verify', side_effect=AssertionError('Must not enter verify')):
                with self.assertRaises(FileExistsError):
                    t16.fit_all('unused', out)
            self.assertEqual((out/'RUNNING.lock').read_text(), 'other process')

    def test_final_assembly_can_resume_and_detects_oof_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, design = self.fresh(tmp)
            folds = {s: self.fold for s in t16.SCHEMES}
            for f in range(5):
                t16.fit_fold(self.X, self.F, self.y, self.ids, folds, out, 'geo2_s101', 'NO_ROS', f, design)
            first = t16.assemble(out, self.ids, self.y, folds, 'geo2_s101', 'NO_ROS')
            second = t16.assemble(out, self.ids, self.y, folds, 'geo2_s101', 'NO_ROS')
            pd.testing.assert_frame_equal(first, second)
            first.loc[0, 'y_pred'] = 999
            t16.write_pred(out/'geo2_s101/NO_ROS/oof.csv.gz', first)
            with self.assertRaises(AssertionError):
                t16.assemble(out, self.ids, self.y, folds, 'geo2_s101', 'NO_ROS')


class RuleTests(unittest.TestCase):
    def test_boundary_and_paired_split_requirements(self):
        rule = t16.config()['rule']
        self.assertEqual(t16.judge([0.003]*20, [0.003]*4, -0.01, rule)['label'], 'SUPPORTED_CONTRAST')
        self.assertEqual(t16.judge([0.0029]*20, [0.003]*4, 0, rule)['label'], 'NOT_SUPPORTED')
        self.assertEqual(t16.judge([0.003]*20, [0.003]*4, -0.0101, rule)['label'], 'NOT_SUPPORTED')
        self.assertEqual(t16.judge([0.003]*19, [0.003]*4, 0, rule)['label'], 'NOT_SUPPORTED')
        self.assertEqual(t16.judge([0.003]*20, [0.003]*2+[-0.003]*2, 0, rule)['label'], 'NOT_SUPPORTED')


if __name__ == '__main__':
    unittest.main()
