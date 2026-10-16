"""T30 합성 계약: 실제 개발 데이터/홀드아웃/기존 모델을 읽지 않고 실제 study fit도 하지 않는다.

RF는 5-트리 모델이며 계약 점검용이다. git 점검은 임시 저장소에서만 수행한다.
"""
import copy
import inspect
import json
from pathlib import Path
import shutil
import subprocess
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
from experiments.rf_factors_strat import run as t30
from experiments.te_ros_rf_repro import run as t14
from preprocess import GEO, ID, sha256_file


def small_config(X, F, y, fold, baseline_path=None):
    c = copy.deepcopy(t30.config())
    c['rf_params'] = dict(n_jobs=1, random_state=0, n_estimators=5)
    c['min_free_disk_gb_per_fold'] = 0
    a, b = np.flatnonzero(fold != 0), np.flatnonzero(fold == 0)
    for arm in c['expected_model_columns']:
        built = t30.build_fold_matrices(F.iloc[a], X.iloc[a], y.iloc[a], F.iloc[b], 'strat', c, arm)
        c['expected_model_columns'][arm] = len(built['selected'])
    return c


def baseline_frame(X, F, y, ids, fold, c):
    """Independent T14-pipeline OOF on the synthetic data (the stand-in for the audited baseline file)."""
    parts = []
    for f in range(5):
        a, b = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
        c14 = copy.deepcopy(t14.config())
        c14['rf_params'] = c['rf_params']
        c14['expected_model_columns'] = c['expected_model_columns']['C0_BASE_REFIT']
        built = t14.build_fold_matrices(F.iloc[a], X.iloc[a], y.iloc[a], F.iloc[b], 'strat', c14)
        model, _, _ = t14.fit_rf(built['Xos'], built['yos'], c14)
        p = t14.predict_proba_deterministic(model, built['Bo'])
        parts.append(pd.DataFrame({ID: ids.iloc[b].to_numpy(), 'fold': f, 'y_true': y.iloc[b].to_numpy(),
                                   'y_pred': p.argmax(1)+1, **{n: p[:, i] for i, n in enumerate(t14.PROBS)}}))
    return pd.concat(parts).set_index(ID).loc[ids.to_numpy()].reset_index()


def write_baseline(path, frame, c):
    t14.write_pred(path, frame)
    c['baseline_oof'] = dict(path=str(path), sha256=sha256_file(path))


class ConfigTests(unittest.TestCase):
    def test_plan_and_pins(self):
        c = t30.config()
        tg = t30.targets()
        self.assertEqual(len(tg), 41)
        self.assertEqual(len({t30.key(*t) for t in tg}), 41)
        self.assertEqual(tg[0], ('strat', 'C0_BASE_REFIT', 0))
        self.assertEqual([t for t in tg if t[1] == 'C0_BASE_REFIT'], [tg[0]])
        self.assertEqual([t[1] for t in tg[1:11]], ['N1_RS1']*5+['N2_RS2']*5)
        self.assertEqual([t[1] for t in tg[-5:]], ['ONEHOT']*5)
        self.assertEqual(list(dict.fromkeys(t[1] for t in tg)), t30.ARM_ORDER)
        self.assertEqual(c['planned_completed_fits'], 41)
        self.assertEqual(c['max_new_study_fits'], 45)
        self.assertEqual(c['max_infrastructure_retries'], 4)
        self.assertEqual(c['automatic_retries'], 0)
        self.assertEqual(c['outer_parallel_jobs'], 1)
        self.assertFalse(c['holdout_access'] or c['tuning'] or c['early_stopping'])
        self.assertEqual(c['rf_params'], dict(n_jobs=4, random_state=0))
        self.assertEqual(c['arm_rf_params']['N1_RS1'], dict(random_state=1))
        self.assertEqual(c['arm_rf_params']['N2_RS2'], dict(random_state=2))
        self.assertEqual(c['arm_rf_params']['CW_NO_ROS'], dict(class_weight='balanced'))
        self.assertEqual(c['baseline_oof']['sha256'], '69a74d410da08bf06c9da35211e3fe6f1e424bd4afbc72c4de2d2d3c21c09969')
        self.assertRegex(c['plan_sha256'], '^[0-9a-f]{64}$')  # equality with the file is enforced by `prepare`
        for p in t30.SOURCE:
            self.assertTrue((ROOT/p).is_file(), p)
        self.assertIn('experiments/rf_factors/run.py', t30.SOURCE)
        self.assertIn('experiments/te_ros_rf_repro/run.py', t30.SOURCE)

    def test_environment_pin_is_enforced(self):
        env = dict(t14.environment(), numpy='0.0.0')
        with mock.patch.object(t14, 'environment', return_value=env):
            with self.assertRaises(ValueError):
                t30.config()

    def test_rf_and_te_snapshots_are_enforced(self):
        base = t14.config()
        for k, v in [('rf_params', dict(n_jobs=4, random_state=3)),
                     ('te', dict(base['te'], smoothing_m=5))]:
            with mock.patch.object(t14, 'config', return_value=dict(copy.deepcopy(base), **{k: v})):
                with self.assertRaises(ValueError):
                    t30.config()

    def test_arm_definitions_are_enforced(self):
        changed = copy.deepcopy(t30.ARM_RF)
        changed['CW_NO_ROS'] = dict(class_weight='balanced_subsample')
        with mock.patch.object(t30, 'ARM_RF', changed):
            with self.assertRaises(ValueError):
                t30.config()
        spec = copy.deepcopy(t30.ARM_SPEC)
        spec['DROP_TE_ALL']['drop_all_te'] = False
        with mock.patch.object(t30, 'ARM_SPEC', spec):
            with self.assertRaises(ValueError):
                t30.config()

    def test_no_dense_conversion_or_holdout_literal_access(self):
        text = (ROOT/'experiments/rf_factors_strat/run.py').read_text(encoding='utf-8')
        self.assertNotIn('.toarray(', text)
        self.assertNotIn('final_holdout/', text)
        self.assertNotIn('y_valid', inspect.signature(t30.build_fold_matrices).parameters)
        with self.assertRaises(ValueError):
            t30.input_bundle('C:/x/final_holdout', t30.config())


class SourceBlobTests(unittest.TestCase):
    def git(self, root, *args):
        return subprocess.run(['git', '-c', 'user.name=t', '-c', 'user.email=t@example.invalid', *args],
                              cwd=root, capture_output=True, text=True, check=True).stdout

    def test_only_sealed_files_are_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.git(root, 'init', '-q')
            (root/'a.txt').write_text('a\n')
            (root/'b.txt').write_text('b\n')
            self.git(root, 'add', 'a.txt', 'b.txt')
            self.git(root, 'commit', '-q', '-m', 'x')
            (root/'unrelated.txt').write_text('other experiment\n')  # untracked, not sealed
            self.assertEqual(t30.source_mismatches(['a.txt', 'b.txt'], root), [])
            (root/'a.txt').write_text('changed\n')
            self.assertEqual(t30.source_mismatches(['a.txt', 'b.txt'], root),
                             [('a.txt', 'working-tree blob differs from HEAD')])
            self.git(root, 'checkout', '--', 'a.txt')
            (root/'c.txt').write_text('new\n')
            self.assertEqual(t30.source_mismatches(['a.txt', 'c.txt'], root), [('c.txt', 'not committed at HEAD')])
            self.assertEqual(t30.source_mismatches(['a.txt', 'gone.txt'], root), [('gone.txt', 'missing in worktree')])


class ArmTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.y, cls.ids, cls.fold, _ = data(n=450)
        cls.F = t14.original_frame(cls.X)
        cls.c = small_config(cls.X, cls.F, cls.y, cls.fold)
        cls.a, cls.b = np.flatnonzero(cls.fold != 0), np.flatnonzero(cls.fold == 0)

    def build(self, arm, valid=None):
        return t30.build_fold_matrices(self.F.iloc[self.a], self.X.iloc[self.a], self.y.iloc[self.a],
                                       self.F.iloc[self.b] if valid is None else valid, 'strat', self.c, arm)

    def test_controls_equal_the_t14_matrices_bit_exactly(self):
        c14 = copy.deepcopy(t14.config())
        ref = t14.build_fold_matrices(self.F.iloc[self.a], self.X.iloc[self.a], self.y.iloc[self.a],
                                      self.F.iloc[self.b], 'strat',
                                      dict(c14, expected_model_columns=self.c['expected_model_columns']['C0_BASE_REFIT']))
        for arm in ['C0_BASE_REFIT', 'N1_RS1', 'N2_RS2']:
            built = self.build(arm)
            for name in ['Ao', 'Bo', 'Xos']:
                self.assertTrue(t30.matrix_equal(built[name], ref[name]), (arm, name))
            np.testing.assert_array_equal(built['yos'], ref['yos'])

    def test_rf_parameters_differ_only_where_designed(self):
        c = self.c
        self.assertEqual(t30.merged_rf(c, 'C0_BASE_REFIT'), c['rf_params'])
        self.assertEqual(t30.merged_rf(c, 'N1_RS1')['random_state'], 1)
        self.assertEqual(t30.merged_rf(c, 'N2_RS2')['random_state'], 2)
        self.assertEqual(t30.merged_rf(c, 'CW_NO_ROS')['class_weight'], 'balanced')
        for arm in t30.ARM_ORDER:
            mine, base = t30.merged_rf(c, arm), c['rf_params']
            changed = {k for k in set(mine)|set(base) if mine.get(k) != base.get(k)}
            self.assertEqual(changed, {'N1_RS1': {'random_state'}, 'N2_RS2': {'random_state'},
                                       'CW_NO_ROS': {'class_weight'}}.get(arm, set()), arm)

    def test_t16_arms_delegate_to_the_sealed_builder(self):
        for arm in t16.ARMS:
            mine, ref = self.build(arm), t16.build_fold_matrices(self.F.iloc[self.a], self.X.iloc[self.a],
                                                                 self.y.iloc[self.a], self.F.iloc[self.b],
                                                                 'strat', self.c, arm)
            for name in ['Ao', 'Bo', 'Xos']:
                self.assertTrue(t30.matrix_equal(mine[name], ref[name]), (arm, name))
            np.testing.assert_array_equal(mine['yos'], ref['yos'])

    def test_every_arm_changes_exactly_one_declared_factor(self):
        base = self.build('C0_BASE_REFIT')
        names = {a: self.build(a) for a in t30.ARM_ORDER}
        bs = set(base['selected'])
        te_all, te_23 = set(t30.TE_ALL), {f'{g}_te{k}' for g in GEO[1:] for k in (0, 2)}
        self.assertEqual(set(names['NO_ROS']['selected']), bs)
        self.assertEqual(set(names['CW_NO_ROS']['selected']), bs)
        self.assertEqual(bs-set(names['DROP_GEO23_TE']['selected']), te_23)
        self.assertEqual(bs-set(names['DROP_TE_ALL']['selected']), te_all)
        self.assertEqual(set(names['AGE_C']['selected'])-bs, {'age_is_995'})
        self.assertEqual(bs-set(names['ONEHOT']['state']['columns']), te_all)
        self.assertEqual(len(base['yos']), len(names['AGE_C']['yos']))
        self.assertGreater(len(base['yos']), len(names['NO_ROS']['yos']))
        self.assertEqual(len(names['NO_ROS']['yos']), len(self.a))
        # the validation/train matrices of NO_ROS equal the baseline un-oversampled matrices
        self.assertTrue(t30.matrix_equal(names['NO_ROS']['Ao'], base['Ao']))
        self.assertTrue(t30.matrix_equal(names['CW_NO_ROS']['Xos'], names['NO_ROS']['Xos']))
        for g in GEO:  # raw geo numbers and counts are kept in every non-ONEHOT arm
            for arm in ['DROP_GEO23_TE', 'DROP_TE_ALL']:
                self.assertIn(g, names[arm]['selected'])
                self.assertIn(g+'_cnt', names[arm]['selected'])

    def test_drop_te_all_equals_dropping_before_oversampling_and_keeps_state_in_sync(self):
        base, new = self.build('C0_BASE_REFIT'), self.build('DROP_TE_ALL')
        keep = [n for n in base['selected'] if n not in t30.TE_ALL]
        self.assertEqual(new['selected'], keep)
        self.assertEqual(new['state']['columns'], keep)
        self.assertEqual(new['state']['selected'], keep)
        self.assertEqual(new['state']['drop'], t30.TE_ALL)
        self.assertEqual(len(new['selected']), len(base['selected'])-6)
        pd.testing.assert_frame_equal(new['Ao'], base['Ao'][keep])
        pd.testing.assert_frame_equal(new['Bo'], base['Bo'][keep])
        # ROS is column independent: drop-then-ROS gives the bit-identical matrix
        y0 = pd.Series(np.asarray(self.y.iloc[self.a], dtype=np.int64)-1, index=self.F.iloc[self.a].index)
        Xos, yos = t14.RandomOverSampler(random_state=0).fit_resample(base['Ao'][keep], y0)
        pd.testing.assert_frame_equal(new['Xos'], Xos)
        np.testing.assert_array_equal(new['yos'], yos)
        self.assertTrue(t30.matrix_equal(new['Bo'], t30.apply_state(new['state'], self.F.iloc[self.b])))
        self.assertEqual(list(new['Bo'].columns), keep)
        self.assertEqual(new['Bo'].dtypes.unique().tolist(), [np.dtype('float32')])

    def test_onehot_stays_sparse(self):
        new = self.build('ONEHOT')
        self.assertTrue(sparse.isspmatrix_csr(new['Ao']) and sparse.isspmatrix_csr(new['Xos']))
        self.assertEqual(new['Ao'].dtype, np.float32)

    def test_every_arm_reload_matrix_exact(self):
        for arm in t30.ARM_ORDER:
            built = self.build(arm)
            self.assertTrue(t30.matrix_equal(built['Bo'], t30.apply_state(built['state'], self.F.iloc[self.b])), arm)

    def test_validation_changes_never_change_training_statistics(self):
        V = self.F.iloc[self.b].copy()
        V['age'] = 995
        for g in GEO:
            V[g] = 999999
        for arm in t30.ARM_ORDER:
            base, new = self.build(arm), self.build(arm, V)
            self.assertTrue(t30.matrix_equal(base['Ao'], new['Ao']), arm)
            self.assertTrue(t30.matrix_equal(base['Xos'], new['Xos']), arm)
            np.testing.assert_array_equal(base['yos'], new['yos'])

    def test_unknown_arm_and_group_scheme_are_refused(self):
        with self.assertRaises(ValueError):
            self.build('MYSTERY')
        with self.assertRaises(ValueError):
            t30.build_fold_matrices(self.F.iloc[self.a], self.X.iloc[self.a], self.y.iloc[self.a],
                                    self.F.iloc[self.b], 'geo2_s42', self.c, 'NO_ROS')


class CheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.y, cls.ids, cls.fold, _ = data(n=450)
        cls.F = t14.original_frame(cls.X)
        cls.c = small_config(cls.X, cls.F, cls.y, cls.fold)
        cls.tmp = Path(tempfile.mkdtemp())
        cls.base = baseline_frame(cls.X, cls.F, cls.y, cls.ids, cls.fold, cls.c)
        write_baseline(cls.tmp/'baseline.csv.gz', cls.base, cls.c)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def design(self, c=None):
        return dict(config=c or self.c, study_targets=[t30.key(*t) for t in t30.targets()], max_new_study_fits=45)

    def fresh(self, tmp, c=None):
        out = Path(tmp)
        (out/'backup').mkdir()
        t30.atomic_write(out/'ledger.json', dict(study_id=t30.STUDY_ID, folds={}))
        return out, self.design(c)

    def fit(self, out, design, arm='NO_ROS', f=0, **kwargs):
        return t30.fit_fold(self.X, self.F, self.y, self.ids, self.fold, out, 'strat', arm, f, design, **kwargs)

    def prime(self, out, design):
        self.fit(out, design, 'C0_BASE_REFIT')
        return t30.ensure_gate(out, self.ids, self.y, self.fold, design)

    def altered_config(self, tmp, eps):
        frame = self.base.copy()
        mask = frame.fold == 0
        P = frame.loc[mask, t14.PROBS].to_numpy().copy()
        P[np.arange(len(P)), P.argmax(1)] += eps  # argmax unchanged
        frame.loc[mask, t14.PROBS] = P
        c = copy.deepcopy(self.c)
        write_baseline(Path(tmp)/'altered.csv.gz', frame, c)
        return c

    # ---- checkpoints
    def test_all_arms_checkpoint_reload_backup_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, design = self.fresh(tmp)
            gate = self.prime(out, design)
            self.assertEqual(gate['status'], 'PASS')
            self.assertTrue(gate['bit_identical'])
            for arm in t30.ARM_ORDER:
                entry, fitted = self.fit(out, design, arm)
                self.assertEqual(fitted, arm != 'C0_BASE_REFIT')
                t30.check_saved(out, entry)
                rec = json.loads((out/entry['dir']/'fold_0.json').read_text(encoding='utf-8'))
                self.assertEqual(rec['reload_probability_max_abs_diff'], 0)
                self.assertEqual(rec['study_id'], t30.STUDY_ID)
                self.assertEqual(rec['rf_params'], t30.merged_rf(self.c, arm))
                self.assertEqual(rec['effective_parameters'],
                                 t14.RandomForestClassifier(**t30.merged_rf(self.c, arm)).get_params())
                with mock.patch.object(t14, 'fit_rf', side_effect=AssertionError('No refit')):
                    saved, fitted = self.fit(out, design, arm)
                self.assertFalse(fitted)
                self.assertEqual(entry, saved)

    def test_effective_parameters_of_special_arms(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, design = self.fresh(tmp)
            self.prime(out, design)
            for arm, k, v in [('N1_RS1', 'random_state', 1), ('N2_RS2', 'random_state', 2),
                              ('CW_NO_ROS', 'class_weight', 'balanced'), ('NO_ROS', 'class_weight', None)]:
                self.fit(out, design, arm)
                rec = json.loads((out/'strat'/arm/'fold_0.json').read_text(encoding='utf-8'))
                self.assertEqual(rec['effective_parameters'][k], v, arm)
            model = joblib_load(out/'strat'/'CW_NO_ROS'/'fold_0_model.joblib')
            self.assertEqual(model.class_weight, 'balanced')
            nors = json.loads((out/'strat'/'NO_ROS'/'fold_0.json').read_text(encoding='utf-8'))
            self.assertEqual(nors['fit_rows'], int((self.fold != 0).sum()))

    def test_drop_te_all_checkpoint_has_filtered_columns_and_exact_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, design = self.fresh(tmp)
            self.prime(out, design)
            self.fit(out, design, 'DROP_TE_ALL')
            rec = json.loads((out/'strat'/'DROP_TE_ALL'/'fold_0.json').read_text(encoding='utf-8'))
            base = json.loads((out/'strat'/'C0_BASE_REFIT'/'fold_0.json').read_text(encoding='utf-8'))
            self.assertEqual(set(base['feature_names'])-set(rec['feature_names']), set(t30.TE_ALL))
            state = joblib_load(out/'strat'/'DROP_TE_ALL'/'fold_0_encoder_state.joblib')
            self.assertEqual(state['columns'], rec['feature_names'])
            self.assertEqual(state['selected'], rec['feature_names'])
            self.assertEqual(rec['reload_probability_max_abs_diff'], 0)

    def test_backup_tampering_fails_before_refit(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, design = self.fresh(tmp)
            self.prime(out, design)
            entry, _ = self.fit(out, design)
            (out/'backup'/entry['dir']/'fold_0.json').write_text('{}')
            with mock.patch.object(t14, 'fit_rf', side_effect=AssertionError('No refit')):
                with self.assertRaises(ValueError):
                    self.fit(out, design)

    # ---- C0 gate
    def test_other_fits_are_blocked_until_the_gate_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, design = self.fresh(tmp)
            with mock.patch.object(t14, 'fit_rf', side_effect=AssertionError('No fit before gate')):
                with self.assertRaises(ValueError):
                    self.fit(out, design, 'N1_RS1')
            self.fit(out, design, 'C0_BASE_REFIT')  # C0 done, gate not yet evaluated -> still blocked
            with self.assertRaises(ValueError):
                self.fit(out, design, 'NO_ROS')
            t30.ensure_gate(out, self.ids, self.y, self.fold, design)
            self.fit(out, design, 'NO_ROS')

    def test_gate_failure_stops_every_later_fit(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = self.altered_config(tmp, 1e-9)
            out, design = self.fresh(tempfile.mkdtemp(dir=tmp), c)
            self.fit(out, design, 'C0_BASE_REFIT')
            with self.assertRaises(ValueError):
                t30.ensure_gate(out, self.ids, self.y, self.fold, design)
            self.assertEqual(json.loads((out/'c0_gate.json').read_text(encoding='utf-8'))['status'], 'FAIL')
            with mock.patch.object(t14, 'fit_rf', side_effect=AssertionError('No fit after failed gate')):
                for arm in ['N1_RS1', 'NO_ROS', 'ONEHOT']:
                    with self.assertRaises(ValueError):
                        self.fit(out, design, arm)
            with self.assertRaises(ValueError):  # a later restart cannot re-open the gate
                t30.ensure_gate(out, self.ids, self.y, self.fold, design)
            self.assertEqual(sorted(read_keys(out)), ['strat/C0_BASE_REFIT/0'])

    def test_gate_tolerance_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = self.altered_config(tmp, 5e-13)
            out, design = self.fresh(tempfile.mkdtemp(dir=tmp), c)
            self.fit(out, design, 'C0_BASE_REFIT')
            gate = t30.ensure_gate(out, self.ids, self.y, self.fold, design)
            self.assertEqual(gate['status'], 'PASS')
            self.assertFalse(gate['bit_identical'])
            self.assertLessEqual(gate['max_abs_prob_diff'], 1e-12)

    def test_gate_rejects_label_mismatch_without_exact_tie(self):
        with tempfile.TemporaryDirectory() as tmp:
            frame = self.base.copy()
            top = np.sort(frame[t14.PROBS].to_numpy(), axis=1)
            idx = frame.index[(frame.fold == 0).to_numpy() & (top[:, 2] != top[:, 1])][0]
            frame.loc[idx, 'y_pred'] = 1 + (int(frame.loc[idx, 'y_pred']) % 3)
            c = copy.deepcopy(self.c)
            write_baseline(Path(tmp)/'flipped.csv.gz', frame, c)
            out, design = self.fresh(tempfile.mkdtemp(dir=tmp), c)
            self.fit(out, design, 'C0_BASE_REFIT')
            with self.assertRaises(ValueError):
                t30.ensure_gate(out, self.ids, self.y, self.fold, design)
            g = json.loads((out/'c0_gate.json').read_text(encoding='utf-8'))
            self.assertEqual((g['label_mismatches'], g['label_mismatches_allowed_exact_ties']), (1, 0))

    def test_gate_reads_with_the_round_trip_parser(self):
        self.assertIs(t30.read_pred, t14.read_pred)
        self.assertIn("float_precision='round_trip'", inspect.getsource(t14.read_pred))
        self.assertIn('read_pred', inspect.getsource(t14.reference))
        self.assertEqual(self.c['c0_gate']['float_precision'], 'round_trip')

    # ---- retry / budget / lock
    def test_manual_infrastructure_retry_preserves_partial_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, design = self.fresh(tmp)
            self.prime(out, design)
            d = out/'strat'/'NO_ROS'
            d.mkdir(parents=True)
            (d/'fold_0_model.joblib').write_bytes(b'partial')
            ledger = t30.read_ledger(out)
            ledger['folds']['strat/NO_ROS/0'] = dict(dir='strat/NO_ROS', status='running', attempts=1)
            t30.atomic_write(out/'ledger.json', ledger)
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
            self.prime(out, design)
            for des in [dict(design, study_targets=[]), dict(design, max_new_study_fits=1)]:
                with self.assertRaises(ValueError):
                    self.fit(out, des)
            ledger = t30.read_ledger(out)
            ledger['folds']['strat/NO_ROS/0'] = dict(dir='strat/NO_ROS', status='running', attempts=1)
            ledger['folds'].update({f'failed/{i}': dict(attempts=2) for i in range(4)})
            t30.atomic_write(out/'ledger.json', ledger)
            with self.assertRaises(ValueError):
                self.fit(out, design, retry_interrupted=True, retry_reason='OS interruption')

    def test_attempt_cap_is_45_over_all_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, design = self.fresh(tmp)
            self.prime(out, design)
            ledger = t30.read_ledger(out)
            ledger['folds'].update({f'other/{i}': dict(attempts=1, status='complete') for i in range(43)})
            t30.atomic_write(out/'ledger.json', ledger)  # 1 (C0) + 43 = 44 attempts
            self.fit(out, design, 'NO_ROS')  # 45th attempt allowed
            with self.assertRaises(ValueError):
                self.fit(out, design, 'N1_RS1')  # a 46th is not

    def test_exclusive_lock_prevents_second_executor(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, _ = self.fresh(tmp)
            (out/'RUNNING.lock').write_text('other process')
            with mock.patch.object(t30, 'verify', side_effect=AssertionError('Must not enter verify')):
                with self.assertRaises(FileExistsError):
                    t30.fit_all('unused', out)
            self.assertEqual((out/'RUNNING.lock').read_text(), 'other process')

    def test_lock_is_removed_after_a_failed_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, _ = self.fresh(tmp)
            with mock.patch.object(t30, 'verify', side_effect=ValueError('boom')):
                with self.assertRaises(ValueError):
                    t30.fit_all('unused', out)
            self.assertFalse((out/'RUNNING.lock').exists())

    # ---- whole run on synthetic data
    def test_full_run_order_cap_resume_and_analysis(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, design = self.fresh(tmp)
            args = (self.X, self.y, self.ids, self.fold, out, design)
            t30.run_targets(*args)
            ledger = t30.read_ledger(out)
            self.assertEqual(sum(e['status'] == 'complete' for e in ledger['folds'].values()), 41)
            self.assertEqual(sum(e['attempts'] for e in ledger['folds'].values()), 41)
            events = [json.loads(line) for line in (out/'attempts.jsonl').read_text(encoding='utf-8').splitlines()]
            started = [e['fold'] for e in events if e['action'] == 'fit_start']
            self.assertEqual(started, [t30.key(*t) for t in t30.targets()])
            self.assertLess(started.index('strat/C0_BASE_REFIT/0'), started.index('strat/N1_RS1/0'))
            gate_event = next(i for i, e in enumerate(events) if e['action'] == 'c0_gate')
            self.assertEqual(events[gate_event-1]['fold'], 'strat/C0_BASE_REFIT/0')
            # resume: nothing is retrained
            with mock.patch.object(t14, 'fit_rf', side_effect=AssertionError('No refit')):
                t30.run_targets(*args)
            t30.write(out/'design.json', dict(note='synthetic'))
            design.update(code_commit='0'*40, baseline={})
            res = t30.analyze(out, self.ids, self.y, self.fold, design)
            self.assertEqual((res['fits'], res['attempts'], res['retries']), (41, 41, 0))
            self.assertEqual(set(res['per_arm']), set(t30.JUDGED))
            for arm, v in res['per_arm'].items():
                self.assertIn(v['label'], {'CHANGE_IMPROVES', 'CHANGE_WORSENS', 'NEGLIGIBLE', 'MIXED'})
                self.assertEqual(len(v['fold_delta']), 5)
                self.assertEqual(v['positive_folds']+v['negative_folds'] <= 5, True)
                self.assertAlmostEqual(v['mean_delta'], np.mean(v['fold_delta']))
                self.assertEqual(v['contrast_with'], 'T14_RF_TE_ROS_A')
                self.assertEqual(len(v['stats']['model_columns']), 5)
            self.assertEqual(res['cw_no_ros']['contrast_with'], 'NO_ROS')
            self.assertTrue(res['cw_no_ros']['vs_baseline_reference_only']['reference_only'])
            self.assertEqual(res['cw_no_ros']['reference_error31'], res['to1_reference_counts']['NO_ROS'])
            self.assertEqual(res['per_arm']['NO_ROS']['reference_error31'], res['to1_reference_counts']['baseline'])
            n = res['noise']
            for a in ('N1_RS1', 'N2_RS2'):
                self.assertEqual(n['signed_mean_delta'][a], np.mean(n['signed_fold_delta'][a]))
                self.assertNotIn('label', n['arms'][a])
            self.assertEqual(n['annotation_triggered'], bool(n['max_abs_mean_delta'] >= 0.0015-1e-12
                                                            or n['max_abs_fold_delta'] >= 0.003-1e-12))
            for v in res['per_arm'].values():
                self.assertEqual('annotation' in v, n['annotation_triggered'])
            # final assembly is idempotent and detects OOF corruption
            again = t30.assemble(out, self.ids, self.y, self.fold, 'strat', 'NO_ROS')
            pd.testing.assert_frame_equal(again, t30.assemble(out, self.ids, self.y, self.fold, 'strat', 'NO_ROS'))
            again.loc[0, 'y_pred'] = 999
            t14.write_pred(out/'strat/NO_ROS/oof.csv.gz', again)
            with self.assertRaises(AssertionError):
                t30.assemble(out, self.ids, self.y, self.fold, 'strat', 'NO_ROS')

    def test_gate_failure_stops_run_targets_after_c0(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = self.altered_config(tmp, 1e-9)
            out, design = self.fresh(tempfile.mkdtemp(dir=tmp), c)
            with self.assertRaises(ValueError):
                t30.run_targets(self.X, self.y, self.ids, self.fold, out, design)
            self.assertEqual(read_keys(out), ['strat/C0_BASE_REFIT/0'])

    def test_analysis_refuses_incomplete_or_ungated_study(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, design = self.fresh(tmp)
            self.prime(out, design)
            with self.assertRaises(ValueError):
                t30.analyze(out, self.ids, self.y, self.fold, design)

    def test_status_and_data_resolution(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, design = self.fresh(tmp)
            self.prime(out, design)
            st = t30.status(out)
            self.assertEqual((st['complete'], st['total'], st['c0_gate'], st['lock_present']), (1, 41, 'PASS', False))
            self.assertEqual(st['per_arm']['C0_BASE_REFIT'], dict(complete=1, of=1))
            (out/'design.json').write_text(json.dumps(dict(data_dir='D:/data')), encoding='utf-8')
            args = mock.Mock(data=None, out=str(out))
            self.assertEqual(t30.resolve_data(args), 'D:/data')
            args.data = 'E:/other'
            self.assertEqual(t30.resolve_data(args), 'E:/other')


def joblib_load(path):
    import joblib
    return joblib.load(path)


def read_keys(out):
    return list(t30.read_ledger(out)['folds'])


class T16EquivalenceTests(unittest.TestCase):
    """The copied fit_fold must behave like T16's on the same synthetic input (only study_id/arm labels differ)."""

    @classmethod
    def setUpClass(cls):
        CheckpointTests.setUpClass.__func__(cls)
        cls.c16 = copy.deepcopy(t16.config())
        cls.c16['rf_params'] = cls.c['rf_params']
        cls.c16['min_free_disk_gb_per_fold'] = 0

    tearDownClass = classmethod(CheckpointTests.tearDownClass.__func__)

    def run_both(self, arm, tmp):
        out30, design30 = self.make(Path(tmp)/'s30', self.c, t30)
        out16, design16 = self.make(Path(tmp)/'s16', self.c16, t16)
        CheckpointTests.fit(self, out30, design30, 'C0_BASE_REFIT')
        t30.ensure_gate(out30, self.ids, self.y, self.fold, design30)
        e30, _ = t30.fit_fold(self.X, self.F, self.y, self.ids, self.fold, out30, 'strat', arm, 0, design30)
        e16, _ = t16.fit_fold(self.X, self.F, self.y, self.ids, {'strat': self.fold}, out16, 'strat', arm, 0, design16)
        return (out30, e30), (out16, e16)

    def make(self, out, c, mod):
        out.mkdir(parents=True)
        (out/'backup').mkdir()
        mod.atomic_write(out/'ledger.json', dict(study_id=mod.STUDY_ID, folds={}))
        arms = t30.ARM_ORDER if mod is t30 else mod.ARMS
        design = dict(config=c, study_targets=[mod.key('strat', a, 0) for a in arms], max_new_study_fits=84)
        return out, design

    def test_same_results_as_t16_for_the_four_shared_arms(self):
        for arm in t16.ARMS:
            with tempfile.TemporaryDirectory() as tmp:
                (out30, e30), (out16, e16) = self.run_both(arm, tmp)
                self.assertEqual(set(e30), set(e16))
                self.assertEqual(set(e30['artifact_sha256']), set(e16['artifact_sha256']))
                self.assertEqual(e30['artifact_sha256']['fold_0_pred.csv.gz'], e16['artifact_sha256']['fold_0_pred.csv.gz'])
                self.assertEqual(e30['macro_f1'], e16['macro_f1'])
                r30 = json.loads((out30/'strat'/arm/'fold_0.json').read_text(encoding='utf-8'))
                r16 = json.loads((out16/'strat'/arm/'fold_0.json').read_text(encoding='utf-8'))
                for k in ['train_rows', 'valid_rows', 'fit_rows', 'fit_class_counts', 'feature_names', 'model_columns',
                          'feature_names_sha256', 'effective_parameters', 'metrics', 'age_median',
                          'reload_probability_max_abs_diff', 'csv_round_trip_exact', 'feature_importances']:
                    self.assertEqual(r30[k], r16[k], (arm, k))
                self.assertEqual(r30['study_id'], t30.STUDY_ID)
                self.assertEqual(r16['study_id'], t16.STUDY_ID)
                m30 = joblib_load(out30/'strat'/arm/'fold_0_model.joblib')
                m16 = joblib_load(out16/'strat'/arm/'fold_0_model.joblib')
                built = t30.build_fold_matrices(self.F.iloc[self.fold != 0], self.X.iloc[self.fold != 0],
                                                self.y.iloc[self.fold != 0], self.F.iloc[self.fold == 0],
                                                'strat', self.c, arm)
                np.testing.assert_array_equal(t14.predict_proba_deterministic(m30, built['Bo']),
                                              t14.predict_proba_deterministic(m16, built['Bo']))

    def test_same_rules_as_t16_for_retries_and_resume(self):
        for mod, c in [(t30, self.c), (t16, self.c16)]:
            with tempfile.TemporaryDirectory() as tmp:
                out, design = self.make(Path(tmp)/'r', c, mod)
                if mod is t30:
                    CheckpointTests.fit(self, out, design, 'C0_BASE_REFIT')
                    t30.ensure_gate(out, self.ids, self.y, self.fold, design)
                fold_arg = self.fold if mod is t30 else {'strat': self.fold}
                led = mod.read_ledger(out)
                led['folds'][mod.key('strat', 'NO_ROS', 0)] = dict(dir='strat/NO_ROS', status='running', attempts=1)
                mod.atomic_write(out/'ledger.json', led)
                call = lambda **kw: mod.fit_fold(self.X, self.F, self.y, self.ids, fold_arg, out, 'strat', 'NO_ROS', 0,
                                                design, **kw)
                with self.assertRaises(ValueError):
                    call()
                with self.assertRaises(ValueError):
                    call(retry_interrupted=True)
                entry, fitted = call(retry_interrupted=True, retry_reason='OS interruption')
                self.assertTrue(fitted)
                self.assertEqual(entry['attempts'], 2)
                with mock.patch.object(t14, 'fit_rf', side_effect=AssertionError('No refit')):
                    again, fitted = call()
                self.assertFalse(fitted)
                self.assertEqual(entry, again)


class RuleTests(unittest.TestCase):
    rule = t30.config()['rule']

    def test_label_boundaries(self):
        L = lambda d: t30.label_for(d, self.rule)
        self.assertEqual(L([0.003]*5), 'CHANGE_IMPROVES')
        self.assertEqual(L([0.0029]*5), 'NEGLIGIBLE')
        self.assertEqual(L([0.003]*4+[-0.1]), 'MIXED')  # mean is negative large
        self.assertEqual(L([0.01, 0.01, 0.01, 0.01, -0.02]), 'CHANGE_IMPROVES')  # 4/5 positive, mean 0.004
        self.assertEqual(L([0.02, 0.02, 0.02, -0.001, -0.001]), 'MIXED')          # only 3/5 positive
        self.assertEqual(L([-0.003]*5), 'CHANGE_WORSENS')
        self.assertEqual(L([-0.01, -0.01, -0.01, -0.01, 0.02]), 'CHANGE_WORSENS')
        self.assertEqual(L([-0.02, -0.02, -0.02, 0.01, 0.01]), 'MIXED')
        self.assertEqual(L([0.0]*5), 'NEGLIGIBLE')
        self.assertEqual(L([0.1+0.2-0.3+0.003]*5), 'CHANGE_IMPROVES')  # float noise at the boundary does not flip a label

    def test_flag_thresholds_against_the_baseline_count(self):
        F = lambda r, ref, new: t30.flags_for(r, ref, new, self.rule)
        self.assertEqual(F(0, 744, 744+75), ['TO1_UP'])
        self.assertEqual(F(0, 744, 744+74), [])
        self.assertEqual(F(0, 744, 744-75), ['TO1_DOWN'])
        self.assertEqual(F(0, 744, 744-74), [])
        self.assertEqual(F(-0.01, 744, 744), ['RECALL3_DOWN'])
        self.assertEqual(F(-0.0099, 744, 744), [])
        self.assertEqual(F(0.01, 744, 744), ['RECALL3_UP'])
        self.assertEqual(F(-0.02, 744, 900), ['RECALL3_DOWN', 'TO1_UP'])

    def test_cw_no_ros_flags_use_the_no_ros_count_and_a_ten_row_floor(self):
        F = lambda ref, new: t30.flags_for(0, ref, new, self.rule)
        self.assertEqual(F(40, 49), [])      # 10% of 40 is 4 < 10 -> need |diff| >= 10
        self.assertEqual(F(40, 50), ['TO1_UP'])
        self.assertEqual(F(40, 30), ['TO1_DOWN'])
        self.assertEqual(F(40, 31), [])
        self.assertEqual(F(2000, 2199), [])  # 10% of 2000 is 200
        self.assertEqual(F(2000, 2200), ['TO1_UP'])

    def test_contrast_tradeoff_and_pairing(self):
        n = 100
        y = pd.Series(np.random.default_rng(3).permutation(np.tile([1, 2, 3, 3, 2], n//5)))
        yv = y.to_numpy()
        fold = np.arange(n) % 5
        frame = lambda pred: pd.DataFrame({'fold': fold, 'y_true': yv, 'y_pred': pred})
        bad = yv.copy()
        bad[yv == 3] = 1                      # every grade 3 becomes 3->1
        con = t30.contrast(frame(yv), frame(bad), y, self.rule)
        self.assertEqual(con['label'], 'CHANGE_IMPROVES')
        self.assertEqual(con['positive_folds'], 5)
        self.assertIn('TO1_DOWN', con['flags'])
        self.assertIn('RECALL3_UP', con['flags'])
        self.assertFalse(con['tradeoff'])
        self.assertEqual(con['reference_error31'], int((yv == 3).sum()))
        mirror = t30.contrast(frame(bad), frame(yv), y, self.rule)
        self.assertEqual(mirror['label'], 'CHANGE_WORSENS')
        self.assertIn('RECALL3_DOWN', mirror['flags'])
        self.assertIn('TO1_UP', mirror['flags'])
        self.assertEqual(mirror['reference_error31'], 0)
        np.testing.assert_allclose(mirror['fold_delta'], -np.array(con['fold_delta']))
        # F1 improves (all grade 2 fixed) while 12 grade-3 rows become 3->1: a trade-off
        ref = yv.copy()
        ref[yv == 2] = 3
        cand = yv.copy()
        cand[np.flatnonzero(yv == 3)[:12]] = 1
        con = t30.contrast(frame(cand), frame(ref), y, self.rule)
        self.assertEqual(con['label'], 'CHANGE_IMPROVES')
        self.assertIn('TO1_UP', con['flags'])
        self.assertIn('RECALL3_DOWN', con['flags'])
        self.assertTrue(con['tradeoff'])

    def test_importance_blocks_sum_to_one(self):
        names = [GEO[0], GEO[0]+'_te0', GEO[1]+'_cnt', 'onehot__geo_level_1_id_3', 'age']
        blocks = t30.importance_blocks(names, [0.1, 0.2, 0.3, 0.1, 0.3])
        self.assertAlmostEqual(sum(blocks.values()), 1.0)
        self.assertEqual({k: round(v, 6) for k, v in blocks.items()},
                         dict(raw_geo=0.1, te=0.2, cnt=0.3, onehot_geo=0.1, other=0.3))


if __name__ == '__main__':
    unittest.main()
