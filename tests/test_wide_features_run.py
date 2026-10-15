"""T27 전용 합성 검사. 개발 자료와 홀드아웃을 읽지 않는다."""
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
from experiments.wide_features import run as t27


def fake_fit(calls=None, fail=False, invalid=False, reload_diff=0):
    def fit(arm, Xa, ya, Xb, c):
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
            paths=[path]
            for suffix in ['preprocessor.joblib','features.joblib']:
                extra=Path(f'{prefix}_{suffix}')
                with extra.open('x',encoding='utf-8') as f: f.write('합성 전처리/피처 상태')
                paths.append(extra)
            return paths

        return dict(p=p, fit_seconds=0.01, save=save, verify=lambda *args: reload_diff,
                    info=dict(train_unknown_combinations={key:dict(count=0,rows=len(Xa),rate=0.) for key in t27.UNKNOWN_KEYS},
                              valid_unknown_combinations={key:dict(count=0,rows=len(Xb),rate=0.) for key in t27.UNKNOWN_KEYS},
                              candidate_encoded_columns=150,kept_columns=100,base_columns=90,full_columns=190,
                              matrix_statistics={'synthetic':True},preprocessing_seconds=dict(C=.01,features=.02,augmentation=.01,total=.04),
                              process_peak_rss=dict(bytes=None,available=False,scope='process_lifetime_high_water')))
    return fit


class Env:
    def __init__(self, root):
        self.out = Path(root)/'run'
        self.out.mkdir()
        (self.out/'backup').mkdir()
        t27.atomic_write(self.out/'ledger.json', dict(fits={}, finalizations={}))
        self.X, self.y, self.ids, _ = synthetic(450)
        self.X.loc[:14, 'age'] = 995
        self.fo = np.arange(len(self.X)) % 5
        self.c = copy.deepcopy(t27.config())
        p = np.full((len(self.X), 3), 0.05)
        p[np.arange(len(self.X)), (self.y.to_numpy()+1) % 3] = 0.9
        self.ref = pd.DataFrame({t27.ID: self.ids, 'fold': self.fo, 'y_true': self.y, 'y_pred': p.argmax(1)+1,
                                 **{n: p[:, j] for j, n in enumerate(t27.PROBS)}})
        self.design = dict(config=self.c, code_commit='a'*40)
        self.inputs = (self.X, self.y, self.ids, {s: self.fo for s in self.c['schemes']},
                       {s: self.ref for s in self.c['schemes']}, self.design)

    def fit(self, scheme='geo2_s42', arm='WIDE_FEATURES', f=0, fit_fn=None):
        a, b = np.flatnonzero(self.fo != f), np.flatnonzero(self.fo == f)
        return t27.checkpoint_fit(self.out, scheme, arm, f, self.X.iloc[a], self.y.iloc[a], self.X.iloc[b],
                                  self.y.iloc[b], self.ids.iloc[b], self.c, fit_fn or fake_fit())


class RunnerTests(unittest.TestCase):
    def test_failed_or_running_in_any_journal_blocks_other_filter_and_direct_fit(self):
        for section in ['fits','finalizations']:
            for status in ['failed','running']:
                with self.subTest(section=section,status=status),tempfile.TemporaryDirectory() as d:
                    e,calls=Env(d),[]
                    ledger=t27.read_ledger(e.out)
                    ledger[section]['existing']=dict(status=status,attempts=1)
                    t27.atomic_write(e.out/'ledger.json',ledger)
                    before=(e.out/'ledger.json').read_bytes()
                    with mock.patch.object(t27,'verify',return_value=e.inputs):
                        with self.assertRaises(ValueError):
                            t27.fit_all('synthetic',e.out,schemes=['geo2_s202'],folds=[2],fit_fn=fake_fit(calls))
                    with self.assertRaises(ValueError): e.fit(scheme='geo2_s101',f=1,fit_fn=fake_fit(calls))
                    self.assertEqual(calls,[])
                    self.assertEqual((e.out/'ledger.json').read_bytes(),before)

    def test_initial_fit_ledger_write_error_is_preserved_before_training(self):
        with tempfile.TemporaryDirectory() as d:
            e,calls=Env(d),[]
            original=t27.atomic_write
            writes=[]

            def fail_first(path,record):
                writes.append(1)
                if len(writes) == 1: raise OSError('합성 초기 원장 쓰기 실패')
                return original(path,record)

            with mock.patch.object(t27,'atomic_write',side_effect=fail_first):
                with self.assertRaises(OSError): e.fit(fit_fn=fake_fit(calls))
            self.assertEqual(calls,[])
            entry=t27.read_ledger(e.out)['fits']['geo2_s42/WIDE_FEATURES/f0']
            self.assertEqual(entry['status'],'failed')
            self.assertIn('합성 초기 원장 쓰기 실패',entry['failure_traceback'])
            with self.assertRaises(ValueError): e.fit(scheme='geo2_s202',f=2)

    def test_finalize_failure_is_journaled_and_blocks_new_work(self):
        with tempfile.TemporaryDirectory() as d,mock.patch.object(t27.t15,'log'):
            e=Env(d)
            with mock.patch.object(t27,'verify',return_value=e.inputs):
                t27.fit_all('synthetic',e.out,fit_fn=fake_fit())
                t27.write(e.out/'design.json',e.design)
                original=t27.write

                def fail_analysis(path,record):
                    if Path(path).name == 'analysis.json': raise OSError('합성 최종 집계 쓰기 실패')
                    return original(path,record)

                with mock.patch.object(t27,'write',side_effect=fail_analysis):
                    with self.assertRaises(OSError): t27.finalize_all('synthetic',e.out)
                ledger=t27.read_ledger(e.out)
                self.assertEqual(ledger['finalizations']['analysis']['status'],'failed')
                self.assertEqual(len(ledger['fits']),20)
                for entry in ledger['fits'].values(): t27.check_saved(e.out,entry)
                with self.assertRaises(ValueError): t27.fit_all('synthetic',e.out,schemes=['geo2_s303'],folds=[4])
                with self.assertRaises(ValueError): t27.finalize_all('synthetic',e.out)

    def test_initial_finalize_ledger_write_failure_blocks_followup(self):
        with tempfile.TemporaryDirectory() as d,mock.patch.object(t27.t15,'log'):
            e=Env(d)
            with mock.patch.object(t27,'verify',return_value=e.inputs):
                t27.fit_all('synthetic',e.out,fit_fn=fake_fit())
                original=t27.atomic_write;writes=[]
                def fail_first(path,record):
                    writes.append(1)
                    if len(writes)==1: raise OSError('합성 집계 시작 원장 쓰기 실패')
                    return original(path,record)
                with mock.patch.object(t27,'atomic_write',side_effect=fail_first):
                    with self.assertRaises(OSError): t27.finalize_all('synthetic',e.out)
                self.assertEqual(t27.read_ledger(e.out)['finalizations']['analysis']['status'],'failed')
                self.assertFalse((e.out/'analysis.json').exists())
                with self.assertRaises(ValueError): t27.fit_all('synthetic',e.out,folds=[4])

    def test_source_must_match_commit_blob_even_if_gitignored(self):
        digest = t27.hashlib.sha256(b'sealed').hexdigest()
        with mock.patch.object(t27.subprocess, 'run', return_value=mock.Mock(stdout=b'changed')):
            with self.assertRaises(ValueError):
                t27.verify_committed_sources({'ignored.py': digest}, 'a'*40)

    def test_frozen_config_and_exact_reference_paths_hashes(self):
        c, b = t27.config(), t27.t15.config()
        self.assertEqual([s['seed'] for s in c['schemes'].values()], [42, 101, 202, 303])
        self.assertEqual(c['lgbm_params'], b['lgbm_params'])
        self.assertEqual(c['lgbm_params']['n_estimators'], 300)
        self.assertEqual(c['lgbm_params']['num_leaves'], 31)
        self.assertEqual(c['max_new_study_fits'], 20)
        self.assertIsNone(c['sample_weight'])
        self.assertEqual(c['arms'], t27.ARMS)
        for s in c['schemes']:
            self.assertEqual(c['schemes'][s]['manifest'], b['schemes'][s]['manifest'])
            self.assertEqual(c['schemes'][s]['refs']['LGBM_O'], b['schemes'][s]['refs']['LGBM_O'])

    def test_config_rejects_weight_model_rule_and_seed_drift(self):
        mutations = [('sample_weight', [1., 1., 1.5]),
                     ('class_weight', 'balanced'), ('max_new_study_fits', 21),
                     ('decision_rule', 'grade3_offset'), ('early_stopping', True)]
        for name, value in mutations:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as d:
                c = copy.deepcopy(t27.config())
                c[name] = value
                path = Path(d)/'config.json'
                t27.write(path, c)
                with self.assertRaises(ValueError):
                    t27.config(path)

    def test_validation_labels_not_a_fit_argument_and_matrix_is_sparse(self):
        self.assertEqual(list(inspect.signature(t27.fit_enhanced).parameters), ['arm', 'Xa', 'ya', 'Xb', 'c'])
        from scipy import sparse
        mat = sparse.csr_matrix(np.eye(3, dtype=np.float32))
        evidence = t27.matrix_evidence(mat)
        self.assertEqual(evidence['shape'], [3, 3])
        self.assertEqual(evidence['dtype'], 'float32')
        self.assertEqual(set(evidence['arrays']), {'data', 'indices', 'indptr'})
        with self.assertRaises(ValueError):
            t27.matrix_evidence(np.eye(3))

    def test_source_import_closure_includes_custom_preprocessor_and_tests(self):
        sources = t27.sources()
        for path in ['experiments/age995/run.py', 'experiments/lgbm_native_cat/run.py',
                     'experiments/grade3_decision_rule/run.py', 'tests/test_modeling.py', *t27.OWN_SOURCE]:
            self.assertIn(path, sources)
        self.assertTrue(all(len(s) == 64 for s in sources.values()))

    def test_guard_blocks_raw_and_resolved_holdout(self):
        for p in ['../../earthquake/data/final_holdout', 'FINAL_HOLDOUT/data.csv']:
            with self.assertRaises(ValueError):
                t27.guard(p)

    def test_checkpoint_skip_backup_and_probability_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            e, calls = Env(d), []
            result, fresh = e.fit(fit_fn=fake_fit(calls))
            self.assertTrue(fresh)
            self.assertEqual(set(result['artifact_sha256']),{'fit_model.txt','fit_preprocessor.joblib',
                             'fit_features.joblib','pred.csv.gz','metrics.json'})
            same, fresh = e.fit(fit_fn=fake_fit(calls))
            self.assertFalse(fresh)
            self.assertEqual(len(calls), 1)
            self.assertEqual(result, same)
            t27.check_saved(e.out, same)
            pred = t27.read_pred(e.out/same['dir']/'pred.csv.gz')
            self.assertTrue((pred.y_pred.to_numpy() == pred[t27.PROBS].to_numpy().argmax(1)+1).all())

    def test_crash_records_failure_never_retries_and_preserves_original(self):
        with tempfile.TemporaryDirectory() as d:
            e = Env(d)
            with self.assertRaises(RuntimeError):
                e.fit(fit_fn=fake_fit(fail=True))
            before = (e.out/'geo2_s42/WIDE_FEATURES/f0/attempt1/failure.txt').read_bytes()
            with self.assertRaises(ValueError):
                e.fit()
            ledger = t27.read_ledger(e.out)
            self.assertEqual(ledger['fits']['geo2_s42/WIDE_FEATURES/f0']['status'], 'failed')
            self.assertEqual(ledger['fits']['geo2_s42/WIDE_FEATURES/f0']['attempts'], 1)
            self.assertEqual((e.out/'geo2_s42/WIDE_FEATURES/f0/attempt1/failure.txt').read_bytes(), before)

    def test_invalid_probability_reload_and_budget_fail_closed(self):
        for fit in [fake_fit(invalid=True), fake_fit(reload_diff=1e-6)]:
            with self.subTest(fit=fit), tempfile.TemporaryDirectory() as d:
                e = Env(d)
                with self.assertRaises(ValueError):
                    e.fit(fit_fn=fit)
                self.assertEqual(t27.read_ledger(e.out)['fits']['geo2_s42/WIDE_FEATURES/f0']['status'], 'failed')
        with tempfile.TemporaryDirectory() as d:
            e = Env(d)
            e.c['max_new_study_fits'] = 0
            with self.assertRaises(ValueError):
                e.fit()
            self.assertEqual(t27.read_ledger(e.out)['fits'], {})

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
            with mock.patch.object(t27, 'verify') as verify:
                with self.assertRaises(FileExistsError):
                    t27.fit_all('synthetic', e.out)
                verify.assert_not_called()
            self.assertEqual((e.out/'RUNNING.lock').read_text(encoding='utf-8'), '다른 실행')

    def test_filters_one_fold_then_remaining_no_duplicate_and_finalize(self):
        with tempfile.TemporaryDirectory() as d:
            e, calls = Env(d), []
            with mock.patch.object(t27, 'verify', return_value=e.inputs):
                t27.fit_all('synthetic', e.out, [t27.ARMS[0]], ['geo2_s42'], [0], fake_fit(calls))
                self.assertEqual(len(calls), 1)
                t27.fit_all('synthetic', e.out, fit_fn=fake_fit(calls))
                self.assertEqual(len(calls), 20)
                t27.write(e.out/'design.json', e.design)
                result = t27.finalize_all('synthetic', e.out)
                self.assertEqual(result['fit_attempts'], 20)
                arm_result=result['arms']['WIDE_FEATURES']
                self.assertEqual(arm_result['feature_width_range']['kept'],dict(min=100,max=100))
                self.assertAlmostEqual(arm_result['preprocessing_seconds_sum']['total'],.8)
                self.assertEqual(set(arm_result['unknown_valid']),t27.UNKNOWN_KEYS)
                for value in arm_result['unknown_valid'].values(): self.assertEqual(value['rows'],1800)
                for value in arm_result['unknown_train'].values(): self.assertEqual(value['rows'],7200)
                self.assertEqual(result['pooled_denominator']['predictions'], 1800)
                self.assertEqual(result['pooled_denominator']['unique_development_rows'], 450)
                self.assertEqual(len(result['fold_records']), 20)
                for arm in t27.ARMS:
                    self.assertEqual(len(result['arms'][arm]['fold_records']), 20)
                    self.assertEqual(result['arms'][arm]['turnover']['rows'], 1800)
                    self.assertEqual(result['arms'][arm]['pooled_profiles']['age995']['enhanced']['rows'], 60)
                self.assertEqual(set(result['arms'][t27.ARMS[0]]['judgment']['gates']), {'mean_delta_ok', 'fold_wins_ok',
                                  'positive_seeds_ok', 'pooled_recall3_ok', 'pooled_error31_rate_ok'})
                self.assertEqual((e.out/'analysis.json').read_bytes(), (e.out/'backup/analysis.json').read_bytes())
                with self.assertRaises(ValueError):
                    t27.finalize_all('synthetic', e.out)
            for a, _, b in calls:
                self.assertFalse(set(a) & set(b))

    def test_finalize_incomplete_or_empty_filter_stops(self):
        with tempfile.TemporaryDirectory() as d:
            e = Env(d)
            with mock.patch.object(t27, 'verify', return_value=e.inputs):
                with self.assertRaises(ValueError):
                    t27.finalize_all('synthetic', e.out)
                for schemes, folds in [([], None), (None, []), (['invalid'], None), (None, [5])]:
                    with self.assertRaises(ValueError):
                        t27.fit_all('synthetic', e.out, schemes=schemes, folds=folds)

    def test_bad_seal_stops_before_reading_any_data(self):
        with tempfile.TemporaryDirectory() as d:
            t27.write(Path(d)/'design.json', {})
            t27.write(Path(d)/'design_seal.json', {'design_sha256': 'bad'})
            with mock.patch.object(t27, 'load') as load:
                with self.assertRaises(ValueError):
                    t27.verify('synthetic', d)
                load.assert_not_called()

    def test_judgment_strict_wins_seed_support_and_safety_gates(self):
        c = t27.config()
        cm = [[10, 0, 0], [0, 10, 0], [1, 3, 6]]
        baseline = dict(macro_f1=.7, confusion=cm)
        records = [dict(scheme=s, fold=f, baseline=baseline,
                        enhanced=dict(macro_f1=.704, confusion=[[10,0,0],[0,10,0],[0,3,7]]))
                   for s in c['schemes'] for f in range(5)]
        self.assertTrue(t27.judge(records, c)['supports_fixed_hypothesis'])
        for record in records[:3]:
            record['enhanced']['confusion'] = [[10,0,0],[0,10,0],[10,0,0]]
        verdict = t27.judge(records, c)
        self.assertFalse(verdict['gates']['pooled_error31_rate_ok'])
        self.assertFalse(verdict['supports_fixed_hypothesis'])
        for record in records:
            record['enhanced'] = copy.deepcopy(baseline)
        self.assertEqual(t27.judge(records, c)['fold_wins'], 0)

    def test_mocked_model_keeps_C_bytes_and_uses_actual_wide_filter_and_reload(self):
        from scipy import sparse
        X,y,_,_=synthetic(900)
        X.loc[:2,'age']=995
        X.loc[700:,'geo_level_1_id']=999999
        c=t27.config()
        pre=t27.t8.make_preprocessor('O')
        base=pre.fit_transform(X.iloc[:700])
        calls=[]
        class Native:
            def current_iteration(self): return 300
            def save_model(self,path): Path(path).write_text('synthetic native',encoding='utf-8')
        class Model:
            def __init__(self,**params): self.class_weight=None; self.booster_=Native()
            def fit(self,matrix,labels,**kw):
                self_test.assertEqual(kw,{})
                self_test.assertEqual(t27.matrix_evidence(matrix[:,:base.shape[1]]),t27.matrix_evidence(base))
                np.testing.assert_array_equal(labels,y.iloc[:700].to_numpy()-1)
                self.classes_=np.asarray([0,1,2]);calls.append(matrix.shape)
                return self
            def predict_proba(self,matrix): return np.tile([.1,.3,.6],(matrix.shape[0],1))
        class Booster:
            def __init__(self,**kw): pass
            def num_model_per_iteration(self): return 3
            def predict(self,matrix,**kw): return np.tile([.1,.3,.6],(matrix.shape[0],1))
        self_test=self
        with tempfile.TemporaryDirectory() as d,mock.patch('lightgbm.LGBMClassifier',Model),mock.patch('lightgbm.Booster',Booster):
            result=t27.fit_enhanced('WIDE_FEATURES',X.iloc[:700],y.iloc[:700],X.iloc[700:],c)
            paths=result['save'](Path(d)/'fit')
            self.assertEqual(len(paths),3)
            self.assertEqual(result['verify'](paths,X.iloc[700:],result['p']),0.)
            with self.assertRaises(ValueError): result['save'](Path(d)/'fit')
        info=result['info']
        self.assertEqual(len(calls),1)
        self.assertEqual(info['matrix_evidence']['base_train'],t27.matrix_evidence(base))
        self.assertEqual(set(info['matrix_evidence']),{'base_train','base_valid','extra_train','extra_valid','full_train','full_valid'})
        self.assertEqual(info['feature_metadata']['candidate_numeric_count'],95)
        self.assertEqual(info['feature_metadata']['candidate_logical_count'],101)
        self.assertEqual(info['feature_metadata']['n_train_rows'],700)
        self.assertEqual(info['feature_age995_median'],info['age995_median'])
        self.assertTrue(info['C_prefix_byte_exact'])
        self.assertEqual(set(info['valid_unknown_combinations']),t27.UNKNOWN_KEYS)
        self.assertEqual(info['valid_unknown_combinations']['geo_level_1_id']['count'],200)
        self.assertEqual(info['extra_columns'],info['kept_columns'])
        self.assertGreaterEqual(info['candidate_encoded_columns'],info['kept_columns'])
        self.assertEqual(info['full_columns'],info['base_columns']+info['kept_columns'])
        self.assertEqual(set(info['preprocessing_seconds']),{'C','features','augmentation','total'})
        self.assertGreaterEqual(info['preprocessing_seconds']['total'],0.)
        for key,value in info['matrix_statistics'].items():
            self.assertEqual(value['shape'],info['matrix_evidence'][key]['shape'])
            self.assertGreaterEqual(value['csr_bytes'],0)

    def test_C_and_wide_age_median_mismatch_stops_before_any_model_fit(self):
        X,y,_,_=synthetic(450)
        original=t27.WideFeatures
        class Mismatched(original):
            def fit(self,X,y=None):
                result=super().fit(X,y)
                self.median_+=1
                return result
        with mock.patch.object(t27,'WideFeatures',Mismatched),mock.patch('lightgbm.LGBMClassifier') as model:
            with self.assertRaises(ValueError):
                t27.fit_enhanced('WIDE_FEATURES',X.iloc[:350],y.iloc[:350],X.iloc[350:],t27.config())
            model.assert_not_called()

    def test_feature_filter_policy_drift_is_refused(self):
        for key,value in [('rare_filter',True),('compare_with_C',True),('numeric_candidates',94),('target_used',True)]:
            with self.subTest(key=key),tempfile.TemporaryDirectory() as d:
                c=copy.deepcopy(t27.config());c['feature_policy'][key]=value
                path=Path(d)/'config.json';t27.write(path,c)
                with self.assertRaises(ValueError): t27.config(path)

    def test_actual_feature_metadata_filter_consistency_and_tamper_refusal(self):
        X,_,_,_=synthetic(450)
        feature=t27.WideFeatures('WIDE_FEATURES').fit(X)
        matrix=feature.transform(X)
        metadata=feature.metadata()
        t27.validate_feature_metadata(feature,metadata,matrix)
        self.assertEqual(metadata['retained_count'],matrix.shape[1])
        mutations=[('retained_count',-1),('candidate_encoded_count',-1),('age_median',-1),
                   ('target_used',True),('keep_indices',[])]
        for key,value in mutations:
            changed=copy.deepcopy(metadata);changed[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                t27.validate_feature_metadata(feature,changed,matrix)
        changed=copy.deepcopy(metadata)
        changed['candidates'][feature.keep_indices_[0]]['support']=-1
        with self.assertRaises(ValueError): t27.validate_feature_metadata(feature,changed,matrix)

    def test_unknown7_aggregation_uses_summed_denominators_and_refuses_wrong_keys(self):
        records=[{key:dict(count=1,rows=n,rate=1/n) for key in t27.UNKNOWN_KEYS} for n in [2,8]]
        pooled=t27.pool_unknown_counts(records)
        self.assertEqual(set(pooled),t27.UNKNOWN_KEYS)
        for value in pooled.values(): self.assertEqual(value,dict(count=2,rows=10,rate=.2))
        with self.assertRaises(ValueError): t27.pool_unknown_counts([{'geo_level_1_id':dict(count=0,rows=1,rate=0.)}])

    def test_no_t21_t22_t23_sources_config_or_results_are_linked(self):
        self.assertFalse(any(any(name in p for name in ['feature_interactions','material_structure_interactions','ordinal_damage','geo_context_features','catboost_c','catboost_native_c']) for p in t27.sources()))
        self.assertEqual(len(t27.sources()), 49)
        c = copy.deepcopy(t27.config())
        c['arms'] = ['WIDE_FEATURES','OTHER_ARM']
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'config.json'
            t27.write(p,c)
            with self.assertRaises(ValueError):
                t27.config(p)

    def test_augment_rejects_dense_wrong_dtype_and_nonfinite(self):
        from scipy import sparse
        base = sparse.csr_matrix(np.eye(3, dtype=np.float32))
        for extra in [np.eye(3), sparse.csr_matrix(np.eye(3, dtype=np.float64)), sparse.csr_matrix(np.full((3,1),np.nan,dtype=np.float32))]:
            with self.assertRaises(ValueError):
                t27.augment(base, extra)

    def test_turnover_has_correct_actual_grade3_denominator(self):
        y = [1,2,3,3,3,2]
        base = [1,1,1,2,3,2]
        enhanced = [2,2,3,1,3,2]
        record = t27.turnover(y, base, enhanced)
        self.assertEqual(record['corrected'], 2)
        self.assertEqual(record['newly_wrong'], 1)
        self.assertEqual(record['error31_removed'], 1)
        self.assertEqual(record['error31_introduced'], 1)
        self.assertEqual(record['actual_grade3_rows'], 3)
        merged = t27.merge_turnovers([record, record])
        self.assertEqual(merged['actual_grade3_rows'], 6)
        self.assertEqual(merged['rows'], 12)

    def test_profile_can_have_missing_actual_class_without_failure(self):
        result = t27.pooled_metrics([[[0,0,0],[0,1,0],[1,0,2]]])
        self.assertEqual(result['support'], [0,1,3])
        self.assertEqual(result['error31_rate'], 1/3)


if __name__ == '__main__':
    unittest.main()
