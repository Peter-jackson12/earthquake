"""T23 실행기 합성 검사. 개발 자료·홀드아웃을 읽지 않는다."""
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
from test_modeling import synthetic
from experiments.ordinal_damage import run as t23


def fake_fit(calls=None, fail_threshold=None, invalid=False, reload_diff=0, different_matrix=False):
    def fit(threshold, Xa, ya, Xb, c):
        if calls is not None:
            calls.append((threshold, Xa.index.to_numpy(), np.asarray(ya), Xb.index.to_numpy()))
        if threshold == fail_threshold:
            raise RuntimeError('합성 임계모델 중단')
        q = np.asarray(([.8, .6, .2] if threshold == 2 else [.2, .7, .9]))[Xb.index.to_numpy() % 3]
        if invalid:
            q[0] = np.nan

        def save(prefix):
            path = Path(f'{prefix}_model.txt')
            with path.open('x', encoding='utf-8') as f:
                f.write(f'합성 이진 모델 {threshold}')
            pre = Path(f'{prefix}_preprocessor.joblib')
            with pre.open('x', encoding='utf-8') as f:
                f.write('동일한 합성 C 전처리')
            return [path, pre]

        cols = threshold if different_matrix else 2
        matrices = {name: t23.matrix_evidence(sparse.csr_matrix((rows, cols), dtype=np.float32))
                    for name, rows in [('train',len(Xa)),('valid',len(Xb))]}
        return dict(q=q, fit_seconds=.01, save=save, verify=lambda *a: reload_diff,
                    info=dict(matrix_evidence=matrices))
    return fit


class Env:
    def __init__(self, root):
        self.out = Path(root)/'run'
        self.out.mkdir()
        (self.out/'backup').mkdir()
        t23.atomic_write(self.out/'ledger.json', dict(fits={}, units={}))
        self.X, self.y, self.ids, _ = synthetic(450)
        self.X.loc[:14,'age'] = 995
        self.fo = np.arange(len(self.X)) % 5
        self.c = copy.deepcopy(t23.config())
        p = np.full((len(self.X),3), .05)
        p[np.arange(len(self.X)), self.y.to_numpy() % 3] = .9
        self.ref = pd.DataFrame({t23.ID:self.ids,'fold':self.fo,'y_true':self.y,'y_pred':p.argmax(1)+1,
                                 **{n:p[:,j] for j,n in enumerate(t23.PROBS)}})
        self.design = dict(config=self.c, code_commit='a'*40)
        self.inputs = self.X,self.y,self.ids,{s:self.fo for s in self.c['schemes']},{s:self.ref for s in self.c['schemes']},self.design

    def fit(self, threshold=2, scheme='geo2_s42', f=0, fit_fn=None):
        a,b = np.flatnonzero(self.fo != f),np.flatnonzero(self.fo == f)
        return t23.checkpoint_fit(self.out,scheme,f,threshold,self.X.iloc[a],self.y.iloc[a],self.X.iloc[b],
                                   self.y.iloc[b],self.ids.iloc[b],self.c,fit_fn or fake_fit())

    def merge(self, scheme='geo2_s42', f=0):
        b = np.flatnonzero(self.fo == f)
        return t23.merge_unit(self.out,scheme,f,self.ids.iloc[b],self.y.iloc[b],self.c)


class RunnerTests(unittest.TestCase):
    def test_exact_binary_config_baseline_and_reference_pins(self):
        c,b = t23.config(),t23.t15.config()
        expected = dict(b['lgbm_params'],objective='binary')
        expected.pop('num_class')
        self.assertEqual(c['binary_lgbm_params'],expected)
        self.assertEqual(c['lgbm_params'],b['lgbm_params'])
        self.assertEqual(c['arms'],['CUMULATIVE_BINARY'])
        self.assertEqual(c['thresholds'],[2,3])
        self.assertEqual(c['max_new_study_fits'],40)
        self.assertEqual(c['max_outer_units'],20)
        self.assertEqual([s['seed'] for s in c['schemes'].values()],[42,101,202,303])
        for s in c['schemes']:
            self.assertEqual(c['schemes'][s]['manifest'],b['schemes'][s]['manifest'])
            self.assertEqual(c['schemes'][s]['refs']['LGBM_O'],b['schemes'][s]['refs']['LGBM_O'])

    def test_config_rejects_threshold_weight_objective_or_budget_drift(self):
        for key,value in [('thresholds',[1,3]),('class_weight','balanced'),('sample_weight',[1,1,1.5]),
                          ('max_new_study_fits',41),('decision_rule','argmax'),('early_stopping',True)]:
            with self.subTest(key=key),tempfile.TemporaryDirectory() as d:
                c=copy.deepcopy(t23.config())
                c[key]=value
                p=Path(d)/'config.json'
                t23.write(p,c)
                with self.assertRaises(ValueError): t23.config(p)

    def test_fit_signature_has_no_validation_labels(self):
        self.assertEqual(list(inspect.signature(t23.fit_binary).parameters),['threshold','Xa','ya','Xb','c','prepared'])
        self.assertNotIn('yb',inspect.signature(t23.prepare_c).parameters)

    def test_source_closure_has_adapter_and_no_t21_t22_runner(self):
        paths=t23.sources()
        self.assertTrue(set(t23.OWN_SOURCE) <= set(paths))
        self.assertIn('experiments/age995/run.py',paths)
        self.assertIn('experiments/ordinal_damage/model.py',paths)
        self.assertFalse(any('feature_interactions' in p or 'material_structure_interactions' in p for p in paths))

    def test_source_must_match_commit_blob(self):
        with mock.patch.object(t23.subprocess,'run',return_value=mock.Mock(stdout=b'changed')):
            with self.assertRaises(ValueError): t23.verify_committed_sources({'ignored.py':'a'*64},'b'*40)

    def test_bad_seal_blocks_data_loading(self):
        with tempfile.TemporaryDirectory() as d:
            t23.write(Path(d)/'design.json',{})
            t23.write(Path(d)/'design_seal.json',dict(design_sha256='bad'))
            with mock.patch.object(t23,'load') as load:
                with self.assertRaises(ValueError): t23.verify('synthetic',d)
                load.assert_not_called()

    def test_existing_lock_blocks_without_deleting_original(self):
        with tempfile.TemporaryDirectory() as d:
            e=Env(d)
            (e.out/'RUNNING.lock').write_text('원래 실행',encoding='utf-8')
            with mock.patch.object(t23,'verify') as verify:
                with self.assertRaises(FileExistsError): t23.fit_all('synthetic',e.out)
                verify.assert_not_called()
            self.assertEqual((e.out/'RUNNING.lock').read_text(encoding='utf-8'),'원래 실행')

    def test_guard_and_cumulative_probability_fail_closed(self):
        with self.assertRaises(ValueError): t23.guard('../../earthquake/data/final_holdout')
        for q in [[np.nan],[1.1],[-.1],[[.3]]]:
            with self.assertRaises(ValueError): t23.cumulative(q,1)

    def test_binary_checkpoint_skips_complete_and_raw_q_roundtrips(self):
        with tempfile.TemporaryDirectory() as d:
            e,calls=Env(d),[]
            entry,fresh=e.fit(fit_fn=fake_fit(calls))
            self.assertTrue(fresh)
            same,fresh=e.fit(fit_fn=fake_fit(calls))
            self.assertFalse(fresh)
            self.assertEqual(entry,same)
            self.assertEqual(len(calls),1)
            raw=t23.read_pred(e.out/entry['dir']/'raw_q.csv.gz')
            np.testing.assert_array_equal(raw.y_binary,(raw.y_true >= 2).astype(int))
            t23.check_saved(e.out,entry)

    def test_first_failure_does_not_start_second(self):
        with tempfile.TemporaryDirectory() as d:
            e,calls=Env(d),[]
            with mock.patch.object(t23,'verify',return_value=e.inputs):
                with self.assertRaises(RuntimeError):
                    t23.fit_all('synthetic',e.out,schemes=['geo2_s42'],folds=[0],fit_fn=fake_fit(calls,fail_threshold=2))
            self.assertEqual([x[0] for x in calls],[2])
            ledger=t23.read_ledger(e.out)
            self.assertEqual(len(ledger['fits']),1)
            self.assertEqual(ledger['units'],{})

    def test_second_failure_preserves_first_and_forbids_all_retries(self):
        with tempfile.TemporaryDirectory() as d:
            e,calls=Env(d),[]
            with mock.patch.object(t23,'verify',return_value=e.inputs):
                with self.assertRaises(RuntimeError):
                    t23.fit_all('synthetic',e.out,schemes=['geo2_s42'],folds=[0],fit_fn=fake_fit(calls,fail_threshold=3))
                ledger=t23.read_ledger(e.out)
                unit='geo2_s42/CUMULATIVE_BINARY/f0'
                first=ledger['fits'][unit+'/ge2']
                self.assertEqual(first['status'],'complete')
                self.assertEqual(ledger['fits'][unit+'/ge3']['status'],'failed')
                t23.check_saved(e.out,first)
                marker=(e.out/first['dir']/'raw_q.csv.gz').read_bytes()
                with self.assertRaises(ValueError):
                    t23.fit_all('synthetic',e.out,schemes=['geo2_s42'],folds=[0],fit_fn=fake_fit(calls))
                self.assertEqual([x[0] for x in calls],[2,3])
                self.assertEqual((e.out/first['dir']/'raw_q.csv.gz').read_bytes(),marker)
                self.assertEqual(sum(x['attempts'] for x in t23.read_ledger(e.out)['fits'].values()),2)

    def test_stop_between_models_resumes_only_unstarted_second(self):
        with tempfile.TemporaryDirectory() as d:
            e,calls=Env(d),[]
            e.fit(2,fit_fn=fake_fit(calls))
            with mock.patch.object(t23,'verify',return_value=e.inputs):
                t23.fit_all('synthetic',e.out,schemes=['geo2_s42'],folds=[0],fit_fn=fake_fit(calls))
            self.assertEqual([x[0] for x in calls],[2,3])
            self.assertEqual(len(t23.read_ledger(e.out)['units']),1)

    def test_global_failed_or_running_entry_blocks_other_filters(self):
        for status in ['failed','running']:
            with self.subTest(status=status),tempfile.TemporaryDirectory() as d:
                e,calls=Env(d),[]
                ledger=t23.read_ledger(e.out)
                ledger['fits']['geo2_s42/CUMULATIVE_BINARY/f0/ge3']=dict(status=status,attempts=1)
                t23.atomic_write(e.out/'ledger.json',ledger)
                before=(e.out/'ledger.json').read_bytes()
                with mock.patch.object(t23,'verify',return_value=e.inputs):
                    with self.assertRaises(ValueError):
                        t23.fit_all('synthetic',e.out,schemes=['geo2_s101'],folds=[1],fit_fn=fake_fit(calls))
                self.assertEqual(calls,[])
                self.assertEqual((e.out/'ledger.json').read_bytes(),before)

    def test_direct_checkpoint_cannot_bypass_global_failure_or_running(self):
        for status in ['failed','running']:
            with self.subTest(status=status),tempfile.TemporaryDirectory() as d:
                e,calls=Env(d),[]
                ledger=t23.read_ledger(e.out)
                ledger['fits']['geo2_s42/CUMULATIVE_BINARY/f0/ge3']=dict(status=status,attempts=1)
                t23.atomic_write(e.out/'ledger.json',ledger)
                with self.assertRaises(ValueError):
                    e.fit(2,scheme='geo2_s202',f=2,fit_fn=fake_fit(calls))
                self.assertEqual(calls,[])
                self.assertEqual(len(t23.read_ledger(e.out)['fits']),1)

    def test_merge_failure_records_stop_and_blocks_other_unit_fit(self):
        with tempfile.TemporaryDirectory() as d:
            e,calls=Env(d),[]
            first,_=e.fit(2)
            second,_=e.fit(3)
            with mock.patch.object(t23,'write_pred',side_effect=OSError('합성 결합 저장 실패')):
                with self.assertRaises(OSError): e.merge()
            ledger=t23.read_ledger(e.out)
            self.assertEqual(ledger['units']['geo2_s42/CUMULATIVE_BINARY/f0']['status'],'failed')
            self.assertIn('합성 결합 저장 실패',ledger['units']['geo2_s42/CUMULATIVE_BINARY/f0']['failure_traceback'])
            t23.check_saved(e.out,first)
            t23.check_saved(e.out,second)
            with mock.patch.object(t23,'verify',return_value=e.inputs):
                with self.assertRaises(ValueError):
                    t23.fit_all('synthetic',e.out,schemes=['geo2_s303'],folds=[4],fit_fn=fake_fit(calls))
            with self.assertRaises(ValueError): e.fit(2,scheme='geo2_s202',f=2,fit_fn=fake_fit(calls))
            with self.assertRaises(ValueError): e.merge()
            self.assertEqual(calls,[])

    def test_running_merge_also_blocks_other_unit(self):
        with tempfile.TemporaryDirectory() as d:
            e,calls=Env(d),[]
            ledger=t23.read_ledger(e.out)
            ledger['units']['geo2_s42/CUMULATIVE_BINARY/f0']=dict(status='running')
            t23.atomic_write(e.out/'ledger.json',ledger)
            with mock.patch.object(t23,'verify',return_value=e.inputs):
                with self.assertRaises(ValueError):
                    t23.fit_all('synthetic',e.out,schemes=['geo2_s101'],folds=[1],fit_fn=fake_fit(calls))
            with self.assertRaises(ValueError): e.fit(2,scheme='geo2_s202',f=2,fit_fn=fake_fit(calls))
            self.assertEqual(calls,[])

    def test_merge_ledger_start_error_records_failed_before_directory_creation(self):
        with tempfile.TemporaryDirectory() as d:
            e=Env(d)
            e.fit(2)
            e.fit(3)
            original=t23.atomic_write
            writes=[]

            def fail_first(path,record):
                writes.append(1)
                if len(writes) == 1:
                    raise OSError('합성 결합 원장 시작 실패')
                return original(path,record)

            with mock.patch.object(t23,'atomic_write',side_effect=fail_first):
                with self.assertRaises(OSError): e.merge()
            ledger=t23.read_ledger(e.out)
            self.assertEqual(ledger['units']['geo2_s42/CUMULATIVE_BINARY/f0']['status'],'failed')
            self.assertFalse((e.out/'geo2_s42/CUMULATIVE_BINARY/f0/merged').exists())
            with self.assertRaises(ValueError): e.fit(2,scheme='geo2_s202',f=2)

    def test_invalid_q_reload_and_actual_binary_budget_stop(self):
        for fit in [fake_fit(invalid=True),fake_fit(reload_diff=1e-5)]:
            with tempfile.TemporaryDirectory() as d:
                e=Env(d)
                with self.assertRaises(ValueError): e.fit(fit_fn=fit)
        with tempfile.TemporaryDirectory() as d:
            e=Env(d)
            e.c['max_new_study_fits']=1
            e.fit(2)
            with self.assertRaises(ValueError): e.fit(3)
            self.assertEqual(len(t23.read_ledger(e.out)['fits']),1)

    def test_backup_tamper_prevents_skip(self):
        with tempfile.TemporaryDirectory() as d:
            e=Env(d)
            entry,_=e.fit()
            (e.out/'backup'/entry['dir']/'metrics.json').write_text('{}',encoding='utf-8')
            with self.assertRaises(ValueError): e.fit()

    def test_merge_requires_both_same_c_inputs_and_proves_projection(self):
        with tempfile.TemporaryDirectory() as d:
            e=Env(d)
            e.fit(2)
            with self.assertRaises(ValueError): e.merge()
            e.fit(3)
            entry,fresh=e.merge()
            self.assertTrue(fresh)
            _,fresh=e.merge()
            self.assertFalse(fresh)
            frame=t23.read_pred(e.out/entry['dir']/'pred.csv.gz')
            ledger=t23.read_ledger(e.out)
            qs=[t23.read_pred(e.out/ledger['fits'][f'geo2_s42/CUMULATIVE_BINARY/f0/ge{t}']['dir']/'raw_q.csv.gz').q_ge for t in [2,3]]
            p,crossing=t23.project_cumulative_probabilities(*qs,return_metadata=True)
            np.testing.assert_array_equal(frame[t23.PROBS],p)
            np.testing.assert_array_equal(frame.y_pred,t23.damage_argmax(p))
            self.assertTrue(crossing['crossing_count'] > 0)
            self.assertTrue((p[np.asarray(qs[1]) > np.asarray(qs[0]),1] == 0).all())
        with tempfile.TemporaryDirectory() as d:
            e=Env(d)
            e.fit(2,fit_fn=fake_fit(different_matrix=True))
            e.fit(3,fit_fn=fake_fit(different_matrix=True))
            with self.assertRaises(ValueError): e.merge()

    def test_filters_complete40_binary20_units_and_finalize(self):
        with tempfile.TemporaryDirectory() as d,mock.patch.object(t23.t15,'log'):
            e,calls=Env(d),[]
            with mock.patch.object(t23,'verify',return_value=e.inputs):
                t23.fit_all('synthetic',e.out,schemes=['geo2_s42'],folds=[0],fit_fn=fake_fit(calls))
                self.assertEqual(len(calls),2)
                t23.fit_all('synthetic',e.out,fit_fn=fake_fit(calls))
                self.assertEqual(len(calls),40)
                self.assertEqual(len(t23.read_ledger(e.out)['units']),20)
                t23.write(e.out/'design.json',e.design)
                result=t23.finalize_all('synthetic',e.out)
                self.assertEqual(result['binary_fit_attempts'],40)
                self.assertEqual(result['completed_outer_units'],20)
                arm=result['arms']['CUMULATIVE_BINARY']
                self.assertEqual(arm['binary_fit_attempts'],40)
                self.assertEqual(arm['completed_outer_units'],20)
                self.assertEqual(len(arm['fold_records']),20)
                self.assertEqual(arm['crossing']['rows'],1800)
                self.assertEqual(arm['pooled_profiles']['age995']['enhanced']['rows'],60)
                self.assertEqual(arm['enhanced_ordinal']['rows'],1800)
                self.assertEqual(set(arm['judgment']['gates']),{'mean_delta_ok','fold_wins_ok','positive_seeds_ok','pooled_recall3_ok','pooled_error31_rate_ok'})
                self.assertEqual((e.out/'analysis.json').read_bytes(),(e.out/'backup/analysis.json').read_bytes())
                with self.assertRaises(ValueError): t23.finalize_all('synthetic',e.out)
            for _,a,_,b in calls: self.assertFalse(set(a) & set(b))

    def test_empty_filters_and_incomplete_finalize_stop(self):
        with tempfile.TemporaryDirectory() as d:
            e=Env(d)
            with mock.patch.object(t23,'verify',return_value=e.inputs):
                with self.assertRaises(ValueError): t23.finalize_all('synthetic',e.out)
                for arms,schemes,folds in [([],None,None),(None,[],None),(None,None,[]),(['wrong'],None,None),(None,None,[5])]:
                    with self.assertRaises(ValueError): t23.fit_all('synthetic',e.out,arms,schemes,folds)

    def test_ordinal_and_crossing_denominators_and_tie_rule(self):
        metrics=t23.ordinal_metrics([1,2,3],[3,2,1])
        self.assertEqual(metrics['ordinal_mae'],4/3)
        self.assertEqual(metrics['two_grade_error_rate'],2/3)
        self.assertEqual(metrics['class2_precision'],1.)
        self.assertEqual(t23.pool_ordinal([metrics,metrics])['rows'],6)
        p,crossing=t23.project_cumulative_probabilities([.2,.5],[.8,.5],return_metadata=True)
        np.testing.assert_array_equal(t23.damage_argmax(p),[1,1])
        pooled=t23.pool_crossings([crossing,crossing])
        self.assertEqual(pooled['crossing_rate'],.5)
        self.assertEqual(pooled['mean_crossing_excess_denominator'],2)
        self.assertEqual(pooled['mean_crossing_excess'],crossing['mean_crossing_excess'])

    def test_real_two_binary_share_c_native_reload_projection(self):
        from lightgbm import LGBMClassifier
        X,y,_,_=synthetic(900)
        X.loc[:2,'age']=995
        X.loc[700:,'geo_level_2_id']=999999
        c=copy.deepcopy(t23.config())
        c['binary_lgbm_params'].update(n_estimators=5,verbosity=-1)
        prepared=t23.prepare_c(X.iloc[:700],X.iloc[700:],c)
        original_fit=LGBMClassifier.fit
        matrix_ids=[]
        threshold_order=[]

        def spy(model,matrix,labels,**kwargs):
            threshold=[2,3][len(matrix_ids)]
            self.assertTrue(sparse.isspmatrix_csr(matrix))
            self.assertEqual(matrix.dtype,np.float32)
            self.assertEqual(kwargs,{})
            self.assertEqual(model.get_params()['objective'],'binary')
            self.assertNotIn('num_class',model.get_params())
            np.testing.assert_array_equal(labels,(y.iloc[:700].to_numpy() >= threshold).astype(int))
            matrix_ids.append(id(matrix))
            threshold_order.append(threshold)
            return original_fit(model,matrix,labels,**kwargs)

        qs=[]
        pre_shas=[]
        with tempfile.TemporaryDirectory() as d,mock.patch.object(LGBMClassifier,'fit',new=spy):
            for threshold in [2,3]:
                result=t23.fit_binary(threshold,X.iloc[:700],y.iloc[:700],X.iloc[700:],c,prepared=prepared)
                paths=result['save'](Path(d)/f'ge{threshold}')
                self.assertEqual(len(paths),2)
                pre_shas.append(t23.sha256_file(paths[1]))
                self.assertLessEqual(result['verify'](paths,X.iloc[700:],result['q']),1e-12)
                self.assertEqual(result['info']['internal_classes'],[0,1])
                self.assertEqual(result['info']['actual_trees'],5)
                self.assertEqual(result['info']['matrix_evidence']['train'],t23.matrix_evidence(prepared[1]))
                from lightgbm import Booster
                reloaded=Booster(model_file=str(paths[0])).predict(t23.joblib.load(paths[1]).transform(X.iloc[700:]),num_threads=2)
                np.testing.assert_array_equal(reloaded,result['q'])
                qs.append(result['q'])
            before=t23.project_cumulative_probabilities(*qs)
            after=t23.project_cumulative_probabilities(qs[0].copy(),qs[1].copy())
            np.testing.assert_array_equal(t23.damage_argmax(before),t23.damage_argmax(after))
        self.assertEqual(threshold_order,[2,3])
        self.assertEqual(matrix_ids,[id(prepared[1])]*2)
        self.assertEqual(pre_shas[0],pre_shas[1])


if __name__ == '__main__':
    unittest.main()
