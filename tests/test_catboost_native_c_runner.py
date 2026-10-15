"""T26 전용 합성 검사. 개발 자료와 홀드아웃을 읽지 않는다."""
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
from experiments.catboost_native_c import run as t26


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
            path = Path(f'{prefix}_model.cbm')
            with path.open('x', encoding='utf-8') as f:
                f.write('합성 모델')
            pre=Path(f'{prefix}_preprocessor.joblib')
            pre.write_text('합성 C 전처리',encoding='utf-8')
            return [path,pre]

        return dict(p=p, fit_seconds=0.01, save=save, verify=lambda *args: reload_diff,
                    info=dict(train_unknown_categories={f'cat_{j}':dict(count=0,rows=len(Xa),rate=0.) for j in range(11)},
                              valid_unknown_categories={f'cat_{j}':dict(count=0,rows=len(Xb),rate=0.) for j in range(11)}))
    return fit


class Env:
    def __init__(self, root):
        self.out = Path(root)/'run'
        self.out.mkdir()
        (self.out/'backup').mkdir()
        t26.atomic_write(self.out/'ledger.json', dict(fits={}, finalizations={}))
        self.X, self.y, self.ids, _ = synthetic(450)
        self.X.loc[:14, 'age'] = 995
        self.fo = np.arange(len(self.X)) % 5
        self.c = copy.deepcopy(t26.config())
        p = np.full((len(self.X), 3), 0.05)
        p[np.arange(len(self.X)), (self.y.to_numpy()+1) % 3] = 0.9
        self.ref = pd.DataFrame({t26.ID: self.ids, 'fold': self.fo, 'y_true': self.y, 'y_pred': p.argmax(1)+1,
                                 **{n: p[:, j] for j, n in enumerate(t26.PROBS)}})
        self.design = dict(config=self.c, code_commit='a'*40,primary_provenance={'synthetic':True})
        secondary=self.ref.copy()
        secondary['y_pred']=self.y.to_numpy()
        for j,n in enumerate(t26.PROBS): secondary[n]=np.where(self.y.to_numpy()==j+1,.9,.05)
        self.inputs = (self.X, self.y, self.ids, {s:self.fo for s in self.c['schemes']},
                       {s:{'primary':self.ref,'secondary':secondary} for s in self.c['schemes']},self.design)

    def fit(self, scheme='geo2_s42', arm='CATBOOST_NATIVE_C', f=0, fit_fn=None):
        a, b = np.flatnonzero(self.fo != f), np.flatnonzero(self.fo == f)
        return t26.checkpoint_fit(self.out, scheme, arm, f, self.X.iloc[a], self.y.iloc[a], self.X.iloc[b],
                                  self.y.iloc[b], self.ids.iloc[b], self.c, fit_fn or fake_fit())


def provenance_fixture(root):
    root=Path(root)
    c=copy.deepcopy(t26.config())
    code='b'*40
    paths={name:root/f'{name}.json' for name in ['analysis','design','ledger','audit']}
    design=dict(study_id='T25_catboost_c_v1',code_commit=code,
                config=dict(catboost_params=c['catboost_params'],schemes={s:dict(manifest=v['manifest']) for s,v in c['schemes'].items()}),
                data_contract=json.loads((t26.ROOT/'configs/data_contract.json').read_text(encoding='utf-8')),
                source_sha256={p:h for p,h in t26.sources().items() if p not in t26.OWN_SOURCE},
                runtime=dict(manifest_sha256=c['runtime_manifest']['sha256']))
    t26.write(paths['design'],design)
    schemes={}
    for index,scheme in enumerate(c['schemes']):
        directory=f'{scheme}/CATBOOST_C/summary'
        digest=str(index+1)*64
        schemes[scheme]=dict(artifacts=dict(dir=directory,artifact_sha256={'oof.csv.gz':digest}))
        c['schemes'][scheme]['refs']['CATBOOST_C']=dict(path=(root/directory/'oof.csv.gz').as_posix(),sha256=digest)
    analysis=dict(study_id='T25_catboost_c_v1',code_commit=code,fit_attempts=20,
                  design_sha256=t26.sha256_file(paths['design']),arms={'CATBOOST_C':dict(per_scheme=schemes)})
    t26.write(paths['analysis'],analysis)
    ledger=dict(fits={f'{s}/CATBOOST_C/f{f}':dict(status='complete',attempts=1) for s in c['schemes'] for f in range(5)},
                finalizations=dict(analysis=dict(status='complete',artifact_sha256={'analysis.json':t26.sha256_file(paths['analysis'])})))
    t26.write(paths['ledger'],ledger)
    audit=dict(status='PASS_T25',code_commit=code,design_sha256=t26.sha256_file(paths['design']),audited_models=20,
               new_classifier_fits=0,joblib_deserializations=0,repository_runner_imports=0,independent_full_CSR_and_state_match=True,
               tracked_sha256={str(paths[name].resolve()):t26.sha256_file(paths[name]) for name in ['analysis','design','ledger']})
    t26.write(paths['audit'],audit)
    c['primary_provenance']={name:dict(path=str(path),sha256=t26.sha256_file(path)) for name,path in paths.items()}
    c['primary_provenance'].update(study_id='T25_catboost_c_v1',code_commit=code,fit_attempts=20)
    return c,paths


class RunnerTests(unittest.TestCase):
    @unittest.skipUnless(all(Path(t26.PRIMARY_PROVENANCE[k]['path']).is_file() for k in ['analysis','design','ledger','audit'])
                         and Path(sys.executable).resolve() == (t26.ROOT.parents[1]/'local_reproduction/20261014_01/venv/Scripts/python.exe').resolve(),
                         '감사된 비공개 T25 메타데이터와 지정 Python이 있을 때만 실제 참조 결속 검증')
    def test_actual_primary_provenance_without_reading_oof_rows(self):
        with mock.patch.object(t26,'read_pred') as read, mock.patch.object(pd,'read_csv',side_effect=AssertionError('OOF 행 읽기 금지')):
            result=t26.verify_primary_provenance(t26.config())
            read.assert_not_called()
        self.assertEqual(result['audit_status'],'PASS_T25')
        self.assertFalse(result['row_level_oof_read'])

    def test_synthetic_primary_metadata_chain_accepts_without_oof_files(self):
        with tempfile.TemporaryDirectory() as d:
            c,_=provenance_fixture(d)
            self.assertTrue(all(not Path(v['refs']['CATBOOST_C']['path']).exists() for v in c['schemes'].values()))
            with mock.patch.object(t26,'read_pred') as read:
                result=t26.verify_primary_provenance(c)
                read.assert_not_called()
            self.assertEqual(result['audited_models'],20)

    def test_primary_metadata_tamper_failed_audit_or_wrong_oof_pin_stop(self):
        for mode in ['analysis_bytes','failed_audit','oof_pin','ledger_incomplete']:
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as d:
                c,paths=provenance_fixture(d)
                if mode == 'analysis_bytes': paths['analysis'].write_text('{}',encoding='utf-8')
                if mode == 'failed_audit':
                    a=json.loads(paths['audit'].read_text(encoding='utf-8'))
                    a['status']='FAIL_T25'
                    t26.write(paths['audit'],a)
                    c['primary_provenance']['audit']['sha256']=t26.sha256_file(paths['audit'])
                if mode == 'oof_pin': c['schemes']['geo2_s42']['refs']['CATBOOST_C']['sha256']='0'*64
                if mode == 'ledger_incomplete':
                    ledger=json.loads(paths['ledger'].read_text(encoding='utf-8'))
                    next(iter(ledger['fits'].values()))['status']='failed'
                    t26.write(paths['ledger'],ledger)
                    digest=t26.sha256_file(paths['ledger'])
                    c['primary_provenance']['ledger']['sha256']=digest
                    a=json.loads(paths['audit'].read_text(encoding='utf-8'))
                    a['tracked_sha256'][str(paths['ledger'].resolve())]=digest
                    t26.write(paths['audit'],a)
                    c['primary_provenance']['audit']['sha256']=t26.sha256_file(paths['audit'])
                with self.assertRaises(ValueError): t26.verify_primary_provenance(c)

    @unittest.skipUnless(Path(t26.RUNTIME_MANIFEST['path']).is_file()
                         and Path(sys.executable).resolve() == (t26.ROOT.parents[1]/'local_reproduction/20261014_01/venv/Scripts/python.exe').resolve(),
                         'T26 비공개 runtime과 지정 동결 Python이 있을 때만 실제 환경 검증')
    def test_actual_overlay_manifest_and_import_origins(self):
        evidence=t26.activate_and_verify_runtime(t26.config())
        self.assertEqual(evidence['catboost_version'],'1.2.10')
        self.assertEqual(evidence['vendor_file_count'],3014)
        self.assertEqual(len(evidence['distributions']),12)
        self.assertEqual(evidence['manifest_sha256'],t26.RUNTIME_MANIFEST['sha256'])
        self.assertEqual(len(evidence['base_module_files']),7)
        self.assertEqual(len(evidence['base_record_files']),7)

    def test_runtime_manifest_sha_vendor_hash_and_fileset_tamper_stop(self):
        for mode,message in [('manifest_sha','manifest SHA'),('file_hash','vendor 파일 SHA'),('extra_file','파일 폐쇄')]:
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as d:
                root=Path(d)
                vendor=root/'vendor'
                (vendor/'catboost').mkdir(parents=True)
                target=vendor/'catboost/__init__.py'
                target.write_text('sealed',encoding='utf-8')
                manifest=dict(schema_version=1,runtime_id='T25_catboost_c_runtime_v1',
                              distributions={'catboost':'1.2.10'},dependency_requirements_validated=True,
                              no_base_distribution_shadowing=True,vendor_root=str(vendor),
                              files={'catboost/__init__.py':t26.sha256_file(target)})
                path=root/'runtime.json'
                t26.write(path,manifest)
                c=copy.deepcopy(t26.config())
                c['runtime_manifest']=dict(path=str(path),sha256=t26.sha256_file(path))
                if mode == 'manifest_sha': c['runtime_manifest']['sha256']='0'*64
                if mode == 'file_hash': target.write_text('changed',encoding='utf-8')
                if mode == 'extra_file': (vendor/'extra.py').write_text('extra',encoding='utf-8')
                with mock.patch.object(t26,'validate_base_runtime'):
                    with self.assertRaisesRegex(ValueError,message): t26.validate_runtime_manifest(c)

    def test_wrong_python_path_stops_before_module_import(self):
        with tempfile.TemporaryDirectory() as d:
            manifest=dict(python_executable=str(Path(d)/'wrong_python.exe'))
            with mock.patch.object(t26.importlib,'import_module') as imported:
                with self.assertRaisesRegex(ValueError,'Python 실행 경로'):
                    t26.validate_base_runtime(manifest,t26.config())
                imported.assert_not_called()

    def test_wrong_catboost_origin_is_refused_even_with_expected_version(self):
        with tempfile.TemporaryDirectory() as d:
            vendor=Path(d).resolve()
            manifest=dict(vendor_root=str(vendor),distributions={'catboost':'1.2.10'})
            dist=mock.Mock(version='1.2.10',locate_file=lambda p:vendor)
            wrong=mock.Mock(__file__=str(vendor.parent/'wrong/catboost/__init__.py'),__version__='1.2.10')
            with mock.patch.object(t26,'validate_runtime_manifest',return_value=manifest), \
                 mock.patch.object(t26.importlib.metadata,'distribution',return_value=dist), \
                 mock.patch.object(t26.importlib,'import_module',return_value=wrong), \
                 mock.patch.object(t26.sys,'path',list(t26.sys.path)):
                with self.assertRaisesRegex(ValueError,'CatBoost 실제 출처'): t26.activate_and_verify_runtime(t26.config())

    def test_synthetic_runtime_activation_records_verified_origins(self):
        with tempfile.TemporaryDirectory() as d:
            vendor=Path(d).resolve()
            manifest=dict(runtime_id='T25_catboost_c_runtime_v1',vendor_root=str(vendor),
                          distributions={'catboost':'1.2.10'},files={'catboost/__init__.py':'f'*64},
                          base_module_files={},base_record_files={})
            dist=mock.Mock(version='1.2.10',locate_file=lambda p:vendor)
            module=mock.Mock(__file__=str(vendor/'catboost/__init__.py'),__version__='1.2.10')
            with mock.patch.object(t26,'validate_runtime_manifest',return_value=manifest), \
                 mock.patch.object(t26,'validate_base_runtime') as base, \
                 mock.patch.object(t26.importlib.metadata,'distribution',return_value=dist), \
                 mock.patch.object(t26.importlib,'import_module',return_value=module), \
                 mock.patch.object(t26.sys,'path',list(t26.sys.path)):
                record=t26.activate_and_verify_runtime(t26.config())
                self.assertEqual(record['vendor_file_count'],1)
                self.assertEqual(record['catboost_module'],(vendor/'catboost/__init__.py').as_posix())
                self.assertEqual(record['catboost_version'],'1.2.10')
                base.assert_called_once()

    def test_failed_or_running_in_any_journal_blocks_other_filter_and_direct_fit(self):
        for section in ['fits','finalizations']:
            for status in ['failed','running']:
                with self.subTest(section=section,status=status),tempfile.TemporaryDirectory() as d:
                    e,calls=Env(d),[]
                    ledger=t26.read_ledger(e.out)
                    ledger[section]['existing']=dict(status=status,attempts=1)
                    t26.atomic_write(e.out/'ledger.json',ledger)
                    before=(e.out/'ledger.json').read_bytes()
                    with mock.patch.object(t26,'verify',return_value=e.inputs):
                        with self.assertRaises(ValueError):
                            t26.fit_all('synthetic',e.out,schemes=['geo2_s202'],folds=[2],fit_fn=fake_fit(calls))
                    with self.assertRaises(ValueError): e.fit(scheme='geo2_s101',f=1,fit_fn=fake_fit(calls))
                    self.assertEqual(calls,[])
                    self.assertEqual((e.out/'ledger.json').read_bytes(),before)

    def test_initial_fit_ledger_write_error_is_preserved_before_training(self):
        with tempfile.TemporaryDirectory() as d:
            e,calls=Env(d),[]
            original=t26.atomic_write
            writes=[]

            def fail_first(path,record):
                writes.append(1)
                if len(writes) == 1: raise OSError('합성 초기 원장 쓰기 실패')
                return original(path,record)

            with mock.patch.object(t26,'atomic_write',side_effect=fail_first):
                with self.assertRaises(OSError): e.fit(fit_fn=fake_fit(calls))
            self.assertEqual(calls,[])
            entry=t26.read_ledger(e.out)['fits']['geo2_s42/CATBOOST_NATIVE_C/f0']
            self.assertEqual(entry['status'],'failed')
            self.assertIn('합성 초기 원장 쓰기 실패',entry['failure_traceback'])
            with self.assertRaises(ValueError): e.fit(scheme='geo2_s202',f=2)

    def test_finalize_failure_is_journaled_and_blocks_new_work(self):
        with tempfile.TemporaryDirectory() as d,mock.patch.object(t26.t15,'log'):
            e=Env(d)
            with mock.patch.object(t26,'verify',return_value=e.inputs):
                t26.fit_all('synthetic',e.out,fit_fn=fake_fit())
                t26.write(e.out/'design.json',e.design)
                original=t26.write

                def fail_analysis(path,record):
                    if Path(path).name == 'analysis.json': raise OSError('합성 최종 집계 쓰기 실패')
                    return original(path,record)

                with mock.patch.object(t26,'write',side_effect=fail_analysis):
                    with self.assertRaises(OSError): t26.finalize_all('synthetic',e.out)
                ledger=t26.read_ledger(e.out)
                self.assertEqual(ledger['finalizations']['analysis']['status'],'failed')
                self.assertEqual(len(ledger['fits']),20)
                for entry in ledger['fits'].values(): t26.check_saved(e.out,entry)
                with self.assertRaises(ValueError): t26.fit_all('synthetic',e.out,schemes=['geo2_s303'],folds=[4])
                with self.assertRaises(ValueError): t26.finalize_all('synthetic',e.out)

    def test_source_must_match_commit_blob_even_if_gitignored(self):
        digest = t26.hashlib.sha256(b'sealed').hexdigest()
        with mock.patch.object(t26.subprocess, 'run', return_value=mock.Mock(stdout=b'changed')):
            with self.assertRaises(ValueError):
                t26.verify_committed_sources({'ignored.py': digest}, 'a'*40)

    def test_frozen_config_and_exact_reference_paths_hashes(self):
        c, b = t26.config(), t26.t15.config()
        self.assertEqual([s['seed'] for s in c['schemes'].values()], [42, 101, 202, 303])
        self.assertEqual(c['lgbm_params'], b['lgbm_params'])
        self.assertEqual(c['lgbm_params']['n_estimators'], 300)
        self.assertEqual(c['lgbm_params']['num_leaves'], 31)
        self.assertEqual(c['catboost_params']['iterations'], 300)
        self.assertEqual(c['catboost_params']['depth'], 6)
        self.assertEqual(c['catboost_params']['learning_rate'], .05)
        self.assertEqual(c['catboost_params']['thread_count'], 2)
        self.assertEqual(c['catboost_params']['classes_count'], 3)
        self.assertEqual(c['catboost_params']['bootstrap_type'], 'Bayesian')
        self.assertFalse(c['catboost_params']['allow_writing_files'])
        self.assertEqual(c['max_new_study_fits'], 20)
        self.assertIsNone(c['sample_weight'])
        self.assertEqual(c['arms'], t26.ARMS)
        for s in c['schemes']:
            self.assertEqual(c['schemes'][s]['manifest'], b['schemes'][s]['manifest'])
            self.assertEqual(c['schemes'][s]['refs']['LGBM_O'], b['schemes'][s]['refs']['LGBM_O'])

    def test_config_rejects_weight_model_rule_and_seed_drift(self):
        mutations = [('sample_weight', [1., 1., 1.5]),
                     ('class_weight', 'balanced'), ('max_new_study_fits', 21),
                     ('decision_rule', 'grade3_offset'), ('early_stopping', True)]
        for name, value in mutations:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as d:
                c = copy.deepcopy(t26.config())
                c[name] = value
                path = Path(d)/'config.json'
                t26.write(path, c)
                with self.assertRaises(ValueError):
                    t26.config(path)

    def test_validation_labels_not_a_fit_argument_and_matrix_is_sparse(self):
        self.assertEqual(list(inspect.signature(t26.fit_model).parameters), ['arm', 'Xa', 'ya', 'Xb', 'c'])
        from scipy import sparse
        mat = sparse.csr_matrix(np.eye(3, dtype=np.float32))
        evidence = t26.matrix_evidence(mat)
        self.assertEqual(evidence['shape'], [3, 3])
        self.assertEqual(evidence['dtype'], 'float32')
        self.assertEqual(set(evidence['arrays']), {'data', 'indices', 'indptr'})
        with self.assertRaises(ValueError):
            t26.matrix_evidence(np.eye(3))

    def test_source_import_closure_includes_custom_preprocessor_and_tests(self):
        sources = t26.sources()
        for path in ['experiments/age995/run.py', 'experiments/lgbm_native_cat/run.py',
                     'experiments/grade3_decision_rule/run.py', 'tests/test_modeling.py', *t26.OWN_SOURCE]:
            self.assertIn(path, sources)
        self.assertTrue(all(len(s) == 64 for s in sources.values()))

    def test_guard_blocks_raw_and_resolved_holdout(self):
        for p in ['../../earthquake/data/final_holdout', 'FINAL_HOLDOUT/data.csv']:
            with self.assertRaises(ValueError):
                t26.guard(p)

    def test_checkpoint_skip_backup_and_probability_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            e, calls = Env(d), []
            result, fresh = e.fit(fit_fn=fake_fit(calls))
            self.assertTrue(fresh)
            same, fresh = e.fit(fit_fn=fake_fit(calls))
            self.assertFalse(fresh)
            self.assertEqual(len(calls), 1)
            self.assertEqual(result, same)
            t26.check_saved(e.out, same)
            pred = t26.read_pred(e.out/same['dir']/'pred.csv.gz')
            self.assertTrue((pred.y_pred.to_numpy() == pred[t26.PROBS].to_numpy().argmax(1)+1).all())

    def test_crash_records_failure_never_retries_and_preserves_original(self):
        with tempfile.TemporaryDirectory() as d:
            e = Env(d)
            with self.assertRaises(RuntimeError):
                e.fit(fit_fn=fake_fit(fail=True))
            before = (e.out/'geo2_s42/CATBOOST_NATIVE_C/f0/attempt1/failure.txt').read_bytes()
            with self.assertRaises(ValueError):
                e.fit()
            ledger = t26.read_ledger(e.out)
            self.assertEqual(ledger['fits']['geo2_s42/CATBOOST_NATIVE_C/f0']['status'], 'failed')
            self.assertEqual(ledger['fits']['geo2_s42/CATBOOST_NATIVE_C/f0']['attempts'], 1)
            self.assertEqual((e.out/'geo2_s42/CATBOOST_NATIVE_C/f0/attempt1/failure.txt').read_bytes(), before)

    def test_invalid_probability_reload_and_budget_fail_closed(self):
        for fit in [fake_fit(invalid=True), fake_fit(reload_diff=1e-6)]:
            with self.subTest(fit=fit), tempfile.TemporaryDirectory() as d:
                e = Env(d)
                with self.assertRaises(ValueError):
                    e.fit(fit_fn=fit)
                self.assertEqual(t26.read_ledger(e.out)['fits']['geo2_s42/CATBOOST_NATIVE_C/f0']['status'], 'failed')
        with tempfile.TemporaryDirectory() as d:
            e = Env(d)
            e.c['max_new_study_fits'] = 0
            with self.assertRaises(ValueError):
                e.fit()
            self.assertEqual(t26.read_ledger(e.out)['fits'], {})

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
            with mock.patch.object(t26, 'verify') as verify:
                with self.assertRaises(FileExistsError):
                    t26.fit_all('synthetic', e.out)
                verify.assert_not_called()
            self.assertEqual((e.out/'RUNNING.lock').read_text(encoding='utf-8'), '다른 실행')

    def test_filters20_fits_dual_reference_gates_and_shared_artifact(self):
        with tempfile.TemporaryDirectory() as d,mock.patch.object(t26.t15,'log'):
            e,calls=Env(d),[]
            with mock.patch.object(t26,'verify',return_value=e.inputs):
                t26.fit_all('synthetic',e.out,[t26.ARMS[0]],['geo2_s42'],[0],fake_fit(calls))
                self.assertEqual(len(calls),1)
                t26.fit_all('synthetic',e.out,fit_fn=fake_fit(calls))
                self.assertEqual(len(calls),20)
                t26.write(e.out/'design.json',e.design)
                result=t26.finalize_all('synthetic',e.out)
                self.assertEqual(result['fit_attempts'],20)
                self.assertEqual(result['reference_evaluations'],40)
                self.assertEqual(result['comparison_outer_units_per_reference'],20)
                self.assertEqual(result['pooled_denominator']['predictions_per_reference'],1800)
                arm=result['arms']['CATBOOST_NATIVE_C']
                self.assertEqual(len(arm['per_scheme']),4)
                self.assertEqual(len(arm['unknown_valid']),11)
                self.assertEqual(arm['unknown_valid']['cat_0']['rows'],1800)
                for role in ['primary','secondary']:
                    comparison=arm['comparisons'][role]
                    self.assertEqual(len(comparison['fold_records']),20)
                    self.assertEqual(comparison['turnover']['rows'],1800)
                    self.assertEqual(comparison['pooled_profiles']['age995']['enhanced']['rows'],60)
                    self.assertEqual(set(comparison['judgment']['gates']),{'mean_delta_ok','fold_wins_ok','positive_seeds_ok','pooled_recall3_ok','pooled_error31_rate_ok'})
                self.assertTrue(arm['comparisons']['primary']['judgment']['supports_fixed_hypothesis'])
                self.assertFalse(arm['comparisons']['secondary']['judgment']['supports_fixed_hypothesis'])
                self.assertEqual((e.out/'analysis.json').read_bytes(),(e.out/'backup/analysis.json').read_bytes())
                for scheme,block in arm['per_scheme'].items():
                    t26.check_saved(e.out,block['artifacts'])
                    self.assertEqual(set(block['comparisons']),{'primary','secondary'})
                self.assertEqual(t26.read_ledger(e.out)['finalizations']['analysis']['status'],'complete')
                with self.assertRaises(ValueError): t26.finalize_all('synthetic',e.out)
            for a,_,b in calls:self.assertFalse(set(a)&set(b))

    def test_finalize_incomplete_or_empty_filter_stops(self):
        with tempfile.TemporaryDirectory() as d:
            e = Env(d)
            with mock.patch.object(t26, 'verify', return_value=e.inputs):
                with self.assertRaises(ValueError):
                    t26.finalize_all('synthetic', e.out)
                for schemes, folds in [([], None), (None, []), (['invalid'], None), (None, [5])]:
                    with self.assertRaises(ValueError):
                        t26.fit_all('synthetic', e.out, schemes=schemes, folds=folds)

    def test_bad_seal_stops_before_reading_any_data(self):
        with tempfile.TemporaryDirectory() as d:
            t26.write(Path(d)/'design.json', {})
            t26.write(Path(d)/'design_seal.json', {'design_sha256': 'bad'})
            with mock.patch.object(t26, 'load') as load:
                with self.assertRaises(ValueError):
                    t26.verify('synthetic', d)
                load.assert_not_called()

    def test_judgment_strict_wins_seed_support_and_safety_gates(self):
        c = t26.config()
        cm = [[10, 0, 0], [0, 10, 0], [1, 3, 6]]
        baseline = dict(macro_f1=.7, confusion=cm)
        records = [dict(scheme=s, fold=f, baseline=baseline,
                        enhanced=dict(macro_f1=.704, confusion=[[10,0,0],[0,10,0],[0,3,7]]))
                   for s in c['schemes'] for f in range(5)]
        self.assertTrue(t26.judge(records, c)['supports_fixed_hypothesis'])
        for record in records[:3]:
            record['enhanced']['confusion'] = [[10,0,0],[0,10,0],[10,0,0]]
        verdict = t26.judge(records, c)
        self.assertFalse(verdict['gates']['pooled_error31_rate_ok'])
        self.assertFalse(verdict['supports_fixed_hypothesis'])
        for record in records:
            record['enhanced'] = copy.deepcopy(baseline)
        self.assertEqual(t26.judge(records, c)['fold_wins'], 0)

    def test_wrapper_native39_preserves_c_numeric28_and_declares11_categories(self):
        from scipy import sparse
        X,y,_,_=synthetic(450)
        X.loc[:2,'age']=995
        X.loc[350:,'geo_level_2_id']=999999
        c=t26.config()
        baseline=t26.t8.make_preprocessor('O')
        mt=baseline.fit_transform(X.iloc[:350])
        mv=baseline.transform(X.iloc[350:])
        calls=[]

        def adapter(Xa,ya,Xb,params,*,prepared):
            pre,train,valid=prepared
            self.assertEqual(params,c['catboost_params'])
            np.testing.assert_array_equal(ya,y.iloc[:350])
            self.assertEqual(train.shape,(350,39))
            self.assertEqual(valid.shape,(100,39))
            self.assertTrue(all(dtype==np.float32 for dtype in train.iloc[:,:28].dtypes))
            self.assertEqual((mt[:,:28]!=sparse.csr_matrix(train.iloc[:,:28].to_numpy())).nnz,0)
            self.assertEqual((mv[:,:28]!=sparse.csr_matrix(valid.iloc[:,:28].to_numpy())).nnz,0)
            calls.append(1)
            return dict(p=np.full((len(Xb),3),1/3),info=dict(input_evidence=dict(train=t26.input_evidence(train),valid=t26.input_evidence(valid)),cat_feature_indices=list(range(28,39))))

        with mock.patch.object(t26,'fit_catboost_native_c',side_effect=adapter):
            result=t26.fit_model('CATBOOST_NATIVE_C',X.iloc[:350],y.iloc[:350],X.iloc[350:],c)
        self.assertEqual(calls,[1])
        self.assertEqual(result['p'].shape,(100,3))

    def test_wrapper_rejects_adapter_native_input_changes(self):
        X,y,_,_=synthetic(450)
        with mock.patch.object(t26,'fit_catboost_native_c',return_value=dict(info=dict(input_evidence={}))) as adapter:
            with self.assertRaises(ValueError): t26.fit_model('CATBOOST_NATIVE_C',X.iloc[:350],y.iloc[:350],X.iloc[350:],t26.config())
            self.assertEqual(adapter.call_count,1)

    def test_no_t21_t22_t23_sources_config_or_results_are_linked(self):
        self.assertFalse(any('feature_interactions' in p or 'material_structure_interactions' in p or 'ordinal_damage' in p or 'geo_context_features' in p or p.startswith('experiments/catboost_c/') for p in t26.sources()))
        self.assertEqual(len(t26.sources()), 49)
        c = copy.deepcopy(t26.config())
        c['arms'] = ['CATBOOST_NATIVE_C','OTHER_ARM']
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'config.json'
            t26.write(p,c)
            with self.assertRaises(ValueError):
                t26.config(p)

    def test_matrix_evidence_requires_csr(self):
        from scipy import sparse
        with self.assertRaises(ValueError): t26.matrix_evidence(np.eye(3))

    def test_turnover_has_correct_actual_grade3_denominator(self):
        y = [1,2,3,3,3,2]
        base = [1,1,1,2,3,2]
        enhanced = [2,2,3,1,3,2]
        record = t26.turnover(y, base, enhanced)
        self.assertEqual(record['corrected'], 2)
        self.assertEqual(record['newly_wrong'], 1)
        self.assertEqual(record['error31_removed'], 1)
        self.assertEqual(record['error31_introduced'], 1)
        self.assertEqual(record['actual_grade3_rows'], 3)
        merged = t26.merge_turnovers([record, record])
        self.assertEqual(merged['actual_grade3_rows'], 6)
        self.assertEqual(merged['rows'], 12)

    def test_profile_can_have_missing_actual_class_without_failure(self):
        result = t26.pooled_metrics([[[0,0,0],[0,1,0],[1,0,2]]])
        self.assertEqual(result['support'], [0,1,3])
        self.assertEqual(result['error31_rate'], 1/3)


if __name__ == '__main__':
    unittest.main()
