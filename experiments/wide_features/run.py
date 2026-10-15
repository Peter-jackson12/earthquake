"""T27: 고정 LGBM-O의 C 입력 뒤에 넓은 파생 후보와 train-only 필터를 추가한다."""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback
import warnings

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from scipy import sparse
from experiments.wide_features.features import WideFeatures
from preprocess import ID, sha256_file
from experiments.grade3_decision_rule import run as t15
from experiments.lgbm_native_cat import run as t8
from experiments.geo_group_cv.run import PROBS, load, require, score, write

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
STUDY_ID = 'T27_wide_features_v1'
ARMS = ['WIDE_FEATURES']
FEATURE_POLICY = dict(numeric_candidates=95, composite_candidates=6, logical_added=101,
                      original_inputs=38, existing_age995_flag=1, logical_total_candidates=140,
                      filter_scope='extra_only', constant_filter=True, exact_duplicate_filter=True,
                      canonical_dtype='float32', first_nonconstant_candidate_wins=True,
                      compare_with_C=False, rare_filter=False, target_used=False)
UNKNOWN_KEYS = {'foundation_roof','ground_other_floor','foundation_ground_floor','roof_other_floor',
                'position_plan','foundation_land_surface','geo_level_1_id'}
RULES = dict(mean_delta_min=0.003, fold_wins_min=14, positive_seed_means_min=3,
             pooled_recall3_delta_min=0.0, pooled_error31_rate_delta_max=0.0)
OWN_SOURCE = ['experiments/wide_features/__init__.py', 'experiments/wide_features/run.py',
              'experiments/wide_features/config.json', 'experiments/wide_features/features.py',
              'tests/test_wide_features_run.py', 'tests/test_wide_features.py',
              'docs/T27_WIDE_FEATURES_PLAN.md']
now, environment = t15.now, t15.environment
atomic_write, read_pred, write_pred = t15.atomic_write, t15.read_pred, t15.write_pred
read_ledger, check_saved = t15.read_ledger, t15.check_saved


def guard(path):
    p = Path(path)
    require('final_holdout' not in p.as_posix().lower()
            and 'final_holdout' not in p.resolve().as_posix().lower(), '홀드아웃 경로 접근 금지')
    return p


def config(path=HERE/'config.json'):
    c = json.loads(Path(path).read_text(encoding='utf-8'))
    base = t15.config()
    schemes = {s: dict(kind=base['schemes'][s]['kind'], seed=base['schemes'][s]['seed'],
                       manifest=base['schemes'][s]['manifest'],
                       refs={'LGBM_O': base['schemes'][s]['refs']['LGBM_O']}) for s in base['primary_schemes']}
    expected = dict(study_id=STUDY_ID, lgbm_params=base['lgbm_params'], environment=base['environment'],
                    schemes=schemes, group_columns={'group2': base['group_columns']['group2']},
                    arms=ARMS, feature_policy=FEATURE_POLICY, sample_weight=None,
                    class_weight=None, early_stopping=False, sampling=None, decision_rule='argmax',
                    max_new_study_fits=20, automatic_retries=0, outer_parallel_jobs=1,
                    lgbm_threadpool_limit=2, holdout_access=False, rules=RULES,
                    checkpoint=dict(model_reload_probability_atol_lgbm=1e-12, csv_float_format='%.17g'))
    require(c == expected, '사전등록 설정·참조·특성팔·예산 변경')
    require([s['seed'] for s in c['schemes'].values()] == [42, 101, 202, 303], '등록 seed 변경')
    return c


def source_paths():
    """선언된 자료와 로컬 Python import의 전이 폐쇄를 함께 봉인한다."""
    pending, seen = list(dict.fromkeys(t15.SOURCE + OWN_SOURCE)), set()
    while pending:
        rel = pending.pop()
        if rel in seen:
            continue
        p = guard(ROOT/rel)
        require(p.is_file(), f'봉인 소스 누락: {rel}')
        seen.add(rel)
        if p.suffix != '.py':
            continue
        tree = ast.parse(p.read_text(encoding='utf-8-sig'))
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.Import):
                modules = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                prefix = node.module or ''
                if node.level:
                    parts = p.relative_to(ROOT).parent.parts
                    prefix = '.'.join((*parts[:len(parts)-node.level+1], prefix)).strip('.')
                modules = [prefix] + [f'{prefix}.{a.name}'.strip('.') for a in node.names if a.name != '*']
            for module in modules:
                for root in (ROOT, ROOT/'tests'):
                    stem = root.joinpath(*module.split('.'))
                    for candidate in (stem.with_suffix('.py'), stem/'__init__.py'):
                        if candidate.is_file():
                            pending.append(candidate.relative_to(ROOT).as_posix())
                            for parent in candidate.parents:
                                if parent == ROOT:
                                    break
                                if (parent/'__init__.py').is_file():
                                    pending.append((parent/'__init__.py').relative_to(ROOT).as_posix())
    return sorted(seen)


def sources():
    return {p: sha256_file(ROOT/p) for p in source_paths()}


def verify_committed_sources(shas, code_commit):
    for path, digest in shas.items():
        blob = subprocess.run(['git', 'show', f'{code_commit}:{path}'], cwd=ROOT, capture_output=True, check=True).stdout
        require(hashlib.sha256(blob).hexdigest() == digest, f'봉인 대상과 커밋 blob 불일치: {path}')


def probabilities(p, rows):
    p = np.asarray(p, dtype=np.float64)
    require(p.shape == (rows, 3) and np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()
            and np.allclose(p.sum(1), 1.0, atol=1e-12, rtol=0), '예측 확률·클래스 차원 불일치')
    return p


def matrix_evidence(matrix):
    """감사자는 joblib 실행기 import 없이 독립 C전처리 CSR과 비교할 수 있다."""
    from scipy import sparse
    require(sparse.isspmatrix_csr(matrix), 'one-hot CSR 입력만 허용')
    return dict(format='csr', shape=list(matrix.shape), dtype=str(matrix.dtype),
                arrays={name: dict(dtype=str(getattr(matrix, name).dtype),
                                   shape=list(getattr(matrix, name).shape),
                                   sha256=hashlib.sha256(np.ascontiguousarray(getattr(matrix, name)).tobytes()).hexdigest())
                        for name in ['data', 'indices', 'indptr']})


def augment(base, extra):
    require(sparse.isspmatrix_csr(base) and sparse.isspmatrix_csr(extra), '기존·추가 블록 CSR 필수')
    require(base.dtype == np.float32 and extra.dtype == np.float32, '두 블록 float32 필수')
    require(base.shape[0] == extra.shape[0] and np.isfinite(extra.data).all(), '추가 블록 행수·유한값 불일치')
    full = sparse.hstack([base, extra], format='csr', dtype=np.float32)
    require((full[:, :base.shape[1]] != base).nnz == 0, '기존 C 블록 변경')
    return full


def matrix_statistics(matrix):
    return dict(shape=list(matrix.shape), nnz=int(matrix.nnz),
                density=float(matrix.nnz/(matrix.shape[0]*matrix.shape[1])) if all(matrix.shape) else 0.0,
                csr_bytes=int(matrix.data.nbytes+matrix.indices.nbytes+matrix.indptr.nbytes))


def process_peak_rss():
    """현재 프로세스 전체 수명의 최대 사용량이며 Fold만의 사용량이 아니다."""
    try:
        if os.name == 'nt':
            import ctypes
            from ctypes import wintypes
            class MemoryCounters(ctypes.Structure):
                _fields_ = [('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD),
                            *[(name,ctypes.c_size_t) for name in ['PeakWorkingSetSize','WorkingSetSize',
                              'QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage',
                              'QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage']]]
            stats=MemoryCounters(); stats.cb=ctypes.sizeof(stats)
            kernel=ctypes.WinDLL('kernel32',use_last_error=True)
            kernel.GetCurrentProcess.restype=wintypes.HANDLE
            psapi=ctypes.WinDLL('psapi',use_last_error=True)
            psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(MemoryCounters),wintypes.DWORD]
            if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(stats),stats.cb):
                raise OSError(ctypes.get_last_error())
            value=int(stats.PeakWorkingSetSize)
        else:
            import resource, sys
            value=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1 if sys.platform=='darwin' else 1024)
        return dict(bytes=value, scope='process_lifetime_high_water', available=True)
    except (ImportError,OSError,AttributeError):
        return dict(bytes=None, scope='process_lifetime_high_water', available=False)


def validate_feature_metadata(feature, metadata, train_extra):
    keep=np.asarray(feature.keep_indices_,dtype=np.int64)
    names=list(feature.candidate_names_)
    output=list(feature.get_feature_names_out())
    require(metadata['candidate_logical_count']==101 and metadata['candidate_numeric_count']==95
            and metadata['candidate_categorical_count']==6, '고정 후보 수 변경')
    require(metadata['candidate_encoded_count']==len(names)==len(metadata['candidates'])
            and metadata['retained_count']==len(keep)==train_extra.shape[1], '필터 후보·유지 폭 불일치')
    require(metadata['candidate_names']==names and metadata['keep_indices']==keep.tolist()
            and metadata['output_features']==output==[names[i] for i in keep], '필터 열 이름·인덱스 불일치')
    require(metadata['age_median']==feature.median_ and metadata['n_train_rows']==train_extra.shape[0],
            '피처 중앙값·학습 행 수 불일치')
    require(metadata['target_used'] is False and metadata['statistics_scope']=='outer_train_only'
            and metadata['filtering']['compare_with_C'] is False
            and metadata['filtering']['rare_frequency_threshold'] is None, '선택 정보·범위 변경')
    require(train_extra.has_canonical_format and np.isfinite(train_extra.data).all(), '추가 CSR 정규형 불일치')
    csc=train_extra.tocsc()
    for j,i in enumerate(keep):
        record=metadata['candidates'][i]
        require(record['index']==i and record['name']==names[i] and record['retained_index']==j
                and record['support']==int(csc.indptr[j+1]-csc.indptr[j])
                and record['constant_value'] is None and record['removal_reason'] is None
                and record['duplicate_of_index'] is None, '유지 피처 지원 수·필터 상태 불일치')


def fit_enhanced(arm, Xa, ya, Xb, c):
    from lightgbm import LGBMClassifier, Booster
    require(arm in ARMS, '미등록 특성 팔')
    pre, feature = t8.make_preprocessor('O'), WideFeatures(arm)
    with threadpool_limits(limits=c['lgbm_threadpool_limit']):
        preprocessing_started = time.perf_counter()
        bt, bv = pre.fit_transform(Xa), pre.transform(Xb)
        c_preprocessing_seconds = time.perf_counter()-preprocessing_started
        base_before = dict(train=matrix_evidence(bt), valid=matrix_evidence(bv))
        feature_started = time.perf_counter()
        feature.fit(Xa)
        require(feature.median_ == pre['age995'].median_, 'C와 파생 피처 age995 중앙값 불일치')
        et, ev = feature.transform(Xa), feature.transform(Xb)
        feature_preprocessing_seconds = time.perf_counter()-feature_started
        metadata = feature.metadata()
        require(et.shape[1] == ev.shape[1] == len(feature.get_feature_names_out()), '필터 후 열 수 불일치')
        require(np.array_equal(feature.keep_indices_, np.sort(np.unique(feature.keep_indices_))), '선택 열 순서·중복 변경')
        require(len(feature.candidate_names_) >= et.shape[1], '후보·유지 열 수 불일치')
        validate_feature_metadata(feature,metadata,et)
        augmentation_started = time.perf_counter()
        mt, mv = augment(bt, et), augment(bv, ev)
        augmentation_seconds = time.perf_counter()-augmentation_started
        require(base_before == dict(train=matrix_evidence(bt),valid=matrix_evidence(bv)), '추가 계산 중 기존 C 변경')
        require(matrix_evidence(mt[:,:bt.shape[1]]) == base_before['train']
                and matrix_evidence(mv[:,:bv.shape[1]]) == base_before['valid'], 'C 앞 블록 바이트 변경')
        preparation_seconds = time.perf_counter()-preprocessing_started
    for matrix in (bt, bv, mt, mv):
        t8.check_matrix('O', matrix)
    matrices = {name: matrix_evidence(matrix) for name, matrix in
                [('base_train', bt), ('base_valid', bv), ('extra_train', et), ('extra_valid', ev),
                 ('full_train', mt), ('full_valid', mv)]}
    model = LGBMClassifier(**c['lgbm_params'])
    require(model.class_weight is None and c['sample_weight'] is None, '학습 가중치 혼합 금지')
    with threadpool_limits(limits=c['lgbm_threadpool_limit']), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        started = time.perf_counter()
        model.fit(mt, np.asarray(ya, dtype=np.int32)-1)
        elapsed = time.perf_counter()-started
        require(np.array_equal(model.classes_, [0, 1, 2]), '내부 클래스 순서 변경')
        p = probabilities(model.predict_proba(mv), len(Xb))
    require(model.booster_.current_iteration() == c['lgbm_params']['n_estimators'], '고정 학습 예산 미완료')

    def save(prefix):
        paths = [Path(f'{prefix}_model.txt'), Path(f'{prefix}_preprocessor.joblib'), Path(f'{prefix}_features.joblib')]
        require(not any(p.exists() for p in paths), '모델 덮어쓰기 금지')
        model.booster_.save_model(str(paths[0]))
        joblib.dump(pre, paths[1], compress=3)
        joblib.dump(feature, paths[2], compress=3)
        return paths

    def verify(paths, Xv, saved):
        with threadpool_limits(limits=c['lgbm_threadpool_limit']):
            m = Booster(model_file=str(paths[0]))
            require(m.num_model_per_iteration() == 3, '저장 모델 클래스 수 변경')
            base = joblib.load(paths[1]).transform(Xv)
            extra = joblib.load(paths[2]).transform(Xv)
            pp = probabilities(m.predict(augment(base, extra), num_threads=c['lgbm_params']['n_jobs']), len(Xv))
        return float(np.max(np.abs(pp-saved)))

    return dict(p=p, fit_seconds=elapsed, save=save, verify=verify,
                info=dict(age995_median=pre['age995'].median_, base_columns=int(bt.shape[1]),
                          extra_columns=int(et.shape[1]), full_columns=int(mt.shape[1]),
                          matrix_evidence=matrices, feature_metadata=metadata,
                          matrix_statistics={name:matrix_statistics(matrix) for name,matrix in
                              [('base_train',bt),('base_valid',bv),('extra_train',et),('extra_valid',ev),
                               ('full_train',mt),('full_valid',mv)]},
                          candidate_encoded_columns=len(feature.candidate_names_), kept_columns=len(feature.keep_indices_),
                          feature_age995_median=feature.median_, C_prefix_byte_exact=True,
                          preprocessing_seconds=dict(C=c_preprocessing_seconds,features=feature_preprocessing_seconds,
                                                     augmentation=augmentation_seconds,total=preparation_seconds),
                          process_peak_rss=process_peak_rss(),
                          train_unknown_combinations=feature.unknown_counts(Xa),
                          valid_unknown_combinations=feature.unknown_counts(Xb),
                          preprocessor_custom_class='experiments.age995.run.Age995Treatment',
                          feature_class='experiments.wide_features.features.WideFeatures',
                          audit_reload_note='감사자는 일반 joblib.load 없이 독립 C·특성 재구성 CSR 및 native Booster로 검증한다.',
                          internal_classes=model.classes_.tolist(),
                          warnings=[dict(category=w.category.__name__, message=str(w.message)) for w in caught]))


def reference(c, scheme, ids, y, folds):
    return t15.read_reference(c, scheme, 'LGBM_O', ids, y, folds)


def prepare(data, output, code_commit):
    data, out = guard(data), guard(output)
    require(not out.exists(), '기존 출력 경로 덮어쓰기 금지')
    require(isinstance(code_commit, str) and len(code_commit) == 40
            and all(x in '0123456789abcdef' for x in code_commit), '전체 커밋 SHA 필요')
    c, shas = config(), sources()
    require(environment() == c['environment'], '동결 환경 불일치')
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(['git', 'status', '--porcelain', '--', *shas], cwd=ROOT,
                           capture_output=True, text=True, check=True).stdout.strip()
    require(head == code_commit and not dirty, '봉인 대상은 HEAD에 커밋된 깨끗한 코드여야 함')
    verify_committed_sources(shas, code_commit)
    X, y, ids, old, contract = load(data)
    refs, gates = {}, {}
    for s in c['schemes']:
        fo = t15.scheme_folds(c, s, ids, old)
        gates[s] = t15.t5.validate_assignment(X, y, ids, fo, c['group_columns']['group2'], t15.t5.cfg())
        r = reference(c, s, ids, y, fo)
        refs[s] = dict(rows=len(r), sha256=c['schemes'][s]['refs']['LGBM_O']['sha256'])
    out.mkdir(parents=True, exist_ok=False)
    (out/'backup').mkdir()
    design = dict(study_id=STUDY_ID, config=c, source_sha256=shas, code_commit=code_commit,
                  environment=environment(), data_contract=contract, references=refs, scheme_validation=gates,
                  holdout_accessed=False, classifiers_fitted=0, created_at_utc=now())
    write(out/'design.json', design)
    write(out/'design_seal.json', dict(design_sha256=sha256_file(out/'design.json'), code_commit=code_commit))
    for name in ['design.json', 'design_seal.json']:
        shutil.copy2(out/name, out/'backup'/name)
    atomic_write(out/'ledger.json', dict(study_id=STUDY_ID, fits={}, finalizations={}))
    t15.log(out, 'T27 설계 봉인 완료; 신규 학습 0회')
    return design


def verify(data, output):
    data, out = guard(data), guard(output)
    seal = json.loads((out/'design_seal.json').read_text(encoding='utf-8'))
    require(sha256_file(out/'design.json') == seal['design_sha256'], '설계 봉인 변경')
    require(sha256_file(out/'backup/design.json') == seal['design_sha256'], '설계 백업 변경')
    require(sha256_file(out/'design_seal.json') == sha256_file(out/'backup/design_seal.json'), '봉인 백업 변경')
    d = json.loads((out/'design.json').read_text(encoding='utf-8'))
    require(d['study_id'] == STUDY_ID and d['code_commit'] == seal['code_commit'], '설계 ID·커밋 불일치')
    require(d['config'] == config() and d['source_sha256'] == sources(), '봉인 코드·설정 변경')
    require(d['environment'] == environment(), '봉인 환경 변경')
    X, y, ids, old, contract = load(data)
    require(d['data_contract'] == contract, '개발 데이터 계약 변경')
    folds = {s: t15.scheme_folds(d['config'], s, ids, old) for s in d['config']['schemes']}
    refs = {s: reference(d['config'], s, ids, y, folds[s]) for s in folds}
    ledger = read_ledger(out)
    for section in ['fits','finalizations']:
        for e in ledger[section].values():
            if e['status'] == 'complete':
                check_saved(out, e)
    return X, y, ids, folds, refs, d


@contextmanager
def exclusive(output):
    lock = guard(output)/'RUNNING.lock'
    with lock.open('x', encoding='utf-8') as fh:
        fh.write(f'pid={os.getpid()} started={now()}\n')
    try:
        yield
    finally:
        lock.unlink()  # 기존 잠금의 배타적 생성 실패 시 이 코드에 도달하지 않는다.


def save_artifacts(out, directory, paths):
    shas = {p.name: sha256_file(p) for p in paths}
    bd = out/'backup'/directory
    bd.mkdir(parents=True, exist_ok=False)
    for p in paths:
        shutil.copy2(p, bd/p.name)
    e = dict(dir=directory, status='complete', artifact_sha256=shas, finished_at_utc=now())
    check_saved(out, e)
    return e


def require_no_incomplete_work(ledger):
    require(all(e['status'] == 'complete' for section in ['fits','finalizations'] for e in ledger[section].values()),
            '실패·실행 중·미완료 학습 또는 집계 기록 존재; 연구 전체 신규 작업 금지')


def checkpoint_fit(out, scheme, arm, f, Xa, ya, Xb, yb, idb, c, fit_fn=fit_enhanced):
    out = Path(out)
    key = f'{scheme}/{arm}/f{f}'
    ledger = read_ledger(out)
    prior = ledger['fits'].get(key)
    if prior and prior['status'] == 'complete':
        check_saved(out, prior)
        return prior, False
    require(prior is None, '실패·미완료 Fold 재시도 금지; 원본을 보존하고 보고해야 함')
    require_no_incomplete_work(ledger)
    require(sum(e['attempts'] for e in ledger['fits'].values()) < c['max_new_study_fits'], '학습 시도 예산 소진')
    directory = f'{key}/attempt1'
    entry = dict(status='running', attempts=1, dir=directory, scheme=scheme, arm=arm, fold=f, started_at_utc=now())
    ledger['fits'][key] = entry
    d = out/directory
    created = False
    try:
        atomic_write(out/'ledger.json', ledger)
        d.mkdir(parents=True, exist_ok=False)
        created = True
        t15.event(out, fit=key, action='fit_start', attempt=1)
        result = fit_fn(arm, Xa, ya, Xb, c)
        p = probabilities(result['p'], len(Xb))
        paths = result['save'](d/'fit')
        diff = float(result['verify'](paths, Xb, p))
        require(np.isfinite(diff) and diff <= c['checkpoint']['model_reload_probability_atol_lgbm'], '모델 재로딩 불일치')
        pred = pd.DataFrame({ID: np.asarray(idb), 'fold': f, 'y_true': np.asarray(yb), 'y_pred': p.argmax(1)+1,
                             **{n: p[:, j] for j, n in enumerate(PROBS)}})
        write_pred(d/'pred.csv.gz', pred)
        np.testing.assert_array_equal(read_pred(d/'pred.csv.gz')[PROBS].to_numpy(), p)
        write(d/'metrics.json', dict(study_id=STUDY_ID, scheme=scheme, arm=arm, fold=f, train_rows=len(Xa), valid_rows=len(Xb),
                                    metrics=score(yb, pred.y_pred), fit_seconds=result['fit_seconds'],
                                    reload_probability_max_abs_diff=diff, csv_round_trip_exact=True,
                                    sample_weight=None, class_weight=None,
                                    lgbm_params=c['lgbm_params'], info=result.get('info', {})))
        entry.update(save_artifacts(out, directory, [*paths, d/'pred.csv.gz', d/'metrics.json']),
                     fit_seconds=result['fit_seconds'])
    except BaseException:
        entry.update(status='failed', failed_at_utc=now(), failure_traceback=traceback.format_exc())
        ledger = read_ledger(out)
        ledger['fits'][key] = entry
        atomic_write(out/'ledger.json', ledger)
        if created:
            with (d/'failure.txt').open('x', encoding='utf-8') as fh:
                fh.write(entry['failure_traceback'])
        t15.event(out, fit=key, action='fit_failed', attempt=1)
        raise
    ledger = read_ledger(out)
    ledger['fits'][key] = entry
    atomic_write(out/'ledger.json', ledger)
    t15.event(out, fit=key, action='fit_complete', artifact_sha256=entry['artifact_sha256'])
    t15.log(out, f'{key}: 학습 {entry["fit_seconds"]:.1f}초, 저장·재로딩·SHA·백업 확인')
    return entry, True


def fit_all(data, output, arms=None, schemes=None, folds=None, fit_fn=fit_enhanced):
    out = guard(output)
    with exclusive(out):
        X, y, ids, assigned, _, d = verify(data, out)
        require_no_incomplete_work(read_ledger(out))
        require(arms is None or (arms and len(set(arms)) == len(arms) and set(arms) <= set(ARMS)), '미등록·중복·빈 팔 필터')
        require(schemes is None or (schemes and len(set(schemes)) == len(schemes) and set(schemes) <= set(assigned)),
                '등록되지 않은 분할 또는 중복·빈 필터')
        require(folds is None or (folds and len(set(folds)) == len(folds) and set(folds) <= set(range(5))),
                '등록되지 않은 Fold 또는 중복·빈 필터')
        for s, fo in assigned.items():
            if schemes is not None and s not in schemes:
                continue
            for f in range(5):
                if folds is not None and f not in folds:
                    continue
                a, b = np.flatnonzero(fo != f), np.flatnonzero(fo == f)
                for arm in ARMS:
                    if arms is None or arm in arms:
                        checkpoint_fit(out, s, arm, f, X.iloc[a], y.iloc[a], X.iloc[b], y.iloc[b], ids.iloc[b], d['config'], fit_fn)


def pooled_metrics(confusions):
    cm = np.sum(np.asarray(confusions, dtype=np.int64), axis=0)
    require(cm.shape == (3, 3) and (cm >= 0).all() and cm.sum() > 0, '잘못된 혼동행렬')
    tp = np.diag(cm).astype(float)
    p = np.divide(tp, cm.sum(0), out=np.zeros(3), where=cm.sum(0) > 0)
    r = np.divide(tp, cm.sum(1), out=np.zeros(3), where=cm.sum(1) > 0)
    f = np.divide(2*p*r, p+r, out=np.zeros(3), where=p+r > 0)
    return dict(confusion=cm.tolist(), rows=int(cm.sum()), support=cm.sum(1).tolist(), precision=p.tolist(),
                recall=r.tolist(), f1=f.tolist(), macro_f1=float(f.mean()), accuracy=float(tp.sum()/cm.sum()),
                error31=int(cm[2, 0]), error32=int(cm[2, 1]), error23=int(cm[1, 2]),
                error31_rate=float(cm[2, 0]/cm[2].sum()) if cm[2].sum() else None,
                error32_rate=float(cm[2, 1]/cm[2].sum()) if cm[2].sum() else None,
                error23_rate=float(cm[1, 2]/cm[1].sum()) if cm[1].sum() else None)


def judge(records, c):
    deltas = [r['enhanced']['macro_f1']-r['baseline']['macro_f1'] for r in records]
    seed_deltas = {str(c['schemes'][s]['seed']): float(np.mean([deltas[i] for i, r in enumerate(records)
                                                             if r['scheme'] == s])) for s in c['schemes']}
    base = pooled_metrics([r['baseline']['confusion'] for r in records])
    enhanced = pooled_metrics([r['enhanced']['confusion'] for r in records])
    recall_delta = enhanced['recall'][2]-base['recall'][2]
    error_delta = enhanced['error31_rate']-base['error31_rate']
    rules = c['rules']
    gates = dict(mean_delta_ok=bool(np.mean(deltas) >= rules['mean_delta_min']),
                 fold_wins_ok=sum(d > 0 for d in deltas) >= rules['fold_wins_min'],
                 positive_seeds_ok=sum(d > 0 for d in seed_deltas.values()) >= rules['positive_seed_means_min'],
                 pooled_recall3_ok=recall_delta >= rules['pooled_recall3_delta_min'],
                 pooled_error31_rate_ok=error_delta <= rules['pooled_error31_rate_delta_max'])
    return dict(fold_deltas=deltas, mean_delta=float(np.mean(deltas)), fold_wins=sum(d > 0 for d in deltas),
                fold_macro_f1={arm: dict(mean=float(np.mean([r[arm]['macro_f1'] for r in records])),
                                        sd=float(np.std([r[arm]['macro_f1'] for r in records], ddof=1)))
                               for arm in ['baseline', 'enhanced']},
                seed_mean_deltas=seed_deltas, pooled_baseline=base, pooled_enhanced=enhanced,
                pooled_recall3_delta=recall_delta, pooled_error31_rate_delta=error_delta,
                gates=gates, supports_fixed_hypothesis=all(gates.values()))


def turnover(y, baseline, enhanced):
    y, baseline, enhanced = map(np.asarray, (y, baseline, enhanced))
    require(y.shape == baseline.shape == enhanced.shape and len(y) > 0, '오류 전환 입력 불일치')
    bc, ec = baseline == y, enhanced == y
    transitions = np.zeros((3, 3), dtype=np.int64)
    np.add.at(transitions, (baseline.astype(int)-1, enhanced.astype(int)-1), 1)
    was31, is31 = (y == 3) & (baseline == 1), (y == 3) & (enhanced == 1)
    return dict(rows=len(y), prediction_agreement=float(np.mean(baseline == enhanced)),
                changed_predictions=int(np.sum(baseline != enhanced)), both_correct=int(np.sum(bc & ec)),
                corrected=int(np.sum(~bc & ec)), newly_wrong=int(np.sum(bc & ~ec)),
                both_wrong=int(np.sum(~bc & ~ec)), prediction_transition=transitions.tolist(),
                error31_removed=int(np.sum(was31 & ~is31)), error31_introduced=int(np.sum(~was31 & is31)),
                error31_retained=int(np.sum(was31 & is31)), actual_grade3_rows=int(np.sum(y == 3)))


def merge_turnovers(items):
    rows = sum(x['rows'] for x in items)
    require(rows > 0, '빈 오류 전환 집계')
    result = {name: sum(x[name] for x in items) for name in
              ['rows', 'changed_predictions', 'both_correct', 'corrected', 'newly_wrong', 'both_wrong',
               'error31_removed', 'error31_introduced', 'error31_retained', 'actual_grade3_rows']}
    result['prediction_transition'] = np.sum([x['prediction_transition'] for x in items], axis=0).tolist()
    result['prediction_agreement'] = 1-result['changed_predictions']/rows
    return result


def profiles(X, y, base_pred, enhanced_pred):
    return {label: dict(baseline=score(np.asarray(y)[mask], np.asarray(base_pred)[mask]),
                        enhanced=score(np.asarray(y)[mask], np.asarray(enhanced_pred)[mask]),
                        turnover=turnover(np.asarray(y)[mask], np.asarray(base_pred)[mask], np.asarray(enhanced_pred)[mask]))
            for label, mask in [('age995', X.age.to_numpy() == 995), ('non995', X.age.to_numpy() != 995)] if mask.any()}


def pool_unknown_counts(items):
    require(items and all(set(x) == UNKNOWN_KEYS for x in items), '미관측 범주6·geo1 집계 키 불일치')
    result={}
    for key in sorted(UNKNOWN_KEYS):
        require(all(0 <= x[key]['count'] <= x[key]['rows'] for x in items), '미관측 건수·분모 불일치')
        rows=sum(x[key]['rows'] for x in items)
        count=sum(x[key]['count'] for x in items)
        result[key]=dict(count=count,rows=rows,rate=count/rows if rows else None)
    return result


def finalize_all(data, output):
    out = guard(output)
    with exclusive(out):
        X, y, ids, folds, refs, d = verify(data, out)
        require(not (out/'analysis.json').exists() and not (out/'backup/analysis.json').exists(), '집계 덮어쓰기 금지')
        ledger = read_ledger(out)
        expected = {f'{s}/{arm}/f{f}' for s in folds for arm in ARMS for f in range(5)}
        require(set(ledger['fits']) == expected and all(e['status'] == 'complete' for e in ledger['fits'].values()), 'Fold 미완료')
        require(sum(e['attempts'] for e in ledger['fits'].values()) == len(expected) <= d['config']['max_new_study_fits'], '시도 수 불일치')
        # 중단된 집계가 있으면 원본을 유지한 채 학습 없이 수동 검토를 요청한다.
        require(not any((out/f'{s}/{a}/summary').exists() or (out/f'backup/{s}/{a}/summary').exists()
                        for s in folds for a in ARMS), '기존 집계 경로 존재; 원본 보존 후 검토 필요')
        require_no_incomplete_work(ledger)
        require(not ledger['finalizations'], '기존 집계 원장 존재; 덮어쓰기 금지')
        entry = dict(status='running', dir='', started_at_utc=now())
        ledger['finalizations']['analysis'] = entry
        try:
            atomic_write(out/'ledger.json', ledger)
            arms, all_records = {}, []
            for arm in ARMS:
                records, scheme_results = [], {}
                for s, fo in folds.items():
                    frames = []
                    for f in range(5):
                        e = ledger['fits'][f'{s}/{arm}/f{f}']
                        check_saved(out, e)
                        pred = read_pred(out/e['dir']/'pred.csv.gz')
                        b = np.flatnonzero(fo == f)
                        np.testing.assert_array_equal(pred[ID], ids.iloc[b])
                        np.testing.assert_array_equal(pred.y_true, y.iloc[b])
                        np.testing.assert_array_equal(pred.fold, np.full(len(b), f))
                        p = probabilities(pred[PROBS].to_numpy(), len(b))
                        np.testing.assert_array_equal(pred.y_pred, p.argmax(1)+1)
                        baseline = refs[s].loc[refs[s].fold == f]
                        np.testing.assert_array_equal(pred[ID], baseline[ID])
                        info = json.loads((out/e['dir']/'metrics.json').read_text(encoding='utf-8'))['info']
                        records.append(dict(scheme=s, arm=arm, fold=f, baseline=score(y.iloc[b], baseline.y_pred),
                                            enhanced=score(y.iloc[b], pred.y_pred), turnover=turnover(y.iloc[b], baseline.y_pred, pred.y_pred),
                                            unknown_train=info['train_unknown_combinations'], unknown_valid=info['valid_unknown_combinations'],
                                            feature_width=dict(candidate_encoded=info['candidate_encoded_columns'],
                                                               kept=info['kept_columns'],base=info['base_columns'],full=info['full_columns']),
                                            matrix_statistics=info['matrix_statistics'],
                                            preprocessing_seconds=info['preprocessing_seconds'],process_peak_rss=info['process_peak_rss']))
                        frames.append(pred)
                    oof = pd.concat(frames).sort_values(ID, kind='stable').reset_index(drop=True)
                    np.testing.assert_array_equal(oof[ID], ids)
                    np.testing.assert_array_equal(refs[s][ID], ids)
                    directory = f'{s}/{arm}/summary'
                    dest = out/directory
                    dest.mkdir(parents=True, exist_ok=False)
                    write_pred(dest/'oof.csv.gz', oof)
                    np.testing.assert_array_equal(read_pred(dest/'oof.csv.gz')[PROBS], oof[PROBS])
                    selected = [r for r in records if r['scheme'] == s]
                    scheme_results[s] = dict(baseline=pooled_metrics([r['baseline']['confusion'] for r in selected]),
                                             enhanced=pooled_metrics([r['enhanced']['confusion'] for r in selected]),
                                             fold_macro_f1={kind: dict(mean=float(np.mean([r[kind]['macro_f1'] for r in selected])),
                                                                      sd=float(np.std([r[kind]['macro_f1'] for r in selected], ddof=1)))
                                                            for kind in ['baseline', 'enhanced']},
                                             profiles=profiles(X, y, refs[s].y_pred, oof.y_pred),
                                             unknown_train=pool_unknown_counts([r['unknown_train'] for r in selected]),
                                             unknown_valid=pool_unknown_counts([r['unknown_valid'] for r in selected]),
                                             turnover=merge_turnovers([r['turnover'] for r in selected]))
                    write(dest/'metrics.json', scheme_results[s])
                    scheme_results[s]['artifacts'] = save_artifacts(out, directory, [dest/'oof.csv.gz', dest/'metrics.json'])
                profile_names = set().union(*(r['profiles'] for r in scheme_results.values()))
                pooled_profiles = {label: {**{kind: pooled_metrics([r['profiles'][label][kind]['confusion']
                                                                   for r in scheme_results.values() if label in r['profiles']])
                                             for kind in ['baseline', 'enhanced']},
                                          'turnover': merge_turnovers([r['profiles'][label]['turnover']
                                                                      for r in scheme_results.values() if label in r['profiles']])}
                                   for label in sorted(profile_names)}
                arms[arm] = dict(per_scheme=scheme_results, judgment=judge(records, d['config']),
                                 pooled_profiles=pooled_profiles, turnover=merge_turnovers([r['turnover'] for r in records]),
                                 unknown_train=pool_unknown_counts([r['unknown_train'] for r in records]),
                                 unknown_valid=pool_unknown_counts([r['unknown_valid'] for r in records]),
                                 fit_attempts=len(records), fold_records=records,
                                 feature_width_range={key:dict(min=min(r['feature_width'][key] for r in records),
                                                               max=max(r['feature_width'][key] for r in records))
                                                      for key in ['candidate_encoded','kept','base','full']},
                                 preprocessing_seconds_sum={key:sum(r['preprocessing_seconds'][key] for r in records)
                                                            for key in ['C','features','augmentation','total']})
                all_records.extend(records)
            result = dict(study_id=STUDY_ID, code_commit=d['code_commit'], design_sha256=sha256_file(out/'design.json'),
                          holdout_accessed=False, fit_attempts=len(expected), fold_records=all_records, arms=arms,
                          fit_seconds_sum=sum(e['fit_seconds'] for e in ledger['fits'].values()),
                          pooled_denominator=dict(predictions=len(y)*len(folds), unique_development_rows=len(y),
                                                  predictions_per_arm=len(y)*len(folds),
                                                  new_predictions_all_arms=len(y)*len(folds)*len(ARMS),
                                                  repeated_seed_schemes=len(folds), note='predictions는 팔별 값. 동일 개발행 반복 평가; 독립 관측 수가 아님. 오류율 분모는 실제 등급 support.'),
                          note='고정 넓은 파생 피처 묶음 가설의 개발 CV 결과이며 누적 연구의 선택 편향이 있다. '
                               '외부 독립 성능·통계 유의·실무 허용·최종 모델 선정이 아니며 피처 중요도만으로 원인을 확정하지 않는다.')
            write(out/'analysis.json', result)
            shutil.copy2(out/'analysis.json', out/'backup/analysis.json')
            entry.update(status='complete', artifact_sha256={'analysis.json':sha256_file(out/'analysis.json')}, finished_at_utc=now())
            check_saved(out, entry)
        except BaseException:
            entry.update(status='failed', failed_at_utc=now(), failure_traceback=traceback.format_exc())
            ledger = read_ledger(out)
            ledger['finalizations']['analysis'] = entry
            atomic_write(out/'ledger.json', ledger)
            t15.event(out, action='finalize_failed')
            raise
        ledger = read_ledger(out)
        ledger['finalizations']['analysis'] = entry
        atomic_write(out/'ledger.json', ledger)
        t15.event(out, action='finalize_complete', artifact_sha256=entry['artifact_sha256'])
        return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'fit-all', 'finalize', 'status'])
    p.add_argument('--data', default='../../earthquake/data')
    p.add_argument('--output', required=True)
    p.add_argument('--code-commit')
    p.add_argument('--arms', nargs='+')
    p.add_argument('--schemes', nargs='+')
    p.add_argument('--folds', type=int, nargs='+')
    a = p.parse_args()
    if a.action == 'prepare':
        prepare(a.data, a.output, a.code_commit)
    elif a.action == 'fit-all':
        fit_all(a.data, a.output, a.arms, a.schemes, a.folds)
    elif a.action == 'finalize':
        finalize_all(a.data, a.output)
    else:
        out = guard(a.output)
        ledger = read_ledger(out)
        print(json.dumps(dict(완료학습=sum(e['status'] == 'complete' for e in ledger['fits'].values()),
                              집계상태={k:e['status'] for k,e in ledger['finalizations'].items()},
                              학습시도=sum(e['attempts'] for e in ledger['fits'].values()),
                              실패학습=sum(e['status'] == 'failed' for e in ledger['fits'].values()),
                              실행잠금=(out/'RUNNING.lock').exists()), ensure_ascii=False))


if __name__ == '__main__':
    main()
