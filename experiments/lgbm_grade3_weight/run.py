"""T20: LGBM-O의 실제 3등급 학습 행에만 sample_weight=1.5를 적용한다."""
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

from preprocess import ID, sha256_file
from experiments.grade3_decision_rule import run as t15
from experiments.lgbm_native_cat import run as t8
from experiments.geo_group_cv.run import PROBS, load, require, score, write

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
STUDY_ID = 'T20_lgbm_grade3_weight_v1'
RULES = dict(mean_delta_min=0.003, fold_wins_min=14, positive_seed_means_min=3,
             pooled_recall3_delta_min=0.0, pooled_error31_rate_delta_max=0.0)
OWN_SOURCE = ['experiments/lgbm_grade3_weight/__init__.py', 'experiments/lgbm_grade3_weight/run.py',
              'experiments/lgbm_grade3_weight/config.json', 'tests/test_lgbm_grade3_weight.py',
              'docs/T20_LGBM_GRADE3_WEIGHT_PLAN.md']
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
                    sample_weight_by_actual_grade={'1': 1.0, '2': 1.0, '3': 1.5},
                    class_weight=None, early_stopping=False, sampling=None, decision_rule='argmax',
                    max_new_study_fits=20, automatic_retries=0, outer_parallel_jobs=1,
                    lgbm_threadpool_limit=2, holdout_access=False, rules=RULES,
                    checkpoint=dict(model_reload_probability_atol_lgbm=1e-12, csv_float_format='%.17g'))
    require(c == expected, '사전등록 설정·참조·가중치·예산 변경')
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


def weights(y):
    actual = np.asarray(y)
    require(actual.ndim == 1 and np.isin(actual, [1, 2, 3]).all(), '실제 등급은 1/2/3이어야 함')
    return np.where(actual == 3, 1.5, 1.0).astype(np.float64)


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


def fit_weighted(Xa, ya, Xb, c):
    from lightgbm import LGBMClassifier, Booster
    pre = t8.make_preprocessor('O')
    sw = weights(ya)  # 실제 등급에서 생성한 뒤 학습 라벨만 0-based로 변환한다.
    with threadpool_limits(limits=c['lgbm_threadpool_limit']):
        mt, mv = pre.fit_transform(Xa), pre.transform(Xb)
    t8.check_matrix('O', mt)
    t8.check_matrix('O', mv)
    matrices = dict(train=matrix_evidence(mt), valid=matrix_evidence(mv))
    model = LGBMClassifier(**c['lgbm_params'])
    require(model.class_weight is None, 'class_weight 혼합 금지')
    with threadpool_limits(limits=c['lgbm_threadpool_limit']), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        started = time.perf_counter()
        model.fit(mt, np.asarray(ya, dtype=np.int32)-1, sample_weight=sw)
        elapsed = time.perf_counter()-started
        require(np.array_equal(model.classes_, [0, 1, 2]), '내부 클래스 순서 변경')
        p = probabilities(model.predict_proba(mv), len(Xb))
    require(model.booster_.current_iteration() == c['lgbm_params']['n_estimators'], '고정 학습 예산 미완료')

    def save(prefix):
        paths = [Path(f'{prefix}_model.txt'), Path(f'{prefix}_preprocessor.joblib')]
        require(not any(p.exists() for p in paths), '모델 덮어쓰기 금지')
        model.booster_.save_model(str(paths[0]))
        joblib.dump(pre, paths[1], compress=3)
        return paths

    def verify(paths, Xv, saved):
        with threadpool_limits(limits=c['lgbm_threadpool_limit']):
            m = Booster(model_file=str(paths[0]))
            require(m.num_model_per_iteration() == 3, '저장 모델 클래스 수 변경')
            pp = probabilities(m.predict(joblib.load(paths[1]).transform(Xv),
                                         num_threads=c['lgbm_params']['n_jobs']), len(Xv))
        return float(np.max(np.abs(pp-saved)))

    return dict(p=p, fit_seconds=elapsed, save=save, verify=verify,
                info=dict(age995_median=pre['age995'].median_, encoded_columns=int(mt.shape[1]),
                          matrix_evidence=matrices,
                          preprocessor_custom_class='experiments.age995.run.Age995Treatment',
                          audit_reload_note='감사자는 일반 joblib.load를 쓰지 않고 독립 C전처리 CSR과 native Booster로 검증한다.',
                          actual_grade_weight_counts={str(g): int(np.sum(np.asarray(ya) == g)) for g in [1, 2, 3]},
                          training_weight_sum=float(sw.sum()), internal_classes=model.classes_.tolist(),
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
    atomic_write(out/'ledger.json', dict(study_id=STUDY_ID, fits={}))
    t15.log(out, 'T20 설계 봉인 완료; 신규 학습 0회')
    return design


def verify(data, output):
    data, out = guard(data), guard(output)
    seal = json.loads((out/'design_seal.json').read_text(encoding='utf-8'))
    require(sha256_file(out/'design.json') == seal['design_sha256'], '설계 봉인 변경')
    require(sha256_file(out/'backup/design.json') == seal['design_sha256'], '설계 백업 변경')
    require(sha256_file(out/'design_seal.json') == sha256_file(out/'backup/design_seal.json'), '봉인 백업 변경')
    d = json.loads((out/'design.json').read_text(encoding='utf-8'))
    require(d['config'] == config() and d['source_sha256'] == sources(), '봉인 코드·설정 변경')
    require(d['environment'] == environment(), '봉인 환경 변경')
    X, y, ids, old, contract = load(data)
    require(d['data_contract'] == contract, '개발 데이터 계약 변경')
    folds = {s: t15.scheme_folds(d['config'], s, ids, old) for s in d['config']['schemes']}
    refs = {s: reference(d['config'], s, ids, y, folds[s]) for s in folds}
    for e in read_ledger(out)['fits'].values():
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


def checkpoint_fit(out, scheme, f, Xa, ya, Xb, yb, idb, c, fit_fn=fit_weighted):
    out = Path(out)
    key = f'{scheme}/f{f}'
    ledger = read_ledger(out)
    prior = ledger['fits'].get(key)
    if prior and prior['status'] == 'complete':
        check_saved(out, prior)
        return prior, False
    require(prior is None, '실패·미완료 Fold 재시도 금지; 원본을 보존하고 보고해야 함')
    require(sum(e['attempts'] for e in ledger['fits'].values()) < c['max_new_study_fits'], '학습 시도 예산 소진')
    directory = f'{key}/attempt1'
    entry = dict(status='running', attempts=1, dir=directory, scheme=scheme, fold=f, started_at_utc=now())
    ledger['fits'][key] = entry
    atomic_write(out/'ledger.json', ledger)
    d = out/directory
    d.mkdir(parents=True, exist_ok=False)
    t15.event(out, fit=key, action='fit_start', attempt=1)
    try:
        result = fit_fn(Xa, ya, Xb, c)
        p = probabilities(result['p'], len(Xb))
        paths = result['save'](d/'fit')
        diff = float(result['verify'](paths, Xb, p))
        require(np.isfinite(diff) and diff <= c['checkpoint']['model_reload_probability_atol_lgbm'], '모델 재로딩 불일치')
        pred = pd.DataFrame({ID: np.asarray(idb), 'fold': f, 'y_true': np.asarray(yb), 'y_pred': p.argmax(1)+1,
                             **{n: p[:, j] for j, n in enumerate(PROBS)}})
        write_pred(d/'pred.csv.gz', pred)
        np.testing.assert_array_equal(read_pred(d/'pred.csv.gz')[PROBS].to_numpy(), p)
        write(d/'metrics.json', dict(study_id=STUDY_ID, scheme=scheme, fold=f, train_rows=len(Xa), valid_rows=len(Xb),
                                    metrics=score(yb, pred.y_pred), fit_seconds=result['fit_seconds'],
                                    reload_probability_max_abs_diff=diff, csv_round_trip_exact=True,
                                    sample_weight_by_actual_grade=c['sample_weight_by_actual_grade'],
                                    lgbm_params=c['lgbm_params'], info=result.get('info', {})))
        entry.update(save_artifacts(out, directory, [*paths, d/'pred.csv.gz', d/'metrics.json']),
                     fit_seconds=result['fit_seconds'])
    except BaseException:
        entry.update(status='failed', failed_at_utc=now())
        (d/'failure.txt').write_text(traceback.format_exc(), encoding='utf-8')
        ledger = read_ledger(out)
        ledger['fits'][key] = entry
        atomic_write(out/'ledger.json', ledger)
        t15.event(out, fit=key, action='fit_failed', attempt=1)
        raise
    ledger = read_ledger(out)
    ledger['fits'][key] = entry
    atomic_write(out/'ledger.json', ledger)
    t15.event(out, fit=key, action='fit_complete', artifact_sha256=entry['artifact_sha256'])
    t15.log(out, f'{key}: 학습 {entry["fit_seconds"]:.1f}초, 저장·재로딩·SHA·백업 확인')
    return entry, True


def fit_all(data, output, schemes=None, folds=None, fit_fn=fit_weighted):
    out = guard(output)
    with exclusive(out):
        X, y, ids, assigned, _, d = verify(data, out)
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
                checkpoint_fit(out, s, f, X.iloc[a], y.iloc[a], X.iloc[b], y.iloc[b], ids.iloc[b], d['config'], fit_fn)


def pooled_metrics(confusions):
    cm = np.sum(np.asarray(confusions, dtype=np.int64), axis=0)
    require(cm.shape == (3, 3) and (cm >= 0).all() and (cm.sum(1) > 0).all(), '잘못된 혼동행렬')
    tp = np.diag(cm).astype(float)
    p = np.divide(tp, cm.sum(0), out=np.zeros(3), where=cm.sum(0) > 0)
    r = tp/cm.sum(1)
    f = np.divide(2*p*r, p+r, out=np.zeros(3), where=p+r > 0)
    return dict(confusion=cm.tolist(), rows=int(cm.sum()), support=cm.sum(1).tolist(), precision=p.tolist(),
                recall=r.tolist(), f1=f.tolist(), macro_f1=float(f.mean()), accuracy=float(tp.sum()/cm.sum()),
                error31=int(cm[2, 0]), error32=int(cm[2, 1]), error23=int(cm[1, 2]),
                error31_rate=float(cm[2, 0]/cm[2].sum()), error32_rate=float(cm[2, 1]/cm[2].sum()),
                error23_rate=float(cm[1, 2]/cm[1].sum()))


def judge(records, c):
    deltas = [r['weighted']['macro_f1']-r['baseline']['macro_f1'] for r in records]
    seed_deltas = {str(c['schemes'][s]['seed']): float(np.mean([deltas[i] for i, r in enumerate(records)
                                                             if r['scheme'] == s])) for s in c['schemes']}
    base = pooled_metrics([r['baseline']['confusion'] for r in records])
    weighted = pooled_metrics([r['weighted']['confusion'] for r in records])
    recall_delta = weighted['recall'][2]-base['recall'][2]
    error_delta = weighted['error31_rate']-base['error31_rate']
    rules = c['rules']
    gates = dict(mean_delta_ok=bool(np.mean(deltas) >= rules['mean_delta_min']),
                 fold_wins_ok=sum(d > 0 for d in deltas) >= rules['fold_wins_min'],
                 positive_seeds_ok=sum(d > 0 for d in seed_deltas.values()) >= rules['positive_seed_means_min'],
                 pooled_recall3_ok=recall_delta >= rules['pooled_recall3_delta_min'],
                 pooled_error31_rate_ok=error_delta <= rules['pooled_error31_rate_delta_max'])
    return dict(fold_deltas=deltas, mean_delta=float(np.mean(deltas)), fold_wins=sum(d > 0 for d in deltas),
                fold_macro_f1={arm: dict(mean=float(np.mean([r[arm]['macro_f1'] for r in records])),
                                        sd=float(np.std([r[arm]['macro_f1'] for r in records], ddof=1)))
                               for arm in ['baseline', 'weighted']},
                seed_mean_deltas=seed_deltas, pooled_baseline=base, pooled_weighted=weighted,
                pooled_recall3_delta=recall_delta, pooled_error31_rate_delta=error_delta,
                gates=gates, supports_fixed_hypothesis=all(gates.values()))


def finalize_all(data, output):
    out = guard(output)
    with exclusive(out):
        X, y, ids, folds, refs, d = verify(data, out)
        require(not (out/'analysis.json').exists() and not (out/'backup/analysis.json').exists(), '집계 덮어쓰기 금지')
        ledger = read_ledger(out)
        expected = {f'{s}/f{f}' for s in folds for f in range(5)}
        require(set(ledger['fits']) == expected and all(e['status'] == 'complete' for e in ledger['fits'].values()), 'Fold 미완료')
        require(sum(e['attempts'] for e in ledger['fits'].values()) == len(expected) <= d['config']['max_new_study_fits'], '시도 수 불일치')
        records, scheme_results = [], {}
        for s, fo in folds.items():
            frames = []
            for f in range(5):
                e = ledger['fits'][f'{s}/f{f}']
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
                records.append(dict(scheme=s, fold=f, baseline=score(y.iloc[b], baseline.y_pred), weighted=score(y.iloc[b], pred.y_pred)))
                frames.append(pred)
            oof = pd.concat(frames).sort_values(ID, kind='stable').reset_index(drop=True)
            np.testing.assert_array_equal(oof[ID], ids)
            directory = f'{s}/summary'
            dest = out/directory
            dest.mkdir(parents=True, exist_ok=False)
            write_pred(dest/'oof.csv.gz', oof)
            np.testing.assert_array_equal(read_pred(dest/'oof.csv.gz')[PROBS], oof[PROBS])
            selected = [r for r in records if r['scheme'] == s]
            scheme_results[s] = dict(baseline=pooled_metrics([r['baseline']['confusion'] for r in selected]),
                                     weighted=pooled_metrics([r['weighted']['confusion'] for r in selected]),
                                     fold_macro_f1_mean=float(np.mean([r['weighted']['macro_f1'] for r in selected])),
                                     fold_macro_f1_sd=float(np.std([r['weighted']['macro_f1'] for r in selected], ddof=1)),
                                     profiles={label: score(oof.loc[mask, 'y_true'], oof.loc[mask, 'y_pred']) for label, mask in
                                               [('age995', X.age.to_numpy() == 995), ('non995', X.age.to_numpy() != 995)] if mask.any()})
            write(dest/'metrics.json', scheme_results[s])
            scheme_results[s]['artifacts'] = save_artifacts(out, directory, [dest/'oof.csv.gz', dest/'metrics.json'])
        result = dict(study_id=STUDY_ID, code_commit=d['code_commit'], design_sha256=sha256_file(out/'design.json'),
                      holdout_accessed=False, fit_attempts=len(expected), fold_records=records, per_scheme=scheme_results,
                      judgment=judge(records, d['config']), fit_seconds_sum=sum(e['fit_seconds'] for e in ledger['fits'].values()),
                      pooled_denominator=dict(predictions=len(y)*len(folds), unique_development_rows=len(y),
                                              repeated_seed_schemes=len(folds), note='동일 개발행 반복 평가; 독립 관측 수가 아님. 오류율 분모는 실제 등급 support.'),
                      note='고정 가중치 단일가설의 개발 CV 결과. 확률 보정·통계 유의·실무 허용·최종 모델 선정이 아님. T15 결정 규칙과 별개.')
        write(out/'analysis.json', result)
        shutil.copy2(out/'analysis.json', out/'backup/analysis.json')
        return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'fit-all', 'finalize', 'status'])
    p.add_argument('--data', default='../../earthquake/data')
    p.add_argument('--output', required=True)
    p.add_argument('--code-commit')
    p.add_argument('--schemes', nargs='+')
    p.add_argument('--folds', type=int, nargs='+')
    a = p.parse_args()
    if a.action == 'prepare':
        prepare(a.data, a.output, a.code_commit)
    elif a.action == 'fit-all':
        fit_all(a.data, a.output, a.schemes, a.folds)
    elif a.action == 'finalize':
        finalize_all(a.data, a.output)
    else:
        out = guard(a.output)
        ledger = read_ledger(out)
        print(json.dumps(dict(완료학습=sum(e['status'] == 'complete' for e in ledger['fits'].values()),
                              학습시도=sum(e['attempts'] for e in ledger['fits'].values()),
                              실패학습=sum(e['status'] == 'failed' for e in ledger['fits'].values()),
                              실행잠금=(out/'RUNNING.lock').exists()), ensure_ascii=False))


if __name__ == '__main__':
    main()
