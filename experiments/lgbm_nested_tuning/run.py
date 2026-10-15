"""T18: geo2 내부 OOF로만 학습 예산과 leaves를 각각 선택한다. 보정은 없다."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from preprocess import ID, sha256_file
from experiments.grade3_decision_rule import run as t15
from experiments.geo_group_cv.run import PROBS, load, require, score, write

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
STUDY_ID = 'T18_lgbm_nested_tuning_v1'
CANDIDATES = {'B': {'n_estimators': 300, 'num_leaves': 31},
              'R600': {'n_estimators': 600, 'num_leaves': 31},
              'L15': {'n_estimators': 300, 'num_leaves': 15},
              'L63': {'n_estimators': 300, 'num_leaves': 63}}
POLICIES = {'ROUNDS': ['B', 'R600'], 'LEAVES': ['B', 'L15', 'L63']}
SOURCE = list(dict.fromkeys(t15.SOURCE + [
    'experiments/lgbm_nested_tuning/__init__.py', 'experiments/lgbm_nested_tuning/run.py',
    'experiments/lgbm_nested_tuning/config.json', 'tests/test_lgbm_nested_tuning.py', 'docs/T18_LGBM_TUNING_PLAN.md']))
now, environment = t15.now, t15.environment
atomic_write, read_pred, write_pred = t15.atomic_write, t15.read_pred, t15.write_pred
read_ledger, check_saved = t15.read_ledger, t15.check_saved


def config(path=HERE/'config.json'):
    c = json.loads(Path(path).read_text(encoding='utf-8'))
    base = t15.config()
    require(c['study_id'] == STUDY_ID and c['candidates'] == CANDIDATES and c['policies'] == POLICIES,
            '사전 등록 후보 또는 정책 변경')
    require(c['lgbm_params'] == base['lgbm_params'], '기존 기준선 모델 파라미터 변경')
    require(c['environment'] == base['environment'], '동결 환경 변경')
    require(c['schemes'] == {s: base['schemes'][s] for s in base['primary_schemes']}, '기존 geo2 분할/참조 변경')
    require(c['inner'] == dict(n_splits=3, seed=42, shuffle=True, min_class_support=30), '내부 분할 정책 변경')
    require(c['group_columns'] == {'group2': base['group_columns']['group2']}, '그룹 정의 변경')
    require(c['max_new_study_fits'] == 280 and c['max_inner_fits'] == 240 and c['max_outer_refits'] == 40,
            '학습 예산 변경')
    require(c['early_stopping'] is False and c['class_weight'] is None and c['sampling'] is None
            and c['decision_rule'] == 'argmax' and c['outer_parallel_jobs'] == 1
            and c['lgbm_threadpool_limit'] == 2 and c['automatic_retries'] == 0
            and c['holdout_access'] is False, '사전 등록 실행 정책 변경')
    require(c['rules'] == dict(mean_delta_min=0.003, fold_wins_min=14, positive_seed_means_min=3), '판정 규칙 변경')
    require(c['checkpoint']['model_reload_probability_atol_lgbm'] == 1e-12, '재로딩 허용오차 변경')
    return c


def sources():
    return {p: sha256_file(ROOT/p) for p in SOURCE}


def guard(path):
    path = t15.guard(path)
    require('final_holdout' not in path.resolve().as_posix().lower(), '홀드아웃 경로 접근 금지')
    return path


def inner_folds(X, y, c):
    return t15.inner_folds(X, y, 'group2', c)


def inner_digest(folds):
    return hashlib.sha256(np.asarray(folds, dtype=np.int8).tobytes()).hexdigest()


def reference(c, scheme, ids, y, folds):
    return t15.read_reference(c, scheme, 'LGBM_O', ids, y, folds)


def select_candidates(y_inner, probabilities, c):
    """외부 정답을 받지 않고 내부 OOF로 두 정책을 각각 고정한다."""
    scores = {k: score(y_inner, p.argmax(1) + 1)['macro_f1'] for k, p in probabilities.items()}
    require(set(scores) == set(c['candidates']), '내부 후보 부족')
    # max는 첫 최대값을 반환한다. 등록 순서상 B, 그다음 작은 복잡도를 우선한다.
    chosen = {policy: max(options, key=lambda k: scores[k]) for policy, options in c['policies'].items()}
    return chosen, scores


def fit_candidate(candidate, Xa, ya, Xb, c):
    local = copy.deepcopy(c)
    local['lgbm_params'].update(c['candidates'][candidate])
    return t15.fit_inner_arm('LGBM_O', 'group2', Xa, ya, Xb, local)


def log(out, text):
    t15.log(out, text)


def prepare(data, output, code_commit):
    data, out = guard(data), guard(output)
    require(not out.exists(), '기존 출력 경로를 덮어쓸 수 없음')
    require(len(code_commit) == 40 and all(x in '0123456789abcdef' for x in code_commit), '전체 SHA 필요')
    c = config()
    require(environment() == c['environment'], '동결 환경 불일치')
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(['git', 'status', '--porcelain', '--', *SOURCE], cwd=ROOT,
                           capture_output=True, text=True, check=True).stdout.strip()
    require(head == code_commit and not dirty, '봉인 대상은 HEAD에 커밋된 깨끗한 코드여야 함')
    X, y, ids, old, contract = load(data)
    inner_sha, refs, gates = {}, {}, {}
    for s in c['schemes']:
        fo = t15.scheme_folds(c, s, ids, old)
        gates[s] = t15.t5.validate_assignment(X, y, ids, fo, c['group_columns']['group2'], t15.t5.cfg())
        r = reference(c, s, ids, y, fo)
        refs[s] = dict(rows=len(r), sha256=c['schemes'][s]['refs']['LGBM_O']['sha256'])
        for f in range(5):
            tr = np.flatnonzero(fo != f)
            inner = inner_folds(X.iloc[tr], y.iloc[tr], c)
            t15.inner_gate(X.iloc[tr], y.iloc[tr], inner, 'group2', c)
            inner_sha[f'{s}/f{f}'] = inner_digest(inner)
    out.mkdir(parents=True)
    (out/'backup').mkdir()
    design = dict(study_id=STUDY_ID, config=c, code_commit=code_commit, source_sha256=sources(),
                  environment=environment(), data_contract=contract, inner_fold_sha256=inner_sha,
                  references=refs, scheme_validation=gates, holdout_accessed=False, classifiers_fitted=0,
                  created_at_utc=now())
    write(out/'design.json', design)
    write(out/'design_seal.json', dict(design_sha256=sha256_file(out/'design.json'), code_commit=code_commit))
    for name in ('design.json', 'design_seal.json'):
        shutil.copy2(out/name, out/'backup'/name)
    atomic_write(out/'ledger.json', dict(study_id=STUDY_ID, fits={}, selections={}, evaluations={}))
    log(out, 'T18 설계 봉인 완료; 학습 수=0')
    return design


def verify(data, output):
    data, out = guard(data), guard(output)
    seal = json.loads((out/'design_seal.json').read_text(encoding='utf-8'))
    require(sha256_file(out/'design.json') == seal['design_sha256'], '설계 봉인 변경')
    require(sha256_file(out/'backup'/'design.json') == seal['design_sha256'], '설계 백업 변경')
    require(sha256_file(out/'design_seal.json') == sha256_file(out/'backup'/'design_seal.json'), '봉인 백업 변경')
    d = json.loads((out/'design.json').read_text(encoding='utf-8'))
    require(d['config'] == config() and d['source_sha256'] == sources(), '봉인된 코드/설정 변경')
    require(d['environment'] == environment(), '봉인된 환경 변경')
    X, y, ids, old, contract = load(data)
    require(contract == d['data_contract'], '개발 데이터 계약 변경')
    folds = {s: t15.scheme_folds(d['config'], s, ids, old) for s in d['config']['schemes']}
    # 재개할 때에도 원본 OOF의 바이트/행/정답/분할을 다시 확인한다.
    refs = {s: reference(d['config'], s, ids, y, folds[s]) for s in folds}
    ledger = read_ledger(out)
    for section in ('fits', 'selections', 'evaluations'):
        for entry in ledger[section].values():
            if entry['status'] == 'complete':
                check_saved(out, entry)
    return X, y, ids, folds, refs, d


def save_artifacts(out, directory, paths):
    """시도별 새 디렉터리. 중단 시 미완료 파일과 백업도 덮어쓰지 않는다."""
    shas = {p.name: sha256_file(p) for p in paths}
    bd = out/'backup'/directory
    bd.mkdir(parents=True, exist_ok=False)
    for p in paths:
        shutil.copy2(p, bd/p.name)
    entry = dict(dir=directory, status='complete', artifact_sha256=shas, finished_at_utc=now())
    check_saved(out, entry)
    return entry


def checkpoint_fit(out, key, candidate, stage, Xa, ya, Xb, yb, idb, c,
                   fit_fn=fit_candidate, resume_incomplete=False):
    out = Path(out)
    ledger = read_ledger(out)
    prior = ledger['fits'].get(key)
    if prior and prior['status'] == 'complete':
        check_saved(out, prior)
        return prior, False
    require(not prior or resume_incomplete, '미완료 시도 재개에는 --resume-incomplete 필요')
    attempts = sum(e['attempts'] for e in ledger['fits'].values())
    require(attempts < c['max_new_study_fits'], '사전 등록 학습 시도 예산 소진')
    attempt = (prior['attempts'] if prior else 0) + 1
    directory = f'{key}/attempt{attempt}'
    entry = dict(status='running', attempts=attempt, dir=directory, stage=stage, candidate=candidate,
                 started_at_utc=now())
    ledger['fits'][key] = entry
    atomic_write(out/'ledger.json', ledger)
    d = out/directory
    d.mkdir(parents=True, exist_ok=False)
    t15.event(out, fit=key, action='fit_start', attempt=attempt, candidate=candidate, stage=stage)
    res = fit_fn(candidate, Xa, ya, Xb, c)
    p = np.asarray(res['p'], dtype=np.float64)
    require(p.shape == (len(Xb), 3) and np.isfinite(p).all() and (p >= 0).all()
            and np.allclose(p.sum(1), 1, atol=1e-12, rtol=0), '예측 확률 유효성 실패')
    paths = res['save'](d/'fit')
    diff = float(res['verify'](paths, Xb, p))
    require(diff <= c['checkpoint']['model_reload_probability_atol_lgbm'], '저장 모델 재로딩 불일치')
    pred = pd.DataFrame({ID: np.asarray(idb), 'y_true': np.asarray(yb), 'y_pred': p.argmax(1)+1,
                         **{n: p[:, j] for j, n in enumerate(PROBS)}})
    write_pred(d/'pred.csv.gz', pred)
    np.testing.assert_array_equal(read_pred(d/'pred.csv.gz')[PROBS].to_numpy(), p)
    write(d/'metrics.json', dict(study_id=STUDY_ID, key=key, candidate=candidate, stage=stage,
                                train_rows=len(Xa), valid_rows=len(Xb), metrics=score(yb, pred.y_pred),
                                fit_seconds=res['fit_seconds'], reload_probability_max_abs_diff=diff,
                                requested_lgbm_params=dict(c['lgbm_params'], **c['candidates'][candidate]),
                                csv_round_trip_exact=True, warnings=res.get('warnings', [])))
    entry.update(save_artifacts(out, directory, [*paths, d/'pred.csv.gz', d/'metrics.json']),
                 fit_seconds=res['fit_seconds'])
    ledger = read_ledger(out)
    ledger['fits'][key] = entry
    atomic_write(out/'ledger.json', ledger)
    t15.event(out, fit=key, action='fit_complete', artifact_sha256=entry['artifact_sha256'])
    log(out, f'{key}: 학습 {res["fit_seconds"]:.1f}초, 저장·재로딩·SHA·백업 확인')
    return entry, True


def selection_unit(X, y, ids, fo, out, scheme, f, design, fit_fn=fit_candidate, resume_incomplete=False):
    c, out = design['config'], Path(out)
    unit = f'{scheme}/f{f}'
    ledger = read_ledger(out)
    if unit in ledger['selections']:
        entry = ledger['selections'][unit]
        check_saved(out, entry)
        return json.loads((out/entry['dir']/'selection.json').read_text(encoding='utf-8'))
    tr, va = np.flatnonzero(fo != f), np.flatnonzero(fo == f)
    inf = inner_folds(X.iloc[tr], y.iloc[tr], c)
    require(inner_digest(inf) == design['inner_fold_sha256'][unit], '봉인된 내부 Fold 변경')
    probabilities = {}
    for candidate in c['candidates']:
        p = np.full((len(tr), 3), np.nan)
        for j in range(c['inner']['n_splits']):
            a, b = tr[inf != j], tr[inf == j]
            require(not (set(a) | set(b)) & set(va), '외부 검증 행 내부 학습 누수')
            e, _ = checkpoint_fit(out, f'{unit}/inner/{candidate}/i{j}', candidate, 'inner',
                                  X.iloc[a], y.iloc[a], X.iloc[b], y.iloc[b], ids.iloc[b], c, fit_fn, resume_incomplete)
            saved = read_pred(out/e['dir']/'pred.csv.gz')
            np.testing.assert_array_equal(saved[ID].to_numpy(), ids.iloc[b].to_numpy())
            np.testing.assert_array_equal(saved.y_true.to_numpy(), y.iloc[b].to_numpy())
            p[inf == j] = saved[PROBS].to_numpy()
        require(np.isfinite(p).all(), '내부 OOF 미완료')
        probabilities[candidate] = p
    chosen, scores = select_candidates(y.iloc[tr].to_numpy(), probabilities, c)
    record = dict(study_id=STUDY_ID, scheme=scheme, outer_fold=f, chosen=chosen, inner_scores=scores,
                  inner_rows=len(tr), selection_uses_outer_labels=False, created_at_utc=now())
    directory = f'{unit}/selection'
    d = out/directory
    d.mkdir(parents=True, exist_ok=True)
    path = d/'selection.json'
    if path.exists():
        prior = json.loads(path.read_text(encoding='utf-8'))
        require(prior['chosen'] == chosen and prior['inner_scores'] == scores, '미완료 선택 기록 불일치')
        path.rename(d/f'selection.incomplete.{os.getpid()}.json')
    write(path, record)
    # 결정 단계가 백업 후 중단되어도 같은 결과를 검증하여 재개할 수 있다.
    if (out/'backup'/directory).exists():
        old = out/'backup'/directory/'selection.json'
        prior = json.loads(old.read_text(encoding='utf-8'))
        require(prior['chosen'] == chosen and prior['inner_scores'] == scores, '백업 선택 불일치')
        shutil.copy2(old, path)
        record = json.loads(path.read_text(encoding='utf-8'))
        entry = dict(dir=directory, status='complete', artifact_sha256={'selection.json': sha256_file(path)})
        check_saved(out, entry)
    else:
        entry = save_artifacts(out, directory, [path])
    ledger = read_ledger(out)
    ledger['selections'][unit] = entry
    atomic_write(out/'ledger.json', ledger)
    return record


def evaluate_unit(X, y, ids, fo, ref, out, scheme, f, design, selection,
                  fit_fn=fit_candidate, resume_incomplete=False):
    c, out = design['config'], Path(out)
    unit = f'{scheme}/f{f}'
    ledger = read_ledger(out)
    if unit in ledger['evaluations']:
        check_saved(out, ledger['evaluations'][unit])
        return
    # 선택이 저장·봉인된 다음에만 외부 학습 및 평가를 실행한다.
    check_saved(out, ledger['selections'][unit])
    persisted = json.loads((out/ledger['selections'][unit]['dir']/'selection.json').read_text(encoding='utf-8'))
    require(selection == persisted, '저장된 선택과 불일치')
    tr, va = np.flatnonzero(fo != f), np.flatnonzero(fo == f)
    rf = ref.loc[ref.fold == f].reset_index(drop=True)
    np.testing.assert_array_equal(rf[ID].to_numpy(), ids.iloc[va].to_numpy())
    baseline = score(rf.y_true.to_numpy(), rf.y_pred.to_numpy())
    blocks = {}
    for policy, candidate in selection['chosen'].items():
        if candidate == 'B':
            pred = rf.y_pred.to_numpy()
        else:
            e, _ = checkpoint_fit(out, f'{unit}/outer/{candidate}', candidate, 'outer',
                                  X.iloc[tr], y.iloc[tr], X.iloc[va], y.iloc[va], ids.iloc[va], c, fit_fn, resume_incomplete)
            pred = read_pred(out/e['dir']/'pred.csv.gz').y_pred.to_numpy()
        metrics = score(y.iloc[va], pred)
        blocks[policy] = dict(candidate=candidate, metrics=metrics,
                              delta_macro_f1=metrics['macro_f1'] - baseline['macro_f1'])
    directory = f'{unit}/evaluation'
    d = out/directory
    d.mkdir(parents=True, exist_ok=True)
    path = d/'evaluation.json'
    record = dict(study_id=STUDY_ID, scheme=scheme, outer_fold=f, baseline=baseline, policies=blocks,
                  outer_labels_used_only_after_selection=True)
    if path.exists():
        require(json.loads(path.read_text(encoding='utf-8')) == record, '미완료 평가 불일치')
    else:
        write(path, record)
    if (out/'backup'/directory).exists():
        entry = dict(dir=directory, status='complete', artifact_sha256={'evaluation.json': sha256_file(path)})
        check_saved(out, entry)
    else:
        entry = save_artifacts(out, directory, [path])
    ledger = read_ledger(out)
    ledger['evaluations'][unit] = entry
    atomic_write(out/'ledger.json', ledger)


def fit_all(data, output, schemes=None, resume_incomplete=False, fit_fn=fit_candidate, outer_folds=None):
    out = guard(output)
    lock = out/'RUNNING.lock'
    # 배타적 생성으로 동시 실행을 막는다. 죽은 프로세스의 잠금은 별도 확인 후 해제한다.
    with lock.open('x', encoding='utf-8') as fh:
        fh.write(f'pid={os.getpid()} started={now()}\n')
    try:
        X, y, ids, folds, refs, design = verify(data, out)
        require(schemes is None or set(schemes) <= set(folds), '등록되지 않은 분할')
        require(outer_folds is None or set(outer_folds) <= set(range(5)), '등록되지 않은 외부 Fold')
        for scheme, fo in folds.items():
            if schemes is not None and scheme not in schemes:
                continue
            for f in range(5):
                if outer_folds is not None and f not in outer_folds:
                    continue
                selection = selection_unit(X, y, ids, fo, out, scheme, f, design, fit_fn, resume_incomplete)
                evaluate_unit(X, y, ids, fo, refs[scheme], out, scheme, f, design, selection, fit_fn, resume_incomplete)
    finally:
        lock.unlink(missing_ok=True)


def pooled_metrics(confusions):
    """반복 분할의 혼동행렬 합에서 클래스별 지표와 방향별 오류를 계산한다."""
    cm = np.sum(np.asarray(confusions, dtype=np.int64), axis=0)
    require(cm.shape == (3, 3) and (cm >= 0).all(), '잘못된 혼동행렬')
    tp = np.diag(cm).astype(float)
    precision = np.divide(tp, cm.sum(0), out=np.zeros(3), where=cm.sum(0) > 0)
    recall = np.divide(tp, cm.sum(1), out=np.zeros(3), where=cm.sum(1) > 0)
    f1 = np.divide(2*precision*recall, precision+recall, out=np.zeros(3), where=(precision+recall) > 0)
    return dict(confusion=cm.tolist(), rows=int(cm.sum()), support=cm.sum(1).tolist(),
                precision=precision.tolist(), recall=recall.tolist(), f1=f1.tolist(), macro_f1=float(f1.mean()),
                error31=int(cm[2, 0]), error32=int(cm[2, 1]), error23=int(cm[1, 2]),
                error31_rate=float(cm[2, 0]/cm[2].sum()) if cm[2].sum() else 0.0,
                error32_rate=float(cm[2, 1]/cm[2].sum()) if cm[2].sum() else 0.0,
                error23_rate=float(cm[1, 2]/cm[1].sum()) if cm[1].sum() else 0.0)


def finalize_all(data, output):
    out = guard(output)
    _, y, _, folds, _, design = verify(data, out)
    require(not (out/'analysis.json').exists(), '집계 덮어쓰기 금지')
    ledger = read_ledger(out)
    c = design['config']
    units = [f'{s}/f{f}' for s in folds for f in range(5)]
    require(set(ledger['selections']) == set(units) == set(ledger['evaluations']), '외부 Fold 미완료')
    require(all(e['status'] == 'complete' for e in ledger['fits'].values()), '미완료 학습 존재')
    require(sum(e['stage'] == 'inner' for e in ledger['fits'].values()) == c['max_inner_fits'], '내부 fit 수 불일치')
    for section in ('fits', 'selections', 'evaluations'):
        for e in ledger[section].values():
            check_saved(out, e)
    records = [json.loads((out/ledger['evaluations'][u]['dir']/'evaluation.json').read_text(encoding='utf-8')) for u in units]
    pooled_baseline = pooled_metrics([r['baseline']['confusion'] for r in records])
    policies = {}
    for policy in c['policies']:
        deltas = [r['policies'][policy]['delta_macro_f1'] for r in records]
        by_seed = {str(c['schemes'][s]['seed']): float(np.mean([r['policies'][policy]['delta_macro_f1']
                    for r in records if r['scheme'] == s])) for s in folds}
        gates = dict(mean_delta_ok=bool(np.mean(deltas) >= c['rules']['mean_delta_min']),
                     fold_wins_ok=sum(d > 0 for d in deltas) >= c['rules']['fold_wins_min'],
                     positive_seeds_ok=sum(d > 0 for d in by_seed.values()) >= c['rules']['positive_seed_means_min'])
        pooled = pooled_metrics([r['policies'][policy]['metrics']['confusion'] for r in records])
        policies[policy] = dict(fold_deltas=deltas, mean_delta=float(np.mean(deltas)), fold_wins=sum(d > 0 for d in deltas),
                                seed_mean_deltas=by_seed, selected_candidates=[r['policies'][policy]['candidate'] for r in records],
                                pooled=pooled, supports_policy=all(gates.values()), gates=gates)
    result = dict(study_id=STUDY_ID, code_commit=design['code_commit'], design_sha256=sha256_file(out/'design.json'),
                  holdout_accessed=False, policies=policies, fold_records=records, pooled_baseline=pooled_baseline,
                  pooled_denominator=dict(predictions=pooled_baseline['rows'], unique_development_rows=len(y),
                                          repeated_seed_schemes=len(folds),
                                          note='같은 개발행을 seed마다 반복 평가했다. 예측 수를 독립 관측 수로 해석하지 않는다. 오류율 분모는 실제 해당 등급 support다.'),
                  actual_inner_fits=sum(e['stage'] == 'inner' for e in ledger['fits'].values()),
                  actual_outer_refits=sum(e['stage'] == 'outer' for e in ledger['fits'].values()),
                  fit_attempts=sum(e['attempts'] for e in ledger['fits'].values()),
                  fit_seconds_sum=sum(e['fit_seconds'] for e in ledger['fits'].values()),
                  note='ROUNDS와 LEAVES는 별도 정책이다. 조합 최적화, 결정 규칙 결합, 최종 모델 선택 또는 독립 표본 20개로 해석하지 않는다.')
    write(out/'analysis.json', result)
    shutil.copy2(out/'analysis.json', out/'backup'/'analysis.json')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'fit-all', 'finalize', 'status'])
    p.add_argument('--data', default='../../earthquake/data')
    p.add_argument('--output', required=True)
    p.add_argument('--code-commit')
    p.add_argument('--schemes', nargs='*')
    p.add_argument('--outer-folds', type=int, nargs='*')
    p.add_argument('--resume-incomplete', action='store_true')
    a = p.parse_args()
    if a.action == 'prepare':
        prepare(a.data, a.output, a.code_commit)
    elif a.action == 'fit-all':
        fit_all(a.data, a.output, a.schemes, a.resume_incomplete, outer_folds=a.outer_folds)
    elif a.action == 'finalize':
        finalize_all(a.data, a.output)
    else:
        ledger = read_ledger(guard(a.output))
        print(json.dumps(dict(완료학습=sum(e['status'] == 'complete' for e in ledger['fits'].values()),
                              학습시도=sum(e['attempts'] for e in ledger['fits'].values()),
                              선택완료=len(ledger['selections']), 평가완료=len(ledger['evaluations'])), ensure_ascii=False))


if __name__ == '__main__':
    main()
