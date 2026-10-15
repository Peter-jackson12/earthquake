"""T17: 검증된 전체 개발 Group OOF 재사용으로 b만 고정(새 학습 없음)."""
from __future__ import annotations
import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from experiments.geo_group_cv import run as t5
from experiments.geo_hier_te import run as io
from preprocess import ID, sha256_file

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
STUDY_ID = 'T17_grade3_rule_freeze_v1'
ARMS = ['LGBM_O', 'LGBM_TE_G']
SCHEMES = ['geo2_s42', 'geo2_s101', 'geo2_s202', 'geo2_s303']
GRID = [v / 10 for v in range(21)]
SOURCE = list(dict.fromkeys(t5.SOURCE + [
    'configs/data_contract.json', 'experiments/geo_hier_te/run.py',
    'experiments/grade3_decision_rule/config.json',
    'experiments/grade3_rule_freeze/__init__.py', 'experiments/grade3_rule_freeze/run.py',
    'experiments/grade3_rule_freeze/config.json', 'tests/test_grade3_rule_freeze.py',
    'docs/T17_GRADE3_RULE_FREEZE_PLAN.md']))
require = t5.require


def guard(path):
    p = Path(path)
    require('final_holdout' not in p.as_posix().lower()
            and 'final_holdout' not in p.resolve().as_posix().lower(), '홀드아웃 경로 접근 금지')
    return p


def config():
    c = json.loads((HERE/'config.json').read_text(encoding='utf-8'))
    old = json.loads((ROOT/'experiments/grade3_decision_rule/config.json').read_text(encoding='utf-8'))
    require(c['study_id'] == STUDY_ID and c['arms'] == ARMS and list(c['schemes']) == SCHEMES,
            '사전 등록 팔/분할 변경')
    require(c['grid'] == GRID and c['objective'] == 'mean_seed_pooled_macro_f1'
            and c['tie_break'] == 'smallest_b_exact_maximum', '사전 등록 목적함수 변경')
    require(c['new_model_fits'] == 0 and c['class_weight'] is None and c['sampling'] is None
            and not c['holdout_access'] and not c['model_selection']
            and not c['independent_performance_estimate'], '실험 범위 변경')
    require(c['environment'] == old['environment'] and c['group_columns'] == old['group_columns']['group2'],
            '환경/그룹 정의 변경')
    for s in SCHEMES:
        expected = old['schemes'][s]
        require(c['schemes'][s] == dict(seed=expected['seed'], manifest=expected['manifest'],
                                      refs={a: expected['refs'][a] for a in ARMS}), 'OOF 출처 변경')
        guard(c['schemes'][s]['manifest']['path'])
        for a in ARMS:
            guard(c['schemes'][s]['refs'][a]['path'])
    return c


def sources():
    return {p: sha256_file(ROOT/p) for p in SOURCE}


def adjust_predict(p, b):
    p = np.asarray(p, dtype=np.float64)
    require(p.ndim == 2 and p.shape[1] == 3 and np.isfinite(p).all() and (p >= 0).all()
            and np.allclose(p.sum(axis=1), 1, atol=1e-10, rtol=0), '확률 형식 오류')
    require(np.isfinite(b) and b in GRID, '등록 그리드 밖 b')
    return (p * [1., 1., np.exp(b)]).argmax(axis=1) + 1


def curve(y, p, grid=GRID):
    return [t5.score(y, adjust_predict(p, b))['macro_f1'] for b in grid]


def metrics_curve(y, p):
    """선택용 통계에도 클래스별 P/R/F1과 혼동행렬을 남긴다."""
    return [t5.score(y, adjust_predict(p, b)) for b in GRID]


def choose(curves, grid=GRID):
    values = np.asarray(curves, dtype=float)
    require(values.ndim == 2 and values.shape[1] == len(grid) and len(values) > 0
            and np.isfinite(values).all(), '선택 곡선 오류')
    mean = values.mean(axis=0)
    j = int(np.argmax(mean))  # 정렬된 그리드: 정확 동률일 때 가장 작은 b
    return float(grid[j]), mean.tolist()


def validate_reference(path, digest, ids, y, folds):
    require(sha256_file(guard(path)) == digest, 'OOF 해시 불일치')
    r = io.read_pred(path)
    require(len(r) == len(ids) and r[ID].is_unique
            and set(r[ID]) == set(ids), '전체 개발 OOF 정확한 1회 커버리지 실패')
    r = r.set_index(ID).loc[np.asarray(ids)].reset_index()
    np.testing.assert_array_equal(r.fold.to_numpy(), folds)
    np.testing.assert_array_equal(r.y_true.to_numpy(), np.asarray(y))
    np.testing.assert_array_equal(adjust_predict(r[t5.PROBS].to_numpy(), 0.), r.y_pred.to_numpy())
    return r


def read_inputs(data, c):
    X, y, ids, _, contract = t5.load(guard(data))
    assignments, validation = {}, {}
    for s, sc in c['schemes'].items():
        ref = sc['manifest']
        require(sha256_file(guard(ref['path'])) == ref['sha256'], 'manifest 해시 불일치')
        m = pd.read_csv(ref['path'])
        require(len(m) == len(ids) and m[ID].is_unique and set(m[ID]) == set(ids), 'manifest 커버리지 실패')
        m = m.set_index(ID).loc[np.asarray(ids)].reset_index()
        folds = m.fold.to_numpy()
        validation[s] = t5.validate_assignment(X, y, ids, folds, c['group_columns'], t5.cfg())
        assignments[s] = folds
        for a in ARMS:
            ref = sc['refs'][a]
            validate_reference(ref['path'], ref['sha256'], ids, y, folds)
    return y, ids, assignments, contract, validation


def check_output(output):
    out = guard(output).resolve()
    private = ROOT.parents[1]/'local_reproduction'/'T17_20261015_rule_freeze'
    require(out == private/'run', 'T17 전용 비공개 run 경로만 허용')
    return out


def persist(out, key, record, ledger):
    """집계 체크포인트 재로딩·백업 검증 뒤에만 complete 기록."""
    p = out/'units'/f'{key}.json'
    back = out/'backup'/'units'/p.name
    for q in (p, back):
        if q.exists():
            q.rename(q.with_name(q.name + f'.incomplete_{os.getpid()}_{__import__("time").time_ns()}'))
    t5.write(p, record)
    require(json.loads(p.read_text(encoding='utf-8')) == record, '체크포인트 재로딩 실패')
    back.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(p, back)
    entry = dict(status='complete', dir='units', artifact_sha256={p.name: sha256_file(p)})
    io.check_saved(out, entry)
    ledger['units'][key] = entry
    io.atomic_write(out/'ledger.json', ledger)
    return record


def saved(out, key, ledger):
    entry = ledger['units'].get(key)
    if entry is None:
        return None
    require(entry['status'] == 'complete', '원장 상태 오류')
    io.check_saved(out, entry)
    return json.loads((out/'units'/f'{key}.json').read_text(encoding='utf-8'))


def prepare(data, output, code_commit):
    out = check_output(output)
    require(not out.exists(), '기존 결과 덮어쓰기 금지')
    require(isinstance(code_commit, str) and len(code_commit) == 40
            and all(v in '0123456789abcdef' for v in code_commit), '전체 커밋 SHA 필요')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain', '--', *SOURCE], cwd=ROOT, text=True).strip()
    require(head == code_commit and not dirty, '봉인 소스는 HEAD에 커밋되어야 함')
    c = config()
    require(io.environment() == c['environment'], '동결 환경 불일치')
    y, ids, _, contract, validation = read_inputs(data, c)
    design = dict(study_id=STUDY_ID, config=c, code_commit=code_commit, source_sha256=sources(),
                  data_contract=contract, environment=io.environment(), validation=validation,
                  rows=len(y), ids_sha256=__import__('hashlib').sha256(ids.to_numpy().tobytes()).hexdigest(),
                  classifiers_fitted=0, holdout_accessed=False, created_at_utc=io.now())
    out.mkdir(parents=True)
    t5.write(out/'design.json', design)
    t5.write(out/'design_seal.json', dict(design_sha256=sha256_file(out/'design.json'), code_commit=code_commit))
    (out/'backup').mkdir()
    for name in ('design.json', 'design_seal.json'):
        shutil.copy2(out/name, out/'backup'/name)
    io.atomic_write(out/'ledger.json', dict(study_id=STUDY_ID, new_model_fits=0, units={}))
    return design


def verify(data, out):
    c = config()
    for name in ('design.json', 'design_seal.json'):
        require(sha256_file(out/name) == sha256_file(out/'backup'/name), '봉인 백업 불일치')
    design = json.loads((out/'design.json').read_text(encoding='utf-8'))
    seal = json.loads((out/'design_seal.json').read_text(encoding='utf-8'))
    require(sha256_file(out/'design.json') == seal['design_sha256'], '설계 봉인 변경')
    require(design['config'] == c and design['source_sha256'] == sources()
            and design['environment'] == io.environment(), '봉인 설정/소스/환경 변경')
    y, ids, folds, contract, validation = read_inputs(data, c)
    require(contract == design['data_contract'] and validation == design['validation'], '데이터 계약 변경')
    return y, ids, folds, design


def freeze_all(data, output):
    out = check_output(output)
    lock = out/'RUNNING.lock'
    with lock.open('x', encoding='utf-8') as f:
        f.write(f'pid={os.getpid()}\n')
    try:
        y, ids, folds, design = verify(data, out)
        ledger = io.read_ledger(out)
        for a in ARMS:
            records = []
            for s in SCHEMES:
                k = f'{s}__{a}'
                rec = saved(out, k, ledger)
                if rec is None:
                    ref = design['config']['schemes'][s]['refs'][a]
                    r = validate_reference(ref['path'], ref['sha256'], ids, y, folds[s])
                    p = r[t5.PROBS].to_numpy()
                    pooled = metrics_curve(y, p)
                    per_fold = [metrics_curve(np.asarray(y)[folds[s] == j], p[folds[s] == j]) for j in range(5)]
                    rec = dict(scheme=s, arm=a, rows=len(r), pooled_curve=[v['macro_f1'] for v in pooled],
                               fold_curves=[[v['macro_f1'] for v in row] for row in per_fold],
                               pooled_metrics_by_grid=pooled, fold_metrics_by_grid=per_fold,
                               purpose='selection_only_not_independent_performance', new_model_fits=0)
                    persist(out, k, rec, ledger)
                records.append(rec)
            k = f'fixed__{a}'
            if saved(out, k, ledger) is None:
                b, mean = choose([v['pooled_curve'] for v in records])
                persist(out, k, dict(arm=a, selected_b=b, multiplier_grade3=float(np.exp(b)),
                                    mean_seed_pooled_curve=mean, at_grid_max=b == GRID[-1],
                                    independent_performance_estimate=False, adopted=False,
                                    class_weight=None, model_selection=False), ledger)
    finally:
        lock.unlink()


def finalize(data, output):
    out = check_output(output)
    _, _, _, design = verify(data, out)
    require(not (out/'analysis.json').exists(), '기존 집계 덮어쓰기 금지')
    ledger = io.read_ledger(out)
    expected = {f'{s}__{a}' for s in SCHEMES for a in ARMS} | {f'fixed__{a}' for a in ARMS}
    require(set(ledger['units']) == expected and ledger['new_model_fits'] == 0, '원장 미완료')
    records = {k: saved(out, k, ledger) for k in sorted(expected)}
    for a in ARMS:
        b, mean = choose([records[f'{s}__{a}']['pooled_curve'] for s in SCHEMES])
        require(records[f'fixed__{a}']['selected_b'] == b
                and records[f'fixed__{a}']['mean_seed_pooled_curve'] == mean, 'b 재계산 불일치')
    result = dict(study_id=STUDY_ID, code_commit=design['code_commit'],
                  design_sha256=sha256_file(out/'design.json'), new_model_fits=0, holdout_accessed=False,
                  independent_performance_estimate=False, model_selection=False, adopted=False,
                  note='전체 개발 Group OOF 선택용 통계. 독립 성능/팔간 순위/최종 모델 선정 근거가 아님.',
                  grid=GRID, objective=design['config']['objective'], records=records)
    t5.write(out/'analysis.json', result)
    shutil.copy2(out/'analysis.json', out/'backup'/'analysis.json')
    require(sha256_file(out/'analysis.json') == sha256_file(out/'backup'/'analysis.json'), '집계 백업 불일치')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'freeze-all', 'finalize', 'status'])
    p.add_argument('--data', default='../../earthquake/data')
    p.add_argument('--output', required=True)
    p.add_argument('--code-commit')
    a = p.parse_args()
    if a.action == 'prepare':
        prepare(a.data, a.output, a.code_commit)
    elif a.action == 'freeze-all':
        freeze_all(a.data, a.output)
    elif a.action == 'finalize':
        finalize(a.data, a.output)
    else:
        print(json.dumps(io.read_ledger(check_output(a.output)), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
