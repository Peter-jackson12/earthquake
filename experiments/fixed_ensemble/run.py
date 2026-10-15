"""T19: 학습 없이 원본 OOF의 고정 50:50 확률 평균을 평가한다."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from preprocess import ID, load_development

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PRIVATE = ROOT.parents[1] / 'local_reproduction' / 'T19_20261015_fixed_ensemble'
STUDY = 'T19_fixed_ensemble_v1'
ARMS = ['LGBM_O', 'LGBM_TE_G']
SCHEMES = ['geo2_s42', 'geo2_s101', 'geo2_s202', 'geo2_s303']
PROBS = ['p_grade_1', 'p_grade_2', 'p_grade_3']
SOURCE = ['preprocess.py', 'configs/data_contract.json',
          'experiments/fixed_ensemble/__init__.py', 'experiments/fixed_ensemble/run.py',
          'experiments/fixed_ensemble/config.json', 'tests/test_fixed_ensemble.py',
          'docs/T19_FIXED_ENSEMBLE_PLAN.md']


def require(ok, message):
    if not ok:
        raise ValueError(message)


def guard(path):
    p = Path(path)
    require('final_holdout' not in p.as_posix().lower()
            and 'final_holdout' not in p.resolve().as_posix().lower(), '홀드아웃 경로 금지')
    return p


def digest(path):
    h = hashlib.sha256()
    with guard(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(guard(path).read_text(encoding='utf-8'))


def write_new(path, value):
    path = guard(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    require(read(path) == value, 'JSON 재로딩 불일치')


def environment():
    return dict(python=platform.python_version(), **{
        k: importlib.metadata.version(k) for k in
        ['numpy', 'pandas', 'scipy', 'scikit-learn', 'lightgbm', 'joblib', 'threadpoolctl']})


def sources():
    return {name: digest(ROOT / name) for name in SOURCE}


def config():
    c = read(HERE / 'config.json')
    require(c['study_id'] == STUDY and c['arms'] == ARMS
            and list(c['schemes']) == SCHEMES, '고정 실험/분할 변경')
    require(c['labels'] == [1, 2, 3] and c['weights'] == [.5, .5]
            and c['decision_rule'] == 'arithmetic_probability_mean_argmax_first_label', '고정 규칙 변경')
    require(c['new_model_fits'] == 0 and not c['holdout_access']
            and not c['model_selection'] and not c['independent_performance_estimate'], '범위 변경')
    require(c['criteria'] == dict(mean_matched_fold_macro_f1_delta_min=.003, fold_wins_min=15,
            all_seed_pooled_macro_f1_delta_gt=0.0, each_seed_grade3_recall_delta_min=-.01,
            required_baselines=ARMS), '사전 기술기준 변경')
    return c


def probabilities(p):
    p = np.asarray(p, dtype=np.float64)
    require(p.ndim == 2 and p.shape[1] == 3 and np.isfinite(p).all()
            and (p >= 0).all() and (p <= 1).all()
            and np.allclose(p.sum(axis=1), 1, atol=1e-10, rtol=0), '확률 계약 실패')
    return p


def ensemble(a, b):
    a, b = probabilities(a), probabilities(b)
    require(a.shape == b.shape, '확률 행수 불일치')
    p = .5 * a + .5 * b
    return p, p.argmax(axis=1) + 1


def integer_values(series, allowed, message):
    values = np.asarray(series)
    require(np.issubdtype(values.dtype, np.number) and np.isfinite(values).all()
            and np.equal(values, np.floor(values)).all() and set(values) <= set(allowed), message)
    return values.astype(np.int64)


def align(frame, ids, y=None, folds=None):
    require(ID in frame and frame[ID].notna().all() and frame[ID].is_unique
            and len(frame) == len(ids) and set(frame[ID]) == set(ids), 'ID 정확한 1회 커버리지 실패')
    frame = frame.set_index(ID).loc[np.asarray(ids)].reset_index()
    f = integer_values(frame['fold'], range(5), 'fold 정수/범위 실패')
    if folds is not None:
        require(np.array_equal(f, folds), 'OOF fold 불일치')
    if y is not None:
        yt = integer_values(frame['y_true'], [1, 2, 3], 'OOF 정답 실패')
        yp = integer_values(frame['y_pred'], [1, 2, 3], 'OOF 예측 실패')
        p = probabilities(frame[PROBS].to_numpy())
        require(np.array_equal(yt, y) and np.array_equal(yp, p.argmax(axis=1) + 1),
                'OOF 정답/원본 argmax 불일치')
    return frame


def score(y, pred):
    cm = confusion_matrix(y, pred, labels=[1, 2, 3])
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=[1, 2, 3], zero_division=0)
    errors = {}
    for a, b in [(3, 1), (3, 2), (2, 3)]:
        n, den = int(cm[a-1, b-1]), int(s[a-1])
        errors[f'{a}_to_{b}'] = dict(count=n, denominator_actual_class=den, rate=n/den if den else None)
    return dict(n=len(y), macro_f1=float(f.mean()), accuracy=float(np.trace(cm)/len(y)),
                precision=p.tolist(), recall=r.tolist(), f1=f.tolist(), support=s.tolist(),
                confusion=cm.tolist(), errors=errors)


def error_event_overlap(y, a, b, e, actual, predicted):
    """특정 오분류 사건의 교집합과 감소/증가. 감소가 반드시 정답 전환은 아니다."""
    mask = y == actual
    aa, bb, ee = a[mask] == predicted, b[mask] == predicted, e[mask] == predicted
    categories = dict(both_event=aa & bb, only_O_event=aa & ~bb,
                      only_TE_event=~aa & bb, neither_event=~aa & ~bb)
    den = int(mask.sum())
    result = dict(actual_class=actual, predicted_class=predicted, denominator_actual_class=den,
        total_rows=len(y), categories={k: dict(count=int(v.sum()),
            rate_actual_class=float(v.sum()/den) if den else None,
            fraction_all_rows=float(v.sum()/len(y)), ensemble_event=int((v & ee).sum()),
            ensemble_actual_correct=int((v & (e[mask] == actual)).sum()),
            ensemble_other_wrong=int((v & ~ee & (e[mask] != actual)).sum()))
            for k, v in categories.items()}, by_baseline={})
    for arm, before in zip(ARMS, [aa, bb]):
        removed, introduced = before & ~ee, ~before & ee
        result['by_baseline'][arm] = dict(before_count=int(before.sum()), after_count=int(ee.sum()),
            event_removed=int(removed.sum()), event_introduced=int(introduced.sum()),
            removed_to_actual_correct=int((removed & (e[mask] == actual)).sum()),
            removed_to_other_wrong=int((removed & (e[mask] != actual)).sum()),
            net_event_reduction=int(before.sum()-ee.sum()), denominator_actual_class=den)
    return result


def overlap(y, a, b, e, by_class=True):
    y, a, b, e = map(np.asarray, (y, a, b, e))
    ac, bc, ec = a == y, b == y, e == y
    masks = dict(both_correct=ac & bc, both_wrong=~ac & ~bc,
                 O_rescues_TE=ac & ~bc, TE_rescues_O=~ac & bc)
    result = dict(n=len(y), model_disagreement=int(np.count_nonzero(a != b)),
                  error_union=int(np.count_nonzero(~ac | ~bc)),
                  error_intersection=int(np.count_nonzero(~ac & ~bc)))
    result['categories'] = {k: dict(count=int(m.sum()), denominator=len(y),
        fraction=float(m.sum()/len(y)) if len(y) else None,
        ensemble_correct=int((m & ec).sum()), ensemble_wrong=int((m & ~ec).sum()))
        for k, m in masks.items()}
    require(sum(v['count'] for v in result['categories'].values()) == len(y), '오류 분할 불일치')
    if by_class:
        result['by_actual_class'] = {str(k): overlap(y[y == k], a[y == k], b[y == k], e[y == k], False)
                                     for k in [1, 2, 3]}
        result['error_events'] = {f'{k}_to_{v}': error_event_overlap(y, a, b, e, k, v)
                                   for k, v in [(3, 1), (3, 2), (2, 3)]}
    return result


def evaluate(y, a, b):
    _, ep = ensemble(a, b)
    ap, bp = a.argmax(axis=1) + 1, b.argmax(axis=1) + 1
    metrics = {ARMS[0]: score(y, ap), ARMS[1]: score(y, bp), 'ENSEMBLE_50_50': score(y, ep)}
    return dict(metrics=metrics, overlap=overlap(y, ap, bp, ep),
                deltas={arm: dict(macro_f1=metrics['ENSEMBLE_50_50']['macro_f1']-metrics[arm]['macro_f1'],
                                  grade3_recall=metrics['ENSEMBLE_50_50']['recall'][2]-metrics[arm]['recall'][2])
                        for arm in ARMS})


def provenance(c):
    """모델 로딩 없이 원본 봉인 설계와 완료 fold의 검증 예측에 결속한다."""
    result = {}
    for scheme, sc in c['schemes'].items():
        for arm, ref in sc['refs'].items():
            require(digest(ref['path']) == ref['sha256'], '원본 OOF provenance SHA 실패')
            base = guard(ref['path']).parent
            run = base.parents[1]
            design, seal, ledger = (read(run/n) for n in ['design.json', 'design_seal.json', 'ledger.json'])
            require(digest(run/'design.json') == seal['design_sha256']
                    and design['code_commit'] == seal['code_commit'], '원본 설계 봉인 실패')
            contract = read(ROOT/'configs/data_contract.json')
            require(design['data_contract'] == contract and not design.get('holdout_accessed', True),
                    '원본 데이터/홀드아웃 계약 실패')
            expected_study = ('T8_lgbm_native_cat_v1' if scheme == 'geo2_s42' and arm == ARMS[0]
                              else 'T11_geo_te_group_inner_v1' if scheme == 'geo2_s42'
                              else 'T13_geo2_te_confirm_v1')
            require(design['study_id'] == expected_study and ledger['study_id'] == expected_study,
                    '원본 연구 출처 실패')
            files = {str(run/n): digest(run/n) for n in ['design.json', 'design_seal.json', 'ledger.json']}
            files[str(guard(ref['path']))] = ref['sha256']
            for j in range(5):
                key = f'{base.relative_to(run).as_posix()}/{j}'
                entry = ledger['folds'][key]
                require(entry['status'] == 'complete' and entry['fold'] == j, '원본 fold 미완료')
                require(guard(run/entry['dir']).resolve() == base.resolve(), '원본 fold 디렉터리 불일치')
                for name in [f'fold_{j}.json', f'fold_{j}_pred.csv.gz']:
                    require(digest(base/name) == entry['artifact_sha256'][name], '원본 fold SHA 실패')
                    files[str(base/name)] = digest(base/name)
                metadata = read(base/f'fold_{j}.json')
                require(metadata['study_id'] == expected_study and metadata['fold'] == j
                        and metadata['arm'] == base.name, '원본 fold 메타 출처 실패')
            result[f'{scheme}__{arm}'] = dict(study_id=expected_study, code_commit=design['code_commit'], files=files)
    return result


def inputs(data, c):
    data = guard(data)
    contract = read(ROOT/'configs/data_contract.json')
    for name, value in contract['files'].items():
        require(digest(data/name) == value, '개발 데이터 SHA 실패: '+name)
    X, y, ids, _ = load_development(data)
    require(len(y) == contract['expected_rows'], '개발 행수 실패')
    order = np.argsort(ids.to_numpy(), kind='stable')
    X, y, ids = X.iloc[order].reset_index(drop=True), y.to_numpy()[order], ids.to_numpy()[order]
    manifests, checks = {}, {}
    for scheme, sc in c['schemes'].items():
        ref = sc['manifest']
        require(digest(ref['path']) == ref['sha256'], 'manifest SHA 실패')
        m = align(pd.read_csv(guard(ref['path'])), ids)
        fold = integer_values(m.fold, range(5), 'manifest fold 실패')
        require(set(fold) == set(range(5)), '5fold 커버리지 실패')
        for col in c['group_columns']:
            require(np.array_equal(m[col].astype('int64'), X[col].astype('int64')), 'manifest 지역 불일치')
        require(m.groupby(c['group_columns']).fold.nunique().eq(1).all(), 'geo2 그룹 누출')
        support = [np.bincount(y[fold == j].astype(int), minlength=4)[1:].tolist() for j in range(5)]
        require(np.asarray(support).min() >= 30, 'fold 최소 클래스 지지 실패')
        manifests[scheme] = fold
        checks[scheme] = dict(rows=len(m), groups=len(m.groupby(c['group_columns'])),
                              support_by_fold=support, group_overlap=0)
    return y, ids, manifests, contract, checks


def reference(ref, ids, y, folds):
    require(digest(ref['path']) == ref['sha256'], 'OOF SHA 실패')
    frame = align(pd.read_csv(guard(ref['path']), float_precision='round_trip'), ids, y, folds)
    # OOF 결합 전의 저장 검증 예측과 확률까지 정확히 일치해야 한다.
    for j in range(5):
        original = pd.read_csv(guard(Path(ref['path']).parent/f'fold_{j}_pred.csv.gz'), float_precision='round_trip')
        mask = folds == j
        sub = align(original, ids[mask], y[mask], folds[mask])
        require(np.array_equal(frame.loc[mask, PROBS].to_numpy(), sub[PROBS].to_numpy()), '원본 fold/OOF 확률 불일치')
        metadata = read(Path(ref['path']).parent/f'fold_{j}.json')
        old, recalculated = metadata['metrics'], score(y[mask], sub.y_pred.to_numpy())
        require(old['n'] == int(mask.sum()) == metadata['valid_rows']
                and metadata['train_rows'] + metadata['valid_rows'] == len(y)
                and old['confusion'] == recalculated['confusion']
                and abs(old['macro_f1'] - recalculated['macro_f1']) < 1e-14,
                '원본 baseline 지표/행수 provenance 불일치')
    return frame[PROBS].to_numpy()


def output_path(output):
    out = guard(output).resolve()
    require(out == PRIVATE/'run', 'T19 비공개 run 경로만 허용')
    return out


def backup(out, path):
    dest = out/'backup'/path.relative_to(out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    require(not dest.exists(), '백업 덮어쓰기 금지')
    shutil.copy2(path, dest)
    require(digest(path) == digest(dest), '백업 SHA 실패')


def prepare(data, output, commit):
    out = output_path(output)
    require(not out.exists(), '기존 run 덮어쓰기 금지')
    require(isinstance(commit, str) and len(commit) == 40 and all(v in '0123456789abcdef' for v in commit),
            '전체 코드 커밋 SHA 필요')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain', '--', *SOURCE], cwd=ROOT, text=True).strip()
    require(head == commit and not dirty, '분석 전 계획/코드 커밋 필요')
    c = config()
    require(environment() == c['environment'], '동결 환경 실패')
    y, ids, folds, contract, validation = inputs(data, c)
    prov = provenance(c)
    for scheme, sc in c['schemes'].items():
        for ref in sc['refs'].values():
            reference(ref, ids, y, folds[scheme])
    out.mkdir(parents=True)
    design = dict(study_id=STUDY, code_commit=commit, config=c, source_sha256=sources(),
                  data_contract=contract, environment=environment(), provenance=prov, validation=validation,
                  rows=len(y), ids_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),
                  new_model_fits=0, holdout_accessed=False, created_at_utc=datetime.now(timezone.utc).isoformat())
    write_new(out/'design.json', design)
    write_new(out/'design_seal.json', dict(design_sha256=digest(out/'design.json'), code_commit=commit))
    for name in ['design.json', 'design_seal.json']:
        backup(out, out/name)
    # 원장은 append-only 영수증 목록이며 기존 단위와 이전 상태를 덮어쓰지 않는다.
    write_new(out/'ledger.json', dict(study_id=STUDY, unit_keys=SCHEMES, new_model_fits=0))
    backup(out, out/'ledger.json')
    return design


def verify(data, out):
    for name in ['design.json', 'design_seal.json', 'ledger.json']:
        require(digest(out/name) == digest(out/'backup'/name), '봉인/원장 백업 불일치')
    d, seal = read(out/'design.json'), read(out/'design_seal.json')
    require(digest(out/'design.json') == seal['design_sha256'], '설계 봉인 SHA 실패')
    require(d['code_commit'] == seal['code_commit'] and d['config'] == config()
            and d['source_sha256'] == sources() and d['environment'] == environment(), '설계/코드/환경 변경')
    y, ids, folds, contract, checks = inputs(data, d['config'])
    require(contract == d['data_contract'] and checks == d['validation']
            and provenance(d['config']) == d['provenance'], '원본 근거 변경')
    return y, ids, folds, d


def saved(out, key):
    p, receipt = out/'units'/f'{key}.json', out/'receipts'/f'{key}.json'
    if not receipt.exists():
        # interrupted orphan is retained under a unique name before new work.
        for orphan in [p, out/'backup'/'units'/p.name,
                       out/'backup'/'receipts'/p.name, out/'receipts'/f'{key}.pending.json']:
            if orphan.exists():
                orphan.rename(orphan.with_name(orphan.name+f'.incomplete_{time.time_ns()}'))
        return None
    r = read(receipt)
    require(r['status'] == 'complete' and r['key'] == key
            and digest(p) == r['sha256'] == digest(out/'backup'/'units'/p.name)
            and digest(receipt) == digest(out/'backup'/'receipts'/receipt.name), '완료 단위 SHA/백업 실패')
    return read(p)


def persist_unit(out, key, rec):
    p = out/'units'/f'{key}.json'
    write_new(p, rec)
    backup(out, p)
    # 최종 영수증 rename이 commit 지점이다. 이전 중단은 다음 재개에서 orphan 보존한다.
    rp = out/'receipts'/f'{key}.json'
    pending = rp.with_name(f'{key}.pending.json')
    write_new(pending, dict(status='complete', key=key, sha256=digest(p)))
    back = out/'backup'/'receipts'/rp.name
    require(not back.exists() and not rp.exists(), '완료 영수증 덮어쓰기 금지')
    back.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pending, back)
    require(digest(pending) == digest(back), '영수증 백업 실패')
    pending.rename(rp)
    saved(out, key)


def fold_summary(folds):
    return {arm: dict(mean_macro_f1=float(np.mean([f['metrics'][arm]['macro_f1'] for f in folds])),
                      std_macro_f1_ddof1=float(np.std([f['metrics'][arm]['macro_f1'] for f in folds], ddof=1)))
            for arm in [*ARMS, 'ENSEMBLE_50_50']}


def analyze(data, output):
    out = output_path(output)
    lock = out/'RUNNING.lock'
    with lock.open('x', encoding='utf-8') as f:
        f.write(f'pid={os.getpid()}\n')
    try:
        y, ids, folds, d = verify(data, out)
        for scheme, sc in d['config']['schemes'].items():
            if saved(out, scheme) is not None:
                continue
            a, b = [reference(sc['refs'][arm], ids, y, folds[scheme]) for arm in ARMS]
            rec = dict(scheme=scheme, seed=sc['seed'], rows=len(y), pooled=evaluate(y, a, b),
                       folds=[dict(fold=j, **evaluate(y[folds[scheme] == j],
                             a[folds[scheme] == j], b[folds[scheme] == j])) for j in range(5)],
                       new_model_fits=0, independent_performance_estimate=False)
            rec['fold_summary'] = fold_summary(rec['folds'])
            persist_unit(out, scheme, rec)
    finally:
        lock.unlink()


def summarize(records, criteria):
    result = {}
    for arm in ARMS:
        delta = [r['deltas'][arm]['macro_f1'] for unit in records for r in unit['folds']]
        pooled = [r['pooled']['deltas'][arm]['macro_f1'] for r in records]
        recall = [r['pooled']['deltas'][arm]['grade3_recall'] for r in records]
        require(len(delta) == 20 and len(pooled) == 4, '20fold/4seed 미완료')
        gates = dict(mean_delta=float(np.mean(delta)) >= criteria['mean_matched_fold_macro_f1_delta_min'],
                     fold_wins=sum(v > 0 for v in delta) >= criteria['fold_wins_min'],
                     every_seed_pooled=all(v > criteria['all_seed_pooled_macro_f1_delta_gt'] for v in pooled),
                     every_seed_recall3=all(v >= criteria['each_seed_grade3_recall_delta_min'] for v in recall))
        result[arm] = dict(matched_fold_deltas=delta, mean_delta=float(np.mean(delta)),
            std_delta_ddof1=float(np.std(delta, ddof=1)), wins=sum(v > 0 for v in delta),
            ties=sum(v == 0 for v in delta), losses=sum(v < 0 for v in delta),
            seed_pooled_deltas=pooled, seed_grade3_recall_deltas=recall, gates=gates,
            technical_criteria_pass=all(gates.values()))
    all_folds = [f for unit in records for f in unit['folds']]
    model_summary = fold_summary(all_folds) if all('metrics' in f for f in all_folds) else None
    return dict(by_baseline=result, model_fold_summary=model_summary,
                technical_criteria_pass=all(v['technical_criteria_pass'] for v in result.values()))


def finalize(data, output):
    out = output_path(output)
    require(not (out/'RUNNING.lock').exists(), '분석 실행 중')
    require(not (out/'analysis.json').exists(), '기존 집계 덮어쓰기 금지')
    _, _, _, d = verify(data, out)
    records = [saved(out, s) for s in SCHEMES]
    require(all(r is not None for r in records), '분할 미완료')
    result = dict(study_id=STUDY, code_commit=d['code_commit'], design_sha256=digest(out/'design.json'),
        new_model_fits=0, holdout_accessed=False, independent_performance_estimate=False,
        model_selection=False, adopted=False, weights=[.5, .5], labels=[1, 2, 3],
        summary=summarize(records, d['config']['criteria']), records=records,
        note='재사용 개발 OOF의 고정 앙상블 탐색평가. 독립 검증/최종 모델 선정/채택 근거가 아님.')
    write_new(out/'analysis.json', result)
    backup(out, out/'analysis.json')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'analyze', 'finalize', 'status'])
    p.add_argument('--data', default='../../earthquake/data')
    p.add_argument('--output', required=True)
    p.add_argument('--code-commit')
    args = p.parse_args()
    if args.action == 'prepare':
        prepare(args.data, args.output, args.code_commit)
    elif args.action == 'analyze':
        analyze(args.data, args.output)
    elif args.action == 'finalize':
        finalize(args.data, args.output)
    else:
        out = output_path(args.output)
        print(json.dumps(dict(units={s: saved(out, s) is not None for s in SCHEMES},
                              finalized=(out/'analysis.json').exists()), ensure_ascii=False))


if __name__ == '__main__':
    main()
