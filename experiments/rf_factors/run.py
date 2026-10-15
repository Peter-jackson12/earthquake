"""T16 RF 요인 분리. 기준은 감사된 T14 geo2 OOF를 SHA 고정 재사용한다.

80개 신규 RF fit: 4 arm x 4 geo2 manifest x 5 fold. 홀드아웃 접근 없음.
학습은 fit-all에서만 수행하며 preflight/prepare는 분류기를 학습하지 않는다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder

from preprocess import GEO, ID, TEXT, sha256_file
from experiments.age995.run import Age995Treatment
from experiments.te_ros_rf_repro import run as t14

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
STUDY_ID = 'T16_RF_FACTORS'
SCHEMES = list(t14.GEO_SCHEMES)
ARMS = ['ONEHOT', 'NO_ROS', 'DROP_GEO23_TE', 'AGE_C']
BASE = 'T14_RF_TE_ROS_A'
N_TARGETS = 80
SOURCE = list(dict.fromkeys(t14.SOURCE + [
    'experiments/rf_factors/__init__.py', 'experiments/rf_factors/run.py',
    'experiments/rf_factors/config.json', 'tests/test_rf_factors.py', 'docs/T16_RF_FACTORS_PLAN.md']))
require, score, write = t14.require, t14.score, t14.write
now, log, event = t14.now, t14.log, t14.event
atomic_write, read_pred, write_pred = t14.atomic_write, t14.read_pred, t14.write_pred
read_ledger, check_saved = t14.read_ledger, t14.check_saved


def key(scheme, arm, fold):
    return f'{scheme}/{arm}/{fold}'


def targets():
    return [(s, a, f) for s in SCHEMES for f in range(5) for a in ARMS]


def sources():
    return {p: sha256_file(ROOT/p) for p in SOURCE}


def config():
    c = json.loads((HERE/'config.json').read_text(encoding='utf-8'))
    require(c['study_id'] == STUDY_ID and c['schemes'] == SCHEMES and c['arms'] == ARMS, 'Unplanned design')
    require(c['planned_completed_fits'] == N_TARGETS and c['max_new_study_fits'] == 84
            and c['n_folds'] == 5 and c['max_infrastructure_retries'] == 4, 'Unplanned budget')
    require(c['baseline_design_sha256'] == '965f8741dd48ea0b277a9a512b551fd05ac1f365bd9a4d11fd9769688db09a74',
            'Baseline design pin changed')
    require(c['rf_params'] == t14.config()['rf_params'] and c['te'] == t14.config()['te'], 'T14 settings changed')
    require(c['arm_changes'] == {
        'ONEHOT': dict(geo_encoding='onehot', ros=True, age='A', drop_geo23_te=False),
        'NO_ROS': dict(geo_encoding='te', ros=False, age='A', drop_geo23_te=False),
        'DROP_GEO23_TE': dict(geo_encoding='te', ros=True, age='A', drop_geo23_te=True),
        'AGE_C': dict(geo_encoding='te', ros=True, age='C', drop_geo23_te=False)}, 'Factor contrast changed')
    require(not c['holdout_access'] and not c['tuning'] and not c['early_stopping']
            and not c['threshold_or_decision_rule_change'] and c['outer_parallel_jobs'] == 1
            and c['automatic_retries'] == 0 and c['baseline_refits'] == 0, 'Unplanned execution')
    require(c['environment'] == t14.environment(), 'Pinned environment changed')
    return c


def baseline_design(c):
    path = Path(c['baseline_run'])/'design.json'
    t14.require_not_holdout(path.resolve())
    require(sha256_file(path) == c['baseline_design_sha256'], 'T14 baseline design changed')
    d = json.loads(path.read_text(encoding='utf-8'))
    require(d['config'] == t14.config() and d['source_sha256'] == t14.sources(), 'T14 baseline pipeline changed')
    return d


def build_fold_matrices(F_train, X_train, y_train, F_valid, scheme, c, arm):
    """검증 라벨을 받지 않는다. 원시 geo 숫자와 geo 빈도는 모든 arm에서 유지한다."""
    require(arm in ARMS or arm == BASE, 'Unknown arm')
    change = c['arm_changes'].get(arm, dict(geo_encoding='te', ros=True, age='A', drop_geo23_te=False))
    start = time.perf_counter()
    train, valid = F_train.copy(), F_valid.copy()
    age = None
    if change['age'] == 'C':
        age = Age995Treatment('C').fit(train)
        train, valid = age.transform(train), age.transform(valid)
    y0 = pd.Series(np.asarray(y_train, dtype=np.int64) - 1, index=train.index)
    splits, inner = t14.inner_splits_for(scheme, X_train, y_train)
    A, B, geo = t14.target_enc(train, y0, valid, list(GEO), m=c['te']['smoothing_m'],
                             splits=splits, return_state=True)
    # TE replacement/removal never changes count features or raw numeric geo IDs.
    drop = ([f'{g}_te{k}' for g in GEO for k in (0, 2)] if change['geo_encoding'] == 'onehot' else
            [f'{g}_te{k}' for g in GEO[1:] for k in (0, 2)] if change['drop_geo23_te'] else [])
    A, B = A.drop(columns=drop), B.drop(columns=drop)
    cats = [g for g in F_train.columns if g in TEXT]
    Ao = pd.get_dummies(A, columns=cats, dtype=int).astype('float32')
    Bo = pd.get_dummies(B, columns=cats, dtype=int).reindex(columns=Ao.columns, fill_value=0).astype('float32')
    columns = Ao.columns.tolist()
    encoder = None
    names = columns.copy()
    if change['geo_encoding'] == 'onehot':
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True, dtype=np.float32)
        O = encoder.fit_transform(train[list(GEO)])
        V = encoder.transform(valid[list(GEO)])
        Ao = sparse.hstack([sparse.csr_matrix(Ao.to_numpy()), O], format='csr', dtype=np.float32)
        Bo = sparse.hstack([sparse.csr_matrix(Bo.to_numpy()), V], format='csr', dtype=np.float32)
        names += ['onehot__'+n for n in encoder.get_feature_names_out(GEO)]
    t1 = time.perf_counter()
    if change['ros']:
        Xos, yos = t14.RandomOverSampler(random_state=0).fit_resample(Ao, y0)
    else:
        Xos, yos = Ao, y0
    state = dict(columns=columns, selected=names, cats=cats, geo=geo, age=age, encoder=encoder, drop=drop)
    return dict(Ao=Ao, Bo=Bo, Xos=Xos, yos=yos, selected=names, state=state,
                inner_folds=inner, te_seconds=t1-start, ros_seconds=time.perf_counter()-t1)


def apply_state(state, F_valid):
    V = state['age'].transform(F_valid) if state['age'] is not None else F_valid.copy()
    B = t14.apply_state(state['geo'], V, state['columns'], state['cats'])
    if state['encoder'] is not None:
        B = sparse.hstack([sparse.csr_matrix(B.to_numpy()), state['encoder'].transform(V[list(GEO)])],
                          format='csr', dtype=np.float32)
    return B


def matrix_equal(A, B):
    return A.shape == B.shape and ((A != B).nnz == 0 if sparse.issparse(A) else A.equals(B))


def input_bundle(data, c, out=None):
    X, y, ids, _, contract = t14.load_inputs(data)
    if out is None:
        folds = t14.geo_manifest_folds(c, ids, X, y)
    else:
        folds = {}
        for s in SCHEMES:
            p = Path(out)/'manifests'/s/'manifest.csv'
            require(sha256_file(p) == c['manifests'][s]['sha256'], 'Manifest changed')
            mf = pd.read_csv(p)
            np.testing.assert_array_equal(mf[ID], ids)
            folds[s] = mf['fold'].to_numpy()
            t14.t5.validate_assignment(X, y, ids, folds[s], list(GEO[:2]), t14.t5.cfg())
    return X, y, ids, folds, contract


def baseline_check(c, ids, y, folds):
    baseline_design(c)
    return {s: dict(sha256=c['baseline_oofs'][s]['sha256'],
                    rows=len(t14.reference(c['baseline_oofs'][s]['path'], c['baseline_oofs'][s]['sha256'], ids, y, folds[s])))
            for s in SCHEMES}


def preflight(data, destination):
    dest = Path(destination)
    require(not dest.exists(), 'Preflight destination exists')
    c = config()
    X, y, ids, folds, contract = input_bundle(data, c)
    refs = baseline_check(c, ids, y, folds)
    F = t14.original_frame(X)
    rows = {}
    # One fold per split, all arms: preprocessing only, no classifier fitting.
    for s in SCHEMES:
        a, b = np.flatnonzero(folds[s] != 0), np.flatnonzero(folds[s] == 0)
        base = t14.build_fold_matrices(F.iloc[a], X.iloc[a], y.iloc[a], F.iloc[b], s, t14.config())
        check = build_fold_matrices(F.iloc[a], X.iloc[a], y.iloc[a], F.iloc[b], s, c, BASE)
        require(matrix_equal(base['Ao'], check['Ao']) and matrix_equal(base['Bo'], check['Bo'])
                and matrix_equal(base['Xos'], check['Xos']), 'Baseline reconstruction differs from T14')
        np.testing.assert_array_equal(base['yos'], check['yos'])
        rows[s] = {}
        for arm in ARMS:
            built = build_fold_matrices(F.iloc[a], X.iloc[a], y.iloc[a], F.iloc[b], s, c, arm)
            require(matrix_equal(built['Bo'], apply_state(built['state'], F.iloc[b])), 'Encoder reload differs')
            rows[s][arm] = dict(columns=len(built['selected']), train_rows=len(a), valid_rows=len(b),
                                fit_rows=len(built['yos']), sparse=sparse.issparse(built['Xos']))
        print(f'{s}: 전처리 검증 완료 (분류기 학습 0회)', flush=True)
    result = dict(study_id=STUDY_ID, status='PASS', classifiers_fitted=0, holdout_accessed=False,
                  source_sha256=sources(), config_sha256=sha256_file(HERE/'config.json'),
                  data_contract=contract, baseline_oofs=refs, matrices=rows, created_at_utc=now())
    write(dest/'preflight.json', result)
    return result


def prepare(data, output, code_commit, preflight_path):
    out = Path(output)
    require(not out.exists(), 'Study output exists')
    require(len(code_commit) == 40 and all(ch in '0123456789abcdef' for ch in code_commit), 'Full SHA required')
    c = config()
    require(shutil.disk_usage(out.parent).free/2**30 >= c['min_free_disk_gb_at_prepare'], 'Insufficient disk')
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(['git', 'status', '--porcelain', '--', *SOURCE], cwd=ROOT,
                           capture_output=True, text=True, check=True).stdout.strip()
    require(head == code_commit and not dirty, 'Sources must be committed and clean at HEAD')
    pf = json.loads(Path(preflight_path).read_text(encoding='utf-8'))
    require(pf['study_id'] == STUDY_ID and pf['status'] == 'PASS' and pf['classifiers_fitted'] == 0
            and pf['source_sha256'] == sources(), 'Fresh passing preflight required')
    X, y, ids, folds, contract = input_bundle(data, c)
    refs = baseline_check(c, ids, y, folds)
    require(pf['data_contract'] == contract and pf['baseline_oofs'] == refs, 'Preflight inputs changed')
    out.mkdir(parents=True)
    (out/'backup').mkdir()
    for s in SCHEMES:
        p = out/'manifests'/s/'manifest.csv'
        p.parent.mkdir(parents=True)
        shutil.copyfile(c['manifests'][s]['path'], p)
        require(sha256_file(p) == c['manifests'][s]['sha256'], 'Manifest copy changed')
    design = dict(study_id=STUDY_ID, config=c, code_commit=code_commit, source_sha256=sources(),
                  environment=t14.environment(), data_contract=contract, baseline_oofs=refs,
                  preflight_sha256=sha256_file(preflight_path), study_targets=[key(*t) for t in targets()],
                  max_new_study_fits=c['max_new_study_fits'], classifiers_fitted=0, holdout_accessed=False, created_at_utc=now())
    write(out/'design.json', design)
    write(out/'design_seal.json', dict(design_sha256=sha256_file(out/'design.json'), code_commit=code_commit))
    atomic_write(out/'ledger.json', dict(study_id=STUDY_ID, folds={}))
    for name in ['design.json', 'design_seal.json']:
        shutil.copy2(out/name, out/'backup'/name)
    return design


def verify(data, output):
    out = Path(output)
    seal = json.loads((out/'design_seal.json').read_text(encoding='utf-8'))
    require(sha256_file(out/'design.json') == seal['design_sha256'], 'Design seal changed')
    design = json.loads((out/'design.json').read_text(encoding='utf-8'))
    require(design['config'] == config() and design['source_sha256'] == sources()
            and design['environment'] == t14.environment(), 'Sealed configuration/source/environment changed')
    X, y, ids, folds, contract = input_bundle(data, design['config'], out)
    require(contract == design['data_contract'], 'Data contract changed')
    require(baseline_check(design['config'], ids, y, folds) == design['baseline_oofs'], 'Baseline changed')
    return X, y, ids, folds, design


def fit_fold(X, F, y, ids, folds, out, scheme, arm, f, design, retry_interrupted=False, retry_reason=None):
    out = Path(out)
    k = key(scheme, arm, f)
    require(k in design['study_targets'], 'Unregistered fit')
    ledger = read_ledger(out)
    entry = ledger['folds'].get(k)
    if entry and entry['status'] == 'complete':
        check_saved(out, entry)
        return entry, False
    c = design['config']
    require(shutil.disk_usage(out).free/2**30 >= c['min_free_disk_gb_per_fold'], 'Insufficient disk')
    prior = entry['attempts'] if entry else 0
    if entry:
        require(retry_interrupted and retry_reason and retry_reason.strip(),
                'Interrupted fit: explicit --retry-interrupted and infrastructure --retry-reason required')
        require(prior == 1, 'Only one infrastructure retry per fold is allowed')
        require(sum(max(e['attempts']-1, 0) for e in ledger['folds'].values()) < c['max_infrastructure_retries'],
                'Infrastructure retry reserve exhausted')
    require(sum(e['attempts'] for e in ledger['folds'].values()) < design['max_new_study_fits'], 'Budget exhausted')
    entry = dict(dir=f'{scheme}/{arm}', scheme=scheme, arm=arm, fold=f, status='running', attempts=prior+1,
                 started_at_utc=now())
    ledger['folds'][k] = entry
    atomic_write(out/'ledger.json', ledger)
    d = out/entry['dir']
    d.mkdir(parents=True, exist_ok=True)
    for location in [d, out/'backup'/entry['dir']]:
        stale = list(location.glob(f'fold_{f}[._]*')) if location.exists() else []
        if not prior:
            require(not stale, 'Untracked fold files exist')
        for path in stale:
            target = path.with_name(path.name+f'.incomplete_attempt{prior}')
            require(not target.exists(), 'Preserved incomplete artifact exists')
            path.rename(target)
    event(out, fold=k, action='fit_start', attempt=prior+1, infrastructure_retry_reason=retry_reason if prior else None)
    a, b = np.flatnonzero(folds[scheme] != f), np.flatnonzero(folds[scheme] == f)
    built = build_fold_matrices(F.iloc[a], X.iloc[a], y.iloc[a], F.iloc[b], scheme, c, arm)
    model, seconds, fit_warnings = t14.fit_rf(built['Xos'], built['yos'], c)
    p = t14.predict_proba_deterministic(model, built['Bo'])
    require(p.shape == (len(b), 3) and np.isfinite(p).all(), 'Invalid probabilities')
    pred = p.argmax(1)+1
    model_path, state_path = d/f'fold_{f}_model.joblib', d/f'fold_{f}_encoder_state.joblib'
    joblib.dump(model, model_path, compress=3)
    joblib.dump(built['state'], state_path, compress=3)
    preds = d/f'fold_{f}_pred.csv.gz'
    frame = pd.DataFrame({ID: ids.iloc[b].to_numpy(), 'fold': f, 'y_true': y.iloc[b].to_numpy(), 'y_pred': pred,
                          **{n: p[:, i] for i, n in enumerate(t14.PROBS)}})
    write_pred(preds, frame)
    Bo2 = apply_state(joblib.load(state_path), F.iloc[b])
    require(matrix_equal(built['Bo'], Bo2), 'Reload matrix mismatch')
    p2 = t14.predict_proba_deterministic(joblib.load(model_path), Bo2)
    require(np.array_equal(p, p2), 'Reload probability mismatch')
    np.testing.assert_array_equal(read_pred(preds)[t14.PROBS].to_numpy(), p)
    record = dict(study_id=STUDY_ID, scheme=scheme, arm=arm, fold=f, train_rows=len(a), valid_rows=len(b),
                  fit_rows=len(built['yos']), fit_class_counts=np.bincount(built['yos'], minlength=3).tolist(),
                  feature_names=built['selected'], model_columns=len(built['selected']),
                  feature_names_sha256=hashlib.sha256('\n'.join(built['selected']).encode()).hexdigest(),
                  effective_parameters=model.get_params(), metrics=score(y.iloc[b], pred),
                  fit_seconds=seconds, te_seconds=built['te_seconds'], ros_seconds=built['ros_seconds'],
                  warnings=fit_warnings, age_median=built['state']['age'].median_ if built['state']['age'] else None,
                  reload_probability_max_abs_diff=float(np.max(np.abs(p-p2))), csv_round_trip_exact=True,
                  feature_importances=model.feature_importances_.tolist(), finished_at_utc=now())
    write(d/f'fold_{f}.json', record)
    files = [model_path, state_path, preds, d/f'fold_{f}.json']
    shas = {p.name: sha256_file(p) for p in files}
    backup = out/'backup'/entry['dir']
    backup.mkdir(parents=True, exist_ok=True)
    for path in files:
        require(not (backup/path.name).exists(), 'Backup exists')
        shutil.copy2(path, backup/path.name)
    entry.update(status='complete', artifact_sha256=shas, macro_f1=record['metrics']['macro_f1'],
                 fit_seconds=seconds, finished_at_utc=now())
    check_saved(out, entry)
    ledger['folds'][k] = entry
    atomic_write(out/'ledger.json', ledger)
    event(out, fold=k, action='fit_complete', artifact_sha256=shas)
    log(out, f'{k} F1={entry["macro_f1"]:.9f}, fit={seconds:.1f}s; 저장·SHA·재로딩·백업 검증 완료')
    return entry, True


def fit_all(data, output, retry_interrupted=False, retry_reason=None):
    out = Path(output)
    # Exclusive creation prevents two executors entering the same study.
    with (out/'RUNNING.lock').open('x', encoding='utf-8') as handle:
        handle.write(f'pid={os.getpid()} started={now()}\n')
    try:
        X, y, ids, folds, design = verify(data, out)
        F = t14.original_frame(X)
        for s, arm, f in targets():
            try:
                fit_fold(X, F, y, ids, folds, out, s, arm, f, design, retry_interrupted, retry_reason)
            except Exception:
                event(out, fold=key(s, arm, f), action='stopped', trace=traceback.format_exc())
                raise
    finally:
        (out/'RUNNING.lock').unlink()


def judge(deltas, split_means, recall_delta, rule):
    parts = dict(mean_delta_ok=bool(np.mean(deltas) >= rule['mean_delta_min']),
                 fold_wins_ok=sum(d > 0 for d in deltas) >= rule['fold_wins_min'],
                 split_wins_ok=sum(d > 0 for d in split_means) >= rule['positive_splits_min'],
                 grade3_recall_ok=bool(recall_delta >= rule['grade3_recall_delta_min']))
    return dict(label='SUPPORTED_CONTRAST' if len(deltas) == 20 and len(split_means) == 4 and all(parts.values())
                else 'NOT_SUPPORTED', parts=parts)


def assemble(out, ids, y, folds, scheme, arm):
    ledger = read_ledger(out)
    for f in range(5):
        entry = ledger['folds'].get(key(scheme, arm, f))
        require(entry and entry['status'] == 'complete', 'Incomplete fold')
        check_saved(out, entry)
    d = out/scheme/arm
    frame = pd.concat([read_pred(d/f'fold_{f}_pred.csv.gz') for f in range(5)])
    require(frame[ID].is_unique and len(frame) == len(ids), 'OOF coverage mismatch')
    frame = frame.set_index(ID).loc[ids.to_numpy()].reset_index()
    np.testing.assert_array_equal(frame.fold, folds[scheme])
    np.testing.assert_array_equal(frame.y_true, y)
    path = d/'oof.csv.gz'
    if path.exists():
        saved = read_pred(path)
        pd.testing.assert_frame_equal(saved, frame, check_dtype=False)
    else:
        write_pred(path, frame)
    back = out/'backup'/scheme/arm/'oof.csv.gz'
    if not back.exists():
        shutil.copy2(path, back)
    require(sha256_file(back) == sha256_file(path), 'OOF backup mismatch')
    return frame


def finalize(data, output):
    out = Path(output)
    require(not (out/'analysis.json').exists(), 'Analysis exists')
    X, y, ids, folds, design = verify(data, out)
    c = design['config']
    ledger = read_ledger(out)
    result = dict(study_id=STUDY_ID, design_sha256=sha256_file(out/'design.json'), code_commit=design['code_commit'],
                  holdout_accessed=False, baseline_refits=0, fits=N_TARGETS,
                  attempts=sum(e['attempts'] for e in ledger['folds'].values()), per_arm={},
                  note='조건부 단일 요인 비교. 4분할의 반복 행은 독립 표본이 아니며 유의성·인과성·최종 후보 확정 없음.')
    for arm in ARMS:
        deltas, means, candidate, baseline, per = [], [], [], [], {}
        for s in SCHEMES:
            new = assemble(out, ids, y, folds, s, arm)
            ref = c['baseline_oofs'][s]
            old = t14.reference(ref['path'], ref['sha256'], ids, y, folds[s])
            comp = t14.compare_arms(new, old, y)
            d = comp['fold_delta']
            deltas += d
            means.append(float(np.mean(d)))
            candidate.append(new)
            baseline.append(old)
            per[s] = dict(mean_delta=float(np.mean(d)), comparison=comp,
                          pooled=t14.pooled_scores([new]), baseline_pooled=t14.pooled_scores([old]),
                          oof_sha256=sha256_file(out/s/arm/'oof.csv.gz'))
        new_pool, old_pool = t14.pooled_scores(candidate), t14.pooled_scores(baseline)
        recall_delta = new_pool['grade3_recall']-old_pool['grade3_recall']
        result['per_arm'][arm] = dict(fold_delta=deltas, mean_delta=float(np.mean(deltas)),
            fold_wins=sum(d > 0 for d in deltas), split_mean_delta=means, per_scheme=per,
            pooled=new_pool, baseline_pooled=old_pool, grade3_recall_delta=recall_delta,
            judgment=judge(deltas, means, recall_delta, c['rule']))
    write(out/'analysis.json', result)
    shutil.copy2(out/'analysis.json', out/'backup'/'analysis.json')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['preflight', 'prepare', 'fit-all', 'finalize', 'status'])
    p.add_argument('--data', default='data')
    p.add_argument('--output')
    p.add_argument('--destination')
    p.add_argument('--code-commit')
    p.add_argument('--preflight')
    p.add_argument('--retry-interrupted', action='store_true')
    p.add_argument('--retry-reason')
    a = p.parse_args()
    if a.action == 'preflight':
        preflight(a.data, a.destination)
    elif a.action == 'prepare':
        prepare(a.data, a.output, a.code_commit, a.preflight)
    elif a.action == 'fit-all':
        fit_all(a.data, a.output, a.retry_interrupted, a.retry_reason)
    elif a.action == 'finalize':
        finalize(a.data, a.output)
    else:
        ledger = read_ledger(a.output)
        print(json.dumps(dict(complete=sum(e['status'] == 'complete' for e in ledger['folds'].values()), total=N_TARGETS,
                              attempts=sum(e['attempts'] for e in ledger['folds'].values())), indent=1))


if __name__ == '__main__':
    main()
