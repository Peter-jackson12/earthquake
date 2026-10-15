"""Read-only T6 recovery gate and C-baseline re-audit; NEVER fits a classifier.

Added after the interrupted T6 run. This is not part of the original pre-fit
seal and does not change run.py/audit.py/config.json or any previous evidence.
An absent T6 directory is UNKNOWN historical work, not permission to retrain.
Only checksum-verified, trusted project-generated joblib files may be loaded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

from experiments.geo_ablation.run import config, environment
from experiments.geo_group_cv.run import load
from experiments.geo_group_cv.audit import inspect_preprocessor

PROBS = [f'p_grade_{n}' for n in (1, 2, 3)]
SCHEMES = ('stratified', 'geo2', 'geo1')
ROOT = Path(__file__).resolve().parents[2]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path: Path | str, value: Any, protected: tuple[Path, ...] = ()) -> None:
    target = Path(path).resolve()
    require(not any(target == p.resolve() or p.resolve() in target.parents for p in protected),
            'Recovery output may not be written inside protected evidence')
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x', encoding='utf-8') as stream:
        stream.write(text)


def arithmetic(y: Any, prediction: Any) -> dict[str, Any]:
    """Independent confusion-matrix arithmetic; no sklearn/training metrics."""
    truth, pred = np.asarray(y), np.asarray(prediction)
    require(truth.ndim == pred.ndim == 1 and len(truth) == len(pred) and len(truth) > 0,
            'Metric inputs must be nonempty aligned one-dimensional arrays')
    require(bool(np.isin(truth, [1, 2, 3]).all() and np.isin(pred, [1, 2, 3]).all()),
            'Labels must be exactly 1/2/3')
    cm = np.zeros((3, 3), dtype=np.int64)
    np.add.at(cm, (truth.astype(int) - 1, pred.astype(int) - 1), 1)
    support, predicted, tp = cm.sum(1), cm.sum(0), cm.diagonal()
    precision = np.divide(tp, predicted, out=np.zeros(3), where=predicted != 0)
    recall = np.divide(tp, support, out=np.zeros(3), where=support != 0)
    f1 = np.divide(2 * tp, support + predicted, out=np.zeros(3), where=(support + predicted) != 0)
    normalized = np.divide(cm, support[:, None], out=np.zeros((3, 3)), where=support[:, None] != 0)
    return dict(n=len(truth), macro_f1=float(f1.mean()), accuracy=float(tp.sum() / len(truth)),
                precision=precision.tolist(), recall=recall.tolist(), f1=f1.tolist(), support=support.tolist(),
                confusion=cm.tolist(), normalized_confusion=normalized.tolist(),
                error31=int(cm[2, 0]), error32=int(cm[2, 1]),
                rate31=float(cm[2, 0] / support[2]) if support[2] else None,
                rate32=float(cm[2, 1] / support[2]) if support[2] else None)


def check_oof(frame: pd.DataFrame, ids: Any, y: Any, folds: Any) -> pd.DataFrame:
    require(set(['building_id', 'fold', 'y_true', 'y_pred'] + PROBS) <= set(frame.columns),
            'Required OOF columns missing')
    require(len(frame) == len(ids) and frame.building_id.is_unique, 'Duplicate or missing OOF IDs')
    ordered = frame.sort_values('building_id', kind='stable').reset_index(drop=True)
    for column, expected in [('building_id', ids), ('y_true', y), ('fold', folds)]:
        np.testing.assert_array_equal(ordered[column], expected, err_msg=column + ' mismatch')
    p = ordered[PROBS].to_numpy()
    require(bool(np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()), 'Invalid OOF probabilities')
    np.testing.assert_allclose(p.sum(1), 1, atol=1e-12, rtol=0)
    np.testing.assert_array_equal(ordered.y_pred, p.argmax(1) + 1)
    return ordered


def inventory(root: Path | str) -> dict[str, Any]:
    """Presence only. Neither artifact presence nor absence proves completed fits."""
    root = Path(root)
    original = root / 'outputs/geo_ablation_v1'
    supplement = root / 'outputs/geo_ablation_unattempted_v1'
    required = [original / 'design.json', original / 'design_seal.json',
                original / 'audit_available/audit.json', supplement / 'audit.json']
    found_models = sum(p.is_file() for directory in [original, supplement]
                       for p in directory.rglob('fold_*.joblib'))
    return dict(status='RECOVERED_FILES_REQUIRE_FULL_AUDIT' if all(p.is_file() for p in required)
                else 'BLOCKED_T6_PRIVATE_EVIDENCE_MISSING',
                missing=[str(p.relative_to(root)) for p in required if not p.is_file()],
                model_files_present=found_models, t6_completed_fits_verified=None,
                t6_attempts_verified=None, complete_study=False, training_allowed=False,
                reason='Do not treat missing artifacts as unattempted fits. Recover original and supplement evidence first.')


def audit_c(data: Path, t5: Path, baseline: Path, scheme: str) -> dict[str, Any]:
    require(scheme in SCHEMES, 'Unsupported validation scheme')
    cfg = config()
    require(environment() == cfg['environment'], 'Pinned environment mismatch')
    source_before: dict[str, str] = {}
    t5_design = read_json(t5 / 'design.json')
    require(digest(t5 / 'design.json') == read_json(t5 / 'design_seal.json')['design_sha256'], 'T5 seal mismatch')
    old_meta = read_json(baseline / 'age995_abc_v1/C/run_metadata.json')
    for source_map in [t5_design['source_sha256'], old_meta['source_code_sha256']]:
        for name, expected in source_map.items():
            source = (ROOT / name).resolve()
            require(ROOT in source.parents, 'Source path escaped repository')
            require(digest(source) == expected, 'Frozen C/T5 source mismatch: ' + name)
            source_before[name] = expected
    require(old_meta['config']['models']['lr'] == cfg['lr'] and old_meta['config']['age_policy'] == 'C',
            'Historical C configuration mismatch')
    X, y, ids, old_folds, contract = load(data)
    require(old_meta['data_contract'] == t5_design['data_contract'] == contract, 'C/T5/data contract mismatch')
    evidence_paths = [data / n for n in contract['files']]
    evidence_paths += [t5 / 'design.json', t5 / 'design_seal.json', baseline / 'age995_abc_v1/C/run_metadata.json']
    if scheme == 'stratified':
        folds = old_folds
        directory = baseline / 'age995_abc_v1/C/lr'
        oof_path = directory / 'oof_predictions.csv.gz'
        saved = read_json(directory / 'metrics.json')
        records = saved['fold_results']
        mean_key, std_key = 'cv_macro_f1_mean', 'cv_macro_f1_std'
        manifest = data / 'split_manifest.csv'
        split_info = dict(groups=None, group_overlap_not_applicable=True, exact_fixed_assignment=True)
    else:
        manifest = t5 / scheme / 'manifest.csv'
        require(digest(manifest) == cfg['manifest_sha256'][scheme], 'Group manifest SHA mismatch')
        m = pd.read_csv(manifest)
        np.testing.assert_array_equal(m.building_id, ids)
        columns = ['geo_level_1_id', 'geo_level_2_id'] if scheme == 'geo2' else ['geo_level_1_id']
        np.testing.assert_array_equal(m[columns].astype('int64'), X[columns].astype('int64'))
        keys, group = np.unique(X[columns].astype('int64'), axis=0, return_inverse=True)
        folds = m.fold.to_numpy()
        require(len(keys) == (1408 if scheme == 'geo2' else 31), 'Wrong group count')
        require(bool(pd.DataFrame({'g': group, 'f': folds}).groupby('g').f.nunique().eq(1).all()), 'Group leakage')
        reconstructed = np.full(len(X), -1, dtype=int)
        for f, (_, valid) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(X, y, group)):
            reconstructed[valid] = f
        np.testing.assert_array_equal(folds, reconstructed)
        split_info = dict(groups=len(keys), train_validation_group_overlap=0, exact_reconstruction=True)
        directory = t5 / scheme / 'C'
        oof_path = directory / 'oof.csv.gz'
        saved = read_json(directory / 'metrics.json')
        records = read_json(directory / 'folds.json')
        metadata = read_json(directory / 'metadata.json')
        require(metadata['status'] == 'complete' and metadata['completed_fits'] == metadata['attempted_fits'] == 5,
                'Incomplete historical C run')
        require(metadata['source_sha256'] == t5_design['source_sha256'], 'C model source provenance mismatch')
        for name, expected in metadata['artifact_sha256'].items():
            require(Path(name).name == name, 'Unexpected artifact path')
            require(digest(directory / name) == expected, 'Historical C artifact changed: ' + name)
        mean_key, std_key = 'cv_mean', 'cv_std'
    require(digest(manifest) == cfg['manifest_sha256'][scheme], 'Manifest SHA mismatch')
    require(digest(oof_path) == cfg['C_oof_sha256'][scheme], 'C OOF SHA mismatch')
    evidence_paths += [manifest] + [p for p in directory.iterdir() if p.is_file()]
    evidence_before = {str(p): digest(p) for p in evidence_paths}
    frame = check_oof(pd.read_csv(oof_path, float_precision='round_trip'), ids, y, folds)
    overall = arithmetic(y, frame.y_pred)
    fold_scores, reload_checks = [], []
    require({r['fold'] for r in records} == set(range(5)), 'Missing historical C fold records')
    for f in range(5):
        valid, train = np.flatnonzero(folds == f), np.flatnonzero(folds != f)
        metric = arithmetic(y.iloc[valid], frame.y_pred.iloc[valid])
        record = next(r for r in records if r['fold'] == f)
        original_metric = record['validation'] if scheme == 'stratified' else record['metrics']
        for key in ['macro_f1', 'accuracy']:
            np.testing.assert_allclose(metric[key], original_metric[key], atol=1e-12, rtol=0)
        np.testing.assert_array_equal(metric['confusion'], original_metric['confusion_matrix' if scheme == 'stratified' else 'confusion'])
        fold_scores.append(dict(fold=f, **metric))
        if scheme != 'stratified':
            model_path = directory / f'fold_{f}.joblib'
            require(digest(model_path) == record['model_sha256'], 'C model SHA mismatch')
            with threadpool_limits(limits=1):
                # Trusted original project package; hash checked before deserialization.
                pipe = joblib.load(model_path)
                require(all(pipe['model'].get_params()[k] == v for k, v in cfg['lr'].items()), 'C LR parameters changed')
                require(pipe['model'].class_weight is None, 'Unexpected C class weight')
                np.testing.assert_array_equal(pipe['model'].classes_, [1, 2, 3])
                np.testing.assert_array_equal(pipe.predict(X.iloc[valid]), frame.y_pred.iloc[valid])
                np.testing.assert_allclose(pipe.predict_proba(X.iloc[valid]), frame.iloc[valid][PROBS], atol=1e-12, rtol=0)
                info, _ = inspect_preprocessor(pipe, 'C', X.iloc[train], X.iloc[valid])
            reload_checks.append(dict(fold=f, full_validation_prediction_match=True,
                                      train_only_statistics_match=True, median=info['median'],
                                      unknown_categories=info['categorical'], model_sha256=digest(model_path)))
    values = [entry['macro_f1'] for entry in fold_scores]
    mean, sd = float(np.mean(values)), float(np.std(values, ddof=1))
    np.testing.assert_allclose([mean, sd], [saved[mean_key], saved[std_key]], atol=1e-12, rtol=0)
    saved_overall = saved['oof'] if scheme == 'stratified' else saved['pooled']
    np.testing.assert_allclose(overall['macro_f1'], saved_overall['macro_f1'], atol=1e-12, rtol=0)
    np.testing.assert_array_equal(overall['confusion'], saved_overall['confusion_matrix' if scheme == 'stratified' else 'confusion'])
    groups = {}
    masks = {'age995': X.age.eq(995), 'non995': X.age.ne(995)}
    masks.update({f'geo1_{int(region)}': X.geo_level_1_id.eq(region) for region in X.geo_level_1_id.unique()})
    for name, mask in masks.items():
        groups[name] = arithmetic(y[mask], frame.y_pred[mask])
    require(all(digest(ROOT / p) == expected for p, expected in source_before.items()), 'Source changed during recovery audit')
    require(all(digest(p) == expected for p, expected in evidence_before.items()), 'Evidence changed during recovery audit')
    return dict(status='PASS_C_BASELINE_REAUDIT_ONLY', scheme=scheme, variant='C',
                audited_at_utc=datetime.now(timezone.utc).isoformat(), code_sha256=digest(__file__),
                environment=environment(), data_sha256=contract['files'], source_sha256=source_before,
                manifest_sha256=digest(manifest), oof_sha256=digest(oof_path),
                rows=len(X), cv_mean=mean, cv_std_ddof1=sd, pooled=overall,
                folds=fold_scores, groups=groups, split_checks=split_info, reloaded_models=reload_checks,
                unavailable_historical_stratified_models=5 if scheme == 'stratified' else 0,
                unavailable_model_note='Historical Stratified C binaries were never included; no replacement training.' if scheme == 'stratified' else None,
                classifiers_fitted=0, T6_results_verified=False, holdout_accessed=False,
                source_and_evidence_unchanged=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'data')
    parser.add_argument('--t5', type=Path, default=ROOT / 'outputs/geo_group_cv_v1')
    parser.add_argument('--baseline', type=Path, default=ROOT / 'baseline_private')
    parser.add_argument('--scheme', choices=SCHEMES, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    result = audit_c(args.data, args.t5, args.baseline, args.scheme)
    result['T6_recovery'] = inventory(ROOT)
    protected = (args.data, args.t5, args.baseline, ROOT / 'outputs/geo_ablation_v1',
                 ROOT / 'outputs/geo_ablation_unattempted_v1')
    write_new(args.destination, result, protected)
    print(f'{args.scheme}: C baseline PASS; mean={result["cv_mean"]:.9f}; '
          f'reloaded={len(result["reloaded_models"])}; new classifier fits=0; '
          f'T6={result["T6_recovery"]["status"]}', flush=True)


if __name__ == '__main__':
    main()
