"""Reproduce development-only EDA and strictly validate private full OOF files.

No fitting, holdout access or row-level public outputs. Nonempty output is refused.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from preprocess import ID, TARGET, load_development, sha256_file, write_json

AGE_LABELS = ['0', '1-10', '11-20', '21-30', '31-50', '51-100', '101-994', '995']
PROBABILITY_COLUMNS = ['p_grade_1', 'p_grade_2', 'p_grade_3']


def age_groups(age):
    values = pd.to_numeric(age, errors='raise').to_numpy(dtype=float)
    if (not np.isfinite(values).all() or (values != np.floor(values)).any()
            or (values < 0).any() or (values > 995).any()):
        raise ValueError('Expected integer age in [0, 995]; do not silently bin invalid rows.')
    return pd.Series(pd.cut(values, [-1, 0, 10, 20, 30, 50, 100, 994, 995],
                            labels=AGE_LABELS, right=True), index=age.index)


def development_frame(data_dir):
    contract = json.loads((ROOT / 'configs/data_contract.json').read_text())
    for relative, digest in contract['files'].items():
        if sha256_file(Path(data_dir) / relative) != digest:
            raise ValueError(f'Data contract mismatch: {relative}')
    X, y, ids, cv = load_development(data_dir)
    if len(X) != contract['expected_rows']:
        raise ValueError('Unexpected development row count.')
    frame = X.copy()
    frame[ID], frame[TARGET], frame['fold'] = ids, y, cv.test_fold
    return frame, contract


def eda_summary(frame):
    frame = frame.copy()
    frame['age_group'] = age_groups(frame.age)
    rows = []
    groups = ['foundation_type', 'roof_type', 'geo_level_1_id', 'age_group',
              'has_superstructure_mud_mortar_stone', 'has_superstructure_rc_engineered']
    for feature in groups:
        total = 0
        for value, group in frame.groupby(feature, observed=True, sort=True):
            count = int(group[TARGET].eq(3).sum())
            rows.append(dict(feature=feature, category=str(value), n=len(group),
                             grade3_count=count, grade3_rate=count / len(group)))
            total += len(group)
        if total != len(frame):
            raise AssertionError(f'Incomplete group partition: {feature}')
    for region in ['17', '26']:
        for foundation in ['r', 'i']:
            group = frame.loc[frame.geo_level_1_id.eq(region) & frame.foundation_type.eq(foundation)]
            count = int(group[TARGET].eq(3).sum())
            rows.append(dict(feature='region_foundation', category=f'{region}/{foundation}',
                             n=len(group), grade3_count=count,
                             grade3_rate=count / len(group) if len(group) else None))
    return pd.DataFrame(rows)


def confusion_stats(true, pred):
    true, pred = np.asarray(true), np.asarray(pred)
    if true.ndim != 1 or pred.shape != true.shape or not len(true):
        raise ValueError('Aligned nonempty labels required.')
    if not np.isin(true, [1, 2, 3]).all() or not np.isin(pred, [1, 2, 3]).all():
        raise ValueError('Invalid label.')
    matrix = np.zeros((3, 3), dtype=np.int64)
    np.add.at(matrix, (true.astype(int) - 1, pred.astype(int) - 1), 1)
    support, total, diagonal = matrix.sum(1), matrix.sum(0), matrix.diagonal()
    precision = np.divide(diagonal, total, out=np.zeros(3), where=total != 0)
    recall = np.divide(diagonal, support, out=np.zeros(3), where=support != 0)
    f1 = np.divide(2 * diagonal, support + total, out=np.zeros(3), where=(support + total) != 0)
    result = dict(n=len(true), macro_f1=float(f1.mean()), accuracy=float(diagonal.sum()/len(true)),
                  grade3_count=int(support[2]), error_3_to_1=int(matrix[2, 0]),
                  error_3_to_2=int(matrix[2, 1]), matrix=matrix.tolist())
    for c in range(3):
        for name, values in [('precision', precision), ('recall', recall), ('f1', f1)]:
            result[f'grade{c+1}_{name}'] = float(values[c])
    for label in ['1', '2']:
        result[f'rate_3_to_{label}'] = result[f'error_3_to_{label}'] / support[2] if support[2] else None
    return result


def validate_oof(frame, oof):
    required = [ID, 'fold', 'y_true', 'y_pred', *PROBABILITY_COLUMNS]
    if len(oof) != len(frame) or oof[ID].duplicated().any() or oof[required].isna().any().any():
        raise ValueError('OOF must have one nonmissing prediction per development building.')
    expected = frame[[ID, TARGET, 'fold', 'geo_level_1_id']]
    joined = expected.merge(oof[required], on=ID, how='outer', validate='one_to_one',
                            suffixes=('_expected', ''), indicator=True)
    if not joined['_merge'].eq('both').all():
        raise ValueError('OOF building IDs do not match development.')
    if not np.array_equal(joined['fold_expected'], joined['fold']):
        raise ValueError('OOF fold differs from frozen manifest.')
    if not np.array_equal(joined[TARGET], joined['y_true']):
        raise ValueError('OOF labels differ from development.')
    p = joined[PROBABILITY_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError('Probabilities must be finite and in [0,1].')
    np.testing.assert_allclose(p.sum(axis=1), 1, atol=1e-6, rtol=0)
    if not np.array_equal(np.argmax(p, axis=1) + 1, joined.y_pred):
        raise ValueError('Probability columns/predicted labels disagree (class order 1,2,3).')
    confusion_stats(joined.y_true, joined.y_pred)
    return joined


def verify_metrics(joined, saved):
    if saved['status'] != 'complete' or saved['rows'] != len(joined) or saved['folds'] != 5:
        raise ValueError('Only completed full five-fold results accepted.')
    if saved['holdout_scored']:
        raise ValueError('Holdout scores must not enter development comparison.')
    records = saved['fold_results']
    if [r['fold'] for r in records] != list(range(5)):
        raise ValueError('Invalid fold record coverage.')
    scores = []
    for f, record in enumerate(records):
        part = joined.loc[joined.fold.eq(f)]
        if record['validation_rows'] != len(part) or record['train_rows'] != len(joined)-len(part):
            raise ValueError('Fold row counts do not match.')
        stats = confusion_stats(part.y_true, part.y_pred)
        np.testing.assert_array_equal(stats['matrix'], record['validation']['confusion_matrix'])
        np.testing.assert_allclose(stats['macro_f1'], record['validation']['macro_f1'], atol=1e-12, rtol=0)
        scores.append(stats['macro_f1'])
    whole = confusion_stats(joined.y_true, joined.y_pred)
    np.testing.assert_array_equal(whole['matrix'], saved['oof']['confusion_matrix'])
    for actual, expected in [(np.mean(scores), saved['cv_macro_f1_mean']),
                              (np.std(scores, ddof=1), saved['cv_macro_f1_std']),
                              (whole['macro_f1'], saved['oof']['macro_f1'])]:
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
    return whole


def run(data_dir, output_dir, runs=(), source=None):
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a new audit output directory; historical results are immutable.')
    frame, contract = development_frame(data_dir)
    if source is not None and sha256_file(source) != contract['source_sha256']:
        raise ValueError('Raw CSV SHA does not match the existing source.')
    errors, sources, seen = [], [], set()
    for directory in map(Path, runs):
        meta = json.loads((directory/'run_metadata.json').read_text())
        if (meta['status'] != 'complete' or meta['scope'] != 'full_development_5fold'
                or meta['holdout_scored'] or meta['rows'] != len(frame)
                or meta['data_contract'] != contract):
            raise ValueError('Require completed full-development run with this exact contract.')
        manifest = pd.read_csv(directory/'run_manifest.csv').sort_values(ID).reset_index(drop=True)
        expected = frame[[ID, 'fold']].rename(columns={'fold':'cv_fold'}).sort_values(ID).reset_index(drop=True)
        pd.testing.assert_frame_equal(manifest, expected, check_dtype=False)
        for model in meta['models']:
            experiment = meta['config']['name']
            if (experiment, model) in seen:
                raise ValueError('Duplicate experiment/model.')
            seen.add((experiment, model))
            path = directory/model
            oof_path = path/'oof_predictions.csv.gz'
            saved = json.loads((path/'metrics.json').read_text())
            if saved.get('oof_sha256', sha256_file(oof_path)) != sha256_file(oof_path):
                raise ValueError('OOF checksum mismatch.')
            joined = validate_oof(frame, pd.read_csv(oof_path))
            verify_metrics(joined, saved)
            groups = [('all', joined), *[(str(k), v) for k,v in joined.groupby('geo_level_1_id')]]
            for region, group in groups:
                stats = confusion_stats(group.y_true, group.y_pred)
                stats.pop('matrix')
                errors.append(dict(experiment=experiment, model=model, region=region, **stats))
            sources.append(dict(experiment=experiment, model=model,
                                oof_sha256=sha256_file(oof_path), metrics_sha256=sha256_file(path/'metrics.json'),
                                metadata_sha256=sha256_file(directory/'run_metadata.json')))
    output.mkdir(parents=True, exist_ok=True)
    eda_summary(frame).to_csv(output/'eda_summary.csv', index=False, lineterminator='\n')
    if errors:
        pd.DataFrame(errors).to_csv(output/'oof_error_summary.csv', index=False, lineterminator='\n')
    receipt = dict(eda_status='PASS', oof_status='PASS' if sources else 'NOT_PROVIDED',
                   data_contract=contract, rows=len(frame), source_sha_verified=source is not None,
                   models_checked=len(sources), holdout_opened=False, sources=sources,
                   audit_code_sha256=sha256_file(__file__),
                   limitations=['Does not independently refit models or recover absent historical OOF.'])
    write_json(output/'eda_oof_validation.json', receipt)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='data')
    parser.add_argument('--output', required=True)
    parser.add_argument('--runs', nargs='*', default=[])
    parser.add_argument('--source', help='Optional original CSV: SHA only, never used for EDA.')
    args = parser.parse_args()
    print(json.dumps(run(args.data, args.output, args.runs, args.source), indent=2))
