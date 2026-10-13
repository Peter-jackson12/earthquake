"""Verify completed full-development OOF files and export aggregates only.

Run after baseline.py. This does not fit a model or open holdout files.
F1 is recomputed directly from confusion counts, independently of baseline.metrics.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from preprocess import ID, load_development, sha256_file, write_json


def from_counts(true, predicted):
    true, predicted = np.asarray(true), np.asarray(predicted)
    if true.shape != predicted.shape or true.ndim != 1 or not len(true):
        raise ValueError('Aligned nonempty one-dimensional labels required.')
    if not np.isin(true, [1, 2, 3]).all() or not np.isin(predicted, [1, 2, 3]).all():
        raise ValueError('Invalid label.')
    matrix = np.zeros((3, 3), dtype=np.int64)
    np.add.at(matrix, (true.astype(int) - 1, predicted.astype(int) - 1), 1)
    support, totals, diagonal = matrix.sum(axis=1), matrix.sum(axis=0), matrix.diagonal()
    f1 = np.divide(2 * diagonal, support + totals, out=np.zeros(3), where=(support + totals) != 0)
    recall = np.divide(diagonal, support, out=np.zeros(3), where=support != 0)
    precision = np.divide(diagonal, totals, out=np.zeros(3), where=totals != 0)
    return dict(matrix=matrix, f1=f1, recall=recall, precision=precision,
                macro_f1=float(f1.mean()), accuracy=float(diagonal.sum() / matrix.sum()),
                severe_as_minor_rate=float(matrix[2, 0] / support[2]))


def audit(data_dir, runs, output):
    _, y, ids, cv = load_development(data_dir)
    expected = pd.DataFrame({ID: ids, 'fold': cv.test_fold, 'y_true': y}).sort_values(ID).reset_index(drop=True)
    summaries, fold_rows, sources = [], [], []
    seen = set()
    for directory in map(Path, runs):
        meta = json.loads((directory / 'run_metadata.json').read_text())
        if meta['status'] != 'complete' or meta['scope'] != 'full_development_5fold':
            raise ValueError('Only completed full-development runs may enter this comparison.')
        if meta['holdout_scored'] or meta['rows'] != len(ids):
            raise ValueError('Invalid evaluation scope.')
        for relative, digest in meta['data_contract']['files'].items():
            if sha256_file(Path(data_dir) / relative) != digest:
                raise ValueError('Input file differs from run metadata.')
        experiment = meta['config']['name']
        for model in meta['models']:
            if (experiment, model) in seen:
                raise ValueError('Duplicate experiment/model.')
            seen.add((experiment, model))
            path = directory / model
            result = json.loads((path / 'metrics.json').read_text())
            oof = pd.read_csv(path / 'oof_predictions.csv.gz').sort_values(ID).reset_index(drop=True)
            if oof[ID].duplicated().any() or result['status'] != 'complete':
                raise ValueError('Invalid OOF coverage or model status.')
            pd.testing.assert_frame_equal(oof[[ID, 'fold', 'y_true']], expected, check_dtype=False)
            probabilities = oof[['p_grade_1', 'p_grade_2', 'p_grade_3']].to_numpy()
            if not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any():
                raise ValueError('Invalid probabilities.')
            np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6)
            scores = []
            for f, record in enumerate(result['fold_results']):
                if record['fold'] != f or record['validation_rows'] != int((oof.fold == f).sum()):
                    raise ValueError('Unexpected fold record.')
                part = oof.loc[oof.fold == f]
                checked = from_counts(part.y_true, part.y_pred)
                np.testing.assert_allclose(checked['macro_f1'], record['validation']['macro_f1'], atol=1e-12, rtol=0)
                np.testing.assert_array_equal(checked['matrix'], record['validation']['confusion_matrix'])
                scores.append(checked['macro_f1'])
                fold_rows.append(dict(experiment=experiment, model=model, fold=f,
                                      train_rows=record['train_rows'], validation_rows=record['validation_rows'],
                                      train_macro_f1=record['train_macro_f1'], macro_f1=checked['macro_f1'],
                                      fit_seconds=record['fit_seconds'], warning_count=len(record['warnings'])))
            if len(scores) != 5:
                raise ValueError('Five folds are required.')
            checked = from_counts(oof.y_true, oof.y_pred)
            np.testing.assert_allclose(np.mean(scores), result['cv_macro_f1_mean'], atol=1e-12, rtol=0)
            np.testing.assert_allclose(np.std(scores, ddof=1), result['cv_macro_f1_std'], atol=1e-12, rtol=0)
            np.testing.assert_allclose(checked['macro_f1'], result['oof']['macro_f1'], atol=1e-12, rtol=0)
            np.testing.assert_array_equal(checked['matrix'], result['oof']['confusion_matrix'])
            row = dict(experiment=experiment, model=model, rows=len(oof), folds=5,
                       cv_macro_f1_mean=float(np.mean(scores)), cv_macro_f1_std=float(np.std(scores, ddof=1)),
                       train_macro_f1_mean=float(np.mean([r['train_macro_f1'] for r in result['fold_results']])),
                       oof_macro_f1=checked['macro_f1'], accuracy=checked['accuracy'],
                       severe_as_minor_rate=checked['severe_as_minor_rate'])
            for grade in range(1, 4):
                for metric in ['f1', 'precision', 'recall']:
                    row[f'grade{grade}_{metric}'] = float(checked[metric][grade - 1])
                    np.testing.assert_allclose(row[f'grade{grade}_{metric}'], result['oof']['per_class'][str(grade)]['f1-score' if metric == 'f1' else metric], atol=1e-12, rtol=0)
            summaries.append(row)
            sources.append(dict(experiment=experiment, model=model,
                                metrics_sha256=sha256_file(path / 'metrics.json'),
                                oof_sha256=sha256_file(path / 'oof_predictions.csv.gz'),
                                run_metadata=meta,
                                effective_parameters=result['effective_parameters'],
                                confusion_matrix=checked['matrix'].tolist()))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(output / 'model_summary.csv', index=False, lineterminator='\n')
    pd.DataFrame(fold_rows).to_csv(output / 'fold_summary.csv', index=False, lineterminator='\n')
    write_json(output / 'result_provenance.json', dict(status='PASS', models_checked=len(summaries),
               full_cv_fits_checked=len(fold_rows), rows_per_configuration=len(ids), holdout_scored=False,
               checks=['same IDs, folds and labels as development', 'one OOF prediction per building per configuration',
                       'finite normalized probabilities', 'fold and pooled F1 recomputed from confusion counts',
                       'classwise precision/recall/F1 recomputed', 'data hashes match recorded runs'],
               limitations=['Does not independently refit the models.', 'Training scores and timings are retained from run logs, not independently recomputed.'],
               sources=sources))
    print(pd.DataFrame(summaries)[['experiment', 'model', 'cv_macro_f1_mean', 'cv_macro_f1_std']].to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='data')
    parser.add_argument('--runs', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    audit(args.data, args.runs, args.output)
