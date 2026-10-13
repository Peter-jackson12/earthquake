"""Fixed-fold baselines on development data only. No holdout scoring or tuning."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import platform
import subprocess
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy import sparse
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import PredefinedSplit, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits

from preprocess import (CATEGORICAL, ID, TARGET, load_development,
                        make_model_pipeline, sha256_file, write_json)

ROOT = Path(__file__).resolve().parent
LABELS = [1, 2, 3]
MODEL_NAMES = ['dummy', 'lr', 'dt', 'rf', 'gbc']


def build_pipeline(name: str, config: dict, jobs: int = 2):
    """A fresh unfitted pipeline for each fold; scale only the five LR numerics."""
    constructors = dict(dummy=DummyClassifier, lr=LogisticRegression,
                        dt=DecisionTreeClassifier, rf=RandomForestClassifier,
                        gbc=GradientBoostingClassifier)
    if name not in constructors:
        raise ValueError(f'Unknown model: {name}')
    params = dict(config['models'][name])
    if name == 'rf':
        params['n_jobs'] = jobs
    pipeline = make_model_pipeline(constructors[name](**params), scale_numeric=name == 'lr')
    sampling = config.get('sampling', {}).get(name)
    if sampling is not None:
        if name != 'lr' or sampling != 'random_under':
            raise ValueError('Only LR random_under is supported in this single-factor experiment.')
        if params.get('class_weight') is not None or name in config.get('sample_weight', {}):
            raise ValueError('Do not combine sampling and weighting in a single-factor run.')
        from imblearn.pipeline import Pipeline as SamplingPipeline
        from imblearn.under_sampling import RandomUnderSampler
        # Flatten the unchanged preprocessor: imblearn disallows a nested Pipeline
        # as an intermediate step. Encoding/scaling fit on the original train fold.
        pipeline = SamplingPipeline([
            *pipeline['preprocess'].steps,
            ('sampler', RandomUnderSampler(random_state=config['seed'])),
            ('model', pipeline['model']),
        ])
    return pipeline


def fitted_preprocessor(pipeline):
    if 'preprocess' in pipeline.named_steps:
        return pipeline['preprocess']
    return Pipeline([(key, pipeline[key]) for key in ('schema', 'columns')])


def training_parameters(name, config, y_train):
    """Derive weights only from the supplied training labels, never validation."""
    policy = config.get('sample_weight', {}).get(name)
    if policy is None:
        return {}
    if name != 'gbc' or policy != 'balanced' or config.get('sampling', {}).get(name):
        raise ValueError('Only GBC balanced sample_weight is supported.')
    return {'model__sample_weight': compute_sample_weight('balanced', y_train)}


def class_counts(y):
    return {str(c): int(np.sum(np.asarray(y) == c)) for c in LABELS}


def git_provenance():
    try:
        head = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'],
                                       stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain'],
                                            stderr=subprocess.DEVNULL, text=True).strip())
        return {'head': head, 'dirty': dirty}
    except (OSError, subprocess.CalledProcessError):
        return {'head': None, 'dirty': None, 'note': 'No Git checkout; use source file SHA-256.'}


def select_positions(y: pd.Series, folds: np.ndarray, rows: int, seed: int) -> np.ndarray:
    """Pilot only: fixed stratified subset by original fold AND label, without refolding."""
    if rows == 0:
        return np.arange(len(y))
    if not 75 <= rows < len(y):
        raise ValueError('Pilot rows must be >=75 and smaller than development; use 0 for all.')
    strata = pd.Series(folds).astype(str) + ':' + y.reset_index(drop=True).astype(str)
    positions, _ = train_test_split(np.arange(len(y)), train_size=rows,
                                    stratify=strata, random_state=seed)
    return np.sort(positions)


def metrics(y_true, y_pred) -> dict:
    true, pred = np.asarray(y_true), np.asarray(y_pred)
    if true.ndim != 1 or pred.shape != true.shape or not len(true):
        raise ValueError('Nonempty aligned one-dimensional labels required.')
    if not np.isin(true, LABELS).all() or not np.isin(pred, LABELS).all():
        raise ValueError('Only labels 1, 2 and 3 are allowed.')
    report = classification_report(true, pred, labels=LABELS, output_dict=True, zero_division=0)
    severe = true == 3
    return dict(macro_f1=float(f1_score(true, pred, average='macro', labels=LABELS, zero_division=0)),
                accuracy=float(accuracy_score(true, pred)),
                severe_as_minor_rate=float(np.mean(pred[severe] == 1)) if severe.any() else None,
                per_class=report,
                confusion_matrix=confusion_matrix(true, pred, labels=LABELS).tolist())


def aligned_probabilities(pipeline, X) -> np.ndarray:
    classes = np.asarray(pipeline.named_steps['model'].classes_)
    if set(classes) != set(LABELS):
        raise ValueError('Every training fold must contain classes 1, 2, 3.')
    raw = pipeline.predict_proba(X)
    aligned = raw[:, [int(np.flatnonzero(classes == c)[0]) for c in LABELS]]
    if not np.isfinite(aligned).all() or (aligned < 0).any() or (aligned > 1).any():
        raise ValueError('Invalid class probabilities.')
    np.testing.assert_allclose(aligned.sum(axis=1), 1, atol=1e-6)
    return aligned


def evaluate_model(name, X, y, ids, folds, config, output, jobs=2):
    """OOF prediction exactly once per input row; save aggregate and private outputs."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    predictions = np.full(len(y), -1, dtype=np.int16)
    probabilities = np.full((len(y), 3), np.nan)
    coverage = np.zeros(len(y), dtype=np.int8)
    fold_results = []
    for fold, (train, valid) in enumerate(PredefinedSplit(folds).split()):
        if set(y.iloc[train]) != set(LABELS) or set(y.iloc[valid]) != set(LABELS):
            raise ValueError('A fold is missing a damage grade.')
        if np.intersect1d(ids.iloc[train], ids.iloc[valid]).size:
            raise AssertionError('Training and validation IDs overlap.')
        pipeline = build_pipeline(name, config, jobs)
        fit_params = training_parameters(name, config, y.iloc[train])
        start = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
            warnings.simplefilter('always')
            pipeline.fit(X.iloc[train], y.iloc[train], **fit_params)
        fit_seconds = time.perf_counter() - start
        warning_rows = [{'category': w.category.__name__, 'message': str(w.message)} for w in caught]
        convergence_ok = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
        start = time.perf_counter()
        with threadpool_limits(limits=1):
            pred = pipeline.predict(X.iloc[valid])
            proba = aligned_probabilities(pipeline, X.iloc[valid])
            train_pred = pipeline.predict(X.iloc[train])
        prediction_seconds = time.perf_counter() - start
        predictions[valid], probabilities[valid] = pred, proba
        coverage[valid] += 1
        pre = fitted_preprocessor(pipeline)
        transformed = pre.transform(X.iloc[valid[:8]])
        if not sparse.issparse(transformed):
            raise AssertionError('Sparse preprocessing required.')
        vocabulary = pre['columns'].named_transformers_['categorical'].categories_
        for col, learned in zip(CATEGORICAL, vocabulary):
            if set(learned) != set(X.iloc[train][col]):
                raise AssertionError('Encoder vocabulary is not training-fold-only.')
        train_after = y.iloc[train]
        sampling_record = None
        if 'sampler' in pipeline.named_steps:
            kept = pipeline['sampler'].sample_indices_
            if len(np.unique(kept)) != len(kept):
                raise AssertionError('RUS without replacement must not duplicate rows.')
            train_after = y.iloc[train].iloc[kept]
            sampling_record = dict(method='random_under',
                                   order='schema -> encoding/scaling -> RUS -> LR',
                                   before=class_counts(y.iloc[train]), after=class_counts(train_after),
                                   retained_train_ids_sha256=hashlib.sha256(
                                       np.asarray(ids.iloc[train].iloc[kept], dtype='<i8').tobytes()).hexdigest())
        weights = fit_params.get('model__sample_weight')
        if weights is None and config['models'][name].get('class_weight') == 'balanced':
            weights = compute_sample_weight('balanced', y.iloc[train])
        weight_record = None if weights is None else {
            str(c): float(weights[np.asarray(y.iloc[train]) == c][0]) for c in LABELS}
        restored = False
        if fold == 0:
            # Trusted locally generated model only; model files are never published.
            model_file = output / 'fold0_model.joblib'
            joblib.dump(pipeline, model_file, compress=3)
            loaded = joblib.load(model_file)
            np.testing.assert_array_equal(pred[:32], loaded.predict(X.iloc[valid[:32]]))
            np.testing.assert_allclose(proba[:32], aligned_probabilities(loaded, X.iloc[valid[:32]]))
            del loaded
            restored = True
        row = dict(fold=fold, train_rows=len(train), validation_rows=len(valid),
                   encoded_columns=int(transformed.shape[1]), fit_seconds=fit_seconds,
                   prediction_seconds=prediction_seconds, train_macro_f1=metrics(y.iloc[train], train_pred)['macro_f1'],
                   validation=metrics(y.iloc[valid], pred), warnings=warning_rows,
                   convergence_ok=convergence_ok, train_only_vocabulary=True,
                   reload_prediction_checked=restored,
                   training_class_counts=class_counts(y.iloc[train]),
                   post_sampling_class_counts=class_counts(train_after),
                   class_weights=weight_record, sampling=sampling_record,
                   prediction_seconds_definition='validation predict + predict_proba + full train predict')
        if hasattr(pipeline['model'], 'n_iter_'):
            row['n_iter'] = np.asarray(pipeline['model'].n_iter_).tolist()
        fold_results.append(row)
        write_json(output / 'fold_metrics.json', fold_results)
        print(f'{name} fold {fold}: Macro F1={row["validation"]["macro_f1"]:.5f}, '
              f'fit={fit_seconds:.1f}s, converged={convergence_ok}', flush=True)
        del pipeline, transformed
        gc.collect()
    np.testing.assert_array_equal(coverage, np.ones(len(y)))
    if len(set(ids)) != len(ids) or not np.isfinite(probabilities).all():
        raise AssertionError('Invalid OOF coverage or probabilities.')
    scores = np.asarray([r['validation']['macro_f1'] for r in fold_results])
    result = dict(model=name, rows=len(y), folds=5, status='complete',
                  cv_macro_f1_mean=float(scores.mean()), cv_macro_f1_std=float(scores.std(ddof=1)),
                  std_definition='sample standard deviation across 5 folds; not a confidence interval',
                  total_fit_seconds=float(sum(r['fit_seconds'] for r in fold_results)),
                  all_folds_converged=all(r['convergence_ok'] for r in fold_results),
                  oof=metrics(y, predictions), fold_results=fold_results,
                  exactly_one_oof_per_row=True, holdout_scored=False,
                  effective_parameters=build_pipeline(name, config, jobs)['model'].get_params())
    oof = pd.DataFrame({ID: np.asarray(ids), 'fold': folds, 'y_true': np.asarray(y),
                        'y_pred': predictions, 'p_grade_1': probabilities[:, 0],
                        'p_grade_2': probabilities[:, 1], 'p_grade_3': probabilities[:, 2]})
    oof.to_csv(output / 'oof_predictions.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
    result['oof_sha256'] = sha256_file(output / 'oof_predictions.csv.gz')
    write_json(output / 'metrics.json', result)
    return result


def run(data_dir, output_dir, config_path, rows=0, models=None, jobs=2):
    if jobs < 1:
        raise ValueError('jobs must be positive; use a bounded value such as 2.')
    models = MODEL_NAMES if models is None else models
    if not models or len(set(models)) != len(models) or set(models) - set(MODEL_NAMES):
        raise ValueError('Choose unique valid model names.')
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Output directory is nonempty. Use a new run directory; no silent overwrite.')
    contract = json.loads((ROOT / 'configs/data_contract.json').read_text(encoding='utf-8'))
    data_dir = Path(data_dir)
    for relative, expected in contract['files'].items():
        if sha256_file(data_dir / relative) != expected:
            raise ValueError(f'Data contract mismatch: {relative}')
    X, y, ids, cv = load_development(data_dir)
    if len(X) != contract['expected_rows']:
        raise ValueError('Development row count differs from the approved data contract.')
    config = json.loads(Path(config_path).read_text(encoding='utf-8'))
    positions = select_positions(y, cv.test_fold, rows, config['seed'])
    X, y, ids = (v.iloc[positions].reset_index(drop=True) for v in (X, y, ids))
    folds = cv.test_fold[positions]
    if set(folds) != set(range(5)):
        raise ValueError('All five frozen fold IDs must be present.')
    scope = 'full_development_5fold' if rows == 0 else 'pilot_subset_5fold'
    output.mkdir(parents=True, exist_ok=True)
    metadata = dict(started_at_utc=datetime.now(timezone.utc).isoformat(), scope=scope,
                    rows=len(X), feature_count=X.shape[1], models=models, jobs=jobs,
                    config=config, data_contract=contract, holdout_scored=False,
                    source_code_sha256={p: sha256_file(ROOT / p) for p in ['preprocess.py', 'baseline.py']},
                    config_sha256=sha256_file(config_path), git=git_provenance(),
                    sample_ids_sha256=hashlib.sha256(np.asarray(ids, dtype='<i8').tobytes()).hexdigest(),
                    environment=dict(python=platform.python_version(), numpy=np.__version__,
                                     pandas=pd.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__),
                    status='running')
    if config.get('sampling'):
        import imblearn
        metadata['environment']['imbalanced_learn'] = imblearn.__version__
    write_json(output / 'run_metadata.json', metadata)
    pd.DataFrame({ID: ids, 'cv_fold': folds}).to_csv(output / 'run_manifest.csv', index=False)
    results = []
    try:
        for name in models:
            results.append(evaluate_model(name, X, y, ids, folds, config, output / name, jobs))
            write_json(output / 'results.json', {'scope': scope, 'results': results})
        metadata['status'] = 'complete'
    except BaseException as exc:
        metadata.update(status='incomplete', error=type(exc).__name__ + ': ' + str(exc))
        write_json(output / 'run_metadata.json', metadata)
        raise
    metadata['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
    write_json(output / 'run_metadata.json', metadata)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/baseline.json')
    parser.add_argument('--rows', type=int, default=0, help='0 = full development, otherwise labeled pilot.')
    parser.add_argument('--models', nargs='+', choices=MODEL_NAMES, default=MODEL_NAMES)
    parser.add_argument('--jobs', type=int, default=2)
    args = parser.parse_args()
    run(args.data, args.output, args.config, args.rows, args.models, args.jobs)


if __name__ == '__main__':
    main()
