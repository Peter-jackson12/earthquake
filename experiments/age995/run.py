"""Age995 A/B/C study on frozen development folds. No holdout access.

A = unchanged; B = raw age + indicator; C = train-median age + indicator.
Run from any working directory. All predictive outputs remain private.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import platform
import shutil
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.validation import check_is_fitted
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from baseline import aligned_probabilities, build_pipeline, metrics
from preprocess import BINARY, CATEGORICAL, FEATURES, ID, NUMERIC, sha256_file, write_json
from scripts.audit_eda_oof import development_frame, validate_oof, verify_metrics, confusion_stats

FLAG = 'age_is_995'
MODES = ('A', 'B', 'C')
BASE_COMMIT = '03111b802a9ceebe93a52083f2c9413ce0dc322a'
POLICY_COMMIT = '5272c4d978c2c9525670bbfb72e83e400fee91fc'
PARAMS = dict(solver='lbfgs', C=1.0, max_iter=1500, tol=0.0001, random_state=42)
PROBS = ['p_grade_1', 'p_grade_2', 'p_grade_3']


class Age995Treatment(BaseEstimator, TransformerMixin):
    """Placed AFTER the unchanged raw schema validator, BEFORE encoding/scaling."""
    def __init__(self, mode='B'):
        self.mode = mode

    def fit(self, X, y=None):
        if self.mode not in ('B', 'C'):
            raise ValueError('Age995Treatment accepts B or C only.')
        if FLAG in X.columns:
            raise ValueError('Do not supply an already engineered indicator.')
        age = pd.to_numeric(X['age'], errors='raise')
        if not np.isfinite(age).all():
            raise ValueError('Raw age must be finite.')
        self.median_ = None
        if self.mode == 'C':
            regular = age.loc[age.ne(995)]  # Includes zero; never uses y.
            if regular.empty:
                raise ValueError('No non995 training ages; cannot learn a median.')
            self.median_ = float(regular.median())
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.array(X.columns, dtype=object)
        return self

    def transform(self, X):
        check_is_fitted(self, 'median_')
        if list(X.columns) != list(self.feature_names_in_):
            raise ValueError('Input schema changed after fit.')
        out = X.copy(deep=True)
        flag = out['age'].eq(995)
        out[FLAG] = flag.astype('float32')
        if self.mode == 'C':
            out['age'] = out['age'].astype('float64').mask(flag, self.median_)
        return out

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, 'feature_names_in_')
        return np.append(self.feature_names_in_, FLAG)


def make_pipeline(mode):
    if mode not in MODES:
        raise ValueError('Unknown variant.')
    pipeline = build_pipeline('lr', {'models': {'lr': PARAMS}}, jobs=1)
    if mode != 'A':
        pre = pipeline['preprocess']
        pre.steps.insert(1, ('age995', Age995Treatment(mode)))
        pre['columns'].transformers = [
            (name, transform, list(cols) + ([FLAG] if name == 'binary' else []))
            for name, transform, cols in pre['columns'].transformers]
    return pipeline


def preprocessor_check(pipeline, X_train, X_valid, mode):
    """Inspect learned values and verify validation transform does not mutate them."""
    pre = pipeline['preprocess']
    normal = pre['schema'].transform(X_train)
    engineered = normal if mode == 'A' else pre['age995'].transform(normal)
    if len(engineered) != len(X_train):
        raise AssertionError('No training rows may be removed.')
    cols = pre['columns']
    scaler = cols.named_transformers_['numeric']['scale']
    expected = engineered[NUMERIC].to_numpy(dtype=np.float32).mean(axis=0, dtype=np.float64)
    np.testing.assert_allclose(scaler.mean_, expected, atol=1e-10, rtol=0)
    for col, learned in zip(CATEGORICAL, cols.named_transformers_['categorical'].categories_):
        if set(learned) != set(X_train[col]):
            raise AssertionError('Encoder vocabulary includes nontraining values.')
    before = scaler.mean_.copy()
    sample = pre.transform(X_valid.iloc[:16])
    if not sparse.isspmatrix_csr(sample) or sample.dtype != np.float32:
        raise AssertionError('Expected float32 CSR.')
    np.testing.assert_array_equal(scaler.mean_, before)
    median = pre['age995'].median_ if mode != 'A' else None
    if mode == 'C':
        np.testing.assert_allclose(median, X_train.loc[X_train.age.ne(995), 'age'].median())
    return dict(median=median, numeric_mean=scaler.mean_.tolist(),
                train_age995=int(X_train.age.eq(995).sum()),
                valid_age995=int(X_valid.age.eq(995).sum()),
                engineered_inputs=engineered.shape[1], encoded_columns=sample.shape[1],
                csr=True, train_only_statistics=True)


def reuse_baseline(source, destination, frame, contract):
    """Refuse to reuse a score without aligned OOF and matching source metadata."""
    source, destination = Path(source), Path(destination)
    meta = json.loads((source/'run_metadata.json').read_text())
    saved = json.loads((source/'lr/metrics.json').read_text())
    if (meta['status'] != 'complete' or meta['scope'] != 'full_development_5fold'
            or meta['holdout_scored'] or meta['data_contract'] != contract
            or meta['config']['models']['lr'] != PARAMS
            or not saved['all_folds_converged']):
        raise ValueError('Baseline provenance is not comparable.')
    for path in ('preprocess.py', 'baseline.py'):
        if meta['source_code_sha256'][path] != sha256_file(ROOT/path):
            raise ValueError(f'Baseline source mismatch: {path}')
    if saved['effective_parameters'] != make_pipeline('A')['model'].get_params():
        raise ValueError('Effective model parameters changed.')
    manifest = pd.read_csv(source/'run_manifest.csv').sort_values(ID).reset_index(drop=True)
    expected = frame[[ID, 'fold']].rename(columns={'fold':'cv_fold'}).sort_values(ID).reset_index(drop=True)
    pd.testing.assert_frame_equal(manifest, expected, check_dtype=False)
    if meta['sample_ids_sha256'] != hashlib.sha256(frame[ID].to_numpy(dtype='<i8').tobytes()).hexdigest():
        raise ValueError('Baseline ID hash mismatch.')
    oof_path = source/'lr/oof_predictions.csv.gz'
    if saved['oof_sha256'] != sha256_file(oof_path):
        raise ValueError('Baseline OOF checksum mismatch.')
    verify_metrics(validate_oof(frame, pd.read_csv(oof_path)), saved)
    (destination/'lr').mkdir(parents=True)
    for name in ('run_metadata.json', 'run_manifest.csv'):
        shutil.copyfile(source/name, destination/name)
    for name in ('metrics.json', 'fold_metrics.json', 'oof_predictions.csv.gz'):
        shutil.copyfile(source/'lr'/name, destination/'lr'/name)
    write_json(destination/'reuse_receipt.json', dict(source_oof_sha256=saved['oof_sha256'],
        source_metadata_sha256=sha256_file(source/'run_metadata.json'),
        source_metrics_sha256=sha256_file(source/'lr/metrics.json'),
        ID_fold_label_probability_checks=True, refit_in_this_run=False))
    return saved


def fit_variant(mode, frame, contract, directory):
    directory = Path(directory)
    (directory/'lr').mkdir(parents=True)
    X, y, ids, folds = frame[FEATURES], frame['damage_grade'], frame[ID], frame['fold'].to_numpy()
    before = pd.util.hash_pandas_object(X, index=True).to_numpy()
    meta = dict(status='running', scope='full_development_5fold', rows=len(frame),
        feature_count=38, models=['lr'], holdout_scored=False, data_contract=contract,
        config=dict(name=f'age995_{mode}_v1', models={'lr':PARAMS}, age_policy=mode),
        base_commit=BASE_COMMIT, policy_commit=POLICY_COMMIT,
        started_at_utc=datetime.now(timezone.utc).isoformat(),
        environment=dict(python=platform.python_version(), numpy=np.__version__,
                         pandas=pd.__version__, sklearn=sklearn.__version__),
        source_code_sha256={p:sha256_file(ROOT/p) for p in
                           ('preprocess.py','baseline.py','scripts/audit_eda_oof.py','experiments/age995/run.py')})
    write_json(directory/'run_metadata.json', meta)
    frame[[ID,'fold']].rename(columns={'fold':'cv_fold'}).to_csv(directory/'run_manifest.csv', index=False)
    pred = np.full(len(y), -1, dtype=int)
    proba = np.full((len(y),3), np.nan)
    coverage = np.zeros(len(y), dtype=int)
    records = []
    for fold in range(5):
        train, valid = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        if np.intersect1d(ids.iloc[train], ids.iloc[valid]).size:
            raise AssertionError('Overlapping training/validation IDs.')
        pipeline = make_pipeline(mode)
        started = time.perf_counter()
        with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            pipeline.fit(X.iloc[train], y.iloc[train])
        duration = time.perf_counter()-started
        with threadpool_limits(limits=1):
            pred[valid] = pipeline.predict(X.iloc[valid])
            proba[valid] = aligned_probabilities(pipeline, X.iloc[valid])
            train_pred = pipeline.predict(X.iloc[train])
        coverage[valid] += 1
        checks = preprocessor_check(pipeline, X.iloc[train], X.iloc[valid], mode)
        record = dict(fold=fold, train_rows=len(train), validation_rows=len(valid),
            validation=metrics(y.iloc[valid], pred[valid]),
            train_macro_f1=metrics(y.iloc[train], train_pred)['macro_f1'],
            fit_seconds=duration, convergence_ok=not any(issubclass(w.category, ConvergenceWarning) for w in caught),
            warnings=[{'category':w.category.__name__, 'message':str(w.message)} for w in caught],
            n_iter=pipeline['model'].n_iter_.tolist(), preprocessor_audit=checks)
        records.append(record)
        write_json(directory/'lr/fold_metrics.json', records)
        print(f'{mode} fold {fold}: F1={record["validation"]["macro_f1"]:.9f}, '
              f'fit={duration:.1f}s, median={checks["median"]}, converged={record["convergence_ok"]}', flush=True)
        del pipeline
        gc.collect()
    np.testing.assert_array_equal(coverage, np.ones(len(y)))
    np.testing.assert_array_equal(before, pd.util.hash_pandas_object(X, index=True).to_numpy())
    oof = pd.DataFrame({ID:ids, 'fold':folds, 'y_true':y, 'y_pred':pred,
                        **{c:proba[:,i] for i,c in enumerate(PROBS)}})
    scores = [r['validation']['macro_f1'] for r in records]
    result = dict(status='complete', model='lr', rows=len(y), folds=5, holdout_scored=False,
        cv_macro_f1_mean=float(np.mean(scores)), cv_macro_f1_std=float(np.std(scores, ddof=1)),
        oof=metrics(y,pred), fold_results=records, exactly_one_oof_per_row=True,
        all_folds_converged=all(r['convergence_ok'] for r in records),
        effective_parameters=make_pipeline(mode)['model'].get_params(),
        total_fit_seconds=sum(r['fit_seconds'] for r in records))
    verify_metrics(validate_oof(frame, oof), result)
    oof.to_csv(directory/'lr/oof_predictions.csv.gz', index=False,
               compression={'method':'gzip','mtime':0})
    result['oof_sha256'] = sha256_file(directory/'lr/oof_predictions.csv.gz')
    write_json(directory/'lr/metrics.json', result)
    meta.update(status='complete', finished_at_utc=datetime.now(timezone.utc).isoformat())
    write_json(directory/'run_metadata.json', meta)
    return result


def summarize(frame, output):
    models, fold_rows, groups = [], [], []
    for mode in MODES:
        saved = json.loads((output/mode/'lr/metrics.json').read_text())
        oof = pd.read_csv(output/mode/'lr/oof_predictions.csv.gz')
        joined = validate_oof(frame, oof)
        stats = verify_metrics(joined, saved)
        models.append(dict(variant=mode, cv_mean=saved['cv_macro_f1_mean'],
            cv_std=saved['cv_macro_f1_std'], pooled_f1=stats['macro_f1'],
            accuracy=stats['accuracy'], error_3_to_1=stats['error_3_to_1'],
            error_3_to_2=stats['error_3_to_2'], converged=saved['all_folds_converged'],
            baseline_reused=(output/mode/'reuse_receipt.json').exists()))
        for r in saved['fold_results']:
            fold_rows.append(dict(variant=mode, fold=r['fold'], macro_f1=r['validation']['macro_f1'],
                fit_seconds=r['fit_seconds'], train_rows=r['train_rows'], validation_rows=r['validation_rows'],
                median=r.get('preprocessor_audit',{}).get('median')))
        joined = joined.merge(frame[[ID,'age']], on=ID, validate='one_to_one')
        for label, part in [('all',joined),('age995',joined[joined.age.eq(995)]),
                            ('non995',joined[joined.age.ne(995)])]:
            stats = confusion_stats(part.y_true, part.y_pred)
            stats.pop('matrix')
            groups.append(dict(variant=mode, group=label, **stats))
    pd.DataFrame(models).to_csv(output/'age995_model_summary.csv', index=False)
    pd.DataFrame(fold_rows).to_csv(output/'age995_fold_summary.csv', index=False)
    pd.DataFrame(groups).to_csv(output/'age995_group_summary.csv', index=False)


def run(data_dir, output, baseline_run=None):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a new output directory; previous runs are immutable.')
    frame, contract = development_frame(data_dir)
    before = {n:sha256_file(Path(data_dir)/n) for n in contract['files']}
    output.mkdir(parents=True,exist_ok=True)
    if baseline_run:
        reuse_baseline(baseline_run, output/'A', frame, contract)
    else:
        fit_variant('A', frame, contract, output/'A')
    for mode in ('B','C'):
        fit_variant(mode, frame, contract, output/mode)
    summarize(frame, output)
    if before != {n:sha256_file(Path(data_dir)/n) for n in contract['files']}:
        raise AssertionError('Source data mutated.')
    write_json(output/'verification.json', dict(status='PASS', variants=list(MODES), rows=len(frame),
        new_predictive_fits=10 if baseline_run else 15, baseline_reused=bool(baseline_run),
        verified_oof_rows=len(frame)*3, holdout_scored=False, data_hashes=before,
        base_commit=BASE_COMMIT, policy_commit=POLICY_COMMIT))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT/'data')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline-run', type=Path, help='Optional trusted LR C=1 run with metadata and private OOF.')
    args=parser.parse_args()
    run(args.data_dir,args.output,args.baseline_run)
