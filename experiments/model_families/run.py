"""Fixed A/C x XGBoost/LightGBM study; no holdout, tuning, or row deletion.

Private output contains row predictions and model/preprocessor files. Never publish it.
The public report is produced separately by audit.py after independent OOF checks.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psutil
from scipy import sparse
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from baseline import metrics
from preprocess import FEATURES, ID, TARGET, sha256_file, write_json
from experiments.age995.run import make_pipeline, preprocessor_check, PARAMS, PROBS
from scripts.audit_eda_oof import development_frame, validate_oof, verify_metrics

HERE = Path(__file__).resolve().parent
BASE_SHA = 'a4fc5c82a1907d8fc975f0971deaed8d26a4461f'
EXPERIMENTS = ('XGB_A_v1', 'XGB_C_v1', 'LGBM_A_v1', 'LGBM_C_v1')
LR_OOF_SHA = {
    'A': 'e8dbaa962fe4b2e2fdad160d25f72cbfd0816d73aafbc0419149c53b8a3de221',
    'C': 'de43f9974e536b3a9c5ebac35b984ed886ba2d5426d59dda989ca12f366a2cbc',
}


def peak_rss_mib() -> float:
    if sys.platform == 'win32':
        return float(getattr(psutil.Process().memory_info(), 'peak_wset', 0)) / 2**20
    import resource
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / (2**20 if sys.platform == 'darwin' else 1024)


def environment() -> dict:
    return {'python': platform.python_version(), **{
        name: importlib.metadata.version(name) for name in
        ('numpy', 'pandas', 'scipy', 'scikit-learn', 'xgboost', 'lightgbm', 'joblib', 'psutil')}}


def load_config(path: Path = HERE / 'config.json') -> dict:
    config = json.loads(Path(path).read_text(encoding='utf-8'))
    if (config['experiment_ids'] != list(EXPERIMENTS) or config['n_folds'] != 5
            or config['seed'] != 42 or config['early_stopping']
            or config['class_weight'] is not None or config['sampling'] is not None
            or config['preprocessing'] != 'unchanged_LR_A_C_scaled_OHE_CSR_float32'):
        raise ValueError('This v1 runner requires the preregistered 2x2 contract.')
    for family in ('XGB', 'LGBM'):
        p = config['models'][family]
        if p['n_estimators'] != 300 or p['learning_rate'] != 0.05 or p['random_state'] != 42:
            raise ValueError('Use a new experiment version for parameter changes.')
        if p['subsample'] != 1.0 or p['colsample_bytree'] != 1.0 or p['n_jobs'] != 2:
            raise ValueError('Sampling/thread policy changed.')
    return config


def make_estimator(family: str, config: dict):
    if family == 'XGB':
        from xgboost import XGBClassifier
        return XGBClassifier(**config['models']['XGB'])
    if family == 'LGBM':
        from lightgbm import LGBMClassifier
        return LGBMClassifier(**config['models']['LGBM'])
    raise ValueError('Unknown model family.')


def matrix_sha(matrix) -> str:
    if not sparse.isspmatrix_csr(matrix) or matrix.dtype != np.float32:
        raise ValueError('Expected the unchanged float32 CSR representation.')
    if not np.isfinite(matrix.data).all():
        raise ValueError('Nonfinite encoded feature.')
    h = hashlib.sha256()
    for a in (np.asarray(matrix.shape, dtype='<i8'), matrix.indptr.astype('<i8'),
              matrix.indices.astype('<i8'), matrix.data.astype('<f4')):
        h.update(a.tobytes())
    return h.hexdigest()


def checked_probability(model, matrix) -> np.ndarray:
    if not np.array_equal(model.classes_, [0, 1, 2]):
        raise ValueError('Internal class order must be 0,1,2 (external 1,2,3).')
    p = np.asarray(model.predict_proba(matrix), dtype=np.float64)
    if p.shape != (matrix.shape[0], 3) or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError('Invalid predicted probabilities.')
    np.testing.assert_allclose(p.sum(1), 1.0, atol=1e-6, rtol=0)
    return p


def protected_sources() -> dict:
    expected = json.loads((HERE / 'inherited_sources.json').read_text())['sha256']
    for name, digest in expected.items():
        if sha256_file(ROOT / name) != digest:
            raise ValueError(f'Inherited source changed: {name}')
    names = list(expected) + ['experiments/model_families/run.py', 'experiments/model_families/__init__.py']
    names += ['experiments/model_families/config.json', 'experiments/model_families/PLAN.md',
              'experiments/model_families/inherited_sources.json']
    return {name: sha256_file(ROOT / name) for name in names}


def validate_lr(frame: pd.DataFrame, contract: dict, directory: Path, mode: str) -> dict:
    """Reuse only the pinned historical OOF, not a rounded score or newly fitted LR."""
    directory = Path(directory)
    meta = json.loads((directory / 'run_metadata.json').read_text())
    saved = json.loads((directory / 'lr/metrics.json').read_text())
    if (meta['status'] != 'complete' or meta['scope'] != 'full_development_5fold'
            or meta['holdout_scored'] or meta['data_contract'] != contract
            or meta['config']['models']['lr'] != PARAMS or not saved['all_folds_converged']
            or saved['effective_parameters'] != make_pipeline(mode)['model'].get_params()):
        raise ValueError('Historical LR configuration is not comparable.')
    if mode == 'C' and meta['config'].get('age_policy') != 'C':
        raise ValueError('Expected historical C policy.')
    for name, digest in meta['source_code_sha256'].items():
        if sha256_file(ROOT / name) != digest:
            raise ValueError(f'Historical LR source mismatch: {name}')
    now = environment()
    for name, version in meta['environment'].items():
        if now[{'sklearn': 'scikit-learn'}.get(name, name)] != version:
            raise ValueError(f'Historical LR environment mismatch: {name}')
    actual = pd.read_csv(directory / 'run_manifest.csv').sort_values(ID).reset_index(drop=True)
    expected = frame[[ID, 'fold']].rename(columns={'fold': 'cv_fold'}).sort_values(ID).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
    if 'sample_ids_sha256' in meta:
        digest = hashlib.sha256(frame[ID].to_numpy(dtype='<i8').tobytes()).hexdigest()
        if digest != meta['sample_ids_sha256']:
            raise ValueError('Historical LR ID digest mismatch.')
    p = directory / 'lr/oof_predictions.csv.gz'
    if sha256_file(p) != LR_OOF_SHA[mode] or saved['oof_sha256'] != LR_OOF_SHA[mode]:
        raise ValueError('Historical LR OOF digest mismatch.')
    verify_metrics(validate_oof(frame, pd.read_csv(p)), saved)
    return {'variant': mode, 'refit_in_this_study': False,
            'oof_sha256': sha256_file(p), 'metadata_sha256': sha256_file(directory / 'run_metadata.json'),
            'metrics_sha256': sha256_file(directory / 'lr/metrics.json'),
            'cv_mean': saved['cv_macro_f1_mean'], 'cv_std_ddof1': saved['cv_macro_f1_std']}


def run(data_dir: Path, output: Path, lr_root: Path) -> dict:
    output, data_dir, lr_root = Path(output), Path(data_dir), Path(lr_root)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Refusing to overwrite an existing experiment directory.')
    config = load_config()
    env = environment()
    for name, version in config['required_versions'].items():
        if env[name] != version:
            raise ValueError(f'Pin the tested environment before running: {name}={version}')
    sources = protected_sources()
    frame, contract = development_frame(data_dir)
    before = {str(data_dir / name): sha256_file(data_dir / name) for name in contract['files']}
    for mode in ('A', 'C'):
        for p in sorted((lr_root / mode).rglob('*')):
            if p.is_file():
                before[str(p)] = sha256_file(p)
    references = {mode: validate_lr(frame, contract, lr_root / mode, mode) for mode in ('A', 'C')}
    output.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()
    meta = {'study_id': 'model_families_v1', 'status': 'RUNNING', 'base_commit': BASE_SHA,
            'source_mode': 'Git-blob-verified snapshot; not a successful git clone',
            'config': config, 'config_sha256': sha256_file(HERE / 'config.json'),
            'source_code_sha256': sources, 'environment': env, 'data_contract': contract,
            'lr_reuse': references, 'started_at_utc': started_at,
            'holdout_opened': False, 'new_full_cv_fits': 0,
            'hardware': {'logical_cpu_count': psutil.cpu_count(), 'model_threads': 2,
                         'outer_parallel_jobs': 1, 'platform': platform.platform()},
            'lr_root': str(lr_root.resolve()), 'data_dir': str(data_dir.resolve())}
    write_json(output / 'run_metadata.json', meta)
    y, folds = frame[TARGET].to_numpy(), frame['fold'].to_numpy()
    probabilities = {e: np.full((len(frame), 3), np.nan) for e in EXPERIMENTS}
    coverage = {e: np.zeros(len(frame), dtype=np.int8) for e in EXPERIMENTS}
    records = {e: [] for e in EXPERIMENTS}
    input_hash = pd.util.hash_pandas_object(frame, index=True).to_numpy()
    try:
        for mode in ('A', 'C'):
            for fold in range(5):
                train, valid = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
                if len(train) != 166784 or len(valid) != 41696 or np.intersect1d(train, valid).size:
                    raise ValueError('Frozen fold membership/denominator changed.')
                xt, xv = frame.iloc[train][FEATURES], frame.iloc[valid][FEATURES]
                pre = make_pipeline(mode)['preprocess']
                with threadpool_limits(limits=2):
                    t0 = time.perf_counter()
                    mt, mv = pre.fit_transform(xt), pre.transform(xv)
                    pre_seconds = time.perf_counter() - t0
                audit = preprocessor_check({'preprocess': pre}, xt, xv, mode)
                matrix_hashes = {'train': matrix_sha(mt), 'validation': matrix_sha(mv)}
                pre_path = output / 'preprocessors' / f'{mode}_fold{fold}.joblib'
                pre_path.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(pre, pre_path, compress=3)
                for family in ('XGB', 'LGBM'):
                    experiment = f'{family}_{mode}_v1'
                    destination = output / experiment
                    destination.mkdir(exist_ok=True)
                    model = make_estimator(family, config)
                    t0 = time.perf_counter()
                    with threadpool_limits(limits=2), warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter('always')
                        model.fit(mt, y[train].astype(np.int32) - 1)  # No eval_set / weights / callbacks.
                        fit_seconds = time.perf_counter() - t0
                        t1 = time.perf_counter()
                        p = checked_probability(model, mv)
                        train_p = checked_probability(model, mt)
                        predict_seconds = time.perf_counter() - t1
                    probabilities[experiment][valid] = p
                    coverage[experiment][valid] += 1
                    extension = 'ubj' if family == 'XGB' else 'txt'
                    model_path = destination / f'fold{fold}.{extension}'
                    if family == 'XGB':
                        model.save_model(model_path)
                        actual_rounds = model.get_booster().num_boosted_rounds()
                    else:
                        model.booster_.save_model(str(model_path))
                        actual_rounds = model.booster_.current_iteration()
                    if actual_rounds != config['models'][family]['n_estimators']:
                        raise ValueError('Full fixed boosting budget was not completed.')
                    pred = np.argmax(p, axis=1) + 1
                    record = {'fold': fold, 'train_rows': len(train), 'validation_rows': len(valid),
                        'train_class_counts': [int(np.sum(y[train] == c)) for c in (1, 2, 3)],
                        'validation_class_counts': [int(np.sum(y[valid] == c)) for c in (1, 2, 3)],
                        'validation': metrics(y[valid], pred),
                        'train_macro_f1': metrics(y[train], train_p.argmax(1) + 1)['macro_f1'],
                        'fit_seconds': fit_seconds, 'predict_train_valid_seconds': predict_seconds,
                        'shared_preprocessing_seconds': pre_seconds, 'rounds_completed': actual_rounds,
                        'process_cumulative_peak_rss_mib': peak_rss_mib(),
                        'rss_after_fit_mib': psutil.Process().memory_info().rss / 2**20,
                        'preprocessor_audit': audit, 'encoded_sha256': matrix_hashes,
                        'preprocessor_file': str(pre_path.relative_to(output)),
                        'preprocessor_sha256': sha256_file(pre_path),
                        'model_file': str(model_path.relative_to(output)), 'model_sha256': sha256_file(model_path),
                        'warnings': [{'category': w.category.__name__, 'message': str(w.message)} for w in caught]}
                    records[experiment].append(record)
                    write_json(destination / 'fold_metrics.json', records[experiment])
                    meta['new_full_cv_fits'] += 1
                    write_json(output / 'run_metadata.json', meta)
                    print(f'{experiment} fold={fold} F1={record["validation"]["macro_f1"]:.9f} '
                          f'fit={fit_seconds:.2f}s median={audit["median"]}', flush=True)
                    del model, train_p
                    gc.collect()
                if matrix_hashes != {'train': matrix_sha(mt), 'validation': matrix_sha(mv)}:
                    raise AssertionError('A model mutated shared encoded input.')
                del pre, mt, mv
                gc.collect()
        for e in EXPERIMENTS:
            np.testing.assert_array_equal(coverage[e], np.ones(len(frame), dtype=np.int8))
            p = probabilities[e]
            oof = pd.DataFrame({ID: frame[ID], 'fold': folds, 'y_true': y, 'y_pred': p.argmax(1) + 1,
                               **{name: p[:, i] for i, name in enumerate(PROBS)}})
            scores = [r['validation']['macro_f1'] for r in records[e]]
            result = {'status': 'complete', 'model': e, 'rows': len(frame), 'folds': 5,
                'holdout_scored': False, 'cv_macro_f1_mean': float(np.mean(scores)),
                'cv_macro_f1_std': float(np.std(scores, ddof=1)), 'oof': metrics(y, oof.y_pred),
                'fold_results': records[e], 'exactly_one_oof_per_row': True,
                'parameters': config['models'][e.split('_')[0]],
                'total_fit_seconds': sum(r['fit_seconds'] for r in records[e])}
            verify_metrics(validate_oof(frame, oof), result)
            path = output / e / 'oof_predictions.csv.gz'
            oof.to_csv(path, index=False, compression={'method': 'gzip', 'mtime': 0})
            result['oof_sha256'] = sha256_file(path)
            write_json(output / e / 'metrics.json', result)
        np.testing.assert_array_equal(input_hash, pd.util.hash_pandas_object(frame, index=True).to_numpy())
        if before != {p: sha256_file(p) for p in before} or sources != protected_sources():
            raise AssertionError('Protected input, historical result, or source changed.')
        meta.update(status='TRAINED_PENDING_INDEPENDENT_AUDIT', finished_at_utc=datetime.now(timezone.utc).isoformat(),
                    protected_files_verified=len(before), protected_files_sha256=before)
        write_json(output / 'run_metadata.json', meta)
        return meta
    except Exception as exc:
        meta.update(status='FAILED', error=f'{type(exc).__name__}: {exc}')
        write_json(output / 'run_metadata.json', meta)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--lr-root', type=Path, required=True, help='Private age995_abc_v1 directory with A and C.')
    args = parser.parse_args()
    run(args.data_dir, args.output, args.lr_root)
