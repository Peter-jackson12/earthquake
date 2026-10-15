"""C-preserving sparse interactions, frozen development folds, private outputs."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import platform
import re
import sys
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
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_is_fitted
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from preprocess import FEATURES, ID, NUMERIC, CATEGORICAL, FeatureSchema, sha256_file, write_json
from baseline import aligned_probabilities, metrics
from experiments.age995.run import make_pipeline as original, PARAMS, PROBS, preprocessor_check
from experiments.age995.audit_followup import audit_variant
from scripts.audit_eda_oof import development_frame, validate_oof

HERE = Path(__file__).parent
SOURCE = ['preprocess.py', 'baseline.py', 'scripts/audit_eda_oof.py',
          'experiments/age995/run.py', 'experiments/age995/audit_followup.py',
          'experiments/age995/interactions/config.json',
          'experiments/age995/interactions/run.py']


def config():
    value = json.loads((HERE/'config.json').read_text(encoding='utf-8'))
    if value['lr'] != PARAMS:
        raise ValueError('Do not change the reference LR parameters.')
    return value


def raw_terms(X):
    """Raw validated ages: 995 remains a category, never mislabeled as age 15."""
    cfg = config()
    band = pd.cut(X.age, cfg['age_edges'], labels=cfg['age_labels'], right=True)
    if band.isna().any():
        raise ValueError('Age outside the fixed band contract.')
    band = band.astype(str)
    geo, foundation = X[cfg['geography']].astype(str), X[cfg['structure']].astype(str)
    # normalize_features limits geography to integer text and foundation to known codes.
    return pd.DataFrame({'age_band': band, 'geo_foundation': geo+'|'+foundation,
                         'geo_age': geo+'|'+band, 'foundation_age': foundation+'|'+band}, index=X.index)


class Terms(BaseEstimator, TransformerMixin):
    def __init__(self, variant='G1'):
        self.variant = variant

    def fit(self, X, y=None):
        if self.variant not in config()['variants']:
            raise ValueError('Unknown fixed variant.')
        self.columns_ = tuple(config()['variants'][self.variant]['extra'])
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.transform(X)
        return self

    def transform(self, X):
        check_is_fitted(self, 'columns_')
        if list(X.columns) != list(self.feature_names_in_):
            raise ValueError('Input schema changed.')
        return raw_terms(X).loc[:, list(self.columns_)].copy()

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, 'columns_')
        return np.asarray(self.columns_, dtype=object)


def make_pipeline(variant):
    if variant not in config()['variants']:
        raise ValueError('Unknown fixed variant.')
    base = original('C')
    extra = Pipeline([('schema', FeatureSchema()), ('terms', Terms(variant)),
                      ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=True, dtype=np.float32))])
    union = FeatureUnion([('C', base['preprocess']), ('extra', extra)], n_jobs=1)
    return Pipeline([('preprocess', union), ('model', base['model'])])


def fitted_checks(pipe, train, valid):
    parts = dict(pipe['preprocess'].transformer_list)
    base = parts['C']
    check = preprocessor_check(Pipeline([('preprocess', base)]), train, valid, 'C')
    # Independent C preprocessor fit, not a second classifier fit.
    reference = original('C')['preprocess'].fit(train)
    for data in (train.iloc[:16], valid):
        a, b = base.transform(data), reference.transform(data)
        if a.shape != b.shape or (a != b).nnz:
            raise AssertionError('C base block changed.')
    extra = parts['extra']
    terms = raw_terms(train)[list(extra['terms'].columns_)]
    encoder = extra['encode']
    before = copy.deepcopy(encoder.categories_)
    rows = []
    for col, learned in zip(terms, encoder.categories_):
        counts = terms[col].value_counts()
        if set(learned) != set(counts.index):
            raise AssertionError('Extra vocabulary is not training-only.')
        valid_values = raw_terms(valid)[col]
        rows.append(dict(feature=col, categories=len(learned), rare_le5=int(counts.le(5).sum()),
                         valid_unseen=int((~valid_values.isin(learned)).sum())))
    matrix = pipe['preprocess'].transform(valid)
    if not sparse.isspmatrix_csr(matrix) or matrix.dtype != np.float32 or matrix.shape[0] != len(valid):
        raise AssertionError('Expected float32 CSR and unchanged validation rows.')
    for old, new in zip(before, encoder.categories_):
        np.testing.assert_array_equal(old, new)
    scaler = base['columns'].named_transformers_['numeric']['scale']
    if int(scaler.n_samples_seen_) != len(train):
        raise AssertionError('Invalid scaler training denominator.')
    check.update(base_C_block_exact=True, extra_coverage=rows,
                 extra_columns=sum(len(c) for c in encoder.categories_),
                 total_columns=matrix.shape[1], numeric_variance=scaler.var_.tolist())
    return check


def verify_C(frame, contract, source):
    source = Path(source)
    saved = json.loads((source/'lr/metrics.json').read_text())
    cfg = json.loads((ROOT/'experiments/age995/followup_v1.json').read_text())
    result = audit_variant(frame, source, 'C', contract, original('C')['model'].get_params(), cfg)
    meta = json.loads((source/'run_metadata.json').read_text())
    current = dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__)
    if any(current[k] != meta['environment'][k] for k in current):
        raise ValueError('Environment differs from the stored C experiment.')
    return dict(status='PASS', model_refitted=False, cv_mean=result['model']['cv_mean'],
                oof_sha256=saved['oof_sha256'], metadata_sha256=sha256_file(source/'run_metadata.json'))


def fit_one(variant, frame, contract, output, code_commit):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    meta = dict(status='running', variant=variant, study_id=config()['study_id'], config=config(),
                code_commit=code_commit, data_contract=contract, holdout_scored=False,
                rows=len(frame), scope='full_development_5fold',
                started_at_utc=datetime.now(timezone.utc).isoformat(),
                environment=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                                 scipy=scipy.__version__, sklearn=sklearn.__version__),
                source_sha256={p:sha256_file(ROOT/p) for p in SOURCE})
    write_json(output/'metadata.json', meta)
    X, y, folds = frame[FEATURES], frame.damage_grade, frame.fold.to_numpy()
    original_hash = pd.util.hash_pandas_object(frame, index=True).to_numpy()
    pred, proba, coverage = np.full(len(y), -1), np.full((len(y),3), np.nan), np.zeros(len(y), dtype=int)
    records = []
    try:
        for fold in range(5):
            a, b = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
            pipe = make_pipeline(variant)
            start = time.perf_counter()
            with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                pipe.fit(X.iloc[a], y.iloc[a])
            duration = time.perf_counter()-start
            with threadpool_limits(limits=1):
                pred[b] = pipe.predict(X.iloc[b])
                proba[b] = aligned_probabilities(pipe, X.iloc[b])
                train_pred = pipe.predict(X.iloc[a])
                check = fitted_checks(pipe, X.iloc[a], X.iloc[b])
                # Keep trained models private and verify full-fold reload, not just a sample.
                model_file = output/f'fold_{fold}.joblib'
                joblib.dump(pipe, model_file)
                loaded = joblib.load(model_file)
                np.testing.assert_array_equal(loaded.predict(X.iloc[b]), pred[b])
                np.testing.assert_allclose(aligned_probabilities(loaded, X.iloc[b]), proba[b], atol=0, rtol=0)
            coverage[b] += 1
            record = dict(fold=fold, train_rows=len(a), valid_rows=len(b),
                          validation=metrics(y.iloc[b],pred[b]), train_macro_f1=metrics(y.iloc[a],train_pred)['macro_f1'],
                          fit_seconds=duration, n_iter=pipe['model'].n_iter_.tolist(),
                          converged=not any(issubclass(w.category,ConvergenceWarning) for w in caught),
                          warnings=[dict(category=w.category.__name__,message=str(w.message)) for w in caught],
                          preprocessing=check, reload_exact=True, model_sha256=sha256_file(model_file))
            records.append(record)
            write_json(output/'folds.json',records)
            print(f'{variant} fold={fold} F1={record["validation"]["macro_f1"]:.9f} '
                  f'fit={duration:.1f}s iterations={record["n_iter"]} converged={record["converged"]}', flush=True)
        np.testing.assert_array_equal(coverage,np.ones(len(y),dtype=int))
        np.testing.assert_array_equal(original_hash,pd.util.hash_pandas_object(frame,index=True).to_numpy())
        oof=pd.DataFrame({ID:frame[ID], 'fold':folds,'y_true':y,'y_pred':pred,
                          **{c:proba[:,i] for i,c in enumerate(PROBS)}})
        validate_oof(frame,oof)
        oof.to_csv(output/'oof.csv.gz',index=False,compression={'method':'gzip','mtime':0})
        scores=[r['validation']['macro_f1'] for r in records]
        result=dict(variant=variant,cv_mean=float(np.mean(scores)),cv_std=float(np.std(scores,ddof=1)),
                    pooled=metrics(y,pred),oof_sha256=sha256_file(output/'oof.csv.gz'),
                    effective_parameters=pipe['model'].get_params(), all_converged=all(r['converged'] for r in records),
                    fit_seconds=sum(r['fit_seconds'] for r in records))
        write_json(output/'metrics.json',result)
        meta.update(status='complete',finished_at_utc=datetime.now(timezone.utc).isoformat())
        write_json(output/'metadata.json',meta)
        return result
    except Exception as exc:
        meta.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        write_json(output/'metadata.json',meta)
        raise


def run(data, C, output, code_commit, variants):
    if not re.fullmatch('[0-9a-f]{40}',code_commit):
        raise ValueError('An actual source commit is required.')
    if not variants or len(set(variants))!=len(variants) or set(variants)-set(config()['variants']):
        raise ValueError('Unique fixed variants required.')
    output=Path(output)
    if any((output/v).exists() for v in variants):
        raise FileExistsError('Do not overwrite previous runs.')
    frame,contract=development_frame(data)
    receipt=verify_C(frame,contract,C)
    output.mkdir(parents=True,exist_ok=True)
    write_json(output/'C_reuse.json',receipt)
    for v in variants:
        fit_one(v,frame,contract,output/v,code_commit)
    for p,h in contract['files'].items():
        if sha256_file(Path(data)/p)!=h:
            raise AssertionError('Protected data changed.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--C',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--code-commit',required=True)
    p.add_argument('--variants',nargs='+',default=list(config()['variants']))
    a=p.parse_args()
    from importlib import import_module
    import_module('experiments.age995.interactions.run').run(a.data,a.C,a.output,a.code_commit,a.variants)
