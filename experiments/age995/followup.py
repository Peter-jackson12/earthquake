"""Frozen D/E/F age experiments; development folds only, immutable private outputs."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import platform
import re
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.validation import check_is_fitted
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from baseline import aligned_probabilities, build_pipeline, metrics
from preprocess import CATEGORICAL, FEATURES, GEO, ID, NUMERIC, sha256_file, write_json
from scripts.audit_eda_oof import development_frame, validate_oof
from experiments.age995.run import PARAMS, FLAG, PROBS

CONFIG = Path(__file__).with_name('followup_v1.json')
SOURCE_PATHS = ('preprocess.py', 'baseline.py', 'configs/data_contract.json',
                'scripts/audit_eda_oof.py', 'experiments/age995/run.py',
                'experiments/age995/followup.py', 'experiments/age995/followup_v1.json')


def config():
    value = json.loads(CONFIG.read_text(encoding='utf-8'))
    if value['lr'] != PARAMS:
        raise ValueError('A/B/C LR parameters must remain unchanged.')
    return value


def id_hash(ids):
    return hashlib.sha256(np.asarray(ids, dtype='<i8').tobytes()).hexdigest()


class UpperCap(BaseEstimator, TransformerMixin):
    """Fit after original integer schema validation; preserve raw frame and flags."""
    def __init__(self, scope='995_only', flag=False, quantile=0.99, interpolation='linear'):
        self.scope = scope
        self.flag = flag
        self.quantile = quantile
        self.interpolation = interpolation

    def fit(self, X, y=None):
        if self.scope not in ('995_only', 'all_above_cap') or not 0 < self.quantile <= 1:
            raise ValueError('Invalid cap policy.')
        if FLAG in X or X.columns.duplicated().any():
            raise ValueError('Already engineered or duplicated columns.')
        age = pd.to_numeric(X.age, errors='raise')
        if not np.isfinite(age).all():
            raise ValueError('Age must be finite.')
        regular = age.loc[age.ne(995)]  # Zero is deliberately retained. y is unused.
        if regular.empty:
            raise ValueError('No non995 training ages to estimate the cap.')
        self.upper_ = float(regular.quantile(self.quantile, interpolation=self.interpolation))
        if self.upper_ > 995:
            raise ValueError('The specified cap must not increase age995.')
        self.fit_non995_rows_ = len(regular)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self, 'upper_')
        if list(X.columns) != list(self.feature_names_in_):
            raise ValueError('Input schema changed after fitting.')
        out = X.copy(deep=True)
        special = out.age.eq(995)
        mask = special if self.scope == '995_only' else out.age.gt(self.upper_)
        out['age'] = out.age.astype('float64').mask(mask, self.upper_)
        if self.flag:
            out[FLAG] = special.astype('float32')
        return out

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, 'feature_names_in_')
        return np.append(self.feature_names_in_, FLAG) if self.flag else self.feature_names_in_


def conditions(X):
    result = {'age_eq_995': X.age.eq(995)}
    result.update({f'{col}_gt_{limit}': X[col].gt(limit)
                   for col, limit in config()['faith_thresholds'].items()})
    return pd.DataFrame(result, index=X.index)


def training_mask(X, variant):
    """Return a label-blind retained-training mask; never apply it to validation."""
    policy = config()['variants'][variant]['drop']
    flags = conditions(X)
    if policy == 'none':
        drop = pd.Series(False, index=X.index)
    elif policy == 'age_eq_995':
        drop = flags['age_eq_995']
    elif policy == 'age_gt_55':
        drop = flags['age_gt_55']
    elif policy == 'faith_four_union':
        drop = flags.drop(columns='age_eq_995').any(axis=1)
    else:
        raise ValueError('Unknown removal policy.')
    if drop.all():
        raise ValueError('Removal policy leaves no training rows.')
    return ~drop


def make_pipeline(variant):
    spec, cfg = config()['variants'][variant], config()
    pipe = build_pipeline('lr', {'models': {'lr': PARAMS}}, jobs=1)
    if spec['cap'] != 'none':
        pre = pipe['preprocess']
        pre.steps.insert(1, ('cap', UpperCap(spec['cap'], spec['flag'],
                                            cfg['quantile'], cfg['interpolation'])))
        if spec['flag']:
            pre['columns'].transformers = [
                (name, transform, list(cols) + ([FLAG] if name == 'binary' else []))
                for name, transform, cols in pre['columns'].transformers]
    return pipe


def class_counts(y):
    return {str(c): int(np.sum(np.asarray(y) == c)) for c in (1, 2, 3)}


def selection_audit(X, y, retained, valid):
    """Public aggregates only; count original-fold criteria before any deletion."""
    kept, dropped = X.loc[retained], X.loc[~retained]
    flags = conditions(X)
    result = dict(original_rows=len(X), retained_rows=len(kept), removed_rows=len(dropped),
        original_class_counts=class_counts(y), retained_class_counts=class_counts(y.loc[retained]),
        removed_class_counts=class_counts(y.loc[~retained]),
        criterion_counts={c:int(flags[c].sum()) for c in flags},
        criterion_class_counts={c:class_counts(y.loc[flags[c]]) for c in flags},
        faith_union_count=int(flags.drop(columns='age_eq_995').any(axis=1).sum()),
        faith_overlap_ge2=int(flags.drop(columns='age_eq_995').sum(axis=1).ge(2).sum()))
    coverage = []
    for col in CATEGORICAL:
        before = X[col].value_counts()
        after = kept[col].value_counts().reindex(before.index, fill_value=0)
        lost = before.index[after.eq(0)]
        rare = before.le(config()['rare_category_max_training_count'])
        coverage.append(dict(feature=col, categories_before=len(before), categories_after=int(after.gt(0).sum()),
            lost_categories=len(lost), rare_categories_before=int(rare.sum()),
            lost_rare_categories=int((rare & after.eq(0)).sum()),
            validation_unseen_before=int((~valid[col].isin(before.index)).sum()),
            validation_unseen_after=int((~valid[col].isin(after.index[after.gt(0)])).sum())))
    result['category_coverage'] = coverage
    regions = []
    for value, part in X.groupby(GEO[0], observed=True, sort=True):
        keep_n = int(retained.loc[part.index].sum())
        regions.append(dict(region=str(value), before=len(part), after=keep_n,
            removed=len(part)-keep_n, removal_rate=(len(part)-keep_n)/len(part),
            share_before=len(part)/len(X), share_after=keep_n/len(kept)))
    result['regions'] = regions
    result['region_total_variation'] = float(sum(abs(r['share_before']-r['share_after']) for r in regions)/2)
    return result


def preprocessor_audit(pipe, train, valid):
    pre = pipe['preprocess']
    normal = pre['schema'].transform(train)
    engineered = pre['cap'].transform(normal) if 'cap' in pre.named_steps else normal
    cols = pre['columns']
    scale = cols.named_transformers_['numeric']['scale']
    values = engineered[NUMERIC].to_numpy(dtype=np.float32).astype(np.float64)
    np.testing.assert_allclose(scale.mean_, values.mean(0), atol=1e-10, rtol=0)
    np.testing.assert_allclose(scale.var_, values.var(0), atol=1e-8, rtol=1e-10)
    if int(scale.n_samples_seen_) != len(train):
        raise AssertionError('Scaler fit count differs from retained training rows.')
    vocab = cols.named_transformers_['categorical'].categories_
    for name, learned in zip(CATEGORICAL, vocab):
        if set(learned) != set(train[name]):
            raise AssertionError('Nontraining categories in encoder.')
    before = (scale.mean_.copy(), scale.var_.copy(), [x.copy() for x in vocab])
    sample = pre.transform(valid.iloc[:16])
    if not sparse.isspmatrix_csr(sample) or sample.dtype != np.float32:
        raise AssertionError('Expected sparse float32 CSR.')
    np.testing.assert_array_equal(scale.mean_, before[0])
    np.testing.assert_array_equal(scale.var_, before[1])
    for old, new in zip(before[2], cols.named_transformers_['categorical'].categories_):
        np.testing.assert_array_equal(old, new)
    upper = None
    modified_train = modified_valid = modified_non995_train = modified_non995_valid = 0
    if 'cap' in pre.named_steps:
        cap = pre['cap']
        upper = cap.upper_
        expected = train.loc[train.age.ne(995), 'age'].quantile(cap.quantile, interpolation=cap.interpolation)
        np.testing.assert_allclose(upper, expected, atol=0, rtol=0)
        for part, is_train in [(train, True), (valid, False)]:
            changed = cap.transform(pre['schema'].transform(part)).age.ne(part.age)
            if is_train:
                modified_train, modified_non995_train = int(changed.sum()), int((changed & part.age.ne(995)).sum())
            else:
                modified_valid, modified_non995_valid = int(changed.sum()), int((changed & part.age.ne(995)).sum())
        if cap.upper_ != upper:
            raise AssertionError('Validation transform mutated cap.')
    return dict(upper=upper, train_modified=modified_train, valid_modified=modified_valid,
        train_non995_modified=modified_non995_train, valid_non995_modified=modified_non995_valid,
        retained_train_age995=int(train.age.eq(995).sum()), valid_age995=int(valid.age.eq(995).sum()),
        numeric_mean=scale.mean_.tolist(), numeric_variance=scale.var_.tolist(),
        fit_rows=int(scale.n_samples_seen_), engineered_inputs=engineered.shape[1],
        encoded_columns=sample.shape[1], train_only_statistics=True, float32_csr=True)


def fit_variant(variant, frame, contract, directory, code_commit):
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(f'Immutable run already exists: {directory}')
    (directory/'lr').mkdir(parents=True)
    cfg = config()
    meta = dict(status='running', scope='full_development_5fold', study_id=cfg['study_id'],
        experiment_id=cfg['variants'][variant]['id'], variant=variant, rows=len(frame),
        feature_count=38, models=['lr'], holdout_scored=False, data_contract=contract,
        config=cfg, code_commit=code_commit, base_commit=cfg['base_commit'],
        started_at_utc=datetime.now(timezone.utc).isoformat(),
        environment=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                         scipy=scipy.__version__, sklearn=sklearn.__version__),
        source_code_sha256={p:sha256_file(ROOT/p) for p in SOURCE_PATHS})
    write_json(directory/'run_metadata.json', meta)
    frame[[ID,'fold']].rename(columns={'fold':'cv_fold'}).to_csv(directory/'run_manifest.csv', index=False)
    X, y, folds = frame[FEATURES], frame.damage_grade, frame.fold.to_numpy()
    original = pd.util.hash_pandas_object(frame, index=True).to_numpy()
    pred, probabilities = np.full(len(y), -1, dtype=int), np.full((len(y),3), np.nan)
    coverage, records = np.zeros(len(y), dtype=int), []
    try:
        for fold in range(contract['n_folds']):
            train_pos, valid_pos = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
            train, valid = X.iloc[train_pos], X.iloc[valid_pos]
            mask = training_mask(train, variant)
            selected = train.loc[mask]
            y_selected = y.loc[selected.index]
            if set(y_selected) != {1,2,3}:
                raise ValueError('Policy removed an entire training class.')
            if set(frame.loc[selected.index,ID]) & set(frame.iloc[valid_pos][ID]):
                raise AssertionError('Train/validation ID overlap.')
            selection = selection_audit(train, y.loc[train.index], mask, valid)
            selection['retained_ids_sha256'] = id_hash(frame.loc[selected.index,ID])
            selection['removed_ids_sha256'] = id_hash(frame.loc[train.index[~mask],ID])
            pipe = make_pipeline(variant)
            start = time.perf_counter()
            with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                pipe.fit(selected, y_selected)
            fit_seconds = time.perf_counter()-start
            start = time.perf_counter()
            with threadpool_limits(limits=1):
                pred[valid_pos] = pipe.predict(valid)
                probabilities[valid_pos] = aligned_probabilities(pipe, valid)
                train_pred = pipe.predict(selected)
            prediction_seconds = time.perf_counter()-start
            coverage[valid_pos] += 1
            check = preprocessor_audit(pipe, selected, valid)
            record = dict(fold=fold, original_train_rows=len(train), train_rows=len(selected),
                validation_rows=len(valid), validation=metrics(y.iloc[valid_pos], pred[valid_pos]),
                train_macro_f1=metrics(y_selected,train_pred)['macro_f1'],
                fit_seconds=fit_seconds, prediction_seconds_including_train=prediction_seconds,
                convergence_ok=not any(issubclass(w.category,ConvergenceWarning) for w in caught),
                warnings=[dict(category=w.category.__name__,message=str(w.message)) for w in caught],
                n_iter=pipe['model'].n_iter_.tolist(), selection_audit=selection, preprocessor_audit=check)
            records.append(record)
            write_json(directory/'lr/fold_metrics.json', records)
            print(f'{variant} fold {fold}: F1={record["validation"]["macro_f1"]:.9f}; '
                  f'removed={selection["removed_rows"]}; cap={check["upper"]}; '
                  f'fit={fit_seconds:.1f}s; converged={record["convergence_ok"]}', flush=True)
            del pipe
            gc.collect()
        np.testing.assert_array_equal(coverage, np.ones(len(frame), dtype=int))
        np.testing.assert_array_equal(original, pd.util.hash_pandas_object(frame, index=True).to_numpy())
        oof = pd.DataFrame({ID:frame[ID], 'fold':folds, 'y_true':y, 'y_pred':pred,
                           **{c:probabilities[:,i] for i,c in enumerate(PROBS)}})
        validate_oof(frame, oof)  # Original validation denominator, not retained rows.
        scores = [r['validation']['macro_f1'] for r in records]
        result = dict(status='complete', variant=variant, model='lr', rows=len(frame), folds=5,
            holdout_scored=False, cv_macro_f1_mean=float(np.mean(scores)),
            cv_macro_f1_std=float(np.std(scores,ddof=1)), oof=metrics(y,pred), fold_results=records,
            exactly_one_oof_per_row=True, all_folds_converged=all(r['convergence_ok'] for r in records),
            effective_parameters=make_pipeline(variant)['model'].get_params(),
            total_fit_seconds=sum(r['fit_seconds'] for r in records))
        oof.to_csv(directory/'lr/oof_predictions.csv.gz',index=False,compression={'method':'gzip','mtime':0})
        result['oof_sha256'] = sha256_file(directory/'lr/oof_predictions.csv.gz')
        write_json(directory/'lr/metrics.json',result)
        meta.update(status='complete',finished_at_utc=datetime.now(timezone.utc).isoformat())
        write_json(directory/'run_metadata.json',meta)
        return result
    except Exception as exc:
        meta.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        write_json(directory/'run_metadata.json',meta)
        raise


def run(data_dir, output, variants, code_commit):
    cfg = config()
    if not re.fullmatch(r'[0-9a-f]{40}',code_commit):
        raise ValueError('Supply the actual source commit, not an unverified label.')
    if not variants or len(set(variants)) != len(variants) or set(variants)-set(cfg['variants']):
        raise ValueError('Select unique, predefined variants.')
    output = Path(output)
    # Refuse all collisions before reading data or beginning any training.
    for variant in variants:
        if (output/variant).exists():
            raise FileExistsError(f'Immutable variant exists: {output/variant}')
    frame, contract = development_frame(data_dir)
    before = {n:sha256_file(Path(data_dir)/n) for n in contract['files']}
    for variant in variants:
        fit_variant(variant,frame,contract,output/variant,code_commit)
    after = {n:sha256_file(Path(data_dir)/n) for n in contract['files']}
    if before != after:
        raise AssertionError('Source data changed during run.')
    return before


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,default=ROOT/'data')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--variants',nargs='+',default=['D','E1','E2','E3'])
    p.add_argument('--code-commit',required=True)
    args = p.parse_args()
    run(args.data_dir,args.output,args.variants,args.code_commit)
