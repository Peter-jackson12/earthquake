"""Context-constrained SMOTE interpolation with explicit random-copy fallback.

Not stock imblearn.SMOTENC. Only three originally numeric features are allowed
fractional values in synthetic training vectors. Public inference always uses
the unmodified original schema. No raw row or ID is stored on the fitted model.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted
from imblearn.over_sampling import RandomOverSampler
from preprocess import (CATEGORICAL, BINARY, NUMERIC, normalize_features,
                        make_preprocessor)

PROTECTED = CATEGORICAL + BINARY + ['count_floors_pre_eq', 'count_families']
CONTINUOUS = ['age', 'area_percentage', 'height_percentage']


def make_plan(raw: pd.DataFrame, y, scale, seed=42, k=5, interpolate=True):
    """Parents are train positions; all constraints are learned from train only.

    Use identical class-balanced RandomOverSampler donors for the ROS control
    and interpolation arm. Eligible neighbors have identical protected fields,
    age state and label. Singletons remain explicit copies, never discarded.
    """
    if not isinstance(k, int) or k < 1:
        raise ValueError('k must be a positive integer')
    raw = normalize_features(raw).reset_index(drop=True)
    y = np.asarray(y)
    if y.shape != (len(raw),) or set(y) != {1, 2, 3}:
        raise ValueError('Aligned training labels 1, 2, 3 required')
    numeric = raw[CONTINUOUS].to_numpy(dtype=float)
    scale = np.asarray(scale, dtype=float)
    if scale.shape != (3,) or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError('Three positive training-only distance scales required')
    if (raw.age > 995).any():
        raise ValueError('Age outside the declared 0..995 domain')
    ros = RandomOverSampler(random_state=seed)
    ros.fit_resample(np.arange(len(raw)).reshape(-1, 1), y)
    a = ros.sample_indices_[len(raw):].astype(np.int64)
    b = a.copy()
    alpha = np.zeros(len(a), dtype=np.float64)
    context = raw[PROTECTED].copy()
    context['_age_state'] = np.select([raw.age.eq(0), raw.age.eq(995)], [0, 995], default=1)
    context['_class'] = y
    codes, _ = pd.factorize(pd.MultiIndex.from_frame(context), sort=False)
    # No hash-only grouping: MultiIndex factorization compares full tuples.
    groups = pd.Series(np.arange(len(raw))).groupby(codes, sort=False).indices
    rng = np.random.default_rng(seed)
    grouped_draws = pd.Series(np.arange(len(a))).groupby(codes[a], sort=False).indices
    eligible = 0
    if interpolate:
        for code, draw_positions in grouped_draws.items():
            members = groups[code]
            if len(members) < 2:
                continue
            if len(members) > 4000:
                raise ValueError('Context exceeds bounded distance budget; review policy, do not silently truncate')
            distances = cdist(numeric[members] / scale, numeric[members] / scale, 'sqeuclidean')
            np.fill_diagonal(distances, np.inf)
            neighbors = np.argsort(distances, axis=1, kind='stable')[:, :min(k, len(members)-1)]
            local = np.searchsorted(members, a[draw_positions])
            chosen = rng.integers(0, neighbors.shape[1], size=len(draw_positions))
            b[draw_positions] = members[neighbors[local, chosen]]
            alpha[draw_positions] = rng.random(len(draw_positions))
            eligible += len(draw_positions)
    # Independent invariant checks on the complete plan, before construction.
    if not np.array_equal(codes[a], codes[b]) or not np.array_equal(y[a], y[b]):
        raise AssertionError('Protected context or class mismatch')
    syn = numeric[a] + alpha[:, None] * (numeric[b] - numeric[a])
    if not np.isfinite(syn).all():
        raise AssertionError('Nonfinite synthetic value')
    bounds = (syn >= np.minimum(numeric[a], numeric[b])-1e-12) & (syn <= np.maximum(numeric[a], numeric[b])+1e-12)
    if not bounds.all():
        raise AssertionError('Synthetic value outside its two training parents')
    protected_age = np.isin(numeric[a, 0], [0, 995])
    if not np.array_equal(syn[protected_age, 0], numeric[a[protected_age], 0]):
        raise AssertionError('Special age was interpolated across states')
    changed = np.any(syn != numeric[a], axis=1)
    counts = lambda z: {str(c): int(np.sum(z == c)) for c in [1, 2, 3]}
    diagnostics = {
        'method': 'context_constrained_smote_ros_fallback' if interpolate else 'random_over_control',
        'before': counts(y), 'after': counts(np.concatenate([y, y[a]])),
        'synthetic_or_copied_rows': len(a), 'eligible_interpolation_rows': eligible,
        'actually_changed_rows': int(changed.sum()), 'exact_copy_rows': int((~changed).sum()),
        'singleton_fallback_rows': len(a)-eligible if interpolate else 0,
        'protected_context_violations': 0, 'age_state_violations': 0,
        'out_of_parent_bounds_rows': 0,
        'fractional_continuous_rows': int(np.any(syn != np.floor(syn), axis=1).sum()),
        'original_rows_preserved': True, 'k_neighbors_max': k, 'seed': seed,
        'protected_columns': PROTECTED, 'synthetic_continuous_columns': CONTINUOUS,
        'distance_scale_source': 'original training fold StandardScaler',
        'note': 'Constraint validity is not evidence that generated buildings are physically real.'}
    return a, b, alpha, diagnostics


class PersonalSamplingLR(ClassifierMixin, BaseEstimator):
    """Raw integer input -> original train-only preprocessing -> sampler -> LR."""
    def __init__(self, mode='context_smote', C=1.0, seed=42, k_neighbors=5, max_iter=1500):
        self.mode = mode
        self.C = C
        self.seed = seed
        self.k_neighbors = k_neighbors
        self.max_iter = max_iter

    def fit(self, X, y):
        if self.mode not in {'none', 'random_over', 'context_smote'}:
            raise ValueError('Unknown sampling mode')
        raw = normalize_features(X)
        y = np.asarray(y)
        self.preprocess_ = make_preprocessor(scale_numeric=True)
        Z = self.preprocess_.fit_transform(raw, y)
        if not sparse.isspmatrix_csr(Z):
            raise AssertionError('CSR required')
        fit_z, fit_y = Z, y
        self.sampling_diagnostics_ = None
        if self.mode != 'none':
            scaler = self.preprocess_['columns'].named_transformers_['numeric']['scale']
            scale = scaler.scale_[[NUMERIC.index(c) for c in CONTINUOUS]]
            a, b, weight, self.sampling_diagnostics_ = make_plan(
                raw, y, scale, self.seed, self.k_neighbors, self.mode == 'context_smote')
            delta = (Z[b] - Z[a]).tocsr()
            delta.eliminate_zeros()
            if not np.isin(delta.indices, [NUMERIC.index(c) for c in CONTINUOUS]).all():
                raise AssertionError('Interpolation attempted to change protected encoded columns')
            generated = (Z[a] + delta.multiply(weight.astype(np.float32)[:, None])).tocsr()
            fit_z = sparse.vstack([Z, generated], format='csr', dtype=np.float32)
            fit_y = np.concatenate([y, y[a]])
            if not np.isfinite(fit_z.data).all():
                raise AssertionError('Invalid synthetic matrix')
        self.model_ = LogisticRegression(solver='lbfgs', C=self.C, max_iter=self.max_iter,
                                         tol=1e-4, random_state=self.seed)
        self.model_.fit(fit_z, fit_y)
        self.classes_ = self.model_.classes_
        self.n_features_in_ = raw.shape[1]
        self.feature_names_in_ = np.asarray(raw.columns, dtype=object)
        self.fit_rows_ = len(fit_y)
        return self

    def predict_proba(self, X):
        check_is_fitted(self, 'model_')
        return self.model_.predict_proba(self.preprocess_.transform(X))

    def predict(self, X):
        check_is_fitted(self, 'model_')
        return self.model_.predict(self.preprocess_.transform(X))
