"""T11 group-aware inner-OOF hierarchical geo target encoder (importable module; never the `-m` entry point).

Single change vs T10: the splitter that produces the inner-OOF encodings of the TRAINING rows mimics the outer
split. For the outer geo2 scheme the inner splitter is StratifiedGroupKFold(5, shuffle=True, random_state=42)
with groups = (geo1, geo2) composite; for the outer geo1 scheme the groups are geo1. All statistics, the
encoding formula, smoothing, parent fallback and validation-row encoding (full outer-training-fold statistics)
are T10's, imported unchanged (`fit_stats`, `encode`, `oof_encode`).

This class lives here, not in the runner, so that pickled artifacts reference
`experiments.geo_te_group_inner.encoder.GroupInnerTargetEncoder` and load from any entry point.
"""
from __future__ import annotations
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.validation import check_is_fitted

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.geo_hier_te.run import (LEVELS, TE_NAMES, encode, fit_stats, geo_keys,  # noqa: E402
                                         oof_encode)
from experiments.geo_group_cv.run import require  # noqa: E402

GROUP_LEVELS = {'geo1': 0, 'geo2': 1}  # index into geo_keys(): geo1 -> k1, geo2 -> geo1|geo2 composite


def group_ids(G, level):
    """Outer-split group identity of each row, derived from the geo columns only."""
    require(level in GROUP_LEVELS, f'Unknown group level {level!r}')
    return geo_keys(G)[GROUP_LEVELS[level]]


def inner_group_fold_ids(y, groups, n_splits=5, seed=42):
    """StratifiedGroupKFold assignment; stops (no fallback) if five non-empty, group-disjoint folds cannot be formed."""
    y = np.asarray(y)
    groups = np.asarray(groups, dtype=object)
    require(len(y) == len(groups), 'Labels/groups misaligned')
    n_groups = len(np.unique(groups))
    require(n_groups >= n_splits, f'Only {n_groups} groups for {n_splits} inner folds; refusing to change n_splits')
    folds = np.full(len(y), -1, dtype=np.int64)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        splitter = StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed)
        for i, (_, v) in enumerate(splitter.split(np.zeros(len(y)), y, groups)):
            folds[v] = i
    require((folds >= 0).all(), 'Inner fold assignment incomplete')
    sizes = np.bincount(folds, minlength=n_splits)
    require(len(sizes) == n_splits and (sizes > 0).all(), 'Fewer than n_splits non-empty inner folds; study must stop')
    owner = {}
    for g, f in zip(groups, folds):
        require(owner.setdefault(g, f) == f, 'Group split across inner folds')
    return folds


class GroupInnerTargetEncoder(BaseEstimator, TransformerMixin):
    """fit_transform(X, y) -> group-aware inner-OOF encodings of the training rows; transform(X) -> full-fit statistics."""

    def __init__(self, m=20.0, group_level='geo2', inner_splits=5, inner_seed=42):
        self.m = m
        self.group_level = group_level
        self.inner_splits = inner_splits
        self.inner_seed = inner_seed

    def fit(self, X, y=None):
        require(y is not None, 'Target encoding needs training labels')
        self.stats_ = fit_stats(X, y, self.m)
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        yy = np.asarray(y)
        self.inner_groups_ = group_ids(X, self.group_level)
        self.inner_folds_ = inner_group_fold_ids(yy, self.inner_groups_, self.inner_splits, self.inner_seed)
        out, unseen = oof_encode(X, yy, self.inner_folds_, self.m, return_unseen=True)
        self.train_oof_unseen_ = unseen
        self.train_oof_mean_ = out.mean(0).tolist()
        self.train_oof_std_ = out.std(0).tolist()
        self.train_groups_ = int(len(np.unique(self.inner_groups_)))
        self.inner_fold_groups_ = [int(len(np.unique(self.inner_groups_[self.inner_folds_ == i])))
                                   for i in range(self.inner_splits)]
        self.inner_fold_class_counts_ = [[int((yy[self.inner_folds_ == i] == v).sum()) for v in (1, 2, 3)]
                                         for i in range(self.inner_splits)]
        del self.inner_groups_  # not needed after fit; keeps the pickle small
        return out.astype(np.float32)

    def transform(self, X):
        check_is_fitted(self, 'stats_')
        return encode(self.stats_, X).astype(np.float32)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(TE_NAMES, dtype=object)
