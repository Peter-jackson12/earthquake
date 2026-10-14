"""T12 structure-derived features (stateless, importable module; never the `python -m` entry point).

Bundle S (structure summary) and bundle D (size/ratio) are computed per row from raw binary / numeric
inputs only. No region id, no age, no fitted statistic. Column order inside the output is fixed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

SUPER = ['has_superstructure_' + s for s in (
    'adobe_mud', 'mud_mortar_stone', 'stone_flag', 'cement_mortar_stone', 'mud_mortar_brick',
    'cement_mortar_brick', 'timber', 'bamboo', 'rc_non_engineered', 'rc_engineered', 'other')]
WEAK = ['has_superstructure_' + s for s in (
    'adobe_mud', 'mud_mortar_stone', 'stone_flag', 'mud_mortar_brick', 'timber', 'bamboo')]
STRONG = ['has_superstructure_' + s for s in (
    'cement_mortar_stone', 'cement_mortar_brick', 'rc_non_engineered', 'rc_engineered')]
USE = ['has_secondary_use_' + s for s in (
    'agriculture', 'hotel', 'rental', 'institution', 'school', 'industry', 'health_post', 'gov_office',
    'use_police', 'other')]  # excludes the aggregate has_secondary_use
S_NAMES = ['n_super', 'n_weak', 'n_strong', 'n_use']
D_NAMES = ['height_per_floor', 'area_x_floors', 'families_per_floor']
S_INPUTS = SUPER + USE
D_INPUTS = ['count_floors_pre_eq', 'height_percentage', 'area_percentage', 'count_families']
BUNDLES = ('S', 'D', 'SD')
NAMES = {'S': S_NAMES, 'D': D_NAMES, 'SD': S_NAMES + D_NAMES}
INPUTS = {'S': S_INPUTS, 'D': D_INPUTS, 'SD': S_INPUTS + D_INPUTS}


def derive(X: pd.DataFrame, bundle: str) -> np.ndarray:
    """float64 matrix (rows x len(NAMES[bundle])). Floors are guarded with max(floors, 1)."""
    if bundle not in BUNDLES:
        raise ValueError('Unknown bundle.')
    cols = []
    if 'S' in bundle:
        f = lambda names: X[names].to_numpy(dtype=np.float64).sum(axis=1)  # noqa: E731
        cols += [f(SUPER), f(WEAK), f(STRONG), f(USE)]
    if 'D' in bundle:
        floors = X['count_floors_pre_eq'].to_numpy(dtype=np.float64)
        guard = np.maximum(floors, 1.0)
        cols += [X['height_percentage'].to_numpy(dtype=np.float64) / guard,
                 X['area_percentage'].to_numpy(dtype=np.float64) * floors,
                 X['count_families'].to_numpy(dtype=np.float64) / guard]
    return np.column_stack(cols)


class StructFeatures(BaseEstimator, TransformerMixin):
    """Stateless: fit only records the input schema. Output float64; the scaler/cast follow in a Pipeline."""
    def __init__(self, bundle='SD'):
        self.bundle = bundle

    def fit(self, X, y=None):
        if self.bundle not in BUNDLES:
            raise ValueError('Unknown bundle.')
        missing = [c for c in INPUTS[self.bundle] if c not in X.columns]
        if missing:
            raise ValueError(f'Missing inputs: {missing}')
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self, 'feature_names_in_')
        if list(X.columns) != list(self.feature_names_in_):
            raise ValueError('Input schema changed after fit.')
        return derive(X, self.bundle)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(NAMES[self.bundle], dtype=object)
