"""T35: feature construction ported from the teammate (EY) notebook's revised pipeline, for fold-internal use.

Provenance: the logic follows cell 28 (0-based) of the teammate notebook `지진.ipynb`
(SHA-256 633f6e6b3c1c3bdf669af71df29d31581cb97df63a303ab72b0bbc21b8eed33d, read-only, never modified; private copy only).
The code below is our own re-implementation; comments are ours.  Faithfulness is checked outside Git by extracting the
notebook's own definitions (AST) from the private copy and requiring bit-identical outputs (`run.py preflight`).

Design: docs/T35_EY_PIPELINE_V2_BENCHMARK_PLAN.md.
* Region composition, combination damage rates and the one-hot column list come from the training frame only.
* The evaluation frame's label column (if any) is dropped before anything is computed.
* No holdout access and no model fitting in this module.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

SEED = 42
GRADE = 'damage_grade'
MIN_REGION = 20
TEXT_CAT = ['land_surface_condition', 'foundation_type', 'roof_type', 'ground_floor_type',
            'other_floor_type', 'position', 'plan_configuration', 'legal_ownership_status']
DUMMY_COLS = TEXT_CAT + ['material_group', 'structure_type']
WEAK_FLAGS = ['has_superstructure_adobe_mud', 'has_superstructure_mud_mortar_stone',
              'has_superstructure_stone_flag', 'has_superstructure_mud_mortar_brick']
# combination key -> minimum building count for its own rate (smaller groups get the base-wide rate)
RATE_KEYS = {'k_region_material': 50, 'k_region_structure': 50, 'k_material_structure': 50,
             'k_region_material_floor': 50, 'k_g2': 10, 'k_g3': 10,
             'k_g2_material': 10, 'k_g3_material': 10, 'k_material': 50}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def original_columns():
    from preprocess import FEATURES
    return list(FEATURES)


def derived_columns():
    return (['material_group', 'structure_type', 'n_weak_material', 'n_material', 'weak_age']
            + [f'g{lv}_{c}' for lv in (1, 2, 3) for c in ('n_buildings', 'weak_ratio', 'rc_ratio', 'age_vs_region')]
            + [f'{k[2:]}_{h}_rate' for k in RATE_KEYS if k != 'k_material' for h in ('high', 'low')]
            + [f'{lv}_material_{h}_gap' for lv in ('g2', 'g3') for h in ('high', 'low')])


def feature_columns():
    return original_columns() + derived_columns()


# ---------------------------------------------------------------- per-row derivations (no labels, no statistics)
def building_features(d):
    d = d.copy()
    d['n_weak_material'] = d[WEAK_FLAGS].sum(axis=1)
    d['n_material'] = d[[c for c in d.columns if c.startswith('has_superstructure_')]].sum(axis=1)
    # representative material: later assignments override earlier ones (strongest material wins)
    d['material_group'] = 'other'
    d.loc[d['has_superstructure_timber'] + d['has_superstructure_bamboo'] > 0, 'material_group'] = 'timber_bamboo'
    d.loc[d['n_weak_material'] > 0, 'material_group'] = 'mud_stone'
    d.loc[d['has_superstructure_cement_mortar_stone'] + d['has_superstructure_cement_mortar_brick'] > 0,
          'material_group'] = 'cement'
    d.loc[d['has_superstructure_rc_non_engineered'] + d['has_superstructure_rc_engineered'] > 0,
          'material_group'] = 'RC'
    d['structure_type'] = d['foundation_type'] + '_' + d['roof_type'] + '_' + d['ground_floor_type']
    d['floor_band'] = d['count_floors_pre_eq'].clip(upper=3).astype(str)
    d['weak_age'] = np.where(d['material_group'] == 'mud_stone', d['age'], 0)
    g1, g2, g3 = (d[f'geo_level_{i}_id'].astype(str) for i in (1, 2, 3))
    mat = d['material_group']
    d['k_region_material'] = g1 + '|' + mat
    d['k_region_structure'] = g1 + '|' + d['structure_type']
    d['k_material_structure'] = mat + '|' + d['structure_type']
    d['k_region_material_floor'] = g1 + '|' + mat + '|' + d['floor_band']
    d['k_g2'], d['k_g3'] = g2, g3
    d['k_g2_material'], d['k_g3_material'] = g2 + '|' + mat, g3 + '|' + mat
    d['k_material'] = mat
    return d


# ---------------------------------------------------------------- statistics from the base (training) frame
def region_summary(base, geo):
    """Per-region count, median age, mud/stone share and RC share; regions below MIN_REGION get base-wide values."""
    work = pd.DataFrame({'g': base[geo].to_numpy(), 'age': base['age'].to_numpy(),
                         'weak': (base['material_group'] == 'mud_stone').to_numpy(),
                         'rc': (base['material_group'] == 'RC').to_numpy()})
    grp = work.groupby('g')
    s = pd.DataFrame({'n': grp['age'].size(), 'med_age': grp['age'].median(),
                      'weak': grp['weak'].mean(), 'rc': grp['rc'].mean()})
    s.index.name = geo
    small = s['n'] < MIN_REGION
    s.loc[small, 'weak'] = (base['material_group'] == 'mud_stone').mean()
    s.loc[small, 'rc'] = (base['material_group'] == 'RC').mean()
    s.loc[small, 'med_age'] = base['age'].median()
    return s


def attach_region(d, geo, lv, s, base_median_age):
    # unseen regions: count 0, shares = plain mean over the region table rows, age vs the base median
    d[f'g{lv}_n_buildings'] = d[geo].map(s['n']).fillna(0)
    d[f'g{lv}_weak_ratio'] = d[geo].map(s['weak']).fillna(s['weak'].mean())
    d[f'g{lv}_rc_ratio'] = d[geo].map(s['rc']).fillna(s['rc'].mean())
    d[f'g{lv}_age_vs_region'] = d['age'] - d[geo].map(s['med_age']).fillna(base_median_age)


def rate_table(data, key, min_count):
    """Grade-3 and grade-1 share per key value (exact integer sums / counts); thin keys get the data-wide share."""
    grade = data['damage_grade'].to_numpy()
    work = pd.DataFrame({'k': data[key].to_numpy(), 'hi': (grade == 3).astype(np.int64),
                         'lo': (grade == 1).astype(np.int64)})
    g = work.groupby('k')
    n = g['hi'].size()
    t = pd.DataFrame({'n': n, 'high': g['hi'].sum() / n, 'low': g['lo'].sum() / n})
    t.index.name = key
    t.loc[t['n'] < min_count, 'high'] = (data['damage_grade'] == 3).mean()
    t.loc[t['n'] < min_count, 'low'] = (data['damage_grade'] == 1).mean()
    return t


def make_features(base, others):
    """(base with labels, list of frames) -> (base features, list of other features); all statistics from `base`."""
    require(GRADE in base.columns, 'Base frame needs the label column')
    base = building_features(base).reset_index(drop=True)
    others = [building_features(o.drop(columns=[GRADE], errors='ignore')) for o in others]
    for lv in (1, 2, 3):
        geo = f'geo_level_{lv}_id'
        s = region_summary(base, geo)
        base_median = base['age'].median()
        for d in [base] + others:
            attach_region(d, geo, lv, s, base_median)
    all_high, all_low = (base[GRADE] == 3).mean(), (base[GRADE] == 1).mean()
    parts = list(KFold(5, shuffle=True, random_state=SEED).split(base))
    for key, min_count in RATE_KEYS.items():
        name = key[2:]
        base[f'{name}_high_rate'], base[f'{name}_low_rate'] = 0.0, 0.0
        for other_idx, this_idx in parts:
            t = rate_table(base.loc[other_idx], key, min_count)
            k = base.loc[this_idx, key]
            base.loc[this_idx, f'{name}_high_rate'] = k.map(t['high']).fillna(all_high).values
            base.loc[this_idx, f'{name}_low_rate'] = k.map(t['low']).fillna(all_low).values
        t = rate_table(base, key, min_count)
        for d in others:
            d[f'{name}_high_rate'] = d[key].map(t['high']).fillna(all_high)
            d[f'{name}_low_rate'] = d[key].map(t['low']).fillna(all_low)
    for d in [base] + others:
        for lv in ('g2', 'g3'):
            d[f'{lv}_material_high_gap'] = d[f'{lv}_material_high_rate'] - d['material_high_rate']
            d[f'{lv}_material_low_gap'] = d[f'{lv}_material_low_rate'] - d['material_low_rate']
    return base, others


def to_matrix(frames, cols):
    """One-hot the categorical columns; later frames are aligned to the first frame's columns (absent -> 0)."""
    cats = [c for c in DUMMY_COLS if c in cols]
    first = pd.get_dummies(frames[0][cols], columns=cats, dtype=int)
    rest = [pd.get_dummies(f[cols], columns=cats, dtype=int).reindex(columns=first.columns, fill_value=0)
            for f in frames[1:]]
    return [first] + rest


def class_weights(y0, strength):
    """Inverse class frequency raised to `strength`; `y0` is a Series of 0/1/2 labels."""
    counts = y0.value_counts()
    return y0.map((len(y0) / (3 * counts)) ** strength).values


def fold_matrices(train_frame, valid_frame):
    """Training fold (with labels) + outer validation fold -> (train matrix, valid matrix, 75-col frames, diagnostics)."""
    base_f, (valid_f,) = make_features(train_frame, [valid_frame.drop(columns=[GRADE], errors='ignore')])
    cols = feature_columns()
    X_tr, X_va = to_matrix([base_f, valid_f], cols)
    require(list(X_tr.columns) == list(X_va.columns), 'Validation matrix columns differ from training columns')
    diag = dict(
        columns=int(X_tr.shape[1]),
        dummy_columns=int(sum(1 for c in X_tr.columns if c not in cols)),
        nan_cells_train=int(X_tr.isna().sum().sum()), nan_cells_valid=int(X_va.isna().sum().sum()),
        valid_rows_unseen_region={lv: int((~valid_f[f'geo_level_{lv}_id'].isin(base_f[f'geo_level_{lv}_id'])).sum())
                                  for lv in (1, 2, 3)},
        valid_unseen_category_rows={c: int((~valid_f[c].isin(base_f[c])).sum()) for c in DUMMY_COLS})
    return X_tr, X_va, base_f[cols], valid_f[cols], diag


def matrix_sha256(df):
    """Fingerprint of a numeric matrix: column names, row count, then each column as float64 bytes."""
    h = hashlib.sha256()
    h.update(json.dumps(list(df.columns), ensure_ascii=False).encode('utf-8'))
    h.update(str(len(df)).encode())
    for c in df.columns:
        h.update(np.ascontiguousarray(df[c].to_numpy(dtype=np.float64)).tobytes())
    return h.hexdigest()


def label_probe(train_frame, valid_frame_with_labels, seed=0):
    """No model: shuffle the outer-validation labels and require a bit-identical validation matrix."""
    rng = np.random.default_rng(seed)
    shuffled = valid_frame_with_labels.copy()
    shuffled[GRADE] = rng.permutation(shuffled[GRADE].to_numpy())
    _, a, _, _, _ = fold_matrices(train_frame, valid_frame_with_labels)
    _, b, _, _, _ = fold_matrices(train_frame, shuffled)
    changed = float((shuffled[GRADE].to_numpy() != valid_frame_with_labels[GRADE].to_numpy()).mean())
    same = bool(list(a.columns) == list(b.columns) and a.equals(b))
    return dict(outer_labels_changed_share=changed, valid_matrix_identical=same, passes=bool(same and changed > 0))
