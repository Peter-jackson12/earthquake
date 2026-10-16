"""T33: feature construction ported from a teammate (EY) notebook pipeline, rewritten for fold-internal use.

Provenance: the logic follows cell 25 of the teammate notebook `earthquake/data/지진.ipynb`
(SHA-256 480d63cd6adc3f68f70142ff159d229654db5d17a27d3af093baa38b831ac07b, read-only, never modified).
The code below is a re-implementation in our own words; comments are ours.  Faithfulness is checked outside Git by
extracting the notebook's own function definitions (AST) from the private copy and requiring bit-identical outputs
(`run.py preflight`; also an optional unit test that is skipped when the private copy is absent).

Rules kept from the design (docs/T33_EY_PIPELINE_BENCHMARK_PLAN.md):
* Every statistic (region composition, target rates, category vocabulary) is computed from the training frame handed to
  `build_features` only.  The evaluation frame is only looked up, and its label column is dropped before anything runs.
* No holdout access, no model fitting in this module.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

SEED = 42
MIN_BUILDINGS = 20
MIN_COMBO = 50
GRADE = 'damage_grade'

ORIGINAL_CAT = ['land_surface_condition', 'foundation_type', 'roof_type', 'ground_floor_type',
                'other_floor_type', 'position', 'plan_configuration', 'legal_ownership_status']
NEW_CAT = ['material_group', 'structure_type']
CAT_COLS = ORIGINAL_CAT + NEW_CAT

OLD_KEYS = ['key_region_material', 'key_region_structure', 'key_material_structure', 'key_region_material_floor']
GEO_KEYS = ['key_g2', 'key_g3', 'key_g2_material', 'key_g3_material']
ALL_KEYS = OLD_KEYS + GEO_KEYS + ['key_material']
KEY_MIN = {**{k: 50 for k in OLD_KEYS}, **{k: 10 for k in GEO_KEYS}, 'key_material': 50}
LEVELS = [('geo_level_1_id', 'g1'), ('geo_level_2_id', 'g2'), ('geo_level_3_id', 'g3')]

REGION_COLS = [f'g{lv}_{c}' for lv in '123' for c in ['weak_ratio', 'rc_ratio', 'age_vs_region', 'n_buildings']]
OLD_RATES = [f'{k.replace("key_", "")}_{h}_rate' for k in OLD_KEYS for h in ('high', 'low')]
GEO_RATES = [f'{k.replace("key_", "")}_{h}_rate' for k in GEO_KEYS for h in ('high', 'low')]
NEW_FEATURES = ['weak_age'] + [f'{lv}_material_{h}_gap' for lv in ('g2', 'g3') for h in ('high', 'low')]
WEAK_FLAGS = ['has_superstructure_adobe_mud', 'has_superstructure_mud_mortar_stone',
              'has_superstructure_stone_flag', 'has_superstructure_mud_mortar_brick']


def original_columns():
    """The 38 raw inputs in the order of the source table (identical to preprocess.FEATURES)."""
    from preprocess import FEATURES
    return list(FEATURES)


def feature_columns():
    return (original_columns() + NEW_CAT + ['n_weak_material', 'n_material'] + REGION_COLS + OLD_RATES
            + GEO_RATES + NEW_FEATURES)


def require(ok, message):
    if not ok:
        raise ValueError(message)


# ---------------------------------------------------------------- input frame
def notebook_frame(X, y=None):
    """Our normalized inputs -> the dtypes the notebook sees after reading its CSV.

    Region IDs go back to integers (checked to round-trip exactly), binary flags and counts become int64, and the
    label column (1/2/3) is attached only when `y` is given.  The index is reset so positions are explicit.
    """
    from preprocess import BINARY, GEO, NUMERIC
    cols = original_columns()
    require(list(X.columns) == cols, 'Input columns must be the 38 raw inputs in source order')
    F = X.loc[:, cols].reset_index(drop=True).copy()
    for c in GEO:
        s = X[c].reset_index(drop=True)
        as_int = pd.to_numeric(s, errors='raise').astype('int64')
        require((as_int.astype(str) == s.astype(str)).all(), f'{c}: str(int(x)) round trip failed')
        F[c] = as_int
    for c in BINARY:
        v = X[c].reset_index(drop=True)
        require(v.isin([0, 1]).all(), f'{c}: expected only 0/1')
        F[c] = v.astype('int64')
    for c in NUMERIC:
        F[c] = X[c].reset_index(drop=True).astype('int64')
    if y is not None:
        labels = np.asarray(y).astype('int64')
        require(len(labels) == len(F) and set(np.unique(labels)) <= {1, 2, 3}, 'Labels must be 1/2/3 and aligned')
        F[GRADE] = labels
    return F


# ---------------------------------------------------------------- step 1: per-row derivations (no statistics)
def building_features(df):
    df = df.copy()
    df['n_weak_material'] = df[WEAK_FLAGS[0]] + df[WEAK_FLAGS[1]] + df[WEAK_FLAGS[2]] + df[WEAK_FLAGS[3]]
    material_cols = [c for c in df.columns if c.startswith('has_superstructure_')]
    df['n_material'] = df[material_cols].sum(axis=1)

    # material group: assignments run weakest to strongest, the last matching one wins
    df['material_group'] = 'other'
    df.loc[(df['has_superstructure_timber'] == 1) | (df['has_superstructure_bamboo'] == 1),
           'material_group'] = 'timber_bamboo'
    df.loc[df['n_weak_material'] > 0, 'material_group'] = 'mud_stone'
    df.loc[(df['has_superstructure_cement_mortar_stone'] == 1) |
           (df['has_superstructure_cement_mortar_brick'] == 1), 'material_group'] = 'cement'
    df.loc[(df['has_superstructure_rc_non_engineered'] == 1) |
           (df['has_superstructure_rc_engineered'] == 1), 'material_group'] = 'RC'

    df['structure_type'] = df['foundation_type'] + '_' + df['roof_type'] + '_' + df['ground_floor_type']

    df['floor_band'] = '3F+'
    df.loc[df['count_floors_pre_eq'] == 2, 'floor_band'] = '2F'
    df.loc[df['count_floors_pre_eq'] == 1, 'floor_band'] = '1F'

    region = df['geo_level_1_id'].astype(str)
    df['key_region_material'] = region + '|' + df['material_group']
    df['key_region_structure'] = region + '|' + df['structure_type']
    df['key_material_structure'] = df['material_group'] + '|' + df['structure_type']
    df['key_region_material_floor'] = region + '|' + df['material_group'] + '|' + df['floor_band']
    g2 = df['geo_level_2_id'].astype(str)
    g3 = df['geo_level_3_id'].astype(str)
    df['key_g2'] = g2
    df['key_g3'] = g3
    df['key_g2_material'] = g2 + '|' + df['material_group']
    df['key_g3_material'] = g3 + '|' + df['material_group']
    df['key_material'] = df['material_group']
    df['weak_age'] = 0
    df.loc[df['material_group'] == 'mud_stone', 'weak_age'] = df['age']
    return df


# ---------------------------------------------------------------- step 2: region composition tables
def region_table(train, level):
    """Per-region summary computed from the training frame only; small regions get the training-wide value."""
    weak = (train['material_group'] == 'mud_stone').astype(int)
    rc = (train['material_group'] == 'RC').astype(int)
    work = pd.DataFrame({'level': train[level].to_numpy(), 'weak': weak.to_numpy(), 'rc': rc.to_numpy(),
                         'age': train['age'].to_numpy()})
    summary = work.groupby('level').agg(n_buildings=('weak', 'count'), weak_ratio=('weak', 'mean'),
                                        rc_ratio=('rc', 'mean'), mean_age=('age', 'median'))
    overall = {'weak_ratio': weak.mean(), 'rc_ratio': rc.mean(), 'mean_age': train['age'].median()}
    small = summary['n_buildings'] < MIN_BUILDINGS
    for col, value in overall.items():
        summary.loc[small, col] = value
    return dict(summary=summary, overall={k: float(v) for k, v in overall.items()}, small_regions=int(small.sum()),
                regions=int(len(summary)))


def attach_region(d, level, name, table):
    """Look the stored summary up for every row of `d` (train or evaluation); unseen regions fall back."""
    summary, overall = table['summary'], table['overall']
    d[f'{name}_n_buildings'] = d[level].map(summary['n_buildings']).fillna(0)
    d[f'{name}_weak_ratio'] = d[level].map(summary['weak_ratio']).fillna(overall['weak_ratio'])
    d[f'{name}_rc_ratio'] = d[level].map(summary['rc_ratio']).fillna(overall['rc_ratio'])
    region_age = d[level].map(summary['mean_age']).fillna(overall['mean_age'])
    d[f'{name}_age_vs_region'] = d['age'] - region_age


# ---------------------------------------------------------------- step 3: label-derived combination rates
def rate_table(keys, grade, min_count):
    """Share of grade 3 and grade 1 per combination key (vectorized; bit-identical to a groupby/lambda mean because
    the group sums are exact integers divided by exact integer counts)."""
    grade = np.asarray(grade)
    frame = pd.DataFrame({'k': np.asarray(keys, dtype=object), 'hi': (grade == 3).astype(np.int64),
                          'lo': (grade == 1).astype(np.int64)})
    g = frame.groupby('k')
    count = g['hi'].size()
    table = pd.DataFrame({'count': count, 'high_rate': g['hi'].sum() / count, 'low_rate': g['lo'].sum() / count})
    small = table['count'] < min_count
    table.loc[small, 'high_rate'] = pd.Series(grade == 3).mean()
    table.loc[small, 'low_rate'] = pd.Series(grade == 1).mean()
    return table


def key_name(key):
    return key.replace('key_', '')


def rate_tables(train, keys):
    """Training-wide table per key plus the training-wide fallback rates (used for evaluation rows)."""
    overall_high = (train[GRADE] == 3).mean()
    overall_low = (train[GRADE] == 1).mean()
    out = {}
    for key in keys:
        t = rate_table(train[key].to_numpy(), train[GRADE].to_numpy(), KEY_MIN[key])
        out[key] = dict(table=t, overall_high=float(overall_high), overall_low=float(overall_low),
                        min_count=KEY_MIN[key], combinations=int(len(t)), small_combinations=int((t['count'] < KEY_MIN[key]).sum()))
    return out


def attach_rates_lookup(d, tables):
    for key, spec in tables.items():
        name = key_name(key)
        t = spec['table']
        d[f'{name}_high_rate'] = d[key].map(t['high_rate']).fillna(spec['overall_high'])
        d[f'{name}_low_rate'] = d[key].map(t['low_rate']).fillna(spec['overall_low'])


def attach_rates_oof(train, keys):
    """Training rows: each of five shuffled parts is filled from a table built on the other four parts."""
    train = train.reset_index(drop=True)
    overall_high = (train[GRADE] == 3).mean()
    overall_low = (train[GRADE] == 1).mean()
    parts = list(KFold(n_splits=5, shuffle=True, random_state=SEED).split(train))
    for key in keys:
        name = key_name(key)
        high_col, low_col = f'{name}_high_rate', f'{name}_low_rate'
        train[high_col] = 0.0
        train[low_col] = 0.0
        for other_idx, this_idx in parts:
            t = rate_table(train.loc[other_idx, key].to_numpy(), train.loc[other_idx, GRADE].to_numpy(), KEY_MIN[key])
            this_keys = train.loc[this_idx, key]
            train.loc[this_idx, high_col] = this_keys.map(t['high_rate']).fillna(overall_high).values
            train.loc[this_idx, low_col] = this_keys.map(t['low_rate']).fillna(overall_low).values
    return train


def attach_gaps(d):
    for lv in ('g2', 'g3'):
        d[f'{lv}_material_high_gap'] = d[f'{lv}_material_high_rate'] - d['material_high_rate']
        d[f'{lv}_material_low_gap'] = d[f'{lv}_material_low_rate'] - d['material_low_rate']


# ---------------------------------------------------------------- full construction
def build_features(train_df, eval_df):
    """75-column matrices for `train_df` (with GRADE) and `eval_df` (any GRADE column is dropped first).

    Returns (train_features, eval_features, tables).  Everything fitted comes from `train_df`; `tables` holds it so the
    evaluation side can be rebuilt later with `apply_tables` (reload check, audit).
    """
    require(GRADE in train_df.columns, 'Training frame needs the label column')
    eval_df = eval_df.drop(columns=[GRADE], errors='ignore')
    tr = building_features(train_df.reset_index(drop=True))
    te = building_features(eval_df.reset_index(drop=True))
    tables = dict(region={}, rates={}, categories={})
    for level, name in LEVELS:
        tables['region'][name] = region_table(tr, level)
    for level, name in LEVELS:
        attach_region(tr, level, name, tables['region'][name])
        attach_region(te, level, name, tables['region'][name])
    tr = attach_rates_oof(tr, ALL_KEYS)
    tables['rates'] = rate_tables(tr, ALL_KEYS)
    attach_rates_lookup(te, tables['rates'])
    attach_gaps(tr)
    attach_gaps(te)
    for c in CAT_COLS:
        cats = tr[c].astype('category').cat.categories
        tables['categories'][c] = cats
        tr[c] = pd.Categorical(tr[c], categories=cats)
        te[c] = pd.Categorical(te[c], categories=cats)
    cols = feature_columns()
    return tr[cols], te[cols], tables


def apply_tables(eval_df, tables):
    """Rebuild the evaluation-side matrix from stored tables only (no training rows, no labels)."""
    te = building_features(eval_df.drop(columns=[GRADE], errors='ignore').reset_index(drop=True))
    for level, name in LEVELS:
        attach_region(te, level, name, tables['region'][name])
    attach_rates_lookup(te, tables['rates'])
    attach_gaps(te)
    for c in CAT_COLS:
        te[c] = pd.Categorical(te[c], categories=tables['categories'][c])
    return te[feature_columns()]


def table_summary(tables):
    """Small JSON-able description of the stored tables (sizes and fallback counts only)."""
    return dict(
        region={n: dict(regions=t['regions'], small_regions=t['small_regions']) for n, t in tables['region'].items()},
        rates={k: dict(combinations=s['combinations'], small_combinations=s['small_combinations'],
                       min_count=s['min_count']) for k, s in tables['rates'].items()},
        vocabulary={c: int(len(v)) for c, v in tables['categories'].items()})


def class_weights(y0, strength):
    """Inverse-frequency weights raised to `strength`; `y0` holds labels 0/1/2 as a Series."""
    counts = y0.value_counts()
    return y0.map((len(y0) / (3 * counts)) ** strength).values


def matrix_sha256(df):
    """Order- and dtype-stable fingerprint of a feature matrix (categorical columns by vocabulary + codes)."""
    h = hashlib.sha256()
    h.update(json.dumps(list(df.columns), ensure_ascii=False).encode('utf-8'))
    h.update(str(len(df)).encode())
    for c in df.columns:
        s = df[c]
        if isinstance(s.dtype, pd.CategoricalDtype):
            h.update(b'cat')
            h.update(json.dumps([str(x) for x in s.cat.categories], ensure_ascii=False).encode('utf-8'))
            h.update(np.ascontiguousarray(s.cat.codes.to_numpy().astype(np.int64)).tobytes())
        else:
            h.update(b'num')
            h.update(np.ascontiguousarray(s.to_numpy(dtype=np.float64)).tobytes())
    return h.hexdigest()


def ids_sha256(ids):
    return hashlib.sha256(np.asarray(ids, dtype=np.int64).tobytes()).hexdigest()


# ---------------------------------------------------------------- arm-specific feature sets
def features_B(fold_train_frame, tr_pos, val_pos):
    """E2: split the fold's raw rows first; early-stopping features use the inner training part only.
    The inner validation rows are looked up (their labels never enter this construction)."""
    tr = fold_train_frame.iloc[tr_pos].reset_index(drop=True)
    va = fold_train_frame.iloc[val_pos].drop(columns=[GRADE]).reset_index(drop=True)
    return build_features(tr, va)


def _changed(a, b):
    """Row and column change shares between two same-shaped feature matrices (exact comparison)."""
    ch = np.zeros(a.shape, dtype=bool)
    for j, c in enumerate(a.columns):
        x, y = a[c], b[c]
        if isinstance(x.dtype, pd.CategoricalDtype):
            xo, yo = x.astype(object).to_numpy(), y.astype(object).to_numpy()
            ch[:, j] = ~((xo == yo) | (pd.isna(xo) & pd.isna(yo)))
        else:
            xv, yv = x.to_numpy(dtype=np.float64), y.to_numpy(dtype=np.float64)
            ch[:, j] = ~((xv == yv) | (np.isnan(xv) & np.isnan(yv)))
    return dict(rows_changed_share=float(ch.any(axis=1).mean()), cells_changed_share=float(ch.mean()),
                columns_changed=[str(c) for c, v in zip(a.columns, ch.any(axis=0)) if v],
                n_rows=int(a.shape[0]))


def label_probe(fold_train_frame, outer_valid_frame, tr_pos, val_pos, seed=0):
    """No model: shuffle the labels of the inner-validation rows and see which features move.

    * A order (E1): training-part features should move (the known mixing of validation labels into training rows).
    * B order (E2): training and validation features must not move.
    * Outer evaluation rows: features must not move when the outer labels are shuffled (A order).
    """
    rng = np.random.default_rng(seed)
    attached = GRADE not in outer_valid_frame.columns
    if attached:  # the outer frame normally carries no label; attach a stand-in column so the shuffle below is real
        outer_valid_frame = outer_valid_frame.copy()
        outer_valid_frame[GRADE] = rng.integers(1, 4, len(outer_valid_frame))
    base = fold_train_frame.copy()
    shuffled = fold_train_frame.copy()
    labels = shuffled[GRADE].to_numpy().copy()
    labels[val_pos] = rng.permutation(labels[val_pos])
    shuffled[GRADE] = labels
    A0_tr, A0_va, _ = build_features(base, outer_valid_frame)
    A1_tr, A1_va, _ = build_features(shuffled, outer_valid_frame)
    b0_tr, b0_va, _ = features_B(base, tr_pos, val_pos)
    b1_tr, b1_va, _ = features_B(shuffled, tr_pos, val_pos)
    outer = outer_valid_frame.copy()
    outer[GRADE] = rng.permutation(outer[GRADE].to_numpy())
    outer_changed = float((outer[GRADE].to_numpy() != outer_valid_frame[GRADE].to_numpy()).mean())
    _, A2_va, _ = build_features(base, outer)
    inner_changed_labels = float((base[GRADE].to_numpy()[val_pos] != labels[val_pos]).mean())
    return dict(
        inner_val_labels_changed_share=inner_changed_labels,
        outer_labels_changed_share=outer_changed, outer_label_column_attached=attached,
        outer_invariance_note='build_features drops any label column of the evaluation frame before computing anything '
                              '(structural guarantee); the probe passes a real, shuffled label column to demonstrate it',
        A_order_train_part=_changed(A0_tr.iloc[tr_pos], A1_tr.iloc[tr_pos]),
        A_order_outer_valid=_changed(A0_va, A1_va),
        B_order_train_part=_changed(b0_tr, b1_tr), B_order_inner_val=_changed(b0_va, b1_va),
        A_order_outer_valid_under_outer_label_shuffle=_changed(A0_va, A2_va),
        passes=bool(
            _changed(b0_tr, b1_tr)['cells_changed_share'] == 0 and _changed(b0_va, b1_va)['cells_changed_share'] == 0
            and _changed(A0_va, A2_va)['cells_changed_share'] == 0))


# ---------------------------------------------------------------- non-pickle storage (auditor can read without deserializing)
def state_text_names(prefix):
    """Fixed file names of the text-format state: json (vocabulary, columns, fallbacks), matrix npz, one csv.gz per table."""
    names = [f'{prefix}_state.json', f'{prefix}_matrix.npz']
    names += [f'{prefix}_table_region_{name}.csv.gz' for _, name in LEVELS]
    names += [f'{prefix}_table_rate_{k}.csv.gz' for k in ALL_KEYS]
    return names


def _csv(path, frame, index_label):
    frame.to_csv(path, index=True, index_label=index_label, compression={'method': 'gzip', 'mtime': 0}, float_format='%.17g')


def write_state_text(directory, prefix, tables, matrix=None):
    """Tables as csv.gz (17 significant digits), vocabulary / columns / fallbacks as json, matrix as npz
    (float64 numeric block + int32 category codes, -1 = value outside the vocabulary).  No pickle anywhere."""
    d = Path(directory)
    for _, name in LEVELS:
        _csv(d/f'{prefix}_table_region_{name}.csv.gz', tables['region'][name]['summary'], 'level')
    for k in ALL_KEYS:
        _csv(d/f'{prefix}_table_rate_{k}.csv.gz', tables['rates'][k]['table'], 'key')
    state = dict(
        columns=feature_columns(), categorical_columns=list(CAT_COLS),
        categories={c: [str(x) for x in tables['categories'][c]] for c in CAT_COLS},
        region={n: dict(overall=t['overall'], small_regions=t['small_regions'], regions=t['regions'])
                for n, t in tables['region'].items()},
        rates={k: dict(overall_high=s['overall_high'], overall_low=s['overall_low'], min_count=s['min_count'],
                       combinations=s['combinations'], small_combinations=s['small_combinations'])
               for k, s in tables['rates'].items()},
        matrix_file=f'{prefix}_matrix.npz' if matrix is not None else None,
        matrix_sha256_convention='matrix_sha256 in features.py: column names, row count, then per column float64 bytes or '
                                 'category vocabulary + int64 codes',
        matrix_sha256=matrix_sha256(matrix) if matrix is not None else None)
    (d/f'{prefix}_state.json').write_text(json.dumps(state, ensure_ascii=False, indent=1, allow_nan=False) + '\n',
                                          encoding='utf-8')
    if matrix is not None:
        numeric = [c for c in matrix.columns if c not in CAT_COLS]
        np.savez_compressed(d/f'{prefix}_matrix.npz',
                            numeric=matrix[numeric].to_numpy(dtype=np.float64), numeric_columns=np.array(numeric, dtype=str),
                            category_codes=np.column_stack([matrix[c].cat.codes.to_numpy().astype(np.int32) for c in CAT_COLS]),
                            category_columns=np.array(CAT_COLS, dtype=str),
                            column_order=np.array(list(matrix.columns), dtype=str))
    return state_text_names(prefix) if matrix is not None else [n for n in state_text_names(prefix) if not n.endswith('.npz')]


def read_matrix_text(directory, prefix):
    """Rebuild the stored matrix from npz + json only."""
    d = Path(directory)
    state = json.loads((d/f'{prefix}_state.json').read_text(encoding='utf-8'))
    z = np.load(d/f'{prefix}_matrix.npz', allow_pickle=False)
    cols = {}
    for j, c in enumerate(z['numeric_columns'].tolist()):
        cols[c] = z['numeric'][:, j]
    for j, c in enumerate(z['category_columns'].tolist()):
        cols[c] = pd.Categorical.from_codes(z['category_codes'][:, j], categories=pd.Index(state['categories'][c]))
    return pd.DataFrame({c: cols[c] for c in z['column_order'].tolist()})


def read_tables_text(directory, prefix):
    """Rebuild the lookup tables from csv.gz + json only (exact: 17 significant digits, round-trip float parser)."""
    d = Path(directory)
    state = json.loads((d/f'{prefix}_state.json').read_text(encoding='utf-8'))
    tables = dict(region={}, rates={}, categories={c: pd.Index(v) for c, v in state['categories'].items()})
    for _, name in LEVELS:
        summary = pd.read_csv(d/f'{prefix}_table_region_{name}.csv.gz', float_precision='round_trip', index_col='level')
        summary = summary.astype({'n_buildings': 'int64', 'weak_ratio': 'float64', 'rc_ratio': 'float64', 'mean_age': 'float64'})
        tables['region'][name] = dict(summary=summary, **state['region'][name])
    for k in ALL_KEYS:
        t = pd.read_csv(d/f'{prefix}_table_rate_{k}.csv.gz', float_precision='round_trip', dtype={'key': str}, index_col='key')
        t = t.astype({'count': 'int64', 'high_rate': 'float64', 'low_rate': 'float64'})
        tables['rates'][k] = dict(table=t, **state['rates'][k])
    return tables
