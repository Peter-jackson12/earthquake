"""T34 derived-column bundle: pure functions and fitted-state objects, no file I/O, no labels.

Columns (plan section 2; names follow the plan, order is sealed):
  numeric  #1-#6   height_area_ratio, height_per_floor, material_count, secondary_use_count, concrete_count, mud_count
  age mean #11-#13 geo_level_{1,2,3}_id_age_mean            (S1 / S2 only)
  category #7-#10  foundation_roof, ground_other_floor, foundation_ground, surface_condition
S0 = 6 numeric + 4 category (10 columns), S1 / S2 = 6 numeric + 3 age mean + 4 category (13 columns).

Every statistic (region age tables, fallback means, category vocabulary, scaler) is learned in `fit(X_train)`, whose
signature has no label argument. Validation rows are only looked up (`transform`). The region mean uses the raw
`age` column (before the C-policy replacement of 995), one table per geo level keyed by that level's own column.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from preprocess import GEO, SECONDARY_DETAIL, SUPERSTRUCTURE


class FeatureError(ValueError):
    pass


def _check(ok, message):
    if not ok:
        raise FeatureError(message)


SUP = 'has_superstructure_'
CONCRETE = [SUP + n for n in ('cement_mortar_stone', 'cement_mortar_brick', 'rc_non_engineered', 'rc_engineered')]
MUD = [SUP + n for n in ('adobe_mud', 'mud_mortar_stone', 'mud_mortar_brick')]
NUMERIC_COLS = ['height_area_ratio', 'height_per_floor', 'material_count', 'secondary_use_count', 'concrete_count',
                'mud_count']
AGE_COLS = [f'{g}_age_mean' for g in GEO]
COMBO_COLS = ['foundation_roof', 'ground_other_floor', 'foundation_ground', 'surface_condition']
COMBO_SOURCES = {'foundation_roof': ('foundation_type', 'roof_type'),
                 'ground_other_floor': ('ground_floor_type', 'other_floor_type'),
                 'foundation_ground': ('foundation_type', 'ground_floor_type'),
                 'surface_condition': ('land_surface_condition', 'ground_floor_type')}
ARMS = ('S0', 'S1', 'S2')
KINDS = ('lgbm', 'lr')
AGE_SENTINEL = 995
SEP = '|'
MAX_COMBINATIONS = 75          # 15 + 20 + 25 + 15 from preprocess.ALLOWED_TEXT


def uses_age_means(arm):
    _check(arm in ARMS, f'Unknown arm {arm}')
    return arm in ('S1', 'S2')


def numeric_names(arm):
    return NUMERIC_COLS + (AGE_COLS if uses_age_means(arm) else [])


def derived_names(arm):
    return numeric_names(arm) + COMBO_COLS


def n_numeric(arm):
    return len(numeric_names(arm))


def n_derived(arm):
    return len(derived_names(arm))


def categorical_positions(arm, base_width):
    """Indices of the 4 combination code columns when the block is appended after `base_width` model columns."""
    start = base_width + n_numeric(arm)
    return list(range(start, start + len(COMBO_COLS)))


# ---------------------------------------------------------------------------------------------- stateless columns
def raw_numeric(X):
    """The 6 numeric columns in float64 (no statistic, no epsilon, no clipping)."""
    area = X['area_percentage'].to_numpy(np.float64)
    _check((area != 0).all(), 'area_percentage == 0 found; height_area_ratio has no pre-registered zero rule '
                              '(design revision required before sealing)')
    floors = X['count_floors_pre_eq'].to_numpy(np.float64)
    _check((floors >= 1).all(), 'count_floors_pre_eq must be at least 1')
    height = X['height_percentage'].to_numpy(np.float64)
    out = np.column_stack([
        height / area,
        height / floors,
        X[SUPERSTRUCTURE].to_numpy(np.float64).sum(axis=1),
        X[SECONDARY_DETAIL].to_numpy(np.float64).sum(axis=1),
        X[CONCRETE].to_numpy(np.float64).sum(axis=1),
        X[MUD].to_numpy(np.float64).sum(axis=1),
    ])
    _check(np.isfinite(out).all(), 'Nonfinite raw derived numeric value')
    return out


def to_float32(values, what='derived block'):
    out = np.asarray(values, dtype=np.float64).astype(np.float32)
    _check(np.isfinite(out).all(), f'Nonfinite value after float32 conversion: {what}')
    return out


def combos(X):
    """4 combination columns as collision-free strings `a|b` (single-character vocabularies; the separator may not occur)."""
    out = {}
    for name, (a, b) in COMBO_SOURCES.items():
        sa = pd.Series(X[a].astype(str).to_numpy(object))
        sb = pd.Series(X[b].astype(str).to_numpy(object))
        for values, col in ((sa, a), (sb, b)):
            _check(not values.str.contains(SEP, regex=False).any(), f'Separator {SEP!r} occurs in {col}')
        out[name] = (sa + SEP + sb).to_numpy(object)
    return pd.DataFrame(out, columns=COMBO_COLS)


# ---------------------------------------------------------------------------------------------- region age table
class RegionAgeTable:
    """Per geo level mean of the raw `age` over the training fold. S1 uses every row (995 included); S2 excludes
    `age == 995`. Rows (train or validation) whose region has no usable training age get the fallback mean (S1: all
    training ages; S2: all non-995 training ages). Label-free by construction."""

    def __init__(self, arm):
        _check(arm in ('S1', 'S2'), 'RegionAgeTable needs S1 or S2')
        self.arm = arm

    def fit(self, X_train):
        age = X_train['age'].to_numpy(np.float64)
        keep = np.ones(len(age), dtype=bool) if self.arm == 'S1' else age != AGE_SENTINEL
        _check(keep.any(), 'No usable training ages')
        self.fallback_ = float(age[keep].mean())
        self.tables_, self.seen_, self.counts_ = {}, {}, {}
        for g in GEO:
            key = X_train[g].to_numpy(object)
            self.tables_[g] = pd.Series(age[keep]).groupby(key[keep]).mean()
            self.seen_[g] = frozenset(pd.unique(key))
            self.counts_[g] = pd.Series(key).value_counts()
        self.n_train_rows_ = int(len(age))
        return self

    def transform(self, X, return_stats=False):
        n = len(X)
        out = np.empty((n, len(GEO)), dtype=np.float64)
        stats = {}
        for j, g in enumerate(GEO):
            key = pd.Series(X[g].to_numpy(object))
            mapped = key.map(self.tables_[g])
            missing = mapped.isna().to_numpy()
            unseen = ~key.isin(self.seen_[g]).to_numpy()
            out[:, j] = mapped.fillna(self.fallback_).to_numpy(np.float64)
            stats[g] = dict(rows=int(n), fallback_rows=int(missing.sum()), unseen_region_rows=int(unseen.sum()),
                            known_region_without_usable_age_rows=int((missing & ~unseen).sum()),
                            unseen_regions=int(key[unseen].nunique()))
        _check(np.isfinite(out).all(), 'Nonfinite region age mean')
        return (out, stats) if return_stats else out

    def summary(self):
        s = {}
        for g in GEO:
            t = self.tables_[g]
            s[g] = dict(training_regions=len(self.seen_[g]), regions_with_usable_age=int(len(t)),
                        regions_without_usable_age=int(len(self.seen_[g]) - len(t)),
                        table_min=float(t.min()), table_max=float(t.max()))
        return dict(arm=self.arm, fallback_mean=self.fallback_, n_train_rows=self.n_train_rows_, levels=s)


# ---------------------------------------------------------------------------------------------- fitted bundle
class DerivedBundle:
    """Fitted-on-training-fold block appended after a model's base matrix.

    kind 'lgbm': dense float32 [numeric (unscaled) | combination ordinal codes (NaN = unseen in training)].
    kind 'lr'  : CSR float32   [StandardScaler(train) numeric | one-hot combinations (unseen -> all zero)].
    """

    def __init__(self, arm, kind):
        _check(arm in ARMS, f'Unknown arm {arm}')
        _check(kind in KINDS, f'Unknown kind {kind}')
        self.arm, self.kind = arm, kind

    # -- fit uses the training rows only (no label argument exists)
    def fit(self, X_train):
        _check(len(X_train) > 0, 'Empty training fold')
        self.n_train_rows_ = int(len(X_train))
        self.age_ = RegionAgeTable(self.arm).fit(X_train) if uses_age_means(self.arm) else None
        numeric = self._numeric(X_train)
        combo = combos(X_train)
        self.ord_ = OrdinalEncoder(categories='auto', handle_unknown='use_encoded_value', unknown_value=np.nan,
                                   encoded_missing_value=np.nan, dtype=np.float32).fit(combo)
        _check(sum(len(c) for c in self.ord_.categories_) <= MAX_COMBINATIONS, 'Vocabulary exceeds 75 combinations')
        if self.kind == 'lr':
            self.scaler_ = StandardScaler().fit(numeric)
            self.ohe_ = OneHotEncoder(categories='auto', handle_unknown='ignore', sparse_output=True,
                                      dtype=np.float32).fit(combo)
        return self

    def _numeric(self, X, stats=None):
        raw = raw_numeric(X)
        if self.age_ is not None:
            ages, s = self.age_.transform(X, return_stats=True)
            if stats is not None:
                stats.update(s)
            raw = np.column_stack([raw, ages])
        return to_float32(raw, 'numeric block')

    def combo_codes(self, X):
        return np.asarray(self.ord_.transform(combos(X)), dtype=np.float32)

    def transform(self, X, return_stats=False):
        stats = {}
        numeric = self._numeric(X, stats)
        combo = combos(X)
        if self.kind == 'lgbm':
            block = np.hstack([numeric, np.asarray(self.ord_.transform(combo), dtype=np.float32)]).astype(np.float32)
        else:
            scaled = np.asarray(self.scaler_.transform(numeric), dtype=np.float32)
            _check(np.isfinite(scaled).all(), 'Nonfinite scaled derived value')
            block = sparse.hstack([sparse.csr_matrix(scaled), self.ohe_.transform(combo)], format='csr',
                                  dtype=np.float32)
        return (block, stats) if return_stats else block

    # -- descriptions (aggregate only)
    @property
    def vocabulary(self):
        return {c: [str(v) for v in cats] for c, cats in zip(COMBO_COLS, self.ord_.categories_)}

    def feature_names(self):
        if self.kind == 'lgbm':
            return derived_names(self.arm)
        names = list(numeric_names(self.arm))
        for c, cats in zip(COMBO_COLS, self.ohe_.categories_):
            names += [f'{c}={v}' for v in cats]
        return names

    def summary(self):
        s = dict(arm=self.arm, kind=self.kind, n_train_rows=self.n_train_rows_,
                 vocabulary_sizes={c: len(v) for c, v in self.vocabulary.items()},
                 total_combinations=int(sum(len(v) for v in self.vocabulary.values())))
        if self.age_ is not None:
            s['region_age'] = self.age_.summary()
        if self.kind == 'lr':
            s['scaler_mean'] = self.scaler_.mean_.tolist()
            s['scaler_scale'] = self.scaler_.scale_.tolist()
        return s


def state_signature(bundle):
    """Comparable plain-python snapshot of everything the bundle learned."""
    sig = dict(arm=bundle.arm, kind=bundle.kind, n=bundle.n_train_rows_, vocabulary=bundle.vocabulary)
    if bundle.age_ is not None:
        a = bundle.age_
        sig['fallback'] = a.fallback_
        sig['tables'] = {g: {str(k): float(v) for k, v in a.tables_[g].items()} for g in GEO}
        sig['seen'] = {g: sorted(a.seen_[g]) for g in GEO}
    if bundle.kind == 'lr':
        sig['scaler'] = (bundle.scaler_.mean_.tolist(), bundle.scaler_.scale_.tolist())
    return sig


def verify_fitted_on(bundle, X_train):
    """Refit a fresh bundle on exactly `X_train` and compare every learned value. A bundle that saw any other
    rows (for example validation rows) differs and fails."""
    fresh = DerivedBundle(bundle.arm, bundle.kind).fit(X_train)
    _check(state_signature(fresh) == state_signature(bundle), 'Bundle state is not the training-fold-only state')
    return True


# ---------------------------------------------------------------------------------------------- independent reference
def reference_block(X_train, X_valid, arm):
    """Independent pandas/python formulation (different code path from DerivedBundle) of the unscaled block."""
    def numeric(X):
        sup = [c for c in X.columns if c.startswith('has_superstructure_')]
        use = [c for c in X.columns if c.startswith('has_secondary_use_')]       # trailing '_' excludes the aggregate
        pre = 'has_superstructure_'
        f = lambda cols: X[cols].astype('float64').sum(axis=1)                    # noqa: E731
        cols = {
            'height_area_ratio': X['height_percentage'].astype('float64') / X['area_percentage'].astype('float64'),
            'height_per_floor': X['height_percentage'].astype('float64') / X['count_floors_pre_eq'].astype('float64'),
            'material_count': f(sup), 'secondary_use_count': f(use),
            'concrete_count': f([pre + s for s in ('cement_mortar_stone', 'cement_mortar_brick', 'rc_non_engineered',
                                                    'rc_engineered')]),
            'mud_count': f([pre + s for s in ('adobe_mud', 'mud_mortar_stone', 'mud_mortar_brick')])}
        return pd.DataFrame(cols)

    tr, va = numeric(X_train), numeric(X_valid)
    ages_tr, ages_va = {}, {}
    if uses_age_means(arm):
        age = X_train['age'].astype('float64')
        mask = pd.Series(True, index=age.index) if arm == 'S1' else age.ne(AGE_SENTINEL)
        fallback = float(age[mask].mean())
        for g in GEO:
            lookup = age[mask].groupby(X_train[g][mask].to_numpy()).mean().to_dict()
            ages_tr[g + '_age_mean'] = X_train[g].map(lookup).fillna(fallback).astype('float64')
            ages_va[g + '_age_mean'] = X_valid[g].map(lookup).fillna(fallback).astype('float64')
        if arm == 'S1':                      # self-inclusive transform('mean') gives the same training values
            for g in GEO:
                other = age.groupby(X_train[g].to_numpy()).transform('mean')
                _check(np.allclose(other.to_numpy(), ages_tr[g + '_age_mean'].to_numpy(), rtol=1e-12, atol=1e-9),
                       'Reference S1 mean mismatch between two pandas formulations')
        tr = pd.concat([tr, pd.DataFrame(ages_tr, index=tr.index)], axis=1)
        va = pd.concat([va, pd.DataFrame(ages_va, index=va.index)], axis=1)
    tr, va = tr[numeric_names(arm)], va[numeric_names(arm)]

    def tuples(X):
        return {name: list(zip(X[a].astype(str), X[b].astype(str))) for name, (a, b) in COMBO_SOURCES.items()}

    ct, cv = tuples(X_train), tuples(X_valid)
    vocab = {name: {a + SEP + b for a, b in ct[name]} for name in COMBO_COLS}
    return dict(train_numeric=tr.to_numpy(np.float64), valid_numeric=va.to_numpy(np.float64),
                train_combo={n: [a + SEP + b for a, b in t] for n, t in ct.items()},
                valid_combo={n: [a + SEP + b for a, b in t] for n, t in cv.items()}, vocab=vocab)


def _decode(bundle, codes, name_index):
    cats = np.asarray(bundle.ord_.categories_[name_index], dtype=object)
    out = np.full(len(codes), None, dtype=object)
    known = ~np.isnan(codes)
    out[known] = cats[codes[known].astype(np.int64)]
    return out


def check_ohe_matches_codes(bundle, X, chunk=20000):
    """One-hot block equals the ordinal codes (row by row, chunked; never densifies more than `chunk` rows)."""
    codes = bundle.combo_codes(X)
    block = bundle.transform(X)
    k = n_numeric(bundle.arm)
    ohe = block[:, k:].tocsr()
    sizes = [len(c) for c in bundle.ohe_.categories_]
    offsets = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    _check(ohe.shape[1] == sum(sizes), 'One-hot width differs from the vocabulary')
    for start in range(0, len(codes), chunk):
        stop = min(start + chunk, len(codes))
        dense = ohe[start:stop].toarray()
        expected = np.zeros_like(dense)
        for j in range(len(sizes)):
            c = codes[start:stop, j]
            ok = ~np.isnan(c)
            expected[np.flatnonzero(ok), offsets[j] + c[ok].astype(np.int64)] = 1.0
        _check(np.array_equal(dense, expected), 'One-hot block differs from ordinal codes')
    return True


def check_against_reference(bundle, X_train, X_valid):
    """Plan section 7 item (ii)+(iv): the block equals an independent pandas computation made from the training rows
    only (numeric within float32 resolution, vocabulary and codes exactly). Returns the maximum differences."""
    ref = reference_block(X_train, X_valid, bundle.arm)
    k = n_numeric(bundle.arm)
    ref_t32 = ref['train_numeric'].astype(np.float32).astype(np.float64)
    if bundle.kind == 'lr':                                     # scaler statistics learned from the training fold only
        mean, scale = ref_t32.mean(axis=0), ref_t32.std(axis=0, ddof=0)
        _check(np.allclose(bundle.scaler_.mean_, mean, rtol=1e-6, atol=1e-5), 'Scaler mean differs from the training-fold mean')
        _check(np.allclose(bundle.scaler_.scale_, scale, rtol=1e-5, atol=1e-5), 'Scaler scale differs from the training-fold scale')
    result = {}
    for tag, X, ref_num, ref_combo in (('train', X_train, ref['train_numeric'], ref['train_combo']),
                                       ('valid', X_valid, ref['valid_numeric'], ref['valid_combo'])):
        block = bundle.transform(X)
        if bundle.kind == 'lgbm':
            diff = float(np.max(np.abs(block[:, :k].astype(np.float64) - ref_num) / np.maximum(1.0, np.abs(ref_num))))
            _check(diff <= 5e-6, f'{tag}: derived numeric differs from the independent computation ({diff})')
        else:                                                   # compare in the scaled space (float32 output)
            expected = (ref_num.astype(np.float32).astype(np.float64) - mean) / scale
            got = block[:, :k].toarray().astype(np.float64)
            diff = float(np.max(np.abs(got - expected)))
            _check(np.allclose(got, expected, rtol=1e-5, atol=2e-5), f'{tag}: scaled derived numeric differs ({diff})')
        result[f'{tag}_numeric_max_diff'] = diff
        result[f'{tag}_numeric_max_relative_diff'] = diff
        codes = bundle.combo_codes(X)
        for j, name in enumerate(COMBO_COLS):
            decoded = _decode(bundle, codes[:, j], j)
            expected_c = np.array([v if v in ref['vocab'][name] else None for v in ref_combo[name]], dtype=object)
            _check(all(a == b for a, b in zip(decoded, expected_c)), f'{tag}: {name} codes differ from the reference')
    for j, name in enumerate(COMBO_COLS):
        _check(set(bundle.vocabulary[name]) == ref['vocab'][name], f'{name} vocabulary is not the training vocabulary')
    if bundle.kind == 'lr':
        check_ohe_matches_codes(bundle, X_valid)
    verify_fitted_on(bundle, X_train)
    return result


def check_arm_relations(X_train, X_valid):
    """Plan section 7 item (iii): S0 = S1 without the 3 age mean columns; S1 and S2 differ only in those 3 columns."""
    b = {a: DerivedBundle(a, 'lgbm').fit(X_train) for a in ARMS}
    k0, k1 = n_numeric('S0'), n_numeric('S1')
    result = {}
    for tag, X in (('train', X_train), ('valid', X_valid)):
        m = {a: b[a].transform(X) for a in ARMS}
        s0, s1, s2 = m['S0'], m['S1'], m['S2']
        same_numeric = np.array_equal(s0[:, :k0], s1[:, :k0]) and np.array_equal(s1[:, :k0], s2[:, :k0])
        same_combo = (np.array_equal(s0[:, k0:], s1[:, k1:], equal_nan=True)
                      and np.array_equal(s1[:, k1:], s2[:, k1:], equal_nan=True))
        _check(same_numeric and same_combo, f'{tag}: arms differ outside the region age mean columns')
        result[tag] = dict(non_age_columns_identical=True,
                           s1_s2_age_columns_equal=bool(np.array_equal(s1[:, k0:k1], s2[:, k0:k1])),
                           s1_s2_changed_rows={g: int((s1[:, k0 + j] != s2[:, k0 + j]).sum())
                                               for j, g in enumerate(GEO)})
    return result


# ---------------------------------------------------------------------------------------------- diagnostics
SIZE_BINS = [(1, 1, '1'), (2, 4, '2-4'), (5, 9, '5-9'), (10, 24, '10-24'), (25, 99, '25-99'), (100, None, '100+')]


def _corr(a, b):
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _group_stats(mean_col, age):
    if len(age) == 0:
        return dict(rows=0, correlation=None, mean_abs_diff=None)
    return dict(rows=int(len(age)), correlation=_corr(mean_col, age), mean_abs_diff=float(np.mean(np.abs(mean_col - age))))


def self_inclusion_diagnostics(X_train, X_valid, arm):
    """Relation between the region age mean column and the row's own raw age (training rows contain themselves in the
    mean, validation rows do not). Groups: train / train age==995 / train age!=995 and the same for valid.
    The geo3 mean absolute difference is also split by the geo3 training-fold size bin (valid unseen = size 0)."""
    _check(uses_age_means(arm), 'Only S1 / S2 have region age means')
    table = RegionAgeTable(arm).fit(X_train)
    out = {g: {} for g in GEO}
    by_size = {}
    for part, X in (('train', X_train), ('valid', X_valid)):
        means = table.transform(X)
        age = X['age'].to_numpy(np.float64)
        is995 = age == AGE_SENTINEL
        for j, g in enumerate(GEO):
            for name, mask in ((part, np.ones(len(age), bool)), (part + '_age995', is995), (part + '_age_not995', ~is995)):
                out[g][name] = _group_stats(means[mask, j], age[mask])
        size = pd.Series(X[GEO[2]].to_numpy(object)).map(table.counts_[GEO[2]]).fillna(0).to_numpy(np.int64)
        for lo, hi, label in [(0, 0, '0 (unseen)')] + SIZE_BINS:
            sel = (size >= lo) & ((size <= hi) if hi is not None else True)
            for name, mask in ((part, sel), (part + '_age995', sel & is995), (part + '_age_not995', sel & ~is995)):
                by_size.setdefault(label, {})[name] = _group_stats(means[mask, 2], age[mask])
    out['geo_level_3_id_by_training_size'] = by_size
    return out


def mixed_995_summary(X_train, X_valid):
    """Per level: regions that contain age-995 rows in the training fold, S1/S2 mean difference share (aggregate)."""
    age = X_train['age'].to_numpy(np.float64)
    is995 = age == AGE_SENTINEL
    s1 = RegionAgeTable('S1').fit(X_train)
    s2 = RegionAgeTable('S2').fit(X_train)
    out = {}
    m1t, m2t = s1.transform(X_train), s2.transform(X_train)
    m1v, m2v = s1.transform(X_valid), s2.transform(X_valid)
    for j, g in enumerate(GEO):
        key = pd.Series(X_train[g].to_numpy(object))
        has995 = key[is995].unique()
        regions = key.nunique()
        mixed = pd.Series(is995).groupby(key.to_numpy()).agg(['any', 'all'])
        out[g] = dict(regions=int(regions), regions_with_995=int(len(has995)),
                      regions_mixed_995_and_other=int((mixed['any'] & ~mixed['all']).sum()),
                      regions_all_995=int(mixed['all'].sum()),
                      train_rows_s1_ne_s2=float(np.mean(m1t[:, j] != m2t[:, j])),
                      valid_rows_s1_ne_s2=float(np.mean(m1v[:, j] != m2v[:, j])),
                      train_mean_max_s1=float(m1t[:, j].max()), valid_mean_max_s1=float(m1v[:, j].max()),
                      train_mean_max_s2=float(m2t[:, j].max()), valid_mean_max_s2=float(m2v[:, j].max()))
    return out


def has_995_region_mask(X_train, X_valid):
    """Per level boolean over validation rows: region contains age-995 rows in the training fold."""
    age = X_train['age'].to_numpy(np.float64)
    out = {}
    for g in GEO:
        regions = pd.unique(X_train[g].to_numpy(object)[age == AGE_SENTINEL])
        out[g] = pd.Series(X_valid[g].to_numpy(object)).isin(regions).to_numpy()
    return out
