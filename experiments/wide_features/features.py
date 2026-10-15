"""T27 추가 블록: 고정 산식·train-only 통계·float32 상수/정확중복 정리."""
from __future__ import annotations

import hashlib
from itertools import combinations
import json

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_is_fitted

# 원본 고정 스키마의 상수만 복사한다. 실행기·전처리 모듈을 import하지 않는다.
GEO = ['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id']
NUMERIC = ['count_floors_pre_eq', 'age', 'area_percentage', 'height_percentage', 'count_families']
TEXT = ['land_surface_condition', 'foundation_type', 'roof_type', 'ground_floor_type',
        'other_floor_type', 'position', 'plan_configuration', 'legal_ownership_status']
SUPERSTRUCTURE = ['has_superstructure_' + value for value in (
    'adobe_mud', 'mud_mortar_stone', 'stone_flag', 'cement_mortar_stone', 'mud_mortar_brick',
    'cement_mortar_brick', 'timber', 'bamboo', 'rc_non_engineered', 'rc_engineered', 'other')]
SECONDARY_DETAIL = ['has_secondary_use_' + value for value in (
    'agriculture', 'hotel', 'rental', 'institution', 'school', 'industry', 'health_post',
    'gov_office', 'use_police', 'other')]
BINARY = SUPERSTRUCTURE + ['has_secondary_use'] + SECONDARY_DETAIL
FEATURES = (GEO + NUMERIC[:4] + TEXT[:7] + SUPERSTRUCTURE
            + ['legal_ownership_status', 'count_families', 'has_secondary_use'] + SECONDARY_DETAIL)

ARMS = ('WIDE_FEATURES',)
NUMERIC_TERMS = ('age_clean', 'floors', 'height', 'area', 'families')
NUMERIC_SOURCES = ('age', 'count_floors_pre_eq', 'height_percentage', 'area_percentage', 'count_families')
PRODUCTS = tuple(combinations(range(5), 2))
RATIOS = ((2, 1), (3, 1), (4, 1), (2, 3), (4, 3), (3, 4))
ZERO_DENOMINATORS = (1, 3, 4)
MATERIAL_COLUMNS = tuple(SUPERSTRUCTURE)
MATERIAL_PAIRS = tuple(combinations(range(11), 2))
CONTEXT_COLUMNS = ('count_floors_pre_eq', 'height_percentage', 'area_percentage')
PAIRS = (('foundation_roof', 'foundation_type', 'roof_type', 15),
         ('ground_other_floor', 'ground_floor_type', 'other_floor_type', 20),
         ('foundation_ground_floor', 'foundation_type', 'ground_floor_type', 25),
         ('roof_other_floor', 'roof_type', 'other_floor_type', 12),
         ('position_plan', 'position', 'plan_configuration', 40),
         ('foundation_land_surface', 'foundation_type', 'land_surface_condition', 15))
NUMERIC_NAMES = tuple(
    [f'{NUMERIC_TERMS[i]}__x__{NUMERIC_TERMS[j]}' for i, j in PRODUCTS]
    + [f'{NUMERIC_TERMS[i]}__div__{NUMERIC_TERMS[j]}' for i, j in RATIOS]
    + [f'{NUMERIC_TERMS[i]}__is_zero' for i in ZERO_DENOMINATORS]
    + [f'{MATERIAL_COLUMNS[i]}__AND__{MATERIAL_COLUMNS[j]}' for i, j in MATERIAL_PAIRS]
    + [f'{name}__x__floors' for name in MATERIAL_COLUMNS]
    + ['material_count', 'single_material', 'multiple_materials', 'secondary_detail_count']
    + ['geo1_median_' + name for name in CONTEXT_COLUMNS]
    + ['geo1_difference_' + name for name in CONTEXT_COLUMNS])


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _raw(X):
    _require(isinstance(X, pd.DataFrame) and X.columns.tolist() == FEATURES,
             '원본 38열의 정확한 스키마·순서가 필요합니다.')
    _require(not X.isna().any().any(), '결측 원본 입력은 허용하지 않습니다.')
    numbers = X.loc[:, [*GEO, *NUMERIC, *BINARY]].apply(pd.to_numeric, errors='raise')
    numeric = numbers.to_numpy(dtype=np.float64)
    _require(np.isfinite(numeric).all() and (numeric >= 0).all()
             and np.equal(numeric, np.floor(numeric)).all(), '수치 입력은 유한한 비음수 정수여야 합니다.')
    _require(np.isin(numbers[BINARY].to_numpy(), [0, 1]).all(), '이진 입력은 정확히 0/1이어야 합니다.')
    _require(all(X[c].map(lambda v: isinstance(v, (str, np.str_)) and bool(str(v).strip())).all()
                 for c in TEXT), '문자 범주는 결측·공백 없는 문자열이어야 합니다.')
    groups = np.asarray([int(v) for v in numbers['geo_level_1_id'].to_numpy()], dtype=object)
    return groups, numbers.loc[:, list(NUMERIC_SOURCES)].to_numpy(dtype=np.float64)


def _pair_values(X):
    result = np.empty((len(X), len(PAIRS)), dtype=object)
    for j, (_, left, right, _) in enumerate(PAIRS):
        result[:, j] = [json.dumps([str(a).strip(), str(b).strip()], ensure_ascii=False, separators=(',', ':'))
                        for a, b in zip(X[left], X[right])]
    return result


def canonical_csr(block):
    """추가 후보의 비교·저장 기준. 전체 C 행렬을 밀집화하지 않는다."""
    try:
        with np.errstate(over='raise', invalid='raise'):
            result = sparse.csr_matrix(block, dtype=np.float32)
            result.sum_duplicates()
    except (FloatingPointError, OverflowError) as exc:
        raise ValueError('추가 후보를 유한한 float32로 표현할 수 없습니다.') from exc
    _require(np.isfinite(result.data).all(), '추가 후보의 float32 값이 유한하지 않습니다.')
    result.eliminate_zeros()
    result.sort_indices()
    return result


def filter_candidates(block, names):
    """outer-train 후보만 검사하며 대표는 앞선 보존 비상수 열이다."""
    matrix = canonical_csr(block).tocsc()
    _require(matrix.shape[0] > 0 and matrix.shape[1] == len(names), '필터 후보 행수·이름 수 불일치')
    records, keep, buckets = [], [], {}
    for j, name in enumerate(names):
        a, b = matrix.indptr[j:j+2]
        indices, data = matrix.indices[a:b], matrix.data[a:b]
        support = len(data)
        constant = support == 0 or (support == matrix.shape[0] and np.all(data == data[0]))
        value = 0. if support == 0 else float(data[0]) if constant else None
        reason, duplicate, retained = None, None, None
        if constant:
            reason = 'constant_zero' if support == 0 else 'constant_nonzero'
        else:
            # 해시는 후보를 좁히는 용도이고 실제 제거는 배열의 정확한 동일성으로 결정한다.
            digest = hashlib.sha256(np.asarray(indices, dtype=np.int64).tobytes()+data.tobytes()).hexdigest()
            for prior in buckets.get(digest, []):
                x, z = matrix.indptr[prior:prior+2]
                if np.array_equal(indices, matrix.indices[x:z]) and np.array_equal(data, matrix.data[x:z]):
                    duplicate = prior
                    reason = 'exact_duplicate'
                    break
            if duplicate is None:
                retained = len(keep)
                keep.append(j)
                buckets.setdefault(digest, []).append(j)
        records.append(dict(index=j, name=str(name), support=int(support), constant_value=value,
                            removal_reason=reason, duplicate_of_index=duplicate, retained_index=retained))
    return np.asarray(keep, dtype=np.int64), records


class WideFeatures(BaseEstimator, TransformerMixin):
    """기존 C는 실행기가 보존하며 이 클래스는 추가 블록만 생성한다."""

    def __init__(self, arm='WIDE_FEATURES'):
        self.arm = arm

    def _validate(self, X, fitted=False):
        _require(self.arm in ARMS, '등록하지 않은 T27 피처 팔입니다.')
        if fitted:
            check_is_fitted(self, ['keep_indices_', 'encoder_', 'arm_'])
            _require(self.arm == self.arm_, 'fit 후 arm을 변경할 수 없습니다.')
        return _raw(X)

    def _numeric(self, X, groups, values):
        clean = values.copy()
        clean[:, 0] = np.where(clean[:, 0] == 995, self.median_, clean[:, 0])
        materials = X.loc[:, list(MATERIAL_COLUMNS)].to_numpy(dtype=np.float64)
        medians = np.asarray([self.group_medians_.get(g, self.global_medians_) for g in groups], dtype=np.float64)
        context = values[:, [1, 2, 3]]
        try:
            with np.errstate(over='raise', invalid='raise', divide='raise'):
                columns = [clean[:, i]*clean[:, j] for i, j in PRODUCTS]
                columns += [np.divide(clean[:, i], clean[:, j], out=np.zeros(len(X), dtype=np.float64),
                                      where=clean[:, j] != 0) for i, j in RATIOS]
                columns += [(clean[:, i] == 0).astype(np.float64) for i in ZERO_DENOMINATORS]
                columns += [materials[:, i]*materials[:, j] for i, j in MATERIAL_PAIRS]
                columns += [materials[:, i]*clean[:, 1] for i in range(11)]
                count = materials.sum(axis=1)
                columns += [count, (count == 1).astype(np.float64), (count >= 2).astype(np.float64),
                            X.loc[:, SECONDARY_DETAIL].to_numpy(dtype=np.float64).sum(axis=1)]
                columns += [medians[:, j] for j in range(3)]
                columns += [(context-medians)[:, j] for j in range(3)]
                result = np.column_stack(columns)
        except (FloatingPointError, OverflowError) as exc:
            raise ValueError('추가 후보의 float64 산술 결과가 유한하지 않습니다.') from exc
        _require(result.shape == (len(X), 95) and np.isfinite(result).all(), '수치 후보 95열 산식·유한값 불일치')
        return result

    def _candidates(self, X, groups, values):
        numeric = canonical_csr(self._numeric(X, groups, values))
        encoded = self.encoder_.transform(_pair_values(X))
        return canonical_csr(sparse.hstack([numeric, encoded], format='csr', dtype=np.float32))

    def fit(self, X, y=None):
        groups, values = self._validate(X)
        _require(len(X) > 0, '학습 입력은 비어 있을 수 없습니다.')
        regular = values[values[:, 0] != 995, 0]
        _require(len(regular) > 0, '비995 학습 연령이 없어 중앙값을 계산할 수 없습니다.')
        self.median_ = float(np.median(regular))  # C와 같이 0을 포함하며 y를 읽지 않는다.
        context = values[:, [1, 2, 3]]
        self.global_medians_ = np.median(context, axis=0)
        self.group_ids_ = sorted(set(groups))
        self.group_medians_ = {g: np.median(context[groups == g], axis=0) for g in self.group_ids_}
        self.group_counts_ = {g: int(np.count_nonzero(groups == g)) for g in self.group_ids_}
        self.encoder_ = OneHotEncoder(handle_unknown='ignore', sparse_output=True, dtype=np.float32)
        self.encoder_.fit(_pair_values(X))
        for vocabulary, (_, _, _, upper) in zip(self.encoder_.categories_, PAIRS):
            _require(len(vocabulary) <= upper, '관측 학습 조합 어휘가 사전 범주 상한을 초과했습니다.')
        names = [*NUMERIC_NAMES, *[name+'__'+str(v) for (name, _, _, _), vocabulary
                                  in zip(PAIRS, self.encoder_.categories_) for v in vocabulary]]
        _require(len(NUMERIC_NAMES) == 95 and len(names) <= 222, '사전 후보 수·범주 상한 변경')
        candidates = self._candidates(X, groups, values)
        self.keep_indices_, self.candidate_records_ = filter_candidates(candidates, names)
        self.candidate_names_ = np.asarray(names, dtype=object)
        self.output_features_ = self.candidate_names_[self.keep_indices_].copy()
        self.feature_names_in_ = np.asarray(FEATURES, dtype=object)
        self.n_features_in_ = len(FEATURES)
        self.n_train_rows_ = len(X)
        self.arm_ = self.arm
        return self

    def transform(self, X):
        groups, values = self._validate(X, fitted=True)
        if len(X) == 0:
            return sparse.csr_matrix((0, len(self.keep_indices_)), dtype=np.float32)
        block = self._candidates(X, groups, values)
        _require(block.shape[1] == len(self.candidate_names_), '검증 후보 폭 변경')
        return canonical_csr(block[:, self.keep_indices_])

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, ['output_features_', 'arm_'])
        _require(self.arm == self.arm_, 'fit 후 arm 변경 금지')
        if input_features is not None:
            _require(list(input_features) == FEATURES, '원본 입력 열 순서 불일치')
        return self.output_features_.copy()

    def unknown_counts(self, X):
        groups, _ = self._validate(X, fitted=True)
        values = _pair_values(X)
        rows = len(X)
        result = {}
        for j, (name, _, _, _) in enumerate(PAIRS):
            count = int(np.count_nonzero(~np.isin(values[:, j], self.encoder_.categories_[j])))
            result[name] = dict(count=count, rate=count/rows if rows else None, rows=rows)
        count = sum(g not in self.group_medians_ for g in groups)
        result['geo_level_1_id'] = dict(count=count, rate=count/rows if rows else None, rows=rows)
        return result

    def metadata(self):
        check_is_fitted(self, ['output_features_', 'arm_'])
        _require(self.arm == self.arm_, 'fit 후 arm 변경 금지')
        return dict(arm=self.arm_, input_columns=self.feature_names_in_.tolist(),
            output_features=self.output_features_.tolist(), n_train_rows=self.n_train_rows_, age_median=self.median_,
            candidate_logical_count=101, candidate_numeric_count=95, candidate_categorical_count=6,
            candidate_encoded_count=len(self.candidate_names_), retained_count=len(self.keep_indices_),
            keep_indices=self.keep_indices_.tolist(), candidate_names=self.candidate_names_.tolist(),
            candidates=[dict(item) for item in self.candidate_records_],
            pairs=[dict(name=name, left=left, right=right, upper_bound=upper,
                        vocabulary=[json.loads(str(v)) for v in vocabulary])
                   for (name, left, right, upper), vocabulary in zip(PAIRS, self.encoder_.categories_)],
            geo_context=dict(group_column='geo_level_1_id', context_columns=list(CONTEXT_COLUMNS),
                global_medians=self.global_medians_.tolist(),
                groups=[dict(geo_level_1_id=int(g), rows=self.group_counts_[g], medians=self.group_medians_[g].tolist())
                        for g in self.group_ids_], training_rows_include_self=True, fallback='outer_train_global_medians'),
            scaling='none', target_used=False, statistics_scope='outer_train_only',
            filtering=dict(comparison_dtype='float32', canonical='sum_duplicates_eliminate_zeros_sort_indices',
                scope='additional_block_only', constant_before_duplicate=True, first_nonconstant_candidate_wins=True,
                rare_frequency_threshold=None, compare_with_C=False))
