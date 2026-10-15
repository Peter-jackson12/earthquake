"""T22의 고정 재료·구조 추가 블록. 실행기 import와 데이터 파일 접근은 없다."""
from __future__ import annotations

from itertools import combinations
import json

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_is_fitted

ARMS = ('MATERIAL_PAIRS', 'STRUCTURE_FLOORS')
MATERIAL_COLUMNS = (
    'has_superstructure_adobe_mud', 'has_superstructure_mud_mortar_stone',
    'has_superstructure_stone_flag', 'has_superstructure_cement_mortar_stone',
    'has_superstructure_mud_mortar_brick', 'has_superstructure_cement_mortar_brick',
    'has_superstructure_timber', 'has_superstructure_bamboo',
    'has_superstructure_rc_non_engineered', 'has_superstructure_rc_engineered',
    'has_superstructure_other',
)
MATERIAL_PAIRS = tuple(combinations(MATERIAL_COLUMNS, 2))
STRUCTURE_COLUMNS = ('foundation_type', 'other_floor_type')
FLOOR_COLUMN = 'count_floors_pre_eq'


def _frame(X):
    if not isinstance(X, pd.DataFrame) or not X.columns.is_unique:
        raise ValueError('고유 열 이름을 가진 DataFrame이 필요합니다.')
    if not all(isinstance(column, str) for column in X.columns):
        raise ValueError('입력 열 이름은 문자열이어야 합니다.')


def _materials(X):
    values = X.loc[:, list(MATERIAL_COLUMNS)].apply(pd.to_numeric, errors='raise').to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or not np.isin(values, [0., 1.]).all():
        raise ValueError('11개 재료 플래그는 유한한 0 또는 1이어야 합니다.')
    return values.astype(np.float32)


def _floors(X):
    values = pd.to_numeric(X[FLOOR_COLUMN], errors='raise').to_numpy(dtype=np.float64)
    if (not np.isfinite(values).all() or (values < 0).any()
            or not np.equal(values, np.floor(values)).all()):
        raise ValueError('원본 층수는 0을 포함한 유한한 비음수 정수여야 합니다.')
    with np.errstate(over='raise', invalid='raise'):
        values = values.astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError('float32 층수는 유한해야 합니다.')
    return values


def _categories(X):
    values = X.loc[:, list(STRUCTURE_COLUMNS)].to_numpy(dtype=object)
    if any(not isinstance(v, (str, np.str_)) for v in values.ravel()):
        raise ValueError('구조 범주는 결측 없는 문자열이어야 합니다.')
    return values


def _csr(block):
    result = sparse.csr_matrix(block, dtype=np.float32)
    result.eliminate_zeros()
    result.sort_indices()
    if not np.isfinite(result.data).all():
        raise ValueError('추가 블록의 float32 값은 유한해야 합니다.')
    return result


class MaterialStructureInteractions(BaseEstimator, TransformerMixin):
    """재료의 55개 이진 곱 또는 학습 범주별 원핫×원본 층수. 스케일링 없음."""

    def __init__(self, arm='MATERIAL_PAIRS'):
        self.arm = arm

    def _validate(self, X, fitted=False):
        _frame(X)
        if self.arm not in ARMS:
            raise ValueError('등록하지 않은 T22 피처 팔입니다.')
        required = MATERIAL_COLUMNS if self.arm == ARMS[0] else (*STRUCTURE_COLUMNS, FLOOR_COLUMN)
        missing = [column for column in required if column not in X.columns]
        if missing:
            raise ValueError('추가 피처 입력 열 누락: ' + ', '.join(missing))
        if fitted:
            check_is_fitted(self, ['arm_', 'feature_names_in_', 'output_features_'])
            if self.arm != self.arm_:
                raise ValueError('fit 후 피처 팔을 변경할 수 없습니다.')
            if list(X.columns) != list(self.feature_names_in_):
                raise ValueError('fit 후 입력 열 순서 또는 스키마가 바뀌었습니다.')

    def fit(self, X, y=None):
        self._validate(X)
        if len(X) == 0:
            raise ValueError('학습 입력은 비어 있을 수 없습니다.')
        # y는 사용하지 않는다. A는 행별 공식만, B는 이 X의 범주 어휘만 사용한다.
        if self.arm == ARMS[0]:
            _materials(X)
            self.encoder_ = None
            names = [left + '__AND__' + right for left, right in MATERIAL_PAIRS]
        else:
            _floors(X)
            self.encoder_ = OneHotEncoder(handle_unknown='ignore', sparse_output=True, dtype=np.float32)
            self.encoder_.fit(_categories(X))
            names = [column + '__' + json.dumps(str(value), ensure_ascii=False) + '__x_floors'
                     for column, vocabulary in zip(STRUCTURE_COLUMNS, self.encoder_.categories_)
                     for value in vocabulary]
        self.arm_ = self.arm
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.n_train_rows_ = len(X)
        self.output_features_ = np.asarray(names, dtype=object)
        return self

    def transform(self, X):
        self._validate(X, fitted=True)
        if len(X) == 0:
            return sparse.csr_matrix((0, len(self.output_features_)), dtype=np.float32)
        if self.arm == ARMS[0]:
            flags = _materials(X)
            # 한 열씩 계산하므로 전체 55열의 밀집 행렬을 만들지 않는다.
            columns = [sparse.csr_matrix((flags[:, i] * flags[:, j]).reshape(-1, 1))
                       for i, j in combinations(range(len(MATERIAL_COLUMNS)), 2)]
            block = _csr(sparse.hstack(columns, format='csr', dtype=np.float32))
        else:
            floors = _floors(X)
            encoded = self.encoder_.transform(_categories(X))
            block = _csr(encoded.multiply(floors[:, None]))
        if block.shape != (len(X), len(self.output_features_)):
            raise ValueError('추가 블록의 행 또는 열 수가 바뀌었습니다.')
        return block

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, ['output_features_', 'arm_'])
        if self.arm != self.arm_:
            raise ValueError('fit 후 피처 팔을 변경할 수 없습니다.')
        if input_features is not None and list(input_features) != list(self.feature_names_in_):
            raise ValueError('입력 피처 이름이 fit 스키마와 다릅니다.')
        return self.output_features_.copy()

    def unknown_counts(self, X):
        self._validate(X, fitted=True)
        if self.arm == ARMS[0]:
            return {}
        values = _categories(X)
        result = {}
        for j, column in enumerate(STRUCTURE_COLUMNS):
            count = int(np.count_nonzero(~np.isin(values[:, j], self.encoder_.categories_[j])))
            result[column] = dict(count=count, rate=count/len(X) if len(X) else None, rows=len(X))
        return result

    def metadata(self):
        check_is_fitted(self, ['arm_', 'feature_names_in_', 'output_features_'])
        if self.arm != self.arm_:
            raise ValueError('fit 후 피처 팔을 변경할 수 없습니다.')
        material = self.arm_ == ARMS[0]
        categories = [] if material else [dict(column=column, vocabulary=vocabulary.tolist())
            for column, vocabulary in zip(STRUCTURE_COLUMNS, self.encoder_.categories_)]
        return dict(arm=self.arm_, input_columns=self.feature_names_in_.tolist(),
            output_features=self.output_features_.tolist(), n_train_rows=self.n_train_rows_,
            scaling='none', material_columns=list(MATERIAL_COLUMNS) if material else [],
            pairs=[list(pair) for pair in MATERIAL_PAIRS] if material else [], categories=categories)
