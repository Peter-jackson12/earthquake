"""T21의 추가 블록만 반환한다. 데이터 파일 접근이나 실행기 import는 없다."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

ARMS = ('AGE_HEIGHT_FLOORS', 'STRUCTURE_PAIRS')
PAIRS = (('foundation_roof', 'foundation_type', 'roof_type'),
         ('ground_other_floor', 'ground_floor_type', 'other_floor_type'))
NUMERIC = ('age', 'height_percentage', 'count_floors_pre_eq')


def _frame(X):
    if not isinstance(X, pd.DataFrame) or not X.columns.is_unique:
        raise ValueError('고유 열 이름을 가진 DataFrame이 필요합니다.')
    if not all(isinstance(c, str) for c in X.columns):
        raise ValueError('입력 열 이름은 문자열이어야 합니다.')


def _numbers(X):
    values = X.loc[:, list(NUMERIC)].apply(pd.to_numeric, errors='raise').to_numpy(dtype=np.float64)
    if (not np.isfinite(values).all() or (values < 0).any()
            or not np.equal(values, np.floor(values)).all()):
        raise ValueError('원본 연령·높이·층수는 유한한 비음수 정수여야 합니다.')
    return values


def _pair_values(X):
    """문자열 결합 대신 두 값의 JSON 배열로 튜플을 일대일 직렬화한다."""
    columns = []
    for _, left, right in PAIRS:
        a, b = X[left].to_numpy(), X[right].to_numpy()
        if any(not isinstance(v, (str, np.str_)) for v in a) or any(
                not isinstance(v, (str, np.str_)) for v in b):
            raise ValueError('조합 입력은 결측 없는 문자열 범주여야 합니다.')
        columns.append([json.dumps([str(x), str(y)], ensure_ascii=False, separators=(',', ':'))
                        for x, y in zip(a, b)])
    return np.asarray(columns, dtype=object).T.reshape(len(X), len(PAIRS))


def _csr(values):
    result = sparse.csr_matrix(values, dtype=np.float32)
    if not np.isfinite(result.data).all():
        raise ValueError('추가 블록의 float32 값은 모두 유한해야 합니다.')
    result.sort_indices()
    return result


class FeatureInteractions(BaseEstimator, TransformerMixin):
    """A: C 연령정책의 곱→학습 표준화, B: 학습 튜플 어휘의 희소 원핫."""

    def __init__(self, arm='AGE_HEIGHT_FLOORS'):
        self.arm = arm

    def _validate(self, X, fitted=False):
        _frame(X)
        if self.arm not in ARMS:
            raise ValueError('등록하지 않은 T21 피처 팔입니다.')
        required = NUMERIC if self.arm == ARMS[0] else tuple(c for _, a, b in PAIRS for c in (a, b))
        missing = [c for c in required if c not in X.columns]
        if missing:
            raise ValueError('추가 피처 입력 열 누락: ' + ', '.join(missing))
        if fitted:
            check_is_fitted(self, ['feature_names_in_', 'output_features_', 'arm_'])
            if self.arm != self.arm_:
                raise ValueError('fit 후 피처 팔을 변경할 수 없습니다.')
            if list(X.columns) != list(self.feature_names_in_):
                raise ValueError('fit 후 입력 열 순서 또는 스키마가 바뀌었습니다.')

    def _product(self, X):
        values = _numbers(X)
        age = np.where(values[:, 0] == 995, self.median_, values[:, 0])
        with np.errstate(over='raise', invalid='raise'):
            product = age * values[:, 1] * values[:, 2]
        if not np.isfinite(product).all():
            raise ValueError('곱 상호작용 값이 유한하지 않습니다.')
        return product.reshape(-1, 1)

    def fit(self, X, y=None):
        self._validate(X)
        if len(X) == 0:
            raise ValueError('학습 입력은 비어 있을 수 없습니다.')
        # y를 읽지 않는다. 모든 통계와 어휘는 이 fit의 X에서만 구한다.
        if self.arm == ARMS[0]:
            values = _numbers(X)
            regular = values[values[:, 0] != 995, 0]
            if len(regular) == 0:
                raise ValueError('995 이외의 학습 연령이 없어 중앙값을 계산할 수 없습니다.')
            self.median_ = float(np.median(regular))  # 0 포함, C와 같은 중앙값
            self.scaler_ = StandardScaler().fit(self._product(X))
            self.encoder_ = None
            self.output_features_ = np.asarray(['age_height_floors_product'], dtype=object)
        else:
            self.median_, self.scaler_ = None, None
            self.encoder_ = OneHotEncoder(handle_unknown='ignore', sparse_output=True, dtype=np.float32)
            self.encoder_.fit(_pair_values(X))
            self.output_features_ = np.asarray([
                name + '__' + str(encoded) for (name, _, _), vocabulary in zip(PAIRS, self.encoder_.categories_)
                for encoded in vocabulary], dtype=object)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.n_train_rows_ = len(X)
        self.arm_ = self.arm
        # fit 직후에도 저장 상태로 실제 transform이 유효해야 한다.
        self.transform(X)
        return self

    def transform(self, X):
        self._validate(X, fitted=True)
        if len(X) == 0:
            return sparse.csr_matrix((0, len(self.output_features_)), dtype=np.float32)
        if self.arm == ARMS[0]:
            block = _csr(self.scaler_.transform(self._product(X)))
        else:
            block = _csr(self.encoder_.transform(_pair_values(X)))
        if block.shape != (len(X), len(self.output_features_)):
            raise ValueError('추가 블록의 행 또는 열 수가 바뀌었습니다.')
        return block

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, 'output_features_')
        if input_features is not None and list(input_features) != list(self.feature_names_in_):
            raise ValueError('입력 피처 이름이 fit 스키마와 다릅니다.')
        return self.output_features_.copy()

    def unknown_counts(self, X):
        self._validate(X, fitted=True)
        if self.arm == ARMS[0]:
            return {}
        pairs = _pair_values(X)
        return {name: dict(count=int(np.count_nonzero(~np.isin(pairs[:, j], self.encoder_.categories_[j]))),
                           rate=float(np.count_nonzero(~np.isin(pairs[:, j], self.encoder_.categories_[j]))/len(X))
                           if len(X) else None, rows=len(X))
                for j, (name, _, _) in enumerate(PAIRS)}

    def metadata(self):
        check_is_fitted(self, ['feature_names_in_', 'output_features_', 'arm_'])
        if self.arm != self.arm_:
            raise ValueError('fit 후 피처 팔을 변경할 수 없습니다.')
        scaler = None if self.scaler_ is None else dict(
            mean=self.scaler_.mean_.tolist(), variance=self.scaler_.var_.tolist(),
            scale=self.scaler_.scale_.tolist(), n_samples_seen=int(self.scaler_.n_samples_seen_))
        pairs = [] if self.encoder_ is None else [dict(name=name, left=left, right=right,
            vocabulary=[json.loads(str(encoded)) for encoded in vocabulary])
            for (name, left, right), vocabulary in zip(PAIRS, self.encoder_.categories_)]
        return dict(arm=self.arm_, input_columns=self.feature_names_in_.tolist(),
                    output_features=self.output_features_.tolist(), n_train_rows=self.n_train_rows_,
                    median=self.median_, scaler=scaler, pairs=pairs)
