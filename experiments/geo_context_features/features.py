"""T24: 학습 Fold의 지역별 중앙값과 행별 차이만 추가한다. 실행기 import 없음."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

ARMS = ('GEO1_CONTEXT',)
GROUP_COLUMN = 'geo_level_1_id'
CONTEXT_COLUMNS = ('count_floors_pre_eq', 'height_percentage', 'area_percentage')
OUTPUT_FEATURES = tuple('geo1_median_' + name for name in CONTEXT_COLUMNS) + tuple(
    'geo1_difference_' + name for name in CONTEXT_COLUMNS)


def _frame(X):
    if not isinstance(X, pd.DataFrame) or not X.columns.is_unique:
        raise ValueError('고유 열 이름을 가진 DataFrame이 필요합니다.')
    if not all(isinstance(column, str) for column in X.columns):
        raise ValueError('입력 열 이름은 문자열이어야 합니다.')


def _read_values(X):
    geo = pd.to_numeric(X[GROUP_COLUMN], errors='raise').to_numpy()
    if (not np.isfinite(geo).all() or (geo < 0).any()
            or not np.equal(geo, np.floor(geo)).all()):
        raise ValueError('1차 지역은 0을 포함한 유한한 비음수 정수여야 합니다.')
    # 숫자 문자열은 원래 지역 범주와 동일하게 처리한다. 정수 dtype의 큰 ID도 보존한다.
    groups = np.asarray([int(value) for value in geo], dtype=object)
    values = X.loc[:, list(CONTEXT_COLUMNS)].apply(pd.to_numeric, errors='raise').to_numpy(dtype=np.float64)
    if (not np.isfinite(values).all() or (values < 0).any()
            or not np.equal(values, np.floor(values)).all()):
        raise ValueError('층수·높이·면적은 0을 포함한 유한한 비음수 정수여야 합니다.')
    return groups, values


def _float32(values):
    try:
        with np.errstate(over='raise', invalid='raise'):
            result = np.asarray(values).astype(np.float32)
    except FloatingPointError as exc:
        raise ValueError('추가 피처를 float32로 표현할 수 없습니다.') from exc
    if not np.isfinite(result).all():
        raise ValueError('추가 피처의 float32 값은 유한해야 합니다.')
    return result


class GeoContextFeatures(BaseEstimator, TransformerMixin):
    """정답과 무관한 train geo1 중앙값 3열과 원값−중앙값 3열. 스케일링 없음."""

    def __init__(self, arm='GEO1_CONTEXT'):
        self.arm = arm

    def _validate(self, X, fitted=False):
        _frame(X)
        if self.arm not in ARMS:
            raise ValueError('등록하지 않은 T24 피처 팔입니다.')
        missing = [name for name in (GROUP_COLUMN, *CONTEXT_COLUMNS) if name not in X.columns]
        if missing:
            raise ValueError('지역 문맥 입력 열 누락: ' + ', '.join(missing))
        if fitted:
            check_is_fitted(self, ['arm_', 'feature_names_in_', 'global_medians_', 'group_medians_'])
            if self.arm != self.arm_:
                raise ValueError('fit 후 피처 팔을 변경할 수 없습니다.')
            if list(X.columns) != list(self.feature_names_in_):
                raise ValueError('fit 후 입력 열 순서 또는 스키마가 바뀌었습니다.')

    def fit(self, X, y=None):
        self._validate(X)
        if len(X) == 0:
            raise ValueError('학습 입력은 비어 있을 수 없습니다.')
        groups, values = _read_values(X)
        # y를 읽지 않는다. 모든 outer-train 행을 자신이 속한 그룹 통계에도 포함한다.
        try:
            with np.errstate(over='raise', invalid='raise'):
                global_medians = np.median(values, axis=0)
                ids = sorted(set(groups))
                medians = {geo: np.median(values[groups == geo], axis=0) for geo in ids}
        except FloatingPointError as exc:
            raise ValueError('중앙값 산술 결과가 유한하지 않습니다.') from exc
        _float32(global_medians)
        _float32(np.asarray([medians[geo] for geo in ids]))
        self.global_medians_ = global_medians.copy()
        self.group_medians_ = {geo: medians[geo].copy() for geo in ids}
        self.group_counts_ = {geo: int(np.count_nonzero(groups == geo)) for geo in ids}
        self.group_ids_ = np.asarray(ids, dtype=object)
        self.arm_ = self.arm
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.n_train_rows_ = len(X)
        # 차이 열의 overflow도 학습 단계에서 거절한다.
        self.transform(X)
        return self

    def transform(self, X):
        self._validate(X, fitted=True)
        if len(X) == 0:
            return sparse.csr_matrix((0, len(OUTPUT_FEATURES)), dtype=np.float32)
        groups, values = _read_values(X)
        medians = np.asarray([self.group_medians_.get(geo, self.global_medians_) for geo in groups], dtype=np.float64)
        with np.errstate(over='raise', invalid='raise'):
            differences = values - medians
        result = sparse.csr_matrix(_float32(np.column_stack((medians, differences))), dtype=np.float32)
        result.eliminate_zeros()
        result.sort_indices()
        if result.shape != (len(X), len(OUTPUT_FEATURES)):
            raise ValueError('지역 문맥 블록의 행 또는 열 수가 바뀌었습니다.')
        return result

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, ['arm_', 'feature_names_in_'])
        if self.arm != self.arm_:
            raise ValueError('fit 후 피처 팔을 변경할 수 없습니다.')
        if input_features is not None and list(input_features) != list(self.feature_names_in_):
            raise ValueError('입력 피처 이름이 fit 스키마와 다릅니다.')
        return np.asarray(OUTPUT_FEATURES, dtype=object)

    def unknown_counts(self, X):
        self._validate(X, fitted=True)
        groups, _ = _read_values(X)
        count = sum(geo not in self.group_medians_ for geo in groups)
        return {GROUP_COLUMN: dict(count=count, rate=count/len(X) if len(X) else None, rows=len(X))}

    def metadata(self):
        check_is_fitted(self, ['arm_', 'feature_names_in_', 'global_medians_', 'group_medians_'])
        if self.arm != self.arm_:
            raise ValueError('fit 후 피처 팔을 변경할 수 없습니다.')
        return dict(arm=self.arm_, input_columns=self.feature_names_in_.tolist(),
            output_features=list(OUTPUT_FEATURES), n_train_rows=self.n_train_rows_,
            group_column=GROUP_COLUMN, context_columns=list(CONTEXT_COLUMNS),
            global_medians=self.global_medians_.tolist(),
            groups=[dict(geo_level_1_id=int(geo), rows=self.group_counts_[geo],
                         medians=self.group_medians_[geo].tolist()) for geo in self.group_ids_],
            scaling='none', training_rows_include_self=True, target_used=False,
            fallback='outer_train_global_medians')
