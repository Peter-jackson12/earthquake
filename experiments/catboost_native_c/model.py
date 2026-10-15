"""T26: C의 연령·수치·이진 처리를 보존한 CatBoost native 범주 입력."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from threadpoolctl import threadpool_limits

from preprocess import NUMERIC, BINARY, CATEGORICAL, FEATURES

CATBOOST_VERSION = '1.2.10'
CATBOOST_PARAMS = dict(iterations=300, learning_rate=0.05, depth=6, l2_leaf_reg=3,
    random_strength=1, bootstrap_type='Bayesian', bagging_temperature=1,
    boosting_type='Plain', grow_policy='SymmetricTree', loss_function='MultiClass',
    classes_count=3, random_seed=42, thread_count=2, task_type='CPU',
    allow_writing_files=False, verbose=False)
HEAD_COLUMNS = [f'numeric__{c}' for c in NUMERIC] + [f'binary__{c}' for c in [*BINARY, 'age_is_995']]
NATIVE_COLUMNS = [*HEAD_COLUMNS, *CATEGORICAL]
CAT_FEATURE_INDICES = list(range(28, 39))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _catboost_classifier():
    import catboost
    _require(catboost.__version__ == CATBOOST_VERSION, 'CatBoost 고정 버전 불일치')
    return catboost.CatBoostClassifier


def _fixed_parameters(params):
    _require(isinstance(params, dict) and params == CATBOOST_PARAMS
             and all(type(params[k]) is type(v) for k, v in CATBOOST_PARAMS.items()),
             'CatBoost 고정 설정 변경·추가 인자 금지')
    return dict(params)


class NativeCPreprocessor(TransformerMixin, BaseEstimator):
    """C의 categorical 원핫만 제외한다. 학습 어휘는 진단 전용이며 변환에 사용하지 않는다."""

    def fit(self, X, y=None):
        _require(isinstance(X, pd.DataFrame) and X.columns.tolist() == FEATURES, '원본 38열 스키마·순서 변경')
        _require(len(X) > 0, '빈 outer-train 전처리 학습 금지')
        from experiments.lgbm_native_cat.run import make_preprocessor as original_c
        head = original_c('O')
        columns = head['columns']
        columns.transformers = [item for item in columns.transformers if item[0] != 'categorical']
        head.fit(X)  # y를 전달하지 않으며 모든 outer-train 행을 사용한다.
        _require(head.get_feature_names_out().tolist() == HEAD_COLUMNS and len(HEAD_COLUMNS) == 28,
                 '기존 C 수치·이진 28열 순서 변경')
        normalized = head['schema'].transform(X)
        self.head_ = head
        self.categories_ = {c: sorted(normalized[c].astype(str).unique().tolist()) for c in CATEGORICAL}
        self.n_train_rows_ = len(X)
        self.n_features_in_ = len(FEATURES)
        self.feature_names_in_ = np.asarray(FEATURES, dtype=object)
        return self

    def transform(self, X):
        check_is_fitted(self, 'head_')
        _require(isinstance(X, pd.DataFrame) and X.columns.tolist() == FEATURES, '원본 38열 스키마·순서 변경')
        normalized = self.head_['schema'].transform(X)
        head = self.head_.transform(X) if len(X) else np.empty((0, 28), dtype=np.float32)
        _require(isinstance(head, np.ndarray) and head.dtype == np.float32
                 and head.shape == (len(X), 28) and np.isfinite(head).all(), 'C head28 형식·유한값 변경')
        result = pd.DataFrame(head, index=X.index, columns=HEAD_COLUMNS)
        for c in CATEGORICAL:
            result[c] = normalized[c].astype(str).to_numpy(dtype=object)
        input_evidence(result)
        return result

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, 'head_')
        if input_features is not None:
            _require(list(input_features) == FEATURES, '원본 입력 열 순서 변경')
        return np.asarray(NATIVE_COLUMNS, dtype=object)

    def metadata(self):
        check_is_fitted(self, 'head_')
        scaler = self.head_['columns'].named_transformers_['numeric']['scale']
        return dict(age995_median=float(self.head_['age995'].median_),
                    numeric_mean=scaler.mean_.tolist(), numeric_scale=scaler.scale_.tolist(),
                    numeric_variance=scaler.var_.tolist(),
                    categorical_categories=[list(self.categories_[c]) for c in CATEGORICAL],
                    categorical_columns=list(CATEGORICAL), output_columns=list(NATIVE_COLUMNS),
                    n_train_rows=self.n_train_rows_, numeric_head_columns=28,
                    categorical_encoding='normalized_original_strings', target_used=False,
                    unseen_category_policy='pass_original_string_to_catboost')

    def unknown_counts(self, X):
        check_is_fitted(self, 'head_')
        _require(isinstance(X, pd.DataFrame) and X.columns.tolist() == FEATURES, '원본 38열 스키마·순서 변경')
        normalized = self.head_['schema'].transform(X)
        rows = len(X)
        result = {}
        for c in CATEGORICAL:
            count = int((~normalized[c].astype(str).isin(self.categories_[c])).sum())
            result[c] = dict(count=count, rate=count/rows if rows else None, rows=rows)
        return result


def make_preprocessor():
    return NativeCPreprocessor()


def prepare_native_c(Xa, Xb):
    pre = make_preprocessor()
    with threadpool_limits(limits=2):
        return pre, pre.fit_transform(Xa), pre.transform(Xb)


def input_evidence(frame):
    _require(isinstance(frame, pd.DataFrame) and frame.columns.tolist() == NATIVE_COLUMNS,
             'native 입력 39열 스키마·순서 변경')
    _require(all(frame[c].dtype == np.float32 for c in HEAD_COLUMNS), 'C head28 float32 필수')
    numeric = np.ascontiguousarray(frame[HEAD_COLUMNS].to_numpy(dtype=np.float32))
    _require(np.isfinite(numeric).all(), 'C head28 비유한값 포함')
    _require(all(frame[c].dtype == object and frame[c].map(lambda v: isinstance(v, str)).all()
                 for c in CATEGORICAL), '범주 11열은 원래 문자열 object 필수')
    # 전체 행 목록의 canonical JSON과 동일한 바이트를 순차 해시한다.
    digest = hashlib.sha256()
    digest.update(b'[')
    for i, row in enumerate(frame[CATEGORICAL].itertuples(index=False, name=None)):
        if i:
            digest.update(b',')
        digest.update(json.dumps(row, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8'))
    digest.update(b']')
    return dict(rows=len(frame), column_order=list(NATIVE_COLUMNS), dtypes=[str(v) for v in frame.dtypes],
                numeric_head=dict(dtype='float32', shape=list(numeric.shape),
                                  sha256=hashlib.sha256(numeric.tobytes()).hexdigest()),
                categorical=dict(columns=list(CATEGORICAL), shape=[len(frame), 11],
                                 encoding='canonical_json_utf8_rows', sha256=digest.hexdigest()))


def preprocessor_state(pre):
    _require(isinstance(pre, NativeCPreprocessor) and pre.head_['age995'].mode == 'C',
             'native C 전처리 클래스·연령 정책 변경')
    return pre.metadata()


def probabilities(values, rows):
    raw = np.asarray(values)
    _require(raw.dtype.kind in 'fiu', '확률은 실수 수치형이어야 함')
    p = np.asarray(raw, dtype=np.float64)
    _require(p.shape == (rows, 3) and np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()
             and np.allclose(p.sum(axis=1), 1., atol=1e-12, rtol=0), '예측 확률·클래스 차원 불일치')
    return p


def _target(values, rows):
    y = np.asarray(values)
    _require(y.ndim == 1 and len(y) == rows and rows > 0 and y.dtype.kind in 'iuf', '학습 등급 행수·형식 불일치')
    _require(np.isfinite(y).all() and np.isin(y, [1, 2, 3]).all()
             and np.array_equal(np.unique(y), [1, 2, 3]), '학습 등급은 1/2/3 모두 필요')
    return y.astype(np.int32)-1


def _model_state(model):
    _require(np.array_equal(model.classes_, [0, 1, 2]), '내부 클래스 순서 변경')
    _require(model.tree_count_ == 300, '고정 300회 학습 미완료')
    _require(list(model.feature_names_) == NATIVE_COLUMNS, 'native 모델 열 이름·순서 변경')
    _require(model.get_cat_feature_indices() == CAT_FEATURE_INDICES, 'native 범주 인덱스 변경')
    effective = model.get_all_params()
    for name, expected in CATBOOST_PARAMS.items():
        if name in ('verbose', 'thread_count', 'allow_writing_files'):
            continue
        _require(name in effective, f'학습된 설정 누락: {name}')
        actual = effective[name]
        if isinstance(expected, (float, int)) and not isinstance(expected, bool):
            _require(isinstance(actual, (float, int)) and np.isclose(actual, expected, atol=1e-7, rtol=0),
                     f'학습된 고정 설정 변경: {name}')
        else:
            _require(actual == expected, f'학습된 고정 설정 변경: {name}')
    _require(effective.get('class_weights', [1, 1, 1]) == [1, 1, 1], '외부 학습 가중치 금지')
    return dict(internal_classes=np.asarray(model.classes_).tolist(), tree_count=int(model.tree_count_),
                feature_names=list(model.feature_names_), cat_feature_indices=model.get_cat_feature_indices(),
                effective_parameters=effective)


def _same_native_state(left, right):
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_same_native_state(left[k], right[k]) for k in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_same_native_state(a, b) for a, b in zip(left, right))
    if type(left) is float and type(right) is float:
        return bool(np.isfinite(left) and np.isfinite(right) and abs(left-right) <= 1e-7
                    and np.float32(left) == np.float32(right))
    return type(left) is type(right) and left == right


def fit_catboost_native_c(Xa, ya, Xb, params, *, prepared=None):
    params = _fixed_parameters(params)
    target = _target(ya, len(Xa))
    if prepared is None:
        pre, mt, mv = prepare_native_c(Xa, Xb)
    else:
        _require(isinstance(prepared, tuple) and len(prepared) == 3, 'prepared 형식 불일치')
        pre, mt, mv = prepared
    evidence = dict(train=input_evidence(mt), valid=input_evidence(mv))
    _require(len(mt) == len(Xa) and len(mv) == len(Xb) and len(Xb) > 0, '학습·검증 행수 불일치')
    state = preprocessor_state(pre)
    classifier = _catboost_classifier()
    model = classifier(**params)
    with threadpool_limits(limits=2), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        started = time.perf_counter()
        model.fit(mt, target, cat_features=CAT_FEATURE_INDICES)
        elapsed = time.perf_counter()-started
        native = _model_state(model)
        p = probabilities(model.predict_proba(mv, thread_count=2), len(Xb))
    _require(evidence == dict(train=input_evidence(mt), valid=input_evidence(mv)), '모델이 원래 native 입력을 변경함')
    _require(state == preprocessor_state(pre), '학습 중 전처리 상태 변경')
    init = model.get_params()
    _require(init == params, 'CatBoost 생성 인자 변경')

    def save(prefix):
        paths = [Path(f'{prefix}_model.cbm'), Path(f'{prefix}_preprocessor.joblib')]
        _require(not any(p.exists() for p in paths), '모델·전처리 덮어쓰기 금지')
        model.save_model(str(paths[0]), format='cbm')
        joblib.dump(pre, paths[1], compress=3)
        return paths

    def verify(paths, Xv, saved):
        _require(len(paths) == 2, '모델·전처리 두 파일 필요')
        with threadpool_limits(limits=2):
            reloaded = classifier()
            reloaded.load_model(str(paths[0]), format='cbm')
            loaded_pre = joblib.load(paths[1])
            _require(state == preprocessor_state(loaded_pre), '재로딩 전처리 상태 변경')
            encoded = loaded_pre.transform(Xv)
            _require(input_evidence(encoded) == evidence['valid'], '재로딩 native 입력 불일치')
            _require(_same_native_state(_model_state(reloaded), native), '재로딩 native 모델 상태 변경')
            pp = probabilities(reloaded.predict_proba(encoded, thread_count=2), len(Xv))
            _require(input_evidence(encoded) == evidence['valid'], '재추론이 native 입력을 변경함')
        maximum = float(np.max(np.abs(pp-probabilities(saved, len(Xv)))))
        _require(maximum <= 1e-12, 'native 재로딩 확률 차이 초과')
        return maximum

    return dict(p=p, fit_seconds=elapsed, save=save, verify=verify,
        info=dict(**native, damage_labels=[1, 2, 3], init_parameters=init, catboost_version=CATBOOST_VERSION,
                  input_evidence=evidence, preprocessor_state=state, age995_median=state['age995_median'],
                  input_columns=list(NATIVE_COLUMNS), numeric_head_columns=28, categorical_columns=list(CATEGORICAL),
                  train_unknown_categories=pre.unknown_counts(Xa), valid_unknown_categories=pre.unknown_counts(Xb),
                  preprocessor_custom_class='experiments.catboost_native_c.model.NativeCPreprocessor',
                  audit_reload_note='감사자는 joblib 로딩 없이 독립 C head28·원래범주11 및 native 모델로 검증한다.',
                  warnings=[dict(category=w.category.__name__, message=str(w.message)) for w in caught]))
