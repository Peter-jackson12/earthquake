"""T25: 고정 C 희소 입력의 CatBoost 어댑터. 자료 로딩·환경 설치는 하지 않는다."""
from __future__ import annotations

import hashlib
from pathlib import Path
import time
import warnings

import joblib
import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_limits


CATBOOST_VERSION = '1.2.10'
CATBOOST_PARAMS = dict(iterations=300, learning_rate=0.05, depth=6, l2_leaf_reg=3,
    random_strength=1, bootstrap_type='Bayesian', bagging_temperature=1,
    boosting_type='Plain', grow_policy='SymmetricTree', loss_function='MultiClass',
    classes_count=3, random_seed=42, thread_count=2, task_type='CPU',
    allow_writing_files=False, verbose=False)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _catboost_classifier():
    """호출자가 검증·활성화한 별도 의존성 경로만 사용한다."""
    import catboost
    _require(catboost.__version__ == CATBOOST_VERSION, 'CatBoost 고정 버전 불일치')
    return catboost.CatBoostClassifier


def _fixed_parameters(params):
    _require(isinstance(params, dict) and params == CATBOOST_PARAMS
             and all(type(params[k]) is type(v) for k, v in CATBOOST_PARAMS.items()),
             'CatBoost 고정 설정 변경·추가 인자 금지')
    return dict(params)


def matrix_evidence(matrix):
    _require(sparse.isspmatrix_csr(matrix) and matrix.dtype == np.float32,
             'C 입력은 float32 CSR이어야 함')
    _require(np.isfinite(matrix.data).all(), 'C 희소 입력에 비유한값 포함')
    matrix.check_format(full_check=True)
    return dict(format='csr', shape=list(matrix.shape), dtype=str(matrix.dtype),
        arrays={name: dict(dtype=str(getattr(matrix, name).dtype),
                           shape=list(getattr(matrix, name).shape),
                           sha256=hashlib.sha256(np.ascontiguousarray(getattr(matrix, name)).tobytes()).hexdigest())
                for name in ['data', 'indices', 'indptr']})


def probabilities(values, rows):
    raw = np.asarray(values)
    _require(raw.dtype.kind in 'fiu' and raw.dtype.kind != 'b', '확률은 실수 수치형이어야 함')
    p = np.asarray(raw, dtype=np.float64)
    _require(p.shape == (rows, 3) and np.isfinite(p).all()
             and (p >= 0).all() and (p <= 1).all()
             and np.allclose(p.sum(axis=1), 1., atol=1e-12, rtol=0), '예측 확률·클래스 차원 불일치')
    return p


def _target(values, rows):
    y = np.asarray(values)
    _require(y.ndim == 1 and len(y) == rows and rows > 0 and y.dtype.kind in 'iuf',
             '학습 대상 행수·형식 불일치')
    _require(np.isfinite(y).all() and np.isin(y, [1, 2, 3]).all()
             and np.array_equal(np.unique(y), [1, 2, 3]), '학습 등급은 1/2/3 모두 필요')
    return y.astype(np.int32)-1


def preprocessor_state(pre):
    scaler = pre['columns'].named_transformers_['numeric']['scale']
    encoder = pre['columns'].named_transformers_['categorical']
    _require(pre['age995'].mode == 'C', '기존 C 연령 정책만 허용')
    return dict(age995_median=float(pre['age995'].median_), numeric_mean=scaler.mean_.tolist(),
                numeric_scale=scaler.scale_.tolist(), numeric_variance=scaler.var_.tolist(),
                categorical_categories=[v.tolist() for v in encoder.categories_])


def _model_state(model, columns):
    _require(np.array_equal(model.classes_, [0, 1, 2]), '내부 클래스 순서는 0/1/2 필수')
    _require(model.tree_count_ == CATBOOST_PARAMS['iterations'], '고정 300회 학습 미완료')
    names = list(model.feature_names_)
    _require(names == [str(i) for i in range(columns)], 'CatBoost CSR 열 순서·열수 변경')
    effective = model.get_all_params()
    # CatBoost는 학습률 등을 native float 정밀도로 보관한다. 실행 전 인자는 별도로 엄격 비교한다.
    for name, expected in CATBOOST_PARAMS.items():
        if name in ('verbose', 'thread_count', 'allow_writing_files'):
            continue
        _require(name in effective, f'학습된 CatBoost 설정 누락: {name}')
        actual = effective[name]
        if isinstance(expected, (float, int)) and not isinstance(expected, bool):
            _require(isinstance(actual, (float, int)) and np.isclose(actual, expected, atol=1e-7, rtol=0),
                     f'학습된 CatBoost 설정 변경: {name}')
        else:
            _require(actual == expected, f'학습된 CatBoost 설정 변경: {name}')
    _require(effective.get('class_weights', [1, 1, 1]) == [1, 1, 1], '학습 가중치 혼합 금지')
    return dict(internal_classes=np.asarray(model.classes_).tolist(), tree_count=int(model.tree_count_),
                feature_names=names, effective_parameters=effective)


def _same_native_state(left, right):
    """native JSON 십진 반올림은 동일 float32 표현 안에서만 허용한다."""
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_same_native_state(left[k], right[k]) for k in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_same_native_state(a, b) for a, b in zip(left, right))
    if type(left) is float and type(right) is float:
        return bool(np.isfinite(left) and np.isfinite(right) and abs(left-right) <= 1e-7
                    and np.float32(left) == np.float32(right))
    return type(left) is type(right) and left == right


def fit_catboost_c(Xa, ya, Xb, params, *, prepared=None):
    """검증 target 없이 학습한다. prepared는 기존 C의 (pre, trainCSR, validCSR)이다."""
    params = _fixed_parameters(params)
    target = _target(ya, len(Xa))
    if prepared is None:
        from experiments.lgbm_native_cat.run import make_preprocessor
        pre = make_preprocessor('O')
        with threadpool_limits(limits=2):
            mt, mv = pre.fit_transform(Xa), pre.transform(Xb)
    else:
        _require(isinstance(prepared, tuple) and len(prepared) == 3, 'prepared 형식 불일치')
        pre, mt, mv = prepared
    matrices = dict(train=matrix_evidence(mt), valid=matrix_evidence(mv))
    _require(mt.shape[0] == len(Xa) and mv.shape[0] == len(Xb)
             and mt.shape[1] == mv.shape[1] and mt.shape[1] > 0 and len(Xb) > 0,
             'C 입력 학습·검증 행수 또는 열수 불일치')
    state = preprocessor_state(pre)
    classifier = _catboost_classifier()
    model = classifier(**params)
    with threadpool_limits(limits=2), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        started = time.perf_counter()
        model.fit(mt, target)
        elapsed = time.perf_counter()-started
        native = _model_state(model, mt.shape[1])
        p = probabilities(model.predict_proba(mv, thread_count=2), len(Xb))
    _require(matrices == dict(train=matrix_evidence(mt), valid=matrix_evidence(mv)),
             'CatBoost가 원본 C 희소 입력을 변경함')
    _require(state == preprocessor_state(pre), 'CatBoost 학습 후 C 전처리 상태 변경')
    init = model.get_params()
    _require(init == params, 'CatBoost 생성 인자 변경')

    def save(prefix):
        paths = [Path(f'{prefix}_model.cbm'), Path(f'{prefix}_preprocessor.joblib')]
        _require(not any(path.exists() for path in paths), '기존 모델·전처리 덮어쓰기 금지')
        model.save_model(str(paths[0]), format='cbm')
        joblib.dump(pre, paths[1], compress=3)
        return paths

    def verify(paths, Xv, saved):
        _require(len(paths) == 2, '모델·전처리 파일 두 개 필요')
        with threadpool_limits(limits=2):
            reloaded = classifier()
            reloaded.load_model(str(paths[0]), format='cbm')
            loaded_pre = joblib.load(paths[1])
            _require(state == preprocessor_state(loaded_pre), '재로딩 C 전처리 상태 변경')
            encoded = loaded_pre.transform(Xv)
            _require(matrix_evidence(encoded) == matrices['valid'], '재로딩 검증 C 행렬 불일치')
            reloaded_state = _model_state(reloaded, encoded.shape[1])
            _require(_same_native_state(reloaded_state, native), '재로딩 CatBoost 구조·학습 설정 변경')
            pp = probabilities(reloaded.predict_proba(encoded, thread_count=2), len(Xv))
            _require(matrix_evidence(encoded) == matrices['valid'], '재추론이 C 행렬을 변경함')
        expected = probabilities(saved, len(Xv))
        diff = float(np.max(np.abs(pp-expected)))
        _require(diff <= 1e-12, 'native 모델 재로딩 확률 차이 초과')
        return diff

    return dict(p=p, fit_seconds=elapsed, save=save, verify=verify,
        info=dict(**native, damage_labels=[1, 2, 3], init_parameters=init,
                  catboost_version=CATBOOST_VERSION, matrix_evidence=matrices,
                  preprocessor_state=state, age995_median=state['age995_median'],
                  encoded_columns=int(mt.shape[1]),
                  encoded_feature_names=pre.get_feature_names_out().tolist(),
                  preprocessor_custom_class='experiments.age995.run.Age995Treatment',
                  audit_reload_note='감사자는 joblib 로딩 없이 독립 C CSR 및 native CatBoost로 검증한다.',
                  warnings=[dict(category=w.category.__name__, message=str(w.message)) for w in caught]))
