"""T23: 두 누적 이진확률의 단조 투영과 세 등급 확률 복원. 학습·파일 접근 없음."""
from __future__ import annotations

import numpy as np

LABELS = (1, 2, 3)


def _probability_vector(values, name):
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.dtype.kind not in 'uif':
        raise ValueError(name + ': 1차원 실수 또는 정수 확률 배열이 필요합니다.')
    if not np.isfinite(raw).all() or (raw < 0).any() or (raw > 1).any():
        raise ValueError(name + ': 확률은 유한한 [0,1] 값이어야 합니다.')
    # 입력을 수정하지 않고 모든 산술을 float64로 고정한다.
    return raw.astype(np.float64, copy=True)


def project_cumulative_probabilities(q_ge2, q_ge3, *, return_metadata=False):
    """q2=P(y>=2), q3=P(y>=3)에서 float64 (n,3) 확률을 반환한다.

    q3>q2인 행에만 q2'=q3'=(q2+q3)/2를 적용한다. 이는 동일 가중
    제곱거리 아래 q3'<=q2' 제약으로의 두 값 투영이다. 그 외 행은 유지한다.
    p=[1-q2',q2'-q3',q3']이며 clipping, 재정규화, 허용오차 기반 교차 판정은 없다.
    return_metadata=True이면 (probabilities, metadata)를 반환한다.
    crossing_rate의 분모는 전체 행 수, mean_crossing_excess의 분모는 교차 행 수다.
    빈 입력은 (0,3)을 반환하고 비율은 None, 교차 초과량 통계는 0으로 둔다.
    """
    q2 = _probability_vector(q_ge2, 'q_ge2')
    q3 = _probability_vector(q_ge3, 'q_ge3')
    if q2.shape != q3.shape:
        raise ValueError('두 누적확률 배열의 길이가 다릅니다.')
    crossing = q3 > q2
    excess = q3[crossing] - q2[crossing]
    mean = (q2[crossing] + q3[crossing]) / 2.
    q2[crossing], q3[crossing] = mean, mean
    p = np.column_stack((1.-q2, q2-q3, q3))
    if (not np.isfinite(p).all() or (p < 0).any() or (p > 1).any()
            or not np.allclose(p.sum(axis=1), 1., atol=1e-15, rtol=0)):
        raise ValueError('누적확률 복원의 확률 계약이 실패했습니다.')
    if not return_metadata:
        return p
    count, rows = int(crossing.sum()), len(q2)
    metadata = dict(rows=rows, crossing_count=count,
                    crossing_rate=count/rows if rows else None,
                    crossing_excess_sum=float(excess.sum()),
                    mean_crossing_excess=float(excess.mean()) if count else 0.,
                    mean_crossing_excess_denominator=count,
                    max_crossing_excess=float(excess.max()) if count else 0.)
    return p, metadata


def damage_argmax(probabilities):
    """labels=[1,2,3]의 최대확률 등급. 정확 동률이면 낮은 등급을 택한다."""
    raw = np.asarray(probabilities)
    if raw.ndim != 2 or raw.shape[1] != 3 or raw.dtype.kind not in 'uif':
        raise ValueError('등급 확률은 3열의 실수 또는 정수 배열이어야 합니다.')
    if not np.isfinite(raw).all() or (raw < 0).any() or (raw > 1).any():
        raise ValueError('등급 확률은 유한한 [0,1] 값이어야 합니다.')
    # 외부 float32 확률도 값 자체를 바꾸지 않고 표현 오차 범위에서 검사한다.
    tolerance = 1e-6 if raw.dtype.kind == 'f' and raw.dtype.itemsize <= 4 else 1e-12
    p = raw.astype(np.float64, copy=False)
    if not np.allclose(p.sum(axis=1), 1., atol=tolerance, rtol=0):
        raise ValueError('등급 확률의 행 합은 1이어야 합니다.')
    return np.asarray(LABELS, dtype=np.int64)[p.argmax(axis=1)]
