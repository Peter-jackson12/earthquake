"""T32 target-count arithmetic, sampler contracts, class weights and the G-IDX gate.

Pure functions of training-fold labels only. No validation row, no model and no
data file is touched here. Targets use integer arithmetic (0.5 rounds up) so the
float rounding mode can never move a boundary.

Directions
  UP   : class 2 (majority) fixed, t_c = n_c + add_c for c in {1, 3}.
  DOWN : class 1 (minority) fixed, t_c = n_c - rem_c for c in {2, 3}.
Special UP scenarios: ONE1 (only grade 1 raised to n2) and ONE3 (only grade 3).
"""
from __future__ import annotations

import hashlib

import numpy as np

LABELS = (1, 2, 3)
# Share of the gap that is closed, as an exact fraction p/q.
PHI = {'P25': (1, 4), 'P50': (1, 2), 'P75': (3, 4), 'EQ': (1, 1)}
SCENARIOS = ('ORI', 'P25', 'P50', 'P75', 'EQ', 'ONE1', 'ONE3')
ARM_DIRECTION = {'ORI': None, 'RUP': 'UP', 'RDN': 'DOWN', 'WOBJ': 'UP', 'WNORM': 'UP'}
ARM_SCENARIOS = {'ORI': ('ORI',), 'RUP': ('P25', 'P50', 'P75', 'EQ', 'ONE1', 'ONE3'),
                 'RDN': ('P25', 'P50', 'P75', 'EQ'), 'WOBJ': ('P25', 'P50', 'P75', 'EQ'),
                 'WNORM': ('P25', 'P50', 'P75', 'EQ')}
# Fixed (never touched) grade and the two moved grades, per direction.
FIXED = {'UP': 2, 'DOWN': 1}
MOVED = {'UP': (1, 3), 'DOWN': (2, 3)}


class TargetError(ValueError):
    pass


def count_vector(y) -> tuple[int, int, int]:
    """Training-fold class counts (n1, n2, n3); labels must be exactly {1, 2, 3}."""
    y = np.asarray(y)
    if y.ndim != 1 or len(y) == 0:
        raise TargetError('Nonempty one-dimensional training labels required')
    if not np.isin(y, LABELS).all():
        raise TargetError('Only labels 1, 2, 3 are allowed')
    n = tuple(int(np.sum(y == c)) for c in LABELS)
    check_order(n)
    return n


def check_order(n) -> None:
    """Design assumption: grade 2 is the majority and grade 1 the minority."""
    n1, n2, n3 = (int(v) for v in n)
    if not (0 < n1 < n3 < n2):
        raise TargetError(f'Expected 0 < n1 < n3 < n2, got {(n1, n2, n3)}')


def half_up_share(gap: int, p: int, q: int) -> int:
    """floor((2*p*gap + q) / (2*q)): p/q of the gap, 0.5 rounded up, integers only."""
    if not (isinstance(gap, (int, np.integer)) and gap >= 0 and 0 < p <= q):
        raise TargetError('Invalid gap or fraction')
    return (2 * p * int(gap) + q) // (2 * q)


def scenario_targets(n, direction: str | None, scenario: str) -> tuple[int, int, int]:
    """Targets (t1, t2, t3) of one scenario from the training-fold counts n."""
    n1, n2, n3 = (int(v) for v in n)
    check_order((n1, n2, n3))
    if scenario == 'ORI' or direction is None:
        if scenario != 'ORI':
            raise TargetError('A directionless arm only supports ORI')
        return (n1, n2, n3)
    if direction not in ('UP', 'DOWN'):
        raise TargetError('Unknown direction')
    if scenario in ('ONE1', 'ONE3'):
        if direction != 'UP':
            raise TargetError('ONE1/ONE3 are UP-only decomposition scenarios')
        return (n2, n2, n3) if scenario == 'ONE1' else (n1, n2, n2)
    if scenario not in PHI:
        raise TargetError(f'Unknown scenario {scenario}')
    p, q = PHI[scenario]
    if direction == 'UP':
        return (n1 + half_up_share(n2 - n1, p, q), n2, n3 + half_up_share(n2 - n3, p, q))
    return (n1, n2 - half_up_share(n2 - n1, p, q), n3 - half_up_share(n3 - n1, p, q))


def sampler_strategy(n, t, direction: str) -> dict[int, int]:
    """imblearn dict: moved grades only, ascending insertion, fixed grade never a key.

    Leaving the fixed grade out is what keeps the random stream identical to the
    'auto' strategy (verified by the G-IDX gate). Grades whose target equals their
    count are omitted too (nothing to draw).
    """
    if direction not in ('UP', 'DOWN'):
        raise TargetError('Unknown direction')
    strategy = {}
    for c in MOVED[direction]:                      # ascending: (1, 3) or (2, 3)
        nc, tc = int(n[c - 1]), int(t[c - 1])
        if direction == 'UP':
            if tc < nc:
                raise TargetError('Over-sampling target below the original count')
            if tc > nc:
                strategy[c] = tc
        else:
            if tc > nc:
                raise TargetError('Under-sampling target above the original count')
            if tc < nc:
                strategy[c] = tc
    fixed = FIXED[direction]
    if int(t[fixed - 1]) != int(n[fixed - 1]):
        raise TargetError('The fixed grade must keep its count')
    return strategy


def sample_indices(y, strategy: dict[int, int], direction: str, seed: int) -> np.ndarray:
    """Training positions selected by the real imblearn sampler (labels only).

    UP returns [0..N-1, extra draws ...] (originals first, as RandomOverSampler stacks
    them); DOWN returns kept positions in class order. A feature matrix is not needed:
    the random draws depend on the labels only, so row selection is applied afterwards.
    """
    y = np.asarray(y)
    N = len(y)
    if not strategy:
        return np.arange(N)
    if direction == 'UP':
        from imblearn.over_sampling import RandomOverSampler as Sampler
    else:
        from imblearn.under_sampling import RandomUnderSampler as Sampler
    sampler = Sampler(sampling_strategy=dict(strategy), random_state=seed)
    sampler.fit_resample(np.arange(N).reshape(-1, 1), y)
    return np.asarray(sampler.sample_indices_, dtype=np.int64)


def auto_sample_indices(y, direction: str, seed: int) -> np.ndarray:
    """The stock 'auto' (not-majority / not-minority) path, for the G-IDX comparison."""
    y = np.asarray(y)
    if direction == 'UP':
        from imblearn.over_sampling import RandomOverSampler as Sampler
    else:
        from imblearn.under_sampling import RandomUnderSampler as Sampler
    sampler = Sampler(random_state=seed)
    sampler.fit_resample(np.arange(len(y)).reshape(-1, 1), y)
    return np.asarray(sampler.sample_indices_, dtype=np.int64)


def index_sha256(indices) -> str:
    return hashlib.sha256(np.asarray(indices, dtype='<i8').tobytes()).hexdigest()


def gidx_fold(y_train, seed: int = 42) -> dict:
    """G-IDX for one training fold: dict path == 'auto' path for the fully equal scenario."""
    n = count_vector(y_train)
    out = dict(counts=list(n))
    for direction in ('UP', 'DOWN'):
        t = scenario_targets(n, direction, 'EQ')
        strategy = sampler_strategy(n, t, direction)
        via_dict = sample_indices(y_train, strategy, direction, seed)
        via_auto = auto_sample_indices(y_train, direction, seed)
        equal = bool(via_dict.shape == via_auto.shape and np.array_equal(via_dict, via_auto))
        after = np.bincount(np.asarray(y_train)[via_dict], minlength=4)[1:4].tolist()
        out[direction] = dict(strategy={str(k): v for k, v in strategy.items()}, target=list(t),
                              identical=equal, rows=int(len(via_dict)), after=after,
                              after_equals_target=after == list(t), indices_sha256=index_sha256(via_dict))
    out['pass'] = bool(out['UP']['identical'] and out['DOWN']['identical']
                       and out['UP']['after_equals_target'] and out['DOWN']['after_equals_target'])
    return out


def gidx(y, folds, seed: int = 42) -> dict:
    """G-IDX over all five folds using training-fold labels only (no model, no features)."""
    y, folds = np.asarray(y), np.asarray(folds)
    result = {}
    for f in range(5):
        result[str(f)] = gidx_fold(y[folds != f], seed)
    return dict(seed=seed, folds=result, all_pass=all(v['pass'] for v in result.values()))


def class_weights(n, t, kind: str) -> np.ndarray:
    """Per-class weight (3,): W-OBJ = t_c / n_c; W-NORM = (t_c / n_c) * N / sum(t)."""
    n = np.asarray(n, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    w = t / n
    if kind == 'obj':
        return w
    if kind == 'norm':
        return w * n.sum() / t.sum()
    raise TargetError('Unknown weight kind')


def effective_mass(y, class_w) -> list[float]:
    y = np.asarray(y)
    return [float(np.sum(y == c) * class_w[c - 1]) for c in LABELS]


def describe_targets(n, t) -> dict:
    """before / target / delta / multiplier / share for one fold (the required record)."""
    n, t = [int(v) for v in n], [int(v) for v in t]
    return dict(before=n, target=t, delta=[b - a for a, b in zip(n, t)],
                multiplier=[b / a for a, b in zip(n, t)], total_before=sum(n), total_target=sum(t),
                share_before=[v / sum(n) for v in n], share_target=[v / sum(t) for v in t])
