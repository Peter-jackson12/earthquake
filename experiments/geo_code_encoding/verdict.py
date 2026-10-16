"""T31 pre-registered descriptive verdict rules (plan sections 9.1-9.3). Pure functions; thresholds from config."""
from __future__ import annotations
import numpy as np


def performance_tag(delta, wins, delta_recall3, v):
    """NUM_HIGHER / NUM_LOWER / NO_MATERIAL_DIFFERENCE / MIXED (evaluated in this order)."""
    t, eps = v['delta_threshold'], v['float_eps']
    if delta >= t - eps and wins >= v['wins_higher_min'] and delta_recall3 >= v['recall3_delta_min'] - eps:
        return 'NUM_HIGHER'
    if delta <= -t + eps and wins <= v['wins_lower_max']:
        return 'NUM_LOWER'
    if abs(delta) < t - eps:
        return 'NO_MATERIAL_DIFFERENCE'
    return 'MIXED'


def safety_3to1_exceeds(n_a, n_b, v):
    """True if arm a has relatively more than +10% AND absolutely at least +30 more 3->1 errors than arm b."""
    return bool(n_a > n_b * (1 + v['safety_3to1_relative']) and (n_a - n_b) >= v['safety_3to1_absolute'])


def safety_flags(arm_a, n_a, arm_b, n_b, v):
    """Bidirectional flag list: the arm(s) that carry SAFETY_3to1 within the pair."""
    flags = []
    if safety_3to1_exceeds(n_a, n_b, v):
        flags.append(arm_a)
    if safety_3to1_exceeds(n_b, n_a, v):
        flags.append(arm_b)
    return flags


def ratio_class(r, v):
    return 'LOWER' if r <= v['ratio_lower_max'] else 'HIGHER' if r >= v['ratio_higher_min'] else 'SIMILAR'


def ratio_summary(ratios, v):
    """Median of fold ratios NUM/base with range; BOUNDARY if the per-fold classes are not all the same."""
    r = [float(x) for x in ratios if x is not None and np.isfinite(x)]
    if not r:
        return dict(median=None, min=None, max=None, fold_ratios=[], tag=None, boundary=False)
    med = float(np.median(r))
    classes = {ratio_class(x, v) for x in r}
    return dict(median=med, min=min(r), max=max(r), fold_ratios=r, tag=ratio_class(med, v),
                boundary=bool(len(classes) > 1))


def memory_pair_eligible(inc_a_mib, inc_b_mib, floor_a, floor_b, min_mib):
    """Pairs with an increment below 16 MiB or a FLOOR_LIMITED aux reading are excluded from the memory tag."""
    return bool(inc_a_mib >= min_mib and inc_b_mib >= min_mib and not floor_a and not floor_b)


def safe_ratio(a, b):
    if a is None or b is None or b == 0:
        return None
    return float(a) / float(b)
