"""Deterministic native-spoke subsets with optimal circular angular uniformity."""
from __future__ import annotations

from functools import lru_cache

import numpy as np

POLICY = 'angular_gap_variance_v1'
WINDOW_SIZE = 96
WINDOW_STARTS = (0, 192)
GOLDEN_ANGLE = np.pi / ((1 + np.sqrt(5.0))/2)


def angular_gaps(local_indices):
    angles = np.sort(np.mod(np.asarray(local_indices)*GOLDEN_ANGLE, np.pi))
    return np.diff(np.r_[angles, angles[0]+np.pi])


def temporal_discrepancy(local_indices):
    """Empirical-CDF distance to uniform acquisition times within the window."""
    t = (np.sort(local_indices)+.5)/WINDOW_SIZE
    n = len(t)
    return float(max(np.max(t-np.arange(n)/n), np.max(np.arange(1, n+1)/n-t)))


@lru_cache(maxsize=WINDOW_SIZE)
def _local_subset(n):
    if not 1 <= n <= WINDOW_SIZE:
        raise ValueError('Retain between one and 96 complete spokes')
    if n == WINDOW_SIZE:
        return tuple(range(WINDOW_SIZE))
    angles = np.mod(np.arange(WINDOW_SIZE)*GOLDEN_ANGLE, np.pi)
    order = np.argsort(angles)
    sorted_angles = angles[order]
    candidates = []
    # A selected circular subset is a path with n edges from a starting angle
    # to that angle + pi. Dynamic programming solves its sum of squared gaps.
    # Enumerating all starts covers every possible native subset. Since gaps
    # sum to pi, this also minimizes squared deviation from the ideal pi/n.
    for start in range(WINDOW_SIZE):
        sequence = np.r_[sorted_angles[start:], sorted_angles[:start+1]+np.pi]
        distance = sequence[None, :]-sequence[:, None]
        costs = np.where(distance > 0, distance**2, np.inf)
        previous = np.full(WINDOW_SIZE+1, np.inf); previous[0] = 0
        parents = np.full((n+1, WINDOW_SIZE+1), -1, dtype=np.int16)
        for count in range(1, n+1):
            options = previous[:, None]+costs
            best = options.argmin(axis=0)
            previous = options[best, np.arange(WINDOW_SIZE+1)]
            parents[count] = best
        path, position = [], WINDOW_SIZE
        for count in range(n, 0, -1):
            position = int(parents[count, position]); path.append(position)
        if position != 0 or len(set(path)) != n:
            raise RuntimeError('Invalid angular subset path')
        indices = np.sort(order[(np.asarray(path)+start) % WINDOW_SIZE])
        candidates.append((float(previous[-1]), tuple(map(int, indices))))
    optimum = min(cost for cost, _ in candidates)
    # Geometry ties differ by numerical roundoff. Choose temporal coverage
    # among these equally optimal paths; no measurements, labels or Dice enter.
    tied = {indices for cost, indices in candidates if abs(cost-optimum) <= 1e-11}
    return min(tied, key=lambda indices: (temporal_discrepancy(np.asarray(indices)),
        abs(float(np.mean(indices))-(WINDOW_SIZE-1)/2), indices))


def selected_indices(reduction):
    if int(reduction) != reduction or not 1 <= reduction <= WINDOW_SIZE:
        raise ValueError('Spoke reduction must be an integer from 1 to 96')
    local = np.asarray(_local_subset(WINDOW_SIZE//int(reduction)), dtype=np.int64)
    return np.stack([local+start for start in WINDOW_STARTS])


def selection_metadata(reduction):
    selected = selected_indices(reduction); local = selected[0]
    gaps = angular_gaps(local)
    return dict(policy=POLICY, original_indices=selected.tolist(), retained_spokes=len(local),
        effective_reduction=WINDOW_SIZE/len(local), maximum_gap_degrees=float(np.rad2deg(gaps.max())),
        angular_gap_cv=float(gaps.std()/gaps.mean()), squared_gap_sum=float(np.sum(gaps**2)),
        temporal_discrepancy=temporal_discrepancy(local), mean_local_index=float(local.mean()),
        objective='Global minimum sum of squared circular gaps among subsets of the 96 native line directions; temporal-CDF tie break among optimal paths')
