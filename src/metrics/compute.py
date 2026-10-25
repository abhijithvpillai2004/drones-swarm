"""
src/metrics/compute.py
=======================
Metric functions for evaluating formation controller performance.

Every function here has a corresponding unit test in tests/test_metrics.py.

All metrics operate on arrays logged by SimRunner:
  positions : ndarray (T, N, 3) — per-step, per-drone 3D positions
  targets   : ndarray (N, 3)    — fixed formation targets
  rpms      : ndarray (T, N, 4) — motor RPMs per step
  ctrl_freq : float             — control frequency [Hz]

Design decision D-011 (docs/decisions.md):
  - RMSE computed over the LAST ``eval_fraction`` of steps to skip transients.
  - Settling time: first contiguous ``settling_duration`` seconds within threshold.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------

def tracking_rmse(
    positions: np.ndarray,
    targets: np.ndarray,
    eval_fraction: float = 0.80,
) -> float:
    """Root-mean-square tracking error per drone, averaged over drones.

    Computed over the final ``eval_fraction`` of time-steps to exclude
    the initial transient settling phase.

    Parameters
    ----------
    positions : ndarray (T, N, 3)
        Position history.
    targets : ndarray (N, 3)
        Formation target positions.
    eval_fraction : float
        Fraction of run to evaluate (from the end). Must be in (0, 1].

    Returns
    -------
    rmse : float [m]
    """
    T = positions.shape[0]
    start = int(T * (1.0 - eval_fraction))
    pos_eval = positions[start:]          # (T_eval, N, 3)
    errors = pos_eval - targets[np.newaxis, :, :]  # broadcast
    sq_errors = np.sum(errors ** 2, axis=2)        # (T_eval, N)
    return float(np.sqrt(np.mean(sq_errors)))


def min_inter_drone_distance(positions: np.ndarray) -> float:
    """Minimum pairwise inter-drone distance over the entire run.

    Parameters
    ----------
    positions : ndarray (T, N, 3)

    Returns
    -------
    min_dist : float [m]
        Minimum distance between any pair of drones at any time step.
        Returns infinity if N < 2.
    """
    T, N, _ = positions.shape
    if N < 2:
        return float("inf")

    min_d = float("inf")
    for t in range(T):
        pos_t = positions[t]  # (N, 3)
        for i in range(N):
            for j in range(i + 1, N):
                d = float(np.linalg.norm(pos_t[i] - pos_t[j]))
                if d < min_d:
                    min_d = d
    return min_d


def near_collision_count(
    positions: np.ndarray,
    threshold_m: float = 0.15,
) -> int:
    """Count steps where any pair of drones is closer than *threshold_m*.

    Parameters
    ----------
    positions : ndarray (T, N, 3)
    threshold_m : float
        Near-collision distance threshold [m]. Default 0.15 m.

    Returns
    -------
    count : int
        Number of control steps where at least one pair is too close.
    """
    T, N, _ = positions.shape
    if N < 2:
        return 0

    count = 0
    for t in range(T):
        pos_t = positions[t]
        close = False
        for i in range(N):
            for j in range(i + 1, N):
                if np.linalg.norm(pos_t[i] - pos_t[j]) < threshold_m:
                    close = True
                    break
            if close:
                break
        if close:
            count += 1
    return count


def thrust_energy_proxy(rpms: np.ndarray) -> float:
    """Mean sum-of-squared RPMs per step (proportional to electrical power).

    sum(rpm_i²) is proportional to total electrical power consumed
    (P ∝ Σ I_i * V ∝ Σ F_i ∝ Σ KF * rpm_i²).
    Normalised by number of drones for fair comparison.

    Parameters
    ----------
    rpms : ndarray (T, N, 4)

    Returns
    -------
    energy_proxy : float [RPM²]
    """
    N = rpms.shape[1]
    total_per_step = np.sum(rpms ** 2, axis=(1, 2))   # (T,)
    return float(np.mean(total_per_step)) / N


def settling_time(
    positions: np.ndarray,
    targets: np.ndarray,
    ctrl_freq: float,
    threshold_m: float = 0.05,
    settling_duration_s: float = 1.0,
) -> float:
    """Time until all drones settle within *threshold_m* of their targets.

    Settling is defined as all N drones simultaneously within threshold_m
    of their setpoints, and remaining so for settling_duration_s seconds.

    Parameters
    ----------
    positions : ndarray (T, N, 3)
    targets : ndarray (N, 3)
    ctrl_freq : float — control frequency [Hz]
    threshold_m : float — per-drone distance threshold [m]
    settling_duration_s : float — how long it must remain settled [s]

    Returns
    -------
    t_settle : float [s]
        Time to settle. Returns ``inf`` if drone never settles.
    """
    T = positions.shape[0]
    required_steps = int(settling_duration_s * ctrl_freq)

    errors = np.linalg.norm(positions - targets[np.newaxis, :, :], axis=2)  # (T,N)
    settled_per_step = np.all(errors < threshold_m, axis=1)  # (T,) bool

    for t in range(T - required_steps + 1):
        window = settled_per_step[t: t + required_steps]
        if np.all(window):
            return t / ctrl_freq

    return float("inf")


def task_completion_fraction(
    positions: np.ndarray,
    targets: np.ndarray,
    eval_fraction: float = 0.80,
    radius_m: float = 0.10,
) -> float:
    """Fraction of (step, drone) pairs where drone is within *radius_m* of target.

    Evaluated over the last ``eval_fraction`` of the run.

    Parameters
    ----------
    positions : ndarray (T, N, 3)
    targets : ndarray (N, 3)
    eval_fraction : float
    radius_m : float — success radius [m]

    Returns
    -------
    fraction : float in [0, 1]
    """
    T = positions.shape[0]
    start = int(T * (1.0 - eval_fraction))
    pos_eval = positions[start:]
    errors = np.linalg.norm(pos_eval - targets[np.newaxis, :, :], axis=2)
    return float(np.mean(errors < radius_m))


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def compute_all_metrics(
    positions: np.ndarray,
    targets: np.ndarray,
    rpms: np.ndarray,
    ctrl_freq: float,
    eval_fraction: float = 0.80,
    near_collision_dist: float = 0.15,
    settling_threshold: float = 0.05,
    settling_duration: float = 1.0,
    task_radius: float = 0.10,
) -> Dict[str, float]:
    """Compute all standard metrics and return as a flat dict.

    Parameters
    ----------
    positions : ndarray (T, N, 3)
    targets : ndarray (N, 3)
    rpms : ndarray (T, N, 4)
    ctrl_freq : float
    eval_fraction : float
    near_collision_dist : float [m]
    settling_threshold : float [m]
    settling_duration : float [s]
    task_radius : float [m]

    Returns
    -------
    metrics : dict
        Keys: rmse_m, min_dist_m, near_collisions, energy_proxy,
              settling_time_s, task_completion, rmse_per_drone (list)
    """
    T, N, _ = positions.shape
    start = int(T * (1.0 - eval_fraction))

    # Per-drone RMSE
    errors = positions - targets[np.newaxis, :, :]    # (T,N,3)
    sq_e = np.sum(errors ** 2, axis=2)                # (T,N)
    per_drone_rmse = [
        float(np.sqrt(np.mean(sq_e[start:, j]))) for j in range(N)
    ]

    return {
        "rmse_m": tracking_rmse(positions, targets, eval_fraction),
        "min_dist_m": min_inter_drone_distance(positions),
        "near_collisions": near_collision_count(positions, near_collision_dist),
        "energy_proxy": thrust_energy_proxy(rpms),
        "settling_time_s": settling_time(
            positions, targets, ctrl_freq, settling_threshold, settling_duration
        ),
        "task_completion": task_completion_fraction(
            positions, targets, eval_fraction, task_radius
        ),
        "rmse_per_drone": per_drone_rmse,
    }
