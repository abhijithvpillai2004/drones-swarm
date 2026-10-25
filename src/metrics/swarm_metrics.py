"""Swarm performance metrics module.

Computes:
1. Tracking RMSE (per-drone and swarm aggregate)
2. Minimum inter-drone distance across trajectory
3. Near-collision count (violations of safety sphere threshold)
4. Energy/thrust proxy (integral of squared motor RPMs)
5. Formation settling time (time to enter and remain within error tolerance)
6. Task completion time
"""
from typing import Dict, Any, Tuple, Optional
import numpy as np


def compute_tracking_rmse(
    actual_pos: np.ndarray,
    target_pos: np.ndarray,
) -> Dict[str, Any]:
    """Compute per-drone and swarm-wide Root Mean Square Error (RMSE).

    Args:
        actual_pos: Array of shape (T, N, 3) where T is timesteps, N is drones.
        target_pos: Array of shape (T, N, 3) or (N, 3) of reference positions.

    Returns:
        Dictionary containing:
            - 'per_drone_rmse': 1D array of shape (N,) with RMSE per drone [m].
            - 'swarm_mean_rmse': Mean RMSE across all drones [m].
            - 'swarm_max_rmse': Maximum RMSE among all drones [m].
    """
    actual_pos = np.asarray(actual_pos, dtype=float)
    target_pos = np.asarray(target_pos, dtype=float)

    if target_pos.ndim == 2:
        # Broadcast (N, 3) to (T, N, 3)
        target_pos = np.broadcast_to(target_pos[np.newaxis, :, :], actual_pos.shape)

    errors = actual_pos - target_pos  # (T, N, 3)
    squared_distances = np.sum(errors ** 2, axis=2)  # (T, N)
    per_drone_rmse = np.sqrt(np.mean(squared_distances, axis=0))  # (N,)

    return {
        "per_drone_rmse": per_drone_rmse,
        "swarm_mean_rmse": float(np.mean(per_drone_rmse)),
        "swarm_max_rmse": float(np.max(per_drone_rmse)),
    }


def compute_min_interdrone_distance(
    actual_pos: np.ndarray,
    timestamps: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Compute minimum pairwise Euclidean distance between any two drones.

    Args:
        actual_pos: Array of shape (T, N, 3).
        timestamps: Optional 1D array of shape (T,) with time in seconds.

    Returns:
        Dictionary containing:
            - 'min_distance': Minimum pairwise distance [m].
            - 'time_at_min': Timestamp at which minimum occurred [s].
            - 'drone_pair': Tuple of drone indices (i, j) involved.
    """
    actual_pos = np.asarray(actual_pos, dtype=float)
    T, N, _ = actual_pos.shape

    if N < 2:
        return {
            "min_distance": float("inf"),
            "time_at_min": 0.0,
            "drone_pair": (0, 0),
        }

    min_dist = float("inf")
    min_time = 0.0
    min_pair = (0, 1)

    for i in range(N):
        for j in range(i + 1, N):
            diff = actual_pos[:, i, :] - actual_pos[:, j, :]
            distances = np.linalg.norm(diff, axis=1)  # (T,)
            idx = int(np.argmin(distances))
            if distances[idx] < min_dist:
                min_dist = float(distances[idx])
                min_pair = (i, j)
                min_time = float(timestamps[idx]) if timestamps is not None else float(idx)

    return {
        "min_distance": min_dist,
        "time_at_min": min_time,
        "drone_pair": min_pair,
    }


def compute_near_collisions(
    actual_pos: np.ndarray,
    safety_threshold: float = 0.15,
) -> Dict[str, Any]:
    """Count timesteps and distinct proximity events where inter-drone distance falls below threshold.

    Args:
        actual_pos: Array of shape (T, N, 3).
        safety_threshold: Minimum allowable safe inter-drone distance in meters.

    Returns:
        Dictionary containing:
            - 'near_collision_steps': Total number of timesteps where any pair < threshold.
            - 'near_collision_pairs': Number of drone pairs that violated threshold at least once.
    """
    actual_pos = np.asarray(actual_pos, dtype=float)
    T, N, _ = actual_pos.shape

    if N < 2:
        return {"near_collision_steps": 0, "near_collision_pairs": 0}

    step_violations = np.zeros(T, dtype=bool)
    pairs_violating = 0

    for i in range(N):
        for j in range(i + 1, N):
            diff = actual_pos[:, i, :] - actual_pos[:, j, :]
            distances = np.linalg.norm(diff, axis=1)
            violated = distances < safety_threshold
            if np.any(violated):
                pairs_violating += 1
            step_violations = step_violations | violated

    return {
        "near_collision_steps": int(np.sum(step_violations)),
        "near_collision_pairs": pairs_violating,
    }


def compute_energy_proxy(
    motor_rpms: np.ndarray,
    dt: float,
) -> Dict[str, Any]:
    """Compute energy/thrust proxy metric proportional to aerodynamic work.

    Calculates the integral of squared motor RPMs: Integral( sum_i sum_m RPM_{i,m}^2 dt ).

    Args:
        motor_rpms: Array of shape (T, N, 4) containing motor RPMs.
        dt: Control timestep in seconds.

    Returns:
        Dictionary containing:
            - 'total_energy_proxy': Swarm total energy proxy [RPM^2 * s].
            - 'per_drone_energy': Array of shape (N,) with energy proxy per drone.
    """
    motor_rpms = np.asarray(motor_rpms, dtype=float)
    # Sum over motors (axis 2) -> (T, N)
    squared_rpm_sum = np.sum(motor_rpms ** 2, axis=2)
    # Integrate over time
    per_drone_energy = np.sum(squared_rpm_sum, axis=0) * dt  # (N,)
    total_energy = float(np.sum(per_drone_energy))

    return {
        "total_energy_proxy": total_energy,
        "per_drone_energy": per_drone_energy,
    }


def compute_settling_time(
    actual_pos: np.ndarray,
    target_pos: np.ndarray,
    timestamps: np.ndarray,
    tolerance: float = 0.05,
) -> float:
    """Compute formation settling time.

    Defined as the earliest time t_s such that for all t >= t_s,
    every drone's position error is <= tolerance.
    If the swarm never settles, returns the final simulation time.

    Args:
        actual_pos: Array of shape (T, N, 3).
        target_pos: Array of shape (T, N, 3) or (N, 3).
        timestamps: 1D array of shape (T,) with time in seconds.
        tolerance: Error tolerance sphere radius [m].

    Returns:
        settling_time: Time in seconds.
    """
    actual_pos = np.asarray(actual_pos, dtype=float)
    target_pos = np.asarray(target_pos, dtype=float)
    timestamps = np.asarray(timestamps, dtype=float)

    if target_pos.ndim == 2:
        target_pos = np.broadcast_to(target_pos[np.newaxis, :, :], actual_pos.shape)

    errors = np.linalg.norm(actual_pos - target_pos, axis=2)  # (T, N)
    # Swarm is settled at step t if max error across all drones is <= tolerance
    max_errors = np.max(errors, axis=1)  # (T,)
    within_tolerance = max_errors <= tolerance  # (T,)

    T = len(within_tolerance)
    if not within_tolerance[-1]:
        # Never settled at end of simulation
        return float(timestamps[-1])

    # Scan backward from end: find earliest index where it remained settled continuously
    settled_idx = T - 1
    while settled_idx > 0 and within_tolerance[settled_idx - 1]:
        settled_idx -= 1

    return float(timestamps[settled_idx])


def compute_all_metrics(
    actual_pos: np.ndarray,
    target_pos: np.ndarray,
    timestamps: np.ndarray,
    motor_rpms: np.ndarray,
    dt: float,
    safety_threshold: float = 0.15,
    settling_tolerance: float = 0.05,
) -> Dict[str, Any]:
    """Unified wrapper computing all evaluation metrics for a swarm simulation run.

    Args:
        actual_pos: Array of shape (T, N, 3).
        target_pos: Array of shape (T, N, 3) or (N, 3).
        timestamps: Array of shape (T,).
        motor_rpms: Array of shape (T, N, 4).
        dt: Control timestep in seconds.
        safety_threshold: Near-collision threshold in meters.
        settling_tolerance: Formation settling threshold in meters.

    Returns:
        Comprehensive metrics dictionary.
    """
    rmse_dict = compute_tracking_rmse(actual_pos, target_pos)
    min_dist_dict = compute_min_interdrone_distance(actual_pos, timestamps)
    collision_dict = compute_near_collisions(actual_pos, safety_threshold)
    energy_dict = compute_energy_proxy(motor_rpms, dt)
    settling_t = compute_settling_time(actual_pos, target_pos, timestamps, settling_tolerance)

    return {
        "rmse_mean": rmse_dict["swarm_mean_rmse"],
        "rmse_max": rmse_dict["swarm_max_rmse"],
        "per_drone_rmse": [float(x) for x in rmse_dict["per_drone_rmse"]],
        "min_interdrone_dist": min_dist_dict["min_distance"],
        "time_at_min_dist": min_dist_dict["time_at_min"],
        "near_collision_steps": collision_dict["near_collision_steps"],
        "near_collision_pairs": collision_dict["near_collision_pairs"],
        "total_energy_proxy": energy_dict["total_energy_proxy"],
        "settling_time": settling_t,
        "task_completion_time": settling_t,  # Synonymous for setpoint transitions
    }
