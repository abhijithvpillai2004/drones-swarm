"""Unit tests for swarm metrics computation using analytical ground-truth examples."""
import numpy as np
import pytest

from src.metrics.swarm_metrics import (
    compute_tracking_rmse,
    compute_min_interdrone_distance,
    compute_near_collisions,
    compute_energy_proxy,
    compute_settling_time,
    compute_all_metrics,
)


def test_compute_tracking_rmse_exact() -> None:
    """Verify tracking RMSE matches exact analytical values."""
    T, N = 10, 2
    # Drone 0: constant error vector [0.03, 0.04, 0.0] -> norm = 0.05
    # Drone 1: constant error vector [0.06, 0.08, 0.0] -> norm = 0.10
    target = np.zeros((T, N, 3))
    actual = np.zeros((T, N, 3))
    actual[:, 0, :] = [0.03, 0.04, 0.0]
    actual[:, 1, :] = [0.06, 0.08, 0.0]

    result = compute_tracking_rmse(actual, target)
    assert np.isclose(result["per_drone_rmse"][0], 0.05, atol=1e-6)
    assert np.isclose(result["per_drone_rmse"][1], 0.10, atol=1e-6)
    assert np.isclose(result["swarm_mean_rmse"], 0.075, atol=1e-6)
    assert np.isclose(result["swarm_max_rmse"], 0.10, atol=1e-6)


def test_compute_min_interdrone_distance() -> None:
    """Verify calculation of minimum inter-drone distance and timestamp."""
    timestamps = np.array([0.0, 1.0, 2.0, 3.0])
    actual = np.zeros((4, 2, 3))
    # Timestep 0: dist = 1.0
    actual[0, 0, :] = [0.0, 0.0, 1.0]
    actual[0, 1, :] = [1.0, 0.0, 1.0]
    # Timestep 1: dist = 0.5
    actual[1, 0, :] = [0.0, 0.0, 1.0]
    actual[1, 1, :] = [0.5, 0.0, 1.0]
    # Timestep 2: dist = 0.18 (closest approach)
    actual[2, 0, :] = [0.0, 0.0, 1.0]
    actual[2, 1, :] = [0.18, 0.0, 1.0]
    # Timestep 3: dist = 0.6
    actual[3, 0, :] = [0.0, 0.0, 1.0]
    actual[3, 1, :] = [0.6, 0.0, 1.0]

    result = compute_min_interdrone_distance(actual, timestamps)
    assert np.isclose(result["min_distance"], 0.18, atol=1e-6)
    assert np.isclose(result["time_at_min"], 2.0, atol=1e-6)
    assert result["drone_pair"] == (0, 1)


def test_compute_near_collisions() -> None:
    """Verify identification of safety boundary violations."""
    actual = np.zeros((5, 2, 3))
    # Step 0: dist = 0.50 (safe)
    actual[0, 1, :] = [0.50, 0.0, 0.0]
    # Step 1: dist = 0.12 (violation for threshold 0.15)
    actual[1, 1, :] = [0.12, 0.0, 0.0]
    # Step 2: dist = 0.10 (violation)
    actual[2, 1, :] = [0.10, 0.0, 0.0]
    # Step 3: dist = 0.14 (violation)
    actual[3, 1, :] = [0.14, 0.0, 0.0]
    # Step 4: dist = 0.20 (safe)
    actual[4, 1, :] = [0.20, 0.0, 0.0]

    result = compute_near_collisions(actual, safety_threshold=0.15)
    assert result["near_collision_steps"] == 3
    assert result["near_collision_pairs"] == 1


def test_compute_energy_proxy_analytical() -> None:
    """Verify energy proxy integral matches sum of squared RPMs over time."""
    T, N = 100, 2
    dt = 0.02
    motor_rpms = np.zeros((T, N, 4))
    # Each motor runs at 10,000 RPM
    motor_rpms[:, :, :] = 10000.0

    # Per drone per step: 4 * (10000)^2 = 4 * 10^8
    # Integral over T steps (T * dt = 2.0 sec): (4 * 10^8) * 2.0 = 8.0 * 10^8 per drone
    # Swarm total = 2 * 8.0 * 10^8 = 1.6 * 10^9
    result = compute_energy_proxy(motor_rpms, dt)
    expected_drone = 4.0 * (10000.0 ** 2) * (T * dt)
    expected_total = expected_drone * N

    assert np.isclose(result["per_drone_energy"][0], expected_drone, rtol=1e-5)
    assert np.isclose(result["total_energy_proxy"], expected_total, rtol=1e-5)


def test_compute_settling_time() -> None:
    """Verify settling time accurately finds the entry into steady tolerance."""
    timestamps = np.linspace(0.0, 5.0, 51)
    T = len(timestamps)
    N = 1
    actual = np.zeros((T, N, 3))
    target = np.zeros((T, N, 3))

    # Error starts high (0.5m), drops below tolerance (0.05m) at index 20 (t = 2.0s) and stays there
    for i in range(T):
        if i < 20:
            actual[i, 0, 0] = 0.20  # Error > 0.05m
        else:
            actual[i, 0, 0] = 0.02  # Error < 0.05m

    settling_t = compute_settling_time(actual, target, timestamps, tolerance=0.05)
    assert np.isclose(settling_t, 2.0, atol=1e-5)


def test_compute_all_metrics_wrapper() -> None:
    """Verify that compute_all_metrics packages all keys properly."""
    timestamps = np.linspace(0, 1, 10)
    actual = np.zeros((10, 2, 3))
    target = np.zeros((10, 2, 3))
    rpms = np.full((10, 2, 4), 14000.0)

    summary = compute_all_metrics(actual, target, timestamps, rpms, dt=0.1)
    assert "rmse_mean" in summary
    assert "min_interdrone_dist" in summary
    assert "near_collision_steps" in summary
    assert "total_energy_proxy" in summary
    assert "settling_time" in summary
