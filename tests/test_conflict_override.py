"""Tests for the UA-ADC conflict (closing-speed) override."""
import numpy as np
import pytest

from src.controllers.ua_adc import UAAdaptiveController

cpa = UAAdaptiveController._pair_cpa_risk


def test_head_on_vertical_approach_is_full_risk():
    # 0.8 m apart vertically, closing at 0.5 m/s on the same column -> collision course
    assert cpa(np.array([0, 0, 0.8]), np.array([0, 0, -0.5]), 0.30, 2.5) == pytest.approx(1.0)


def test_static_stack_has_no_risk():
    # Hover column spacing 0.45 m, no relative motion
    assert cpa(np.array([0, 0, 0.45]), np.zeros(3), 0.30, 2.5) == 0.0


def test_separating_pair_has_no_risk():
    assert cpa(np.array([0, 0, 0.5]), np.array([0, 0, 0.5]), 0.30, 2.5) == 0.0


def test_pass_with_large_miss_distance_has_no_risk():
    # Crossing pair: 0.6 m vertical offset, horizontal closing
    assert cpa(np.array([1.0, 0, 0.6]), np.array([-0.5, 0, 0]), 0.30, 2.5) == 0.0


def test_conflict_beyond_horizon_is_ignored():
    # Collision course, but 10 s away with a 2.5 s horizon
    assert cpa(np.array([0, 0, 5.0]), np.array([0, 0, -0.5]), 0.30, 2.5) == 0.0


def test_risk_tapers_with_miss_distance():
    r_close = cpa(np.array([0.05, 0, 0.8]), np.array([0, 0, -0.5]), 0.30, 2.5)
    r_far = cpa(np.array([0.25, 0, 0.8]), np.array([0, 0, -0.5]), 0.30, 2.5)
    assert 1.0 >= r_close > r_far > 0.0


def _states(positions, velocities):
    s = np.zeros((len(positions), 20))
    s[:, 0:3] = positions
    s[:, 3:7] = [0, 0, 0, 1]
    s[:, 10:13] = velocities
    return s


def _force_confident(ctrl):
    for est in ctrl.estimator.estimators:
        est.confidence = 1.0


def test_override_forces_avoidance_on_swap():
    ctrl = UAAdaptiveController(num_drones=2)
    _force_confident(ctrl)
    pos = np.array([[0, 0, 1.6], [0, 0, 0.8]], dtype=float)
    vel = np.array([[0, 0, -0.25], [0, 0, 0.25]], dtype=float)
    ctrl.compute_commands(_states(pos, vel), pos, 1 / 48, formation_velocities=vel)
    assert np.all(ctrl.last_conflict_risk > 0.99)
    assert np.all(ctrl.last_blend_weights < 0.01)


def test_override_disabled_keeps_confidence_blend():
    cfg = {"conflict_override": {"enabled": False}}
    ctrl = UAAdaptiveController(config=cfg, num_drones=2)
    _force_confident(ctrl)
    pos = np.array([[0, 0, 1.6], [0, 0, 0.8]], dtype=float)
    vel = np.array([[0, 0, -0.25], [0, 0, 0.25]], dtype=float)
    ctrl.compute_commands(_states(pos, vel), pos, 1 / 48, formation_velocities=vel)
    assert np.all(ctrl.last_conflict_risk == 0.0)


def test_override_does_not_fire_in_static_hover_column():
    ctrl = UAAdaptiveController(num_drones=3)
    pos = np.array([[0, 0, 1.8], [0, 0, 1.35], [0, 0, 0.9]], dtype=float)
    ctrl.compute_commands(_states(pos, np.zeros((3, 3))), pos, 1 / 48, formation_velocities=np.zeros((3, 3)))
    assert np.all(ctrl.last_conflict_risk == 0.0)


def test_risk_decays_after_conflict_clears():
    ctrl = UAAdaptiveController(num_drones=2)
    pos = np.array([[0, 0, 1.6], [0, 0, 0.8]], dtype=float)
    vel = np.array([[0, 0, -0.25], [0, 0, 0.25]], dtype=float)
    ctrl.compute_commands(_states(pos, vel), pos, 1 / 48, formation_velocities=vel)
    # Now far apart and separating: risk must decay, not drop instantly
    pos2 = np.array([[0.3, 0, 0.8], [0, 0, 1.6]], dtype=float)
    vel2 = -vel
    ctrl.compute_commands(_states(pos2, vel2), pos2, 1 / 48, formation_velocities=vel2)
    hold = ctrl.conflict_hold_s
    assert np.all(ctrl.last_conflict_risk == pytest.approx(1.0 - (1 / 48) / hold))


def test_default_config_is_loaded_from_yaml():
    ctrl = UAAdaptiveController(num_drones=2)
    # Values that only exist in configs/ua_adc.yaml
    assert ctrl.max_compensation_force == pytest.approx(0.25)
    assert ctrl.conflict_enabled is True
