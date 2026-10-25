"""Unit and integration tests for the UA-ADC controller and disturbance estimator.

Tests cover:
  1. DroneDisturbanceEstimator — reset, update convergence, confidence bounds.
  2. SwarmDisturbanceEstimator — multi-drone step, shape contracts.
  3. UAAdaptiveController — interface contract, blend weight correctness,
     safety net always active, N-drone stability, N=5 no-collision check.
"""
from typing import Optional
import pytest
import numpy as np

from src.models.disturbance_estimator import DroneDisturbanceEstimator, SwarmDisturbanceEstimator
from src.controllers.ua_adc import UAAdaptiveController


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _mock_states(num_drones: int, positions: np.ndarray, velocities: Optional[np.ndarray] = None) -> np.ndarray:
    """Build (N, 20) state array from positions and optional velocities."""
    states = np.zeros((num_drones, 20))
    for i in range(num_drones):
        states[i, 0:3] = positions[i]
        states[i, 3:7] = [0.0, 0.0, 0.0, 1.0]  # identity quaternion
        if velocities is not None:
            states[i, 10:13] = velocities[i]
    return states


MINIMAL_ESTIMATOR_CONFIG = {
    "forgetting_factor": 0.98,
    "prior_variance": 1.0,
    "min_confidence_variance": 1.0,
    "max_confidence_variance": 1e-3,
    "confidence_smoothing": 0.90,
    "max_gain": 3.0,
    "gravity": 9.81,
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. DroneDisturbanceEstimator
# ─────────────────────────────────────────────────────────────────────────────

class TestDroneDisturbanceEstimator:

    def test_initial_confidence_is_zero(self):
        """Estimator starts with zero confidence (high uncertainty)."""
        est = DroneDisturbanceEstimator(drone_id=0, mass=0.027, config=MINIMAL_ESTIMATOR_CONFIG)
        assert est.confidence == pytest.approx(0.0)

    def test_reset_restores_state(self):
        """After several updates, reset returns estimator to initial state."""
        est = DroneDisturbanceEstimator(drone_id=0, mass=0.027, config=MINIMAL_ESTIMATOR_CONFIG)
        assumed = np.array([0.0, 0.0, -0.05])
        mg = 0.027 * 9.81
        for _ in range(50):
            est.update(assumed, np.zeros(3), 1.0 / 48, commanded_thrust=mg + 0.05)
        assert est.confidence > 0.5, "confidence should have increased after consistent steps"
        est.reset()
        assert est.confidence == pytest.approx(0.0)
        assert est.P == pytest.approx(MINIMAL_ESTIMATOR_CONFIG["prior_variance"])
        assert est.theta == pytest.approx(1.0)

    def test_confidence_in_valid_range(self):
        """Confidence stays in [0, 1] after many updates."""
        est = DroneDisturbanceEstimator(drone_id=0, mass=0.027, config=MINIMAL_ESTIMATOR_CONFIG)
        assumed = np.array([0.0, 0.0, -0.05])
        mg = 0.027 * 9.81
        for _ in range(200):
            est.update(assumed, np.zeros(3), 1.0 / 48, commanded_thrust=mg + 0.05)
        assert 0.0 <= est.confidence <= 1.0

    def test_confidence_increases_with_consistent_signal(self):
        """Consistent observations should drive confidence upward."""
        est = DroneDisturbanceEstimator(drone_id=0, mass=0.027, config=MINIMAL_ESTIMATOR_CONFIG)
        assumed = np.array([0.0, 0.0, -0.04])
        mg = 0.027 * 9.81
        c_before = est.confidence
        for _ in range(50):
            est.update(assumed, np.zeros(3), 1.0 / 48, commanded_thrust=mg + 0.04)
        assert est.confidence > 0.8, f"Expected confidence > 0.8, got {est.confidence}"
        assert est.confidence > c_before

    def test_confidence_drops_under_inconsistent_observations(self):
        """When observations conflict with the model, confidence drops toward zero."""
        est = DroneDisturbanceEstimator(drone_id=0, mass=0.027, config=MINIMAL_ESTIMATOR_CONFIG)
        assumed = np.array([0.0, 0.0, -0.04])
        mg = 0.027 * 9.81
        # Step 1: establish high confidence
        for _ in range(50):
            est.update(assumed, np.zeros(3), 1.0 / 48, commanded_thrust=mg + 0.04)
        assert est.confidence > 0.8
        c_high = est.confidence

        # Step 2: inject inconsistent observations (thrust opposing opposite direction)
        for _ in range(50):
            est.update(assumed, np.zeros(3), 1.0 / 48, commanded_thrust=mg - 0.15)
        assert est.confidence < c_high
        assert est.confidence < 0.3, f"Expected confidence to drop below 0.3, got {est.confidence}"

    def test_covariance_stays_bounded(self):
        """Posterior variance P must stay strictly within [min_covariance, max_covariance]."""
        cfg = dict(MINIMAL_ESTIMATOR_CONFIG)
        cfg["max_covariance"] = 4.0
        cfg["min_covariance"] = 1e-4
        est = DroneDisturbanceEstimator(drone_id=0, mass=0.027, config=cfg)
        assumed = np.array([0.0, 0.0, -0.05])
        mg = 0.027 * 9.81
        rng = np.random.default_rng(42)
        for _ in range(300):
            # random noisy thrust commands
            rand_t = mg + rng.uniform(-0.1, 0.1)
            est.update(assumed, np.zeros(3), 1.0 / 48, commanded_thrust=rand_t)
            assert 1e-4 <= est.P <= 4.0, f"Covariance P={est.P} violated bounds [1e-4, 4.0]"

    def test_theta_converges_when_assumed_model_matches(self):
        """Gain theta should converge to 1.0 when true disturbance matches assumed model."""
        est = DroneDisturbanceEstimator(drone_id=0, mass=0.027, config=MINIMAL_ESTIMATOR_CONFIG)
        est.theta = 0.2  # start away from 1.0
        assumed = np.array([0.0, 0.0, -0.05])
        mg = 0.027 * 9.81
        for _ in range(150):
            est.update(assumed, np.zeros(3), 1.0 / 48, commanded_thrust=mg + 0.05)
        assert est.theta == pytest.approx(1.0, abs=0.05)

    def test_excitation_threshold_freezes_updates(self):
        """When excitation is below min_excitation, P and theta must remain frozen."""
        est = DroneDisturbanceEstimator(drone_id=0, mass=0.027, config=MINIMAL_ESTIMATOR_CONFIG)
        p_init = est.P
        theta_init = est.theta
        # Tiny assumed force with normalized feature |phi/mg| < 0.01
        tiny_assumed = np.array([0.0, 0.0, -0.0005])
        for _ in range(50):
            est.update(tiny_assumed, np.zeros(3), 1.0 / 48, commanded_thrust=0.027 * 9.81 + 0.02)
        assert est.P == p_init, "P should not update when excitation is below threshold"
        assert est.theta == theta_init, "theta should not update when excitation is below threshold"

    def test_estimate_disturbance_shape(self):
        """estimate_disturbance returns a 3-vector."""
        est = DroneDisturbanceEstimator(drone_id=0, mass=0.027, config=MINIMAL_ESTIMATOR_CONFIG)
        assumed = np.array([0.02, 0.0, -0.09])
        result = est.estimate_disturbance(assumed)
        assert result.shape == (3,)

    def test_zero_assumed_force_gives_zero_estimate(self):
        """Zero assumed force → zero estimated force regardless of theta."""
        est = DroneDisturbanceEstimator(drone_id=0, mass=0.027, config=MINIMAL_ESTIMATOR_CONFIG)
        result = est.estimate_disturbance(np.zeros(3))
        np.testing.assert_allclose(result, np.zeros(3))

    def test_gain_clamped_within_max(self):
        """Gain theta must not exceed max_gain even under extreme noisy observations."""
        config = dict(MINIMAL_ESTIMATOR_CONFIG)
        config["max_gain"] = 2.0
        est = DroneDisturbanceEstimator(drone_id=0, mass=0.027, config=config)
        assumed = np.array([0.0, 0.0, -0.05])
        rng = np.random.default_rng(42)
        for _ in range(500):
            noisy_accel = np.array([0.0, 0.0, rng.normal(0, 5.0)])
            est.update(assumed, noisy_accel, 1.0 / 48)
        assert abs(est.theta) <= 2.0 + 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# 2. SwarmDisturbanceEstimator
# ─────────────────────────────────────────────────────────────────────────────

class TestSwarmDisturbanceEstimator:

    def test_step_output_shapes(self):
        """step() returns arrays of correct shapes."""
        N = 3
        est = SwarmDisturbanceEstimator(num_drones=N, mass=0.027, config=MINIMAL_ESTIMATOR_CONFIG)
        positions = np.array([[0.0, 0.0, 1.5], [0.0, 0.0, 1.0], [0.0, 0.0, 0.5]])
        states = _mock_states(N, positions)
        assumed = np.zeros((N, 3))
        assumed[:, 2] = -0.05
        dists, confs = est.step(states, assumed, 1.0 / 48)
        assert dists.shape == (N, 3)
        assert confs.shape == (N,)

    def test_confidences_in_valid_range(self):
        """All per-drone confidences must be in [0, 1]."""
        N = 4
        est = SwarmDisturbanceEstimator(num_drones=N, mass=0.027, config=MINIMAL_ESTIMATOR_CONFIG)
        positions = np.array([
            [0.0, 0.0, 1.8],
            [0.0, 0.0, 1.2],
            [0.0, 0.0, 0.6],
            [0.0, 0.0, 0.0],
        ])
        states = _mock_states(N, positions)
        assumed = np.zeros((N, 3))
        assumed[:, 2] = -0.07

        for _ in range(50):
            _, confs = est.step(states, assumed, 1.0 / 48)
        assert np.all(confs >= 0.0) and np.all(confs <= 1.0)

    def test_reset_clears_all_estimators(self):
        """After reset, all confidences return to zero."""
        N = 2
        est = SwarmDisturbanceEstimator(num_drones=N, mass=0.027, config=MINIMAL_ESTIMATOR_CONFIG)
        positions = np.array([[0.0, 0.0, 1.5], [0.0, 0.0, 0.8]])
        states = _mock_states(N, positions)
        assumed = np.zeros((N, 3))
        assumed[:, 2] = -0.05
        for _ in range(30):
            est.step(states, assumed, 1.0 / 48)
        est.reset()
        assert all(e.confidence == pytest.approx(0.0) for e in est.estimators)
        assert all(e.P == pytest.approx(MINIMAL_ESTIMATOR_CONFIG["prior_variance"]) for e in est.estimators)


# ─────────────────────────────────────────────────────────────────────────────
# 3. UAAdaptiveController — interface and correctness
# ─────────────────────────────────────────────────────────────────────────────

MINIMAL_UA_CONFIG = {
    "blend_threshold": 0.5,
    "blend_sharpness": 8.0,
    "max_compensation_force": 0.25,
    "drone_mass": 0.027,
    "estimator": MINIMAL_ESTIMATOR_CONFIG,
}

ASSUMED_PARAMS = {
    "radius_xy": 0.25,
    "radius_z": 0.80,
    "max_force_z": 0.15,
    "radial_falloff": "gaussian",
    "vertical_falloff": "exponential",
    "wake_skew_coefficient": 0.1,
    "safe_distance_threshold": 0.12,
    "safety_repulsion_max_force": 0.10,
    "safety_repulsion_displacement": 0.08,
}


class TestUAAdaptiveController:

    def _build_ctrl(self, num_drones: int = 2) -> UAAdaptiveController:
        return UAAdaptiveController(
            config=MINIMAL_UA_CONFIG,
            num_drones=num_drones,
            assumed_config=ASSUMED_PARAMS,
        )

    # ── output shapes ────────────────────────────────────────────────────────

    def test_compute_commands_output_shapes_n2(self):
        """compute_commands returns 3 arrays each of shape (2, 3)."""
        ctrl = self._build_ctrl(num_drones=2)
        positions = np.array([[0.0, 0.0, 1.5], [0.0, 0.0, 0.8]])
        targets = positions.copy()
        states = _mock_states(2, positions)
        p, v, f = ctrl.compute_commands(states, targets, 1.0 / 48)
        assert p.shape == (2, 3)
        assert v.shape == (2, 3)
        assert f.shape == (2, 3)

    def test_compute_commands_output_shapes_n5(self):
        """compute_commands returns correct shapes for N=5 drones."""
        N = 5
        ctrl = self._build_ctrl(num_drones=N)
        positions = np.array([[0.0, 0.0, 1.8 - 0.3 * i] for i in range(N)])
        targets = positions.copy()
        states = _mock_states(N, positions)
        p, v, f = ctrl.compute_commands(states, targets, 1.0 / 48)
        assert p.shape == (N, 3) and v.shape == (N, 3) and f.shape == (N, 3)

    # ── blend weight logic ───────────────────────────────────────────────────

    def test_sigmoid_blend_at_zero_confidence(self):
        """At zero confidence, sigmoid blend weight should be < 0.5."""
        alpha = UAAdaptiveController._sigmoid_blend(
            confidence=np.array([0.0]), threshold=0.5, sharpness=8.0
        )
        assert float(alpha[0]) < 0.5

    def test_sigmoid_blend_at_full_confidence(self):
        """At full confidence, sigmoid blend weight should be > 0.5."""
        alpha = UAAdaptiveController._sigmoid_blend(
            confidence=np.array([1.0]), threshold=0.5, sharpness=8.0
        )
        assert float(alpha[0]) > 0.5

    def test_sigmoid_blend_at_threshold_is_half(self):
        """At threshold confidence, blend weight should be exactly 0.5."""
        alpha = UAAdaptiveController._sigmoid_blend(
            confidence=np.array([0.5]), threshold=0.5, sharpness=8.0
        )
        assert float(alpha[0]) == pytest.approx(0.5, abs=1e-6)

    # ── force bounds ─────────────────────────────────────────────────────────

    def test_feedforward_forces_within_clamp(self):
        """Feedforward forces must not exceed max_compensation_force."""
        ctrl = self._build_ctrl(num_drones=3)
        positions = np.array([[0.0, 0.0, 1.8], [0.0, 0.0, 1.2], [0.0, 0.0, 0.6]])
        states = _mock_states(3, positions)
        targets = positions.copy()
        for _ in range(100):
            _, _, f = ctrl.compute_commands(states, targets, 1.0 / 48)
        # Compensation forces must be within configured max
        assert np.all(np.abs(f) <= MINIMAL_UA_CONFIG["max_compensation_force"] + 0.11)

    # ── safety net always active ─────────────────────────────────────────────

    def test_safety_net_activates_on_close_proximity(self):
        """When two drones are very close, safety forces should be non-zero."""
        ctrl = self._build_ctrl(num_drones=2)
        # Place drones very close together (3 cm apart — well inside 12 cm threshold)
        positions = np.array([[0.0, 0.0, 1.0], [0.03, 0.0, 1.0]])
        states = _mock_states(2, positions)
        targets = positions.copy()
        _, _, f = ctrl.compute_commands(states, targets, 1.0 / 48)
        # At least one drone must receive a non-zero repulsion force
        assert np.any(np.abs(f) > 1e-4), "safety net did not activate for close drones"

    # ── telemetry buffers ─────────────────────────────────────────────────────

    def test_telemetry_buffers_updated(self):
        """last_confidences and last_blend_weights should be updated after step."""
        ctrl = self._build_ctrl(num_drones=2)
        positions = np.array([[0.0, 0.0, 1.5], [0.0, 0.0, 0.8]])
        states = _mock_states(2, positions)
        targets = positions.copy()
        ctrl.compute_commands(states, targets, 1.0 / 48)
        assert ctrl.last_confidences.shape == (2,)
        assert ctrl.last_blend_weights.shape == (2,)
        assert np.all(ctrl.last_blend_weights >= 0.0)
        assert np.all(ctrl.last_blend_weights <= 1.0)

    # ── reset ────────────────────────────────────────────────────────────────

    def test_reset_clears_telemetry(self):
        """Reset restores telemetry and sub-controller state."""
        ctrl = self._build_ctrl(num_drones=2)
        positions = np.array([[0.0, 0.0, 1.5], [0.0, 0.0, 0.8]])
        states = _mock_states(2, positions)
        for _ in range(20):
            ctrl.compute_commands(states, positions, 1.0 / 48)
        ctrl.reset()
        assert np.all(ctrl.last_confidences == pytest.approx(0.0))
        assert np.all(ctrl.last_blend_weights == pytest.approx(0.0))

    # ── N=5 no-collision regression ─────────────────────────────────────────

    def test_n5_no_collision_hover_column(self):
        """N=5 hover column: minimum inter-drone 3D distance must stay > 5 cm."""
        N = 5
        ctrl = self._build_ctrl(num_drones=N)
        # Column: 30 cm vertical spacing
        z_positions = np.array([1.8 - 0.3 * i for i in range(N)])
        positions = np.column_stack([np.zeros(N), np.zeros(N), z_positions])
        states = _mock_states(N, positions)
        targets = positions.copy()
        target_vels = np.zeros((N, 3))
        dt = 1.0 / 48.0

        for _ in range(200):
            ctrl.compute_commands(states, targets, dt, target_vels)

        # Check minimum pairwise distance (should stay well above 5 cm)
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(positions[i] - positions[j])
                assert d > 0.05, (
                    f"Drones {i} and {j} too close: {d:.4f} m — collision risk!"
                )

    # ── scaling: N=2 through N=10 instantiation ─────────────────────────────

    @pytest.mark.parametrize("n", [2, 3, 4, 5, 6, 7, 8, 9, 10])
    def test_scaling_instantiation(self, n: int):
        """UA-ADC instantiates correctly for N in [2, 10]."""
        ctrl = UAAdaptiveController(
            config=MINIMAL_UA_CONFIG,
            num_drones=n,
            assumed_config=ASSUMED_PARAMS,
        )
        assert ctrl.num_drones == n
        assert len(ctrl.estimator.estimators) == n

    @pytest.mark.parametrize("n", [2, 5, 10])
    def test_scaling_one_step(self, n: int):
        """UA-ADC compute_commands runs without error for N in [2, 5, 10]."""
        ctrl = UAAdaptiveController(
            config=MINIMAL_UA_CONFIG,
            num_drones=n,
            assumed_config=ASSUMED_PARAMS,
        )
        dz = 0.3
        positions = np.array([[0.0, 0.0, 1.8 - dz * i] for i in range(n)])
        states = _mock_states(n, positions)
        targets = positions.copy()
        p, v, f = ctrl.compute_commands(states, targets, 1.0 / 48)
        assert p.shape == (n, 3)
        assert v.shape == (n, 3)
        assert f.shape == (n, 3)
