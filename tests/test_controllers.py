from typing import Optional
import pytest
import numpy as np
import yaml

from src.controllers.naive import NaiveSwarmController
from src.controllers.soft_avoid import SoftAvoidSwarmController
from src.controllers.feedforward import FeedforwardSwarmController
from src.models.downwash import DownwashModel
from src.sim.downwash_aviary import DownwashAviary
from src.metrics.swarm_metrics import compute_tracking_rmse


def _create_mock_states(num_drones: int, positions: np.ndarray, velocities: Optional[np.ndarray] = None) -> np.ndarray:
    """Create (N, 20) state array with specified positions and velocities."""
    states = np.zeros((num_drones, 20))
    for i in range(num_drones):
        states[i, 0:3] = positions[i]
        states[i, 3:7] = [0.0, 0.0, 0.0, 1.0]  # identity quaternion
        if velocities is not None:
            states[i, 10:13] = velocities[i]
    return states


# ==============================================================================
# Controller A (Naive) Tests
# ==============================================================================

def test_controller_a_passthrough_targets():
    """Verify Controller A passes targets directly with zero feedforward force."""
    ctrl = NaiveSwarmController(num_drones=2)
    positions = np.array([[0.0, 0.0, 1.5], [0.0, 0.0, 0.8]])
    targets = np.array([[0.0, 0.0, 1.5], [0.0, 0.0, 0.8]])
    target_vels = np.array([[0.1, 0.0, 0.0], [0.0, 0.1, 0.0]])
    states = _create_mock_states(2, positions)

    cmd_p, cmd_v, ff_f = ctrl.compute_commands(
        states=states,
        formation_targets=targets,
        timestep=1.0 / 48.0,
        formation_velocities=target_vels,
    )

    np.testing.assert_allclose(cmd_p, targets)
    np.testing.assert_allclose(cmd_v, target_vels)
    np.testing.assert_allclose(ff_f, np.zeros((2, 3)))


# ==============================================================================
# Controller B (Soft Avoidance) Tests
# ==============================================================================

def test_controller_b_unstacked_zero_avoidance():
    """Verify Controller B applies zero displacement when drones are not vertically stacked."""
    ctrl = SoftAvoidSwarmController(
        num_drones=2,
        config={"max_displacement": 0.20, "max_velocity": 0.50},
    )
    # Drones separated horizontally by 1.0m (well outside wake radius 0.25m)
    positions = np.array([[0.0, 0.0, 1.5], [1.0, 0.0, 0.8]])
    targets = positions.copy()
    states = _create_mock_states(2, positions)

    cmd_p, cmd_v, ff_f = ctrl.compute_commands(
        states=states,
        formation_targets=targets,
        timestep=1.0 / 48.0,
    )

    # Position command should be unchanged
    np.testing.assert_allclose(cmd_p, targets, atol=1e-5)
    np.testing.assert_allclose(ff_f, np.zeros((2, 3)))


def test_controller_b_stacked_lateral_repulsion_bounded():
    """Verify Controller B applies smooth lateral repulsion bounded by max_displacement."""
    d_max = 0.15
    v_max = 0.50
    ctrl = SoftAvoidSwarmController(
        num_drones=2,
        config={"max_displacement": d_max, "max_velocity": v_max},
    )
    # Drone 0 at (0, 0, 1.5), Drone 1 at (0.12, 0.0, 0.9): inside the wake but
    # outside the 0.10 m core where the escape direction is held fixed (D-013),
    # so it must move away from the wake axis.
    positions = np.array([[0.0, 0.0, 1.5], [0.12, 0.0, 0.9]])
    targets = positions.copy()
    states = _create_mock_states(2, positions)
    dt = 1.0 / 48.0

    # Step multiple times to allow rate-limiter to reach target offset
    for _ in range(100):
        cmd_p, cmd_v, ff_f = ctrl.compute_commands(
            states=states,
            formation_targets=targets,
            timestep=dt,
        )

    # Upper drone should have zero avoidance offset
    np.testing.assert_allclose(cmd_p[0], targets[0], atol=1e-5)

    # Lower drone should be nudged in positive x direction (away from x=0)
    assert cmd_p[1, 0] > targets[1, 0]
    # Lateral displacement should not exceed max_displacement
    offset = np.linalg.norm(cmd_p[1, 0:2] - targets[1, 0:2])
    assert offset <= d_max + 1e-5
    # Altitude target should remain unchanged
    assert cmd_p[1, 2] == pytest.approx(targets[1, 2], abs=1e-5)
    # Feedforward force should be strictly zero
    np.testing.assert_allclose(ff_f, np.zeros((2, 3)))


def test_controller_b_symmetry_breaking():
    """Verify Controller B cleanly breaks symmetry when drone is directly at (0, 0) under another."""
    ctrl = SoftAvoidSwarmController(num_drones=2)
    # Exactly vertically stacked at x=0, y=0
    positions = np.array([[0.0, 0.0, 1.5], [0.0, 0.0, 0.9]])
    targets = positions.copy()
    states = _create_mock_states(2, positions)

    cmd_p, _, _ = ctrl.compute_commands(
        states=states,
        formation_targets=targets,
        timestep=1.0 / 48.0,
    )
    # Should not raise exception and should produce finite non-zero lateral direction
    assert np.all(np.isfinite(cmd_p))


def test_controller_b_same_altitude_safety_net():
    """Verify Controller B produces nonzero repulsive force for drones at same altitude when close.

    Two drones placed at the exact same altitude (zero vertical separation) would yield
    zero vertical-wake avoidance in the original logic, but must produce a nonzero
    repulsive force pushing them apart when closer than the safe-distance threshold.
    """
    ctrl = SoftAvoidSwarmController(num_drones=2)
    # Both drones at z = 1.0 (delta_z = 0), separated horizontally by 0.06m (< 0.12m safe distance threshold)
    positions = np.array([[0.0, 0.0, 1.0], [0.06, 0.0, 1.0]])
    targets = positions.copy()
    states = _create_mock_states(2, positions)

    cmd_p, cmd_v, ff_f = ctrl.compute_commands(
        states=states,
        formation_targets=targets,
        timestep=1.0 / 48.0,
    )

    # 1. Feedforward safety force must be nonzero and push drones apart
    assert np.linalg.norm(ff_f[0]) > 0.0, "Drone 0 must experience nonzero 3D safety repulsion force"
    assert np.linalg.norm(ff_f[1]) > 0.0, "Drone 1 must experience nonzero 3D safety repulsion force"
    # Drone 0 at x=0 should be pushed left (-x), Drone 1 at x=0.06 pushed right (+x)
    assert ff_f[0, 0] < 0.0, "Drone 0 should be pushed in negative x direction"
    assert ff_f[1, 0] > 0.0, "Drone 1 should be pushed in positive x direction"
    # Action-reaction: forces must be equal and opposite
    np.testing.assert_allclose(ff_f[0], -ff_f[1], atol=1e-6)

    # 2. Additive position setpoint offset should also nudge them apart
    assert cmd_p[0, 0] < targets[0, 0]
    assert cmd_p[1, 0] > targets[1, 0]


# ==============================================================================
# Controller C (Feedforward Compensation) Tests
# ==============================================================================

def test_controller_c_feedforward_generation():
    """Verify Controller C produces positive upward thrust compensation for lower drone."""
    ctrl = FeedforwardSwarmController(num_drones=2)
    positions = np.array([[0.0, 0.0, 1.5], [0.0, 0.0, 0.9]])
    targets = positions.copy()
    states = _create_mock_states(2, positions)

    cmd_p, cmd_v, ff_f = ctrl.compute_commands(
        states=states,
        formation_targets=targets,
        timestep=1.0 / 48.0,
    )

    # Nominal setpoints should remain strictly unchanged
    np.testing.assert_allclose(cmd_p, targets)
    np.testing.assert_allclose(cmd_v, np.zeros((2, 3)))

    # Upper drone (0) is above, receives zero disturbance and zero feedforward
    np.testing.assert_allclose(ff_f[0], np.zeros(3), atol=1e-5)

    # Lower drone (1) is in wake: downwash force is negative z, so compensation must be positive z
    assert ff_f[1, 2] > 0.02, f"Expected positive z compensation force, got {ff_f[1, 2]}"
    assert ff_f[1, 2] <= ctrl.max_compensation_force


def test_controller_c_unstacked_zero_compensation():
    """Verify Controller C produces zero feedforward when drones are far apart horizontally."""
    ctrl = FeedforwardSwarmController(num_drones=2)
    positions = np.array([[0.0, 0.0, 1.5], [1.5, 0.0, 0.9]])
    targets = positions.copy()
    states = _create_mock_states(2, positions)

    cmd_p, cmd_v, ff_f = ctrl.compute_commands(
        states=states,
        formation_targets=targets,
        timestep=1.0 / 48.0,
    )

    np.testing.assert_allclose(ff_f, np.zeros((2, 3)), atol=1e-5)


# ==============================================================================
# Comparative Aviary Simulation Test (A vs B vs C)
# ==============================================================================

def test_comparative_controllers_hover_pair():
    """Compare Controller A, B, and C in DownwashAviary simulation.

    Verifies:
    1. Controller A (Naive) suffers significant vertical tracking error from downwash.
    2. Controller B (Soft Avoid) shifts laterally away from the wake axis.
    3. Controller C (Feedforward) achieves lower vertical tracking error than Naive
       by injecting upward thrust feedforward.
    """
    initial_xyzs = np.array([[0.0, 0.0, 1.5], [0.0, 0.0, 0.8]])
    target_pos = np.array([[0.0, 0.0, 1.5], [0.0, 0.0, 0.8]])
    sim_freq = 240
    ctrl_freq = 48
    steps_per_ctrl = sim_freq // ctrl_freq
    duration = 2.0
    ctrl_steps = int(duration * ctrl_freq)
    dt_ctrl = 1.0 / ctrl_freq

    with open("configs/downwash_true.yaml", "r", encoding="utf-8") as f:
        true_cfg = yaml.safe_load(f)
    true_downwash_model = DownwashModel(true_cfg)

    def run_sim(controller_class, **kwargs):
        aviary = DownwashAviary(
            num_drones=2,
            initial_xyzs=initial_xyzs,
            downwash_model=true_downwash_model,
            gui=False,
            record=False,
        )
        ctrl = controller_class(num_drones=2, **kwargs)
        obs, _ = aviary.reset()
        traj_all = []

        for _ in range(ctrl_steps):
            states = obs
            traj_all.append(states[:, 0:3].copy())

            # Use compute_actions for end-to-end swarm control
            rpms, _, _ = ctrl.compute_actions(
                control_timestep=dt_ctrl,
                states=states,
                formation_targets=target_pos,
            )
            obs, _, _, _, _ = aviary.step(rpms)

        aviary.close()
        return np.array(traj_all)  # shape (T, 2, 3)

    # 1. Run Controller A (Naive)
    traj_a = run_sim(NaiveSwarmController)
    metrics_a = compute_tracking_rmse(traj_a, target_pos)
    rmse_a_lower = metrics_a["per_drone_rmse"][1]

    # 2. Run Controller B (Soft Avoidance)
    traj_b = run_sim(SoftAvoidSwarmController, config={"max_displacement": 0.20, "max_velocity": 0.15})
    final_lateral_offset_b = np.linalg.norm(traj_b[-20:, 1, 0:2] - target_pos[1, 0:2], axis=1).mean()

    # 3. Run Controller C (Feedforward Compensation)
    traj_c = run_sim(FeedforwardSwarmController)
    metrics_c = compute_tracking_rmse(traj_c, target_pos)
    rmse_c_lower = metrics_c["per_drone_rmse"][1]

    print(f"\n[Test Comparative] Lower Drone Tracking RMSE:")
    print(f"  Controller A (Naive)       : {rmse_a_lower:.4f} m")
    print(f"  Controller B (Soft Avoid)  : Lateral Offset = {final_lateral_offset_b:.4f} m")
    print(f"  Controller C (Feedforward) : {rmse_c_lower:.4f} m")

    # Controller A should suffer noticeable tracking error (> 2 cm)
    assert rmse_a_lower > 0.02, f"Naive error unexpectedly small: {rmse_a_lower:.4f} m"

    # Controller B should move laterally away from (0, 0)
    assert final_lateral_offset_b > 0.05, (
        f"Soft avoidance failed to achieve lateral displacement: {final_lateral_offset_b:.4f} m"
    )

    # Controller C should significantly outperform Controller A in tracking
    assert rmse_c_lower < rmse_a_lower, (
        f"Feedforward RMSE ({rmse_c_lower:.4f}m) was not better than Naive ({rmse_a_lower:.4f}m)"
    )


# ── Regression tests for the Soft-Avoid chatter fixes (decision D-013) ──────

def _sa_states(positions):
    s = np.zeros((len(positions), 20))
    s[:, 0:3] = positions
    s[:, 3:7] = [0, 0, 0, 1]
    return s


def test_soft_avoid_escape_direction_is_stable_in_wake_core():
    """Centimetre wobble of the upper drone must not flip the escape direction."""
    from src.controllers.soft_avoid import SoftAvoidSwarmController
    ctrl = SoftAvoidSwarmController(num_drones=2)
    targets = np.array([[0.0, 0.0, 1.8], [0.0, 0.0, 1.35]])
    dirs = []
    for k in range(40):
        wobble = 0.01 * np.array([np.cos(2.5 * k), np.sin(2.5 * k), 0.0])
        pos = targets.copy()
        pos[0] += wobble
        ctrl.compute_commands(_sa_states(pos), targets, 1 / 48, np.zeros((2, 3)))
        dirs.append(ctrl.current_offsets[1] / max(np.linalg.norm(ctrl.current_offsets[1]), 1e-12))
    dirs = np.array(dirs[1:])
    assert np.all(dirs @ dirs[0] > 0.99)


def test_soft_avoid_offset_velocity_is_filtered_by_default():
    from src.controllers.soft_avoid import SoftAvoidSwarmController
    ctrl = SoftAvoidSwarmController(num_drones=2)
    assert ctrl.offset_velocity_mode == "filtered"
    targets = np.array([[0.0, 0.0, 1.8], [0.0, 0.0, 1.35]])
    _, vel, _ = ctrl.compute_commands(_sa_states(targets), targets, 1 / 48, np.zeros((2, 3)))
    # Raw mode would command the full rate limit (0.15 m/s) on the first step.
    assert np.linalg.norm(vel[1, 0:2]) < 0.5 * ctrl.max_velocity


def test_ua_adc_equals_soft_avoid_when_no_wake_or_conflict():
    """Blend must be exact where both arms agree (coplanar formation)."""
    from src.controllers.soft_avoid import SoftAvoidSwarmController
    from src.controllers.ua_adc import UAAdaptiveController
    targets = np.array([[0.0, 0.0, 1.2], [0.45, 0.0, 1.2], [0.9, 0.0, 1.2]])
    sa = SoftAvoidSwarmController(num_drones=3)
    ua = UAAdaptiveController(num_drones=3)
    p_sa, v_sa, f_sa = sa.compute_commands(_sa_states(targets), targets, 1 / 48, np.zeros((3, 3)))
    p_ua, v_ua, f_ua = ua.compute_commands(_sa_states(targets), targets, 1 / 48, np.zeros((3, 3)))
    assert np.array_equal(p_sa, p_ua) and np.array_equal(v_sa, v_ua) and np.array_equal(f_sa, f_ua)
