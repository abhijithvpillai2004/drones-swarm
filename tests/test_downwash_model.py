"""Unit tests for the aerodynamic downwash disturbance model and physical simulation effects."""
import numpy as np
import pytest
import yaml

from src.models.downwash import DownwashModel
from src.sim.downwash_aviary import DownwashAviary
from src.controllers.single_drone_pid import SingleDronePIDController
from gym_pybullet_drones.utils.enums import DroneModel, Physics


@pytest.fixture
def default_downwash_model() -> DownwashModel:
    """Fixture providing a DownwashModel with default true configuration."""
    with open("configs/downwash_true.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    return DownwashModel(cfg)


def test_unstacked_zero_disturbance(default_downwash_model: DownwashModel) -> None:
    """Verify disturbance is ~0 / zero when drones are unstacked or outside wake boundary."""
    pos_upper = np.array([0.0, 0.0, 1.5])
    vel_upper = np.zeros(3)

    # 1. Lower drone placed ABOVE upper drone (dz > 0)
    pos_above = np.array([0.0, 0.0, 2.0])
    force = default_downwash_model.compute_pair_force(pos_upper, vel_upper, pos_above)
    assert np.all(force == 0.0), f"Expected 0 force when drone is above, got {force}"

    # 2. Lower drone placed far horizontally (dx = 1.0m >> radius_xy = 0.25m)
    pos_far_xy = np.array([1.0, 0.0, 1.0])
    force = default_downwash_model.compute_pair_force(pos_upper, vel_upper, pos_far_xy)
    assert np.all(force == 0.0), f"Expected 0 force when outside radial wake, got {force}"

    # 3. Lower drone placed far vertically (dz = 1.5m >> radius_z = 0.80m)
    pos_far_z = np.array([0.0, 0.0, -0.2])
    force = default_downwash_model.compute_pair_force(pos_upper, vel_upper, pos_far_z)
    assert np.all(force == 0.0), f"Expected 0 force when beyond vertical reach, got {force}"


def test_disturbance_increases_toward_wake_core(default_downwash_model: DownwashModel) -> None:
    """Verify disturbance magnitude monotonically increases as lower drone enters wake core."""
    pos_upper = np.array([0.0, 0.0, 1.5])
    vel_upper = np.zeros(3)
    dz = 0.35  # Constant vertical depth inside wake

    dx_samples = [0.30, 0.20, 0.15, 0.10, 0.05, 0.0]
    forces_z = []

    for dx in dx_samples:
        pos_lower = np.array([dx, 0.0, pos_upper[2] - dz])
        force = default_downwash_model.compute_pair_force(pos_upper, vel_upper, pos_lower)
        forces_z.append(abs(force[2]))

    # Disturbance must be non-decreasing as dx decreases toward center
    for k in range(len(forces_z) - 1):
        assert forces_z[k] <= forces_z[k + 1] + 1e-6, (
            f"Expected monotonic increase toward core: dx={dx_samples[k]} ({forces_z[k]:.4f}N) "
            f"vs dx={dx_samples[k+1]} ({forces_z[k+1]:.4f}N)"
        )

    # Force at core must be strictly positive and directed downward
    core_force = default_downwash_model.compute_pair_force(
        pos_upper, vel_upper, np.array([0.0, 0.0, pos_upper[2] - dz])
    )
    assert core_force[2] < -0.05, f"Expected strong downward force at core, got {core_force[2]}N"


def test_wake_skew_with_forward_velocity(default_downwash_model: DownwashModel) -> None:
    """Verify that upper drone horizontal velocity skews the wake centerline in the velocity direction."""
    pos_upper = np.array([0.0, 0.0, 1.5])
    vel_upper = np.array([1.0, 0.0, 0.0])  # Moving along +X
    dz = 0.40

    # At stationary, max force is at dx = 0.
    # With vx = 1.0, wake center shifts to dx > 0.
    pos_centered = np.array([0.0, 0.0, pos_upper[2] - dz])
    pos_skewed = np.array([default_downwash_model.wake_skew_coeff * dz * 1.0, 0.0, pos_upper[2] - dz])

    f_centered = default_downwash_model.compute_pair_force(pos_upper, vel_upper, pos_centered)
    f_skewed = default_downwash_model.compute_pair_force(pos_upper, vel_upper, pos_skewed)

    assert abs(f_skewed[2]) > abs(f_centered[2]), (
        f"Expected skewed wake location to experience greater force: "
        f"skewed={abs(f_skewed[2]):.4f}N vs centered={abs(f_centered[2]):.4f}N"
    )


def test_simulation_downwash_destabilisation() -> None:
    """Physical simulation test: vertically stacked drone suffers substantially greater error than unstacked."""
    with open("configs/downwash_true.yaml", "r") as f:
        cfg_dw = yaml.safe_load(f)
    dw_model = DownwashModel(cfg_dw)

    def run_two_drone_sim(dx_offset: float) -> float:
        z_upper = 1.5
        z_lower = 1.0
        init_xyzs = np.array([[0.0, 0.0, z_upper], [dx_offset, 0.0, z_lower]])
        target_upper = np.array([0.0, 0.0, z_upper])
        target_lower = np.array([dx_offset, 0.0, z_lower])

        env = DownwashAviary(
            drone_model=DroneModel.CF2X,
            num_drones=2,
            initial_xyzs=init_xyzs,
            physics=Physics.PYB,
            pyb_freq=240,
            ctrl_freq=48,
            gui=False,
            downwash_model=dw_model,
        )
        ctrl_u = SingleDronePIDController(drone_id=0)
        ctrl_l = SingleDronePIDController(drone_id=1)
        action = np.zeros((2, 4))
        obs, _, _, _, _ = env.step(action)

        errors = []
        for step in range(int(2.0 * env.CTRL_FREQ)):
            action[0, :], _, _ = ctrl_u.compute_action(env.CTRL_TIMESTEP, obs[0], target_upper)
            action[1, :], _, _ = ctrl_l.compute_action(env.CTRL_TIMESTEP, obs[1], target_lower)
            obs, _, _, _, _ = env.step(action)
            if step > int(0.5 * env.CTRL_FREQ):
                err = np.linalg.norm(obs[1][0:3] - target_lower)
                errors.append(err)

        env.close()
        return float(np.mean(errors))

    # Error when directly stacked underneath (dx = 0.0)
    err_stacked = run_two_drone_sim(dx_offset=0.0)
    # Error when separated horizontally (dx = 0.50m)
    err_unstacked = run_two_drone_sim(dx_offset=0.50)

    print(f"\nSimulation Comparison: Stacked Error = {err_stacked:.4f}m, Unstacked Error = {err_unstacked:.4f}m")
    # Stacked error should be at least 3x higher due to downwash pushing the lower drone down
    assert err_stacked > 2.5 * err_unstacked, (
        f"Stacked drone error ({err_stacked:.4f}m) should be significantly higher "
        f"than unstacked ({err_unstacked:.4f}m)"
    )


def test_offaxis_wake_does_not_flip_drone() -> None:
    """Regression: a pure downwash FORCE must not induce large torques.

    Bug found in Phase 2: applying the force with WORLD_FRAME and posObj=[0,0,0]
    applied it at the world origin, creating a lever arm that flipped off-axis
    drones. The lower drone here sits off-axis but inside the wake.
    """
    with open("configs/downwash_true.yaml", "r") as f:
        dw_model = DownwashModel(yaml.safe_load(f))

    z_upper, dz, dx = 1.5, 0.35, 0.15
    target_upper = np.array([0.0, 0.0, z_upper])
    target_lower = np.array([dx, 0.0, z_upper - dz])
    env = DownwashAviary(
        num_drones=2,
        initial_xyzs=np.array([target_upper, target_lower]),
        physics=Physics.PYB,
        gui=False,
        downwash_model=dw_model,
    )
    ctrl_u = SingleDronePIDController(drone_id=0)
    ctrl_l = SingleDronePIDController(drone_id=1)
    action = np.zeros((2, 4))
    obs, _, _, _, _ = env.step(action)

    max_tilt_deg = 0.0
    for _ in range(int(2.0 * env.CTRL_FREQ)):
        action[0, :], _, _ = ctrl_u.compute_action(env.CTRL_TIMESTEP, obs[0], target_upper)
        action[1, :], _, _ = ctrl_l.compute_action(env.CTRL_TIMESTEP, obs[1], target_lower)
        obs, _, _, _, _ = env.step(action)
        tilt = np.rad2deg(np.linalg.norm(obs[1][7:9]))
        max_tilt_deg = max(max_tilt_deg, tilt)
    final_err = np.linalg.norm(obs[1][0:3] - target_lower)
    env.close()

    assert max_tilt_deg < 15.0, f"Lower drone tilted {max_tilt_deg:.1f} deg (flip bug?)"
    assert final_err < 0.2, f"Lower drone drifted {final_err:.3f} m (crash?)"


def test_multi_source_stacked_disturbance_greater_than_single(default_downwash_model: DownwashModel) -> None:
    """Verify that a lower drone under two stacked disturbers experiences larger disturbance than under one."""
    vels = np.zeros((3, 3))

    # Single disturber setup: Drone 0 at z=1.0, Drone 1 at z=0.6 (dz = 0.40m)
    pos_single = np.array([
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.6],
    ])
    forces_single = default_downwash_model.compute_swarm_forces(pos_single, vels[:2])
    f_single_mag = abs(forces_single[1, 2])

    # Two disturbers setup: Drone 0 at z=1.4, Drone 1 at z=1.0, Drone 2 at z=0.6
    # Drone 2 is beneath Drone 1 (dz = 0.40m) AND Drone 0 (dz = 0.80m)
    pos_two = np.array([
        [0.0, 0.0, 1.4],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.6],
    ])
    forces_two = default_downwash_model.compute_swarm_forces(pos_two, vels)
    f_two_mag = abs(forces_two[2, 2])

    assert f_single_mag > 0.01, f"Expected non-zero single disturbance, got {f_single_mag:.4f}N"
    assert f_two_mag > f_single_mag, (
        f"Two disturbers ({f_two_mag:.4f}N) must generate greater disturbance than one ({f_single_mag:.4f}N)"
    )


def test_multi_source_combination_modes() -> None:
    """Verify that all configured multi-source combination modes operate correctly."""
    base_cfg = {
        "radius_xy": 0.30,
        "radius_z": 1.0,
        "max_force_z": 0.20,
        "radial_falloff": "gaussian",
        "vertical_falloff": "exponential",
        "wake_skew_coefficient": 0.1,
    }

    pos = np.array([
        [0.0, 0.0, 1.4],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.6],
    ])
    vels = np.zeros((3, 3))

    # 1. Linear sum
    cfg_sum = {**base_cfg, "combination_mode": "linear_sum", "max_combined_force_z": 1.0}
    model_sum = DownwashModel(cfg_sum)
    f_sum = model_sum.compute_swarm_forces(pos, vels)[2, 2]

    # 2. Max mode
    cfg_max = {**base_cfg, "combination_mode": "max", "max_combined_force_z": 1.0}
    model_max = DownwashModel(cfg_max)
    f_max = model_max.compute_swarm_forces(pos, vels)[2, 2]

    # Max magnitude must be less than or equal to linear sum magnitude
    assert abs(f_sum) >= abs(f_max), f"Linear sum ({abs(f_sum):.4f}) must be >= max ({abs(f_max):.4f})"

    # 3. Saturated mode with low ceiling
    ceiling = 0.12
    cfg_sat = {**base_cfg, "combination_mode": "saturated", "max_combined_force_z": ceiling}
    model_sat = DownwashModel(cfg_sat)
    f_sat = model_sat.compute_swarm_forces(pos, vels)[2, 2]
    assert abs(f_sat) <= ceiling + 1e-6, f"Saturated force ({abs(f_sat):.4f}) exceeded ceiling ({ceiling})"

