"""Unit test for single drone hover setpoint acquisition and tracking stability."""
import numpy as np
import pytest
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary

from src.controllers.single_drone_pid import SingleDronePIDController


@pytest.mark.parametrize("target_hover_pos", [
    np.array([0.0, 0.0, 1.0]),
    np.array([0.3, -0.3, 0.8]),
])
def test_drone_reaches_hover_within_tolerance(target_hover_pos: np.ndarray) -> None:
    """Test that a single drone launched from the ground reaches a hover setpoint within 3cm."""
    sim_freq = 240
    ctrl_freq = 48
    initial_xyz = np.array([[0.0, 0.0, 0.1]])

    env = CtrlAviary(
        drone_model=DroneModel.CF2X,
        num_drones=1,
        initial_xyzs=initial_xyz,
        physics=Physics.PYB,
        pyb_freq=sim_freq,
        ctrl_freq=ctrl_freq,
        gui=False,
        obstacles=False,
        user_debug_gui=False,
    )

    ctrl = SingleDronePIDController(drone_id=0, drone_model=DroneModel.CF2X)
    ctrl.reset()

    action = np.zeros((1, 4))
    obs, _, _, _, _ = env.step(action)

    # Simulate for 4.5 seconds to allow full transient settling
    duration_sec = 4.5
    num_steps = int(duration_sec * ctrl_freq)

    for _ in range(num_steps):
        state = obs[0]
        motor_rpms, pos_e, yaw_e = ctrl.compute_action(
            control_timestep=env.CTRL_TIMESTEP,
            state=state,
            target_pos=target_hover_pos,
            target_vel=np.zeros(3),
        )
        action[0, :] = motor_rpms
        obs, _, _, _, _ = env.step(action)

    final_pos = obs[0][0:3]
    final_rpy = obs[0][7:10]
    final_vel = obs[0][10:13]

    env.close()

    pos_error = np.linalg.norm(final_pos - target_hover_pos)
    vel_norm = np.linalg.norm(final_vel)

    print(f"\nHover Test Target={target_hover_pos} -> Final Pos={np.round(final_pos, 4)}, Error={pos_error:.4f}m")

    # Assertions
    # 1. Steady-state position error < 4 cm (0.04 m)
    assert pos_error < 0.04, f"Hover error too high: {pos_error:.4f} m (threshold: 0.04 m)"
    # 2. Attitude is near horizontal (roll and pitch within ~2.8 deg = 0.05 rad)
    assert abs(final_rpy[0]) < 0.05, f"Roll angle too large: {final_rpy[0]:.4f} rad"
    assert abs(final_rpy[1]) < 0.05, f"Pitch angle too large: {final_rpy[1]:.4f} rad"
    # 3. Velocity has settled (< 0.05 m/s)
    assert vel_norm < 0.05, f"Residual velocity too large: {vel_norm:.4f} m/s"


def test_controller_reset() -> None:
    """Verify that controller reset clears accumulated integrator errors."""
    ctrl = SingleDronePIDController(drone_id=0)
    fake_state = np.zeros(20)
    target = np.array([1.0, 1.0, 1.0])

    # Run one step to build error
    action1, pos_e1, _ = ctrl.compute_action(
        control_timestep=0.02,
        state=fake_state,
        target_pos=target,
    )
    assert np.any(ctrl._pid.integral_pos_e != 0.0)

    # Reset
    ctrl.reset()
    assert np.all(ctrl._pid.integral_pos_e == 0.0)
    assert np.all(ctrl._pid.last_pos_e == 0.0)
