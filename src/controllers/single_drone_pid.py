"""Single drone position and attitude tracking PID controller.

Wraps gym-pybullet-drones DSLPIDControl under the BaseLowLevelController interface
with feedforward force compensation support.
"""
import math
from typing import Optional, Tuple
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation

from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel

from src.controllers.base import BaseLowLevelController


class FeedforwardDSLPIDControl(DSLPIDControl):
    """Cascaded DSL PID controller extended with 3D feedforward force injection.

    In standard DSLPIDControl, target_thrust = PID(pos_e, vel_e) + [0, 0, mg].
    Here, feedforward_force is directly added to target_thrust:
        target_thrust = PID(...) + [0, 0, mg] + feedforward_force
    allowing immediate cancellation of known aerodynamic downwash without waiting
    for tracking error accumulation in the feedback loop.
    """

    def computeControl(
        self,
        control_timestep: float,
        cur_pos: np.ndarray,
        cur_quat: np.ndarray,
        cur_vel: np.ndarray,
        cur_ang_vel: np.ndarray,
        target_pos: np.ndarray,
        target_rpy: np.ndarray = np.zeros(3),
        target_vel: np.ndarray = np.zeros(3),
        target_rpy_rates: np.ndarray = np.zeros(3),
        feedforward_force: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Compute control action with optional feedforward force injection."""
        self.control_counter += 1
        if feedforward_force is None:
            ff = np.zeros(3)
        else:
            ff = np.asarray(feedforward_force, dtype=float)

        thrust, computed_target_rpy, pos_e = self._dslPIDPositionControl(
            control_timestep=control_timestep,
            cur_pos=cur_pos,
            cur_quat=cur_quat,
            cur_vel=cur_vel,
            target_pos=target_pos,
            target_rpy=target_rpy,
            target_vel=target_vel,
            feedforward_force=ff,
        )
        rpm = self._dslPIDAttitudeControl(
            control_timestep=control_timestep,
            thrust=thrust,
            cur_quat=cur_quat,
            target_euler=computed_target_rpy,
            target_rpy_rates=target_rpy_rates,
        )
        cur_rpy = np.array(p.getEulerFromQuaternion(cur_quat))
        return rpm, pos_e, computed_target_rpy[2] - cur_rpy[2]

    def _dslPIDPositionControl(
        self,
        control_timestep: float,
        cur_pos: np.ndarray,
        cur_quat: np.ndarray,
        cur_vel: np.ndarray,
        target_pos: np.ndarray,
        target_rpy: np.ndarray,
        target_vel: np.ndarray,
        feedforward_force: Optional[np.ndarray] = None,
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """DSL's CF2.x PID position control with additive feedforward force vector."""
        if feedforward_force is None:
            ff = np.zeros(3)
        else:
            ff = feedforward_force

        cur_rotation = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)
        pos_e = target_pos - cur_pos
        vel_e = target_vel - cur_vel
        self.integral_pos_e = self.integral_pos_e + pos_e * control_timestep
        self.integral_pos_e = np.clip(self.integral_pos_e, -2.0, 2.0)
        self.integral_pos_e[2] = np.clip(self.integral_pos_e[2], -0.15, 0.15)

        #### PID target thrust + feedforward disturbance force #########
        target_thrust = (
            np.multiply(self.P_COEFF_FOR, pos_e)
            + np.multiply(self.I_COEFF_FOR, self.integral_pos_e)
            + np.multiply(self.D_COEFF_FOR, vel_e)
            + np.array([0.0, 0.0, self.GRAVITY])
            + ff
        )
        scalar_thrust = max(0.0, float(np.dot(target_thrust, cur_rotation[:, 2])))
        thrust = (math.sqrt(scalar_thrust / (4.0 * self.KF)) - self.PWM2RPM_CONST) / self.PWM2RPM_SCALE

        norm_thrust = float(np.linalg.norm(target_thrust))
        if norm_thrust < 1e-6:
            target_z_ax = np.array([0.0, 0.0, 1.0])
        else:
            target_z_ax = target_thrust / norm_thrust

        target_x_c = np.array([math.cos(target_rpy[2]), math.sin(target_rpy[2]), 0.0])
        cross_zx = np.cross(target_z_ax, target_x_c)
        norm_cross = float(np.linalg.norm(cross_zx))
        if norm_cross < 1e-6:
            target_y_ax = np.array([0.0, 1.0, 0.0])
        else:
            target_y_ax = cross_zx / norm_cross
        target_x_ax = np.cross(target_y_ax, target_z_ax)
        target_rotation = (np.vstack([target_x_ax, target_y_ax, target_z_ax])).transpose()
        target_euler = (Rotation.from_matrix(target_rotation)).as_euler("XYZ", degrees=False)
        return thrust, target_euler, pos_e


class SingleDronePIDController(BaseLowLevelController):
    """Cascaded PID controller for position and attitude tracking of Crazyflie 2.x."""

    def __init__(
        self,
        drone_id: int = 0,
        drone_model: DroneModel = DroneModel.CF2X,
        g: float = 9.8,
    ) -> None:
        """Initialize PID controller.

        Args:
            drone_id: Drone index.
            drone_model: Drone model enumeration (default CF2X).
            g: Gravitational acceleration in m/s^2.
        """
        super().__init__(drone_id=drone_id)
        self.drone_model: DroneModel = drone_model
        self.g: float = g
        self._pid: FeedforwardDSLPIDControl = FeedforwardDSLPIDControl(
            drone_model=self.drone_model,
            g=self.g,
        )

    def reset(self) -> None:
        """Reset PID internal integrator states and previous errors."""
        self._pid.reset()

    def compute_action(
        self,
        control_timestep: float,
        state: np.ndarray,
        target_pos: np.ndarray,
        target_rpy: Optional[np.ndarray] = None,
        target_vel: Optional[np.ndarray] = None,
        target_rpy_rates: Optional[np.ndarray] = None,
        feedforward_force: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute motor RPMs to track a given 3D setpoint with optional feedforward force.

        Args:
            control_timestep: Time step [s].
            state: Full state vector of shape (20,).
            target_pos: Desired Cartesian position [x, y, z] [m].
            target_rpy: Desired roll, pitch, yaw [rad]. Defaults to zero.
            target_vel: Desired linear velocity [vx, vy, vz] [m/s]. Defaults to zero.
            target_rpy_rates: Desired Euler rates [rad/s]. Defaults to zero.
            feedforward_force: Desired 3D feedforward force vector [N] in world frame.

        Returns:
            Tuple of:
                - rpm: Motor RPM commands of shape (4,).
                - pos_error: Position error [x, y, z] in world frame.
                - yaw_error: Yaw error scalar or array.
        """
        if target_rpy is None:
            target_rpy = np.zeros(3)
        if target_vel is None:
            target_vel = np.zeros(3)
        if target_rpy_rates is None:
            target_rpy_rates = np.zeros(3)
        if feedforward_force is None:
            feedforward_force = np.zeros(3)

        action, pos_e, yaw_e = self._pid.computeControl(
            control_timestep=control_timestep,
            cur_pos=state[0:3],
            cur_quat=state[3:7],
            cur_vel=state[10:13],
            cur_ang_vel=state[13:16],
            target_pos=target_pos,
            target_rpy=target_rpy,
            target_vel=target_vel,
            target_rpy_rates=target_rpy_rates,
            feedforward_force=feedforward_force,
        )
        return action, pos_e, yaw_e

