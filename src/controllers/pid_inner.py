"""
src/controllers/pid_inner.py
=============================
Per-drone cascaded PID position + attitude controller.

Mirrors the DSLPIDControl from gym-pybullet-drones (pure Python, no pybullet
import required) for use in the mock physics backend.

Gains are identical to the validated Crazyflie defaults used by the DSL lab
at UTIAS (see docs/decisions.md D-004):
    Position PID:  P=[0.4,0.4,1.25]  I=[0.05,0.05,0.05]  D=[0.2,0.2,0.5]
    Attitude PID:  P=[70000,70000,60000]  I=[0,0,500]  D=[20000,20000,12000]

The output is a 4-RPM vector for the Crazyflie 2.X X-configuration mixer.

References
----------
DSLPIDControl, gym-pybullet-drones (Zhou et al., DSL, UTIAS).
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation


# PID gains (from DSLPIDControl — validated Crazyflie defaults)
P_FOR = np.array([0.4, 0.4, 1.25])
I_FOR = np.array([0.05, 0.05, 0.05])
D_FOR = np.array([0.2, 0.2, 0.5])
P_TOR = np.array([70000.0, 70000.0, 60000.0])
I_TOR = np.array([0.0, 0.0, 500.0])
D_TOR = np.array([20000.0, 20000.0, 12000.0])

# PWM ↔ RPM mapping (from DSLPIDControl)
PWM2RPM_SCALE = 0.2685
PWM2RPM_CONST = 4070.3
MIN_PWM = 20000
MAX_PWM = 65535

# CF2X motor mixer matrix (maps [roll,pitch,yaw,thrust] to motor PWMs)
# Rows = motors 0-3, columns = [roll, pitch, yaw]
MIXER_MATRIX = np.array(
    [[-0.5, -0.5, -1],
     [-0.5,  0.5,  1],
     [ 0.5,  0.5, -1],
     [ 0.5, -0.5,  1]]
)


class PIDInner:
    """Cascaded position → attitude → RPM PID for one Crazyflie 2.X."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Zero all integrator and derivative states."""
        self.last_pos_e = np.zeros(3)
        self.integral_pos_e = np.zeros(3)
        self.last_rpy = np.zeros(3)
        self.last_rpy_e = np.zeros(3)
        self.integral_rpy_e = np.zeros(3)

    # ------------------------------------------------------------------
    # Main interface
    # ------------------------------------------------------------------

    def compute(
        self,
        dt: float,
        pos: np.ndarray,
        quat: np.ndarray,
        vel: np.ndarray,
        target_pos: np.ndarray,
        target_rpy: np.ndarray = np.zeros(3),
        target_vel: np.ndarray = np.zeros(3),
        hover_rpm: float = 14468.0,
        max_rpm: float = 21702.0,
    ) -> np.ndarray:
        """Compute 4 motor RPMs from current state and position setpoint.

        Parameters
        ----------
        dt : float  — control timestep [s]
        pos : (3,)  — current position [m]
        quat : (4,) — current quaternion [qx,qy,qz,qw]
        vel : (3,)  — current velocity [m/s]
        target_pos : (3,) — desired position [m]
        target_rpy : (3,) — desired roll/pitch/yaw [rad] (usually zeros)
        target_vel : (3,) — feed-forward velocity [m/s]
        hover_rpm : float — hover RPM (for gravity compensation)
        max_rpm   : float — motor saturation limit

        Returns
        -------
        rpm : ndarray (4,)
        """
        cur_rpy = Rotation.from_quat(quat).as_euler("xyz")

        thrust, target_euler = self._pos_pid(dt, pos, vel, target_pos,
                                              target_vel, cur_rpy,
                                              hover_rpm=hover_rpm)
        rpm = self._att_pid(dt, cur_rpy, target_euler, thrust,
                            hover_rpm=hover_rpm, max_rpm=max_rpm)
        return rpm

    # ------------------------------------------------------------------
    # Private: position PID → desired thrust + attitude
    # ------------------------------------------------------------------

    def _pos_pid(
        self,
        dt: float,
        pos: np.ndarray,
        vel: np.ndarray,
        target_pos: np.ndarray,
        target_vel: np.ndarray,
        cur_rpy: np.ndarray,
        hover_rpm: float,
    ):
        """Position PID → thrust scalar and target euler angles."""
        pos_e = target_pos - pos
        d_pos_e = (pos_e - self.last_pos_e) / dt if dt > 0 else np.zeros(3)
        self.integral_pos_e = np.clip(
            self.integral_pos_e + pos_e * dt, -2.0, 2.0
        )
        self.last_pos_e = pos_e

        target_force = (
            P_FOR * pos_e
            + I_FOR * self.integral_pos_e
            + D_FOR * d_pos_e
            + target_vel * D_FOR
        )

        # Gravity compensation
        G = 9.8
        thrust_raw = target_force[2] + G     # [m/s²]

        # Desired tilt angles (small angle, linearised)
        yaw = cur_rpy[2]
        cy, sy = math.cos(yaw), math.sin(yaw)
        target_roll = (
            (1.0 / max(thrust_raw, 0.1)) *
            (target_force[0] * sy - target_force[1] * cy)
        )
        target_pitch = (
            (1.0 / max(thrust_raw, 0.1)) *
            (target_force[0] * cy + target_force[1] * sy)
        )
        target_roll = float(np.clip(target_roll, -math.pi / 6, math.pi / 6))
        target_pitch = float(np.clip(target_pitch, -math.pi / 6, math.pi / 6))

        thrust_scalar = float(np.clip(thrust_raw, 0.0, G * 2.0))
        return thrust_scalar, np.array([target_roll, target_pitch, 0.0])

    # ------------------------------------------------------------------
    # Private: attitude PID → RPMs
    # ------------------------------------------------------------------

    def _att_pid(
        self,
        dt: float,
        cur_rpy: np.ndarray,
        target_rpy: np.ndarray,
        thrust: float,
        hover_rpm: float,
        max_rpm: float,
    ) -> np.ndarray:
        """Attitude PID → 4 motor RPMs."""
        rpy_e = target_rpy - cur_rpy
        d_rpy_e = (rpy_e - self.last_rpy_e) / dt if dt > 0 else np.zeros(3)
        self.integral_rpy_e = np.clip(
            self.integral_rpy_e + rpy_e * dt, -1500.0, 1500.0
        )
        self.last_rpy_e = rpy_e
        self.last_rpy = cur_rpy

        torques = (
            P_TOR * rpy_e
            + I_TOR * self.integral_rpy_e
            + D_TOR * d_rpy_e
        )
        torques = np.clip(torques, -3200.0, 3200.0)

        # Thrust in normalised [0,1]
        thrust_norm = thrust / (9.8 * 2.0)  # normalise by 2g ceiling
        # Motor PWM from mixer
        pwm = np.clip(
            thrust_norm * (MAX_PWM - MIN_PWM) + MIN_PWM + MIXER_MATRIX @ torques,
            MIN_PWM,
            MAX_PWM,
        )
        rpm = PWM2RPM_SCALE * pwm + PWM2RPM_CONST
        rpm = np.clip(rpm, 0.0, max_rpm)
        return rpm
