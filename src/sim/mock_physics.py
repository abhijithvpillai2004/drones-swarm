"""
src/sim/mock_physics.py
========================
Pure-Python physics mock for the Crazyflie 2.X quadrotor.

Reproduces the DYN-mode Euler integration from
``gym_pybullet_drones.envs.BaseAviary._dynamics()`` in plain NumPy.
This allows the full test suite, controller evaluation, and metrics code
to run on any machine, regardless of whether pybullet is installed.

Physical constants come from the Crazyflie 2.X URDF (cf2x.urdf) inspected
in the gym-pybullet-drones repository. See docs/decisions.md D-003.

When pybullet IS installed, ``src/sim/aviary_wrapper.py`` provides a
higher-fidelity implementation through the same interface.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation


# ---------------------------------------------------------------------------
# Crazyflie 2.X constants (from cf2x.urdf — see docs/sim_notes.md §2)
# ---------------------------------------------------------------------------

# These are read from the URDF; not invented. See docs/decisions.md D-003.
CF2X_PARAMS: Dict[str, float] = {
    "mass_kg": 0.027,           # [kg]
    "arm_m": 0.0397,            # [m] motor arm length
    "kf": 3.16e-10,             # [N / RPM²] thrust coefficient
    "km": 7.94e-12,             # [N·m / RPM²] torque coefficient
    "ixx": 1.4e-5,              # [kg·m²]
    "iyy": 1.4e-5,
    "izz": 2.17e-5,
    "g": 9.8,                   # [m/s²] gravitational acceleration
    "max_rpm": 21702.0,         # [RPM]
    "hover_rpm": 14468.0,       # [RPM] sqrt(m*g / (4*kf))
}


@dataclass
class DroneState:
    """Full kinematic state of a single quadrotor.

    Attributes
    ----------
    pos : ndarray (3,)  — position [x,y,z] [m]
    quat : ndarray (4,) — orientation quaternion [qx,qy,qz,qw]
    vel : ndarray (3,)  — linear velocity [m/s]
    ang_vel : ndarray (3,) — angular velocity in body frame [rad/s]
    last_rpm : ndarray (4,) — last motor RPMs
    """

    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    quat: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0, 1.0])  # identity
    )
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    ang_vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    last_rpm: np.ndarray = field(
        default_factory=lambda: np.full(4, CF2X_PARAMS["hover_rpm"])
    )

    def copy(self) -> "DroneState":
        """Return a deep copy."""
        return DroneState(
            pos=self.pos.copy(),
            quat=self.quat.copy(),
            vel=self.vel.copy(),
            ang_vel=self.ang_vel.copy(),
            last_rpm=self.last_rpm.copy(),
        )

    @property
    def rpy(self) -> np.ndarray:
        """Roll, pitch, yaw (extrinsic XYZ) from quaternion [rad]."""
        return Rotation.from_quat(self.quat).as_euler("xyz")

    def as_state_vector(self) -> np.ndarray:
        """Return length-20 state vector matching gym-pybullet-drones format.

        Index mapping:
          0:3   pos
          3:7   quat [qx,qy,qz,qw]
          7:10  rpy
          10:13 vel
          13:16 ang_vel
          16:20 last_rpm
        """
        return np.concatenate(
            [self.pos, self.quat, self.rpy, self.vel, self.ang_vel, self.last_rpm]
        )


class MockPhysics:
    """Pure-Python quadrotor physics for N Crazyflie 2.X drones.

    Implements the same Euler-integration as gym-pybullet-drones'
    DYN physics mode (BaseAviary._dynamics), extended to support
    external force injection (for TRUE downwash disturbances) and
    multi-drone operation.

    Parameters
    ----------
    num_drones : int
        Number of drones to simulate.
    initial_xyzs : ndarray (N,3) or None
        Initial positions. Defaults to N drones along x-axis.
    initial_rpys : ndarray (N,3) or None
        Initial Euler angles [roll,pitch,yaw] in radians.
    pyb_freq : int
        Physics integration frequency [Hz].
    ctrl_freq : int
        Control update frequency [Hz]. Must divide pyb_freq.
    params : dict or None
        Override default CF2X_PARAMS.
    """

    def __init__(
        self,
        num_drones: int,
        initial_xyzs: Optional[np.ndarray] = None,
        initial_rpys: Optional[np.ndarray] = None,
        pyb_freq: int = 240,
        ctrl_freq: int = 48,
        params: Optional[dict] = None,
    ) -> None:
        self.N = num_drones
        self.PYB_FREQ = pyb_freq
        self.CTRL_FREQ = ctrl_freq
        if pyb_freq % ctrl_freq != 0:
            raise ValueError("pyb_freq must be divisible by ctrl_freq")
        self.PYB_STEPS_PER_CTRL = pyb_freq // ctrl_freq
        self.PYB_TIMESTEP: float = 1.0 / pyb_freq
        self.CTRL_TIMESTEP: float = 1.0 / ctrl_freq

        # Physical parameters
        p = {**CF2X_PARAMS, **(params or {})}
        self._m: float = p["mass_kg"]
        self._kf: float = p["kf"]
        self._km: float = p["km"]
        self._g: float = p["g"]
        self._GRAVITY: float = self._m * self._g
        L = p["arm_m"]
        self._J: np.ndarray = np.diag([p["ixx"], p["iyy"], p["izz"]])
        self._J_INV: np.ndarray = np.linalg.inv(self._J)
        self._L_sqrt2: float = L / math.sqrt(2)  # for CF2X torque arms
        self._MAX_RPM: float = p["max_rpm"]
        self.HOVER_RPM: float = p["hover_rpm"]
        self.GRAVITY: float = self._GRAVITY  # exposed for controllers

        # Initialise states
        if initial_xyzs is None:
            initial_xyzs = np.array([[i * 0.5, 0.0, 1.0] for i in range(num_drones)])
        if initial_rpys is None:
            initial_rpys = np.zeros((num_drones, 3))

        self.states: List[DroneState] = []
        for i in range(num_drones):
            quat = Rotation.from_euler("xyz", initial_rpys[i]).as_quat()
            s = DroneState(
                pos=initial_xyzs[i].copy().astype(float),
                quat=quat.copy(),
                last_rpm=np.full(4, self.HOVER_RPM),
            )
            self.states.append(s)

        # Pending external forces (cleared after each control step)
        self._ext_forces: np.ndarray = np.zeros((num_drones, 3))

        self.step_counter: int = 0

    # ------------------------------------------------------------------
    # External force injection (TRUE downwash mechanism)
    # ------------------------------------------------------------------

    def apply_external_force(self, drone_idx: int, force_xyz: np.ndarray) -> None:
        """Schedule an external force for drone *drone_idx* [N, world frame].

        Forces accumulate and are applied on the next :meth:`step` call.
        They are cleared automatically after each control step.
        """
        self._ext_forces[drone_idx] += np.asarray(force_xyz, dtype=float)

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------

    def step(self, actions: np.ndarray) -> np.ndarray:
        """Advance simulation by one control step.

        Parameters
        ----------
        actions : ndarray, shape (N, 4)
            Motor RPMs for each drone. Values are clipped to [0, MAX_RPM].

        Returns
        -------
        obs : ndarray, shape (N, 20)
            State vectors for all drones after the step.
        """
        rpms = np.clip(actions, 0.0, self._MAX_RPM)

        # Run PYB_STEPS_PER_CTRL physics sub-steps
        for _ in range(self.PYB_STEPS_PER_CTRL):
            for i in range(self.N):
                self._integrate_drone(i, rpms[i])

        # Clear external forces after the control step
        self._ext_forces[:] = 0.0
        self.step_counter += 1

        return self.get_obs()

    def get_obs(self) -> np.ndarray:
        """Return current state for all drones, shape (N, 20)."""
        return np.array([s.as_state_vector() for s in self.states])

    def reset(
        self,
        initial_xyzs: Optional[np.ndarray] = None,
        initial_rpys: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Reset simulation state."""
        self.step_counter = 0
        self._ext_forces[:] = 0.0
        if initial_xyzs is not None:
            for i, s in enumerate(self.states):
                s.pos = initial_xyzs[i].copy().astype(float)
        if initial_rpys is not None:
            for i, s in enumerate(self.states):
                s.quat = Rotation.from_euler("xyz", initial_rpys[i]).as_quat()
        for s in self.states:
            s.vel[:] = 0.0
            s.ang_vel[:] = 0.0
            s.last_rpm = np.full(4, self.HOVER_RPM)
        return self.get_obs()

    # ------------------------------------------------------------------
    # Internal: single-drone Euler integration (DYN mode)
    # ------------------------------------------------------------------

    def _integrate_drone(self, i: int, rpm: np.ndarray) -> None:
        """Euler-integrate one drone's state by PYB_TIMESTEP.

        Follows BaseAviary._dynamics() exactly, plus external forces.
        """
        s = self.states[i]
        dt = self.PYB_TIMESTEP

        # Rotation matrix from quaternion
        R = Rotation.from_quat(s.quat).as_matrix()

        # Motor forces and torques
        forces = self._kf * rpm ** 2           # shape (4,) [N]
        thrust_body = np.array([0.0, 0.0, np.sum(forces)])
        thrust_world = R @ thrust_body
        gravity = np.array([0.0, 0.0, -self._GRAVITY])
        net_force_world = thrust_world + gravity + self._ext_forces[i]

        # Torques (CF2X layout)
        x_torque = -(forces[0] + forces[1] - forces[2] - forces[3]) * self._L_sqrt2
        y_torque = (-forces[0] + forces[1] + forces[2] - forces[3]) * self._L_sqrt2
        z_torques = self._km * rpm ** 2
        z_torque = -z_torques[0] + z_torques[1] - z_torques[2] + z_torques[3]
        torques = np.array([x_torque, y_torque, z_torque])
        torques -= np.cross(s.ang_vel, self._J @ s.ang_vel)
        ang_acc = self._J_INV @ torques

        # Euler integration
        acc = net_force_world / self._m
        s.vel += dt * acc
        s.ang_vel += dt * ang_acc
        s.pos += dt * s.vel
        s.quat = self._integrate_quaternion(s.quat, s.ang_vel, dt)
        s.last_rpm = rpm.copy()

    @staticmethod
    def _integrate_quaternion(
        quat: np.ndarray, omega: np.ndarray, dt: float
    ) -> np.ndarray:
        """First-order quaternion integration (from BaseAviary._integrateQ).

        quat : [qx, qy, qz, qw]
        omega: angular velocity in body frame [rad/s]
        """
        omega_norm = np.linalg.norm(omega)
        if np.isclose(omega_norm, 0.0):
            return quat

        p_, q_, r_ = omega
        lam = np.array(
            [
                [0.0, r_, -q_, p_],
                [-r_, 0.0, p_, q_],
                [q_, -p_, 0.0, r_],
                [-p_, -q_, -r_, 0.0],
            ]
        ) * 0.5

        theta = omega_norm * dt / 2.0
        mat = np.eye(4) * math.cos(theta) + (2.0 / omega_norm) * lam * math.sin(theta)
        new_quat = mat @ quat
        # Normalise to avoid drift
        norm = np.linalg.norm(new_quat)
        return new_quat / norm if norm > 1e-9 else quat
