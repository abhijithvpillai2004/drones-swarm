"""Base controller interfaces for multi-drone formation and low-level tracking.

This module provides abstract base classes for:
1. BaseLowLevelController: Single-drone controller computing motor inputs (RPMs)
   from full state and reference setpoints.
2. BaseSwarmController: Swarm-level controller taking states of all N drones and
   computing coordination commands (avoidance / compensation / targets).
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import numpy as np


class BaseLowLevelController(ABC):
    """Abstract base class for single-drone low-level attitude and position control."""

    def __init__(self, drone_id: int = 0) -> None:
        """Initialize low-level controller for a single drone.

        Args:
            drone_id: Index of the drone in the swarm.
        """
        self.drone_id: int = drone_id

    @abstractmethod
    def reset(self) -> None:
        """Reset internal integrator states, errors, and memory."""
        pass

    @abstractmethod
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
        """Compute motor commands (RPMs) from current state and reference setpoint.

        Args:
            control_timestep: Time elapsed since last control step [s].
            state: Kinematic state vector of shape (20,) [pos, quat, rpy, vel, ang_vel, rpm].
            target_pos: Desired Cartesian position [x, y, z] in world frame [m].
            target_rpy: Optional desired roll, pitch, yaw [rad].
            target_vel: Optional desired linear velocity [vx, vy, vz] [m/s].
            target_rpy_rates: Optional desired angular velocity rates [rad/s].
            feedforward_force: Optional 3D feedforward force [fx, fy, fz] in world frame [N].

        Returns:
            Tuple of:
                - motor_rpms: Array of shape (4,) containing commanded motor RPMs.
                - pos_error: Position error vector (3,).
                - yaw_error: Yaw error scalar or array.
        """
        pass


class BaseSwarmController(ABC):
    """Abstract base class for N-drone swarm coordination controllers.
    
    Controllers A (Naive), B (Soft-avoidance), and C (Feedforward compensation)
    subclass this interface in Phase 4.
    """

    def __init__(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None,
        num_drones: int = 2,
    ) -> None:
        """Initialize swarm controller.

        Args:
            name: Human-readable identifier for the controller.
            config: Configuration dictionary for controller gains, limits, and models.
            num_drones: Number of drones in the swarm.
        """
        self.name: str = name
        self.config: Dict[str, Any] = config or {}
        self.num_drones: int = num_drones
        self.low_level_controllers: list = []
        self._init_low_level_controllers()

    def _init_low_level_controllers(self) -> None:
        """Initialize low-level PID tracking controllers for each drone."""
        from src.controllers.single_drone_pid import SingleDronePIDController
        self.low_level_controllers = [
            SingleDronePIDController(drone_id=i) for i in range(self.num_drones)
        ]

    def reset(self) -> None:
        """Reset internal controller states for all drones."""
        for ctrl in self.low_level_controllers:
            ctrl.reset()

    @abstractmethod
    def compute_commands(
        self,
        states: np.ndarray,
        formation_targets: np.ndarray,
        timestep: float,
        formation_velocities: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute coordinated target positions, velocities, and feedforward forces.

        Args:
            states: Array of shape (N, 20) with current state vectors.
            formation_targets: Array of shape (N, 3) with nominal formation positions.
            timestep: Control timestep [s].
            formation_velocities: Optional array of shape (N, 3) nominal velocities.

        Returns:
            Tuple of:
                - cmd_positions: Array of shape (N, 3) adapted target positions.
                - cmd_velocities: Array of shape (N, 3) target velocities.
                - feedforward_forces: Array of shape (N, 3) disturbance cancellation forces [N].
        """
        pass

    def compute_actions(
        self,
        control_timestep: float,
        states: np.ndarray,
        formation_targets: np.ndarray,
        formation_velocities: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Full end-to-end swarm step: coordination + low-level motor commands.

        Args:
            control_timestep: Control timestep [s].
            states: Array of shape (N, 20) with current drone states.
            formation_targets: Array of shape (N, 3) nominal formation setpoints.
            formation_velocities: Optional array of shape (N, 3) nominal velocities.

        Returns:
            Tuple of:
                - motor_rpms: Array of shape (N, 4) commanded motor RPMs.
                - pos_errors: Array of shape (N, 3) position errors.
                - yaw_errors: Array of shape (N,) yaw errors.
        """
        cmd_positions, cmd_velocities, ff_forces = self.compute_commands(
            states=states,
            formation_targets=formation_targets,
            timestep=control_timestep,
            formation_velocities=formation_velocities,
        )

        rpms = np.zeros((self.num_drones, 4))
        pos_errors = np.zeros((self.num_drones, 3))
        yaw_errors = np.zeros(self.num_drones)

        for i in range(self.num_drones):
            rpm, p_err, y_err = self.low_level_controllers[i].compute_action(
                control_timestep=control_timestep,
                state=states[i],
                target_pos=cmd_positions[i],
                target_vel=cmd_velocities[i],
                feedforward_force=ff_forces[i],
            )
            rpms[i] = rpm
            pos_errors[i] = p_err
            yaw_errors[i] = y_err

        return rpms, pos_errors, yaw_errors
