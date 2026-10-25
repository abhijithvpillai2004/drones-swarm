"""Controller A: Naive Formation Tracking Swarm Controller.

This baseline controller tracks commanded formation trajectories while completely
ignoring downwash aerodynamic interactions. It sets zero feedforward force compensation
and applies no anti-stacking avoidance maneuvers.
"""
from typing import Dict, Any, Optional, Tuple
import numpy as np

from src.controllers.base import BaseSwarmController


class NaiveSwarmController(BaseSwarmController):
    """Naive N-drone formation controller ignoring downwash interactions."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        num_drones: int = 2,
    ) -> None:
        """Initialize Naive swarm controller.

        Args:
            config: Optional controller configuration dictionary.
            num_drones: Number of drones in the swarm.
        """
        super().__init__(name="Controller_A_Naive", config=config, num_drones=num_drones)

    def compute_commands(
        self,
        states: np.ndarray,
        formation_targets: np.ndarray,
        timestep: float,
        formation_velocities: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute naive tracking setpoints (pass-through nominal targets).

        Args:
            states: Kinematic states array of shape (N, 20).
            formation_targets: Desired Cartesian positions array of shape (N, 3).
            timestep: Control timestep [s].
            formation_velocities: Optional desired velocities array of shape (N, 3).

        Returns:
            Tuple of:
                - cmd_positions: Shape (N, 3), exact copy of formation_targets.
                - cmd_velocities: Shape (N, 3), formation_velocities or zeros.
                - feedforward_forces: Shape (N, 3), all zeros.
        """
        cmd_positions = np.asarray(formation_targets, dtype=float).copy()

        if formation_velocities is not None:
            cmd_velocities = np.asarray(formation_velocities, dtype=float).copy()
        else:
            cmd_velocities = np.zeros_like(cmd_positions)

        feedforward_forces = np.zeros_like(cmd_positions)

        return cmd_positions, cmd_velocities, feedforward_forces
