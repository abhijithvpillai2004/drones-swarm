"""Controller C: Feedforward Compensation Swarm Controller.

Predicts aerodynamic downwash disturbance forces on lower drones using an assumed
aerodynamic model and feeds forward opposing thrust commands directly into the low-level
attitude and thrust mixer while maintaining nominal formation positions.

CRITICAL DESIGN RULE:
This controller strictly utilizes the ASSUMED aerodynamic downwash model parameters
(configs/downwash_assumed.yaml) and never inspects the TRUE simulation parameters.
"""
from src.paths import config_path
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union
import yaml
import numpy as np

from src.controllers.base import BaseSwarmController
from src.models.downwash import DownwashModel


class FeedforwardSwarmController(BaseSwarmController):
    """Swarm controller applying feedforward aerodynamic disturbance compensation."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        num_drones: int = 2,
        assumed_config: Optional[Union[Dict[str, Any], str, Path, DownwashModel]] = None,
    ) -> None:
        """Initialize Feedforward swarm controller.

        Args:
            config: Optional controller configuration dictionary.
            num_drones: Number of drones in the swarm.
            assumed_config: Assumed model config dict, YAML path, or DownwashModel instance.
        """
        super().__init__(name="Controller_C_Feedforward", config=config, num_drones=num_drones)

        # 1. Initialize ASSUMED downwash model (strict anti-circularity)
        if isinstance(assumed_config, DownwashModel):
            self.assumed_model: DownwashModel = assumed_config
            self.assumed_params: Dict[str, Any] = self.assumed_model.config
        else:
            self.assumed_params = self._load_assumed_config(assumed_config)
            self.assumed_model = DownwashModel(config=self.assumed_params)

        # 2. Compensation safety bounds
        cfg = self.config or {}
        default_max_force = 1.5 * float(self.assumed_params.get("max_force_z", 0.15))
        self.max_compensation_force: float = float(cfg.get("max_compensation_force", default_max_force))

    def _load_assumed_config(
        self,
        assumed_config: Optional[Union[Dict[str, Any], str, Path]] = None,
    ) -> Dict[str, Any]:
        """Load assumed downwash parameters from dictionary or YAML file."""
        if assumed_config is not None:
            if isinstance(assumed_config, dict):
                return assumed_config
            path = Path(assumed_config)
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}

        # Default path
        default_path = config_path("downwash_assumed.yaml")
        if default_path.exists():
            with open(default_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}

        # Fallback defaults if file not found
        return {
            "radius_xy": 0.25,
            "radius_z": 0.80,
            "max_force_z": 0.15,
            "radial_falloff": "gaussian",
            "vertical_falloff": "exponential",
            "wake_skew_coefficient": 0.1,
        }

    def reset(self) -> None:
        """Reset internal controller state and low-level controllers."""
        super().reset()

    def compute_commands(
        self,
        states: np.ndarray,
        formation_targets: np.ndarray,
        timestep: float,
        formation_velocities: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute nominal setpoints and feedforward cancellation forces.

        Args:
            states: Kinematic states array of shape (N, 20).
            formation_targets: Desired Cartesian positions array of shape (N, 3).
            timestep: Control timestep [s].
            formation_velocities: Optional desired velocities array of shape (N, 3).

        Returns:
            Tuple of:
                - cmd_positions: Shape (N, 3), exact copy of nominal targets.
                - cmd_velocities: Shape (N, 3), nominal velocities or zeros.
                - feedforward_forces: Shape (N, 3), disturbance cancellation forces [N].
        """
        cmd_positions = np.asarray(formation_targets, dtype=float).copy()

        if formation_velocities is not None:
            cmd_velocities = np.asarray(formation_velocities, dtype=float).copy()
        else:
            cmd_velocities = np.zeros_like(cmd_positions)

        positions = states[:, 0:3]
        velocities = states[:, 10:13]

        # 1. Estimate downwash disturbance forces acting on all drones using ASSUMED model
        estimated_disturbances = self.assumed_model.compute_swarm_forces(
            positions=positions,
            velocities=velocities,
        )

        # 2. Feedforward force cancels the estimated disturbance: F_ff = -F_hat_dw
        feedforward_forces = -estimated_disturbances

        # 3. Apply safety saturation clamp to prevent runaway commands
        feedforward_forces = np.clip(
            feedforward_forces,
            -self.max_compensation_force,
            self.max_compensation_force,
        )

        return cmd_positions, cmd_velocities, feedforward_forces
