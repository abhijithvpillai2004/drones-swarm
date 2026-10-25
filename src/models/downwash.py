"""Aerodynamic downwash disturbance model classes.

Implements standalone analytical models based on:
1. Hönig et al. (IEEE T-RO 2018): Ellipsoidal downwash influence zone.
2. Shi et al. (Neural-Swarm, ICRA 2020): Smooth radial/vertical wake decay.

CRITICAL DESIGN RULE:
This module contains the mathematical representation used for BOTH the TRUE
disturbance (applied by the simulator) and the ASSUMED model (used by controllers).
Instances must be parameterized independently via YAML configs to prevent circularity.
"""
from typing import Dict, Any, Tuple, List
import numpy as np


class DownwashModel:
    """Parametric ellipsoidal aerodynamic downwash model."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize downwash model from configuration dictionary.

        Args:
            config: Dictionary containing model parameters:
                - radius_xy: Horizontal radius of wake influence [m]
                - radius_z: Vertical depth of wake influence [m]
                - max_force_z: Peak downward disturbance force from a single drone [N]
                - radial_falloff: "gaussian", "quadratic", or "linear"
                - vertical_falloff: "exponential" or "linear"
                - wake_skew_coefficient: Horizontal shift per unit vertical depth
                  proportional to upper drone velocity [s]
                - combination_mode: Multi-source wake combination method:
                  "linear_sum" (additive superposition), "max" (peak wake only),
                  "sublinear" (diminishing returns power-law), or "saturated"
                - max_combined_force_z: Maximum allowable combined downward disturbance [N]
        """
        self.config: Dict[str, Any] = config
        self.radius_xy: float = float(config.get("radius_xy", 0.25))  # TODO: verify against Hönig 2018 / Neural-Swarm
        self.radius_z: float = float(config.get("radius_z", 0.80))   # TODO: verify against Hönig 2018 / Neural-Swarm
        self.max_force_z: float = float(config.get("max_force_z", 0.15))  # TODO: verify against Hönig 2018 / Neural-Swarm
        self.radial_falloff: str = str(config.get("radial_falloff", "gaussian")).lower()
        self.vertical_falloff: str = str(config.get("vertical_falloff", "exponential")).lower()
        self.wake_skew_coeff: float = float(config.get("wake_skew_coefficient", 0.1))  # TODO: verify against Hönig 2018 / Neural-Swarm

        # Multi-source downwash combination parameters
        self.combination_mode: str = str(config.get("combination_mode", "linear_sum")).lower()
        default_max_combined = float(config.get("max_combined_force_z", 3.0 * self.max_force_z))
        self.max_combined_force_z: float = default_max_combined
        self.sublinear_gamma: float = float(config.get("sublinear_gamma", 0.8))

    def compute_pair_force(
        self,
        pos_upper: np.ndarray,
        vel_upper: np.ndarray,
        pos_lower: np.ndarray,
    ) -> np.ndarray:
        """Compute disturbance force vector exerted by an upper drone on a lower drone.

        Args:
            pos_upper: Cartesian position [x, y, z] of upper drone in world frame [m].
            vel_upper: Linear velocity [vx, vy, vz] of upper drone in world frame [m/s].
            pos_lower: Cartesian position [x, y, z] of lower drone in world frame [m].

        Returns:
            force_vector: 3D disturbance force [fx, fy, fz] applied on lower drone [N].
        """
        delta_p = pos_lower - pos_upper
        delta_z = delta_p[2]  # < 0 when pos_lower is below pos_upper

        # If lower drone is at or above upper drone, wake does not affect it
        if delta_z >= 0.0:
            return np.zeros(3)

        depth = abs(delta_z)
        if depth > self.radius_z:
            return np.zeros(3)

        # Wake centerline deflection caused by upper drone horizontal velocity
        wake_skew = self.wake_skew_coeff * depth * vel_upper[0:2]
        rel_xy = delta_p[0:2] - wake_skew
        radial_dist = float(np.linalg.norm(rel_xy))

        if radial_dist > self.radius_xy:
            return np.zeros(3)

        # 1. Radial falloff calculation
        r_norm = radial_dist / self.radius_xy
        if self.radial_falloff == "gaussian":
            f_radial = np.exp(-0.5 * (r_norm * 2.0) ** 2)
        elif self.radial_falloff == "quadratic":
            f_radial = max(0.0, 1.0 - r_norm**2)
        elif self.radial_falloff == "linear":
            f_radial = max(0.0, 1.0 - r_norm)
        else:
            f_radial = np.exp(-0.5 * (r_norm * 2.0) ** 2)

        # 2. Vertical falloff calculation
        z_norm = depth / self.radius_z
        if self.vertical_falloff == "exponential":
            # Decays smoothly as depth increases toward wake boundary
            f_vert = np.exp(-1.5 * z_norm)
        elif self.vertical_falloff == "linear":
            f_vert = max(0.0, 1.0 - z_norm)
        else:
            f_vert = np.exp(-1.5 * z_norm)

        # 3. Primary downward disturbance force
        force_z = -self.max_force_z * f_radial * f_vert

        # Small outward radial expansion force (shear layer repulsion)
        if radial_dist > 1e-4:
            radial_dir = rel_xy / radial_dist
            # Radial shear is typically 10-15% of vertical force magnitude
            force_xy = 0.10 * abs(force_z) * radial_dir * (1.0 - r_norm)
        else:
            force_xy = np.zeros(2)

        return np.array([force_xy[0], force_xy[1], force_z])

    def combine_disturbances(self, pair_forces: List[np.ndarray]) -> np.ndarray:
        """Combine multiple disturbance force vectors acting on a single lower drone.

        Supports configurable superposition hypotheses:
        - 'linear_sum' / 'additive': Direct linear superposition sum(F_ij).
        - 'max' / 'peak': Single strongest disturbance vector dominates.
        - 'sublinear': Diminishing returns sum with exponent gamma in (0, 1].
        - 'saturated': Linear sum clamped to max_combined_force_z.

        Args:
            pair_forces: List of 3D force vectors [fx, fy, fz] exerted by each upper drone.

        Returns:
            Net 3D disturbance force vector [N].
        """
        if not pair_forces:
            return np.zeros(3)

        if len(pair_forces) == 1:
            return pair_forces[0].copy()

        forces_arr = np.array(pair_forces)  # Shape (M, 3)

        mode = self.combination_mode
        if mode in ("linear_sum", "additive"):
            net_force = np.sum(forces_arr, axis=0)
        elif mode in ("max", "peak"):
            # Select the force with largest downward component magnitude
            idx_max = int(np.argmax(np.abs(forces_arr[:, 2])))
            net_force = forces_arr[idx_max].copy()
        elif mode == "sublinear":
            # Radial forces sum linearly
            net_xy = np.sum(forces_arr[:, 0:2], axis=0)
            # Vertical forces combine with sublinear power law
            z_mags = np.abs(forces_arr[:, 2])
            combined_z_mag = (np.sum(z_mags ** self.sublinear_gamma)) ** (1.0 / self.sublinear_gamma)
            net_force = np.array([net_xy[0], net_xy[1], -combined_z_mag])
        elif mode == "saturated":
            net_force = np.sum(forces_arr, axis=0)
            if abs(net_force[2]) > self.max_combined_force_z:
                net_force[2] = -self.max_combined_force_z
        else:
            # Default to linear sum
            net_force = np.sum(forces_arr, axis=0)

        # Enforce global safety clamp on combined disturbance if configured
        if self.max_combined_force_z > 0 and abs(net_force[2]) > self.max_combined_force_z:
            net_force[2] = -self.max_combined_force_z

        return net_force

    def compute_swarm_forces(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
    ) -> np.ndarray:
        """Compute net downwash disturbance forces on all N drones in the swarm.

        Evaluates multi-source downwash interactions from all upper drones affecting
        each lower drone, combining contributions via the configured combination_mode.

        Args:
            positions: Array of shape (N, 3) containing 3D positions.
            velocities: Array of shape (N, 3) containing 3D velocities.

        Returns:
            forces: Array of shape (N, 3) where row j is the net disturbance wrench on drone j.
        """
        num_drones = len(positions)
        forces = np.zeros((num_drones, 3))

        for j in range(num_drones):
            contributions: List[np.ndarray] = []
            for i in range(num_drones):
                if i == j:
                    continue
                # Drone i acts on Drone j if i is physically above j
                if positions[i, 2] > positions[j, 2]:
                    f_ij = self.compute_pair_force(
                        pos_upper=positions[i],
                        vel_upper=velocities[i],
                        pos_lower=positions[j],
                    )
                    if np.any(f_ij != 0.0):
                        contributions.append(f_ij)

            if contributions:
                forces[j] = self.combine_disturbances(contributions)

        return forces
