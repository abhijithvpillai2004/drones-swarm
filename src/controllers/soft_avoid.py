"""Controller B: Soft-Constraint Avoidance Swarm Controller.

Penalises and avoids vertical stacking (drone flying directly in another drone's wake)
using a smooth artificial potential field / lateral offset.

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


class SoftAvoidSwarmController(BaseSwarmController):
    """Swarm controller applying smooth lateral avoidance when entering assumed downwash wakes."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        num_drones: int = 2,
        assumed_config: Optional[Union[Dict[str, Any], str, Path]] = None,
    ) -> None:
        """Initialize Soft-Avoidance swarm controller.

        Args:
            config: Optional controller settings (max_displacement, max_velocity, etc.).
            num_drones: Number of drones in the swarm.
            assumed_config: Assumed downwash parameters or path to assumed YAML config.
        """
        super().__init__(name="Controller_B_SoftAvoid", config=config, num_drones=num_drones)

        # 1. Load ASSUMED downwash model parameters (strict anti-circularity)
        self.assumed_params: Dict[str, Any] = self._load_assumed_config(assumed_config)
        self.radius_xy: float = float(self.assumed_params.get("radius_xy", 0.25))
        self.radius_z: float = float(self.assumed_params.get("radius_z", 0.80))
        self.wake_skew_coeff: float = float(self.assumed_params.get("wake_skew_coefficient", 0.1))

        # 2. General 3D safety net parameters (Step 2 hardening)
        self.safe_distance_threshold: float = float(self.assumed_params.get("safe_distance_threshold", 0.12))
        self.safety_repulsion_max_force: float = float(self.assumed_params.get("safety_repulsion_max_force", 0.10))
        self.safety_repulsion_displacement: float = float(self.assumed_params.get("safety_repulsion_displacement", 0.08))

        # 3. Avoidance limits
        cfg = self.config or {}
        default_disp = 0.28 if self.num_drones >= 8 else 0.20
        default_vel = 0.30 if self.num_drones >= 8 else 0.15
        self.max_displacement: float = float(cfg.get("max_displacement", default_disp))  # [m]
        self.max_velocity: float = float(cfg.get("max_velocity", default_vel))          # [m/s]
        if self.num_drones >= 8:
            self.max_displacement = max(self.max_displacement, 0.28)
            self.max_velocity = max(self.max_velocity, 0.30)
        # Radius of the wake core inside which the escape direction is held
        # fixed (see compute_avoidance_offset). Default: 40 % of radius_xy.
        self.direction_lock_radius: float = float(
            cfg.get("direction_lock_radius", 0.4 * self.radius_xy))
        # How the offset's rate of change is fed to the velocity setpoint:
        # "raw" (step-to-step derivative), "filtered" (first-order low-pass
        # with time constant offset_velocity_tau) or "none".
        self.offset_velocity_mode: str = str(cfg.get("offset_velocity_mode", "filtered"))
        self.offset_velocity_tau: float = float(cfg.get("offset_velocity_tau", 0.25))  # [s]
        self.offset_velocity: np.ndarray = np.zeros((self.num_drones, 2))

        # 4. State memory for smooth rate-limited displacements
        # Array of shape (num_drones, 2) storing current lateral avoidance offsets [dx, dy]
        self.current_offsets: np.ndarray = np.zeros((self.num_drones, 2))
        self.last_avoid_dirs: np.ndarray = np.zeros((self.num_drones, 2))
        self.symmetry_dirs: Optional[np.ndarray] = None

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
            "wake_skew_coefficient": 0.1,
            "safe_distance_threshold": 0.12,
            "safety_repulsion_max_force": 0.10,
            "safety_repulsion_displacement": 0.08,
        }

    def reset(self) -> None:
        """Reset internal controller state and low-level controllers."""
        super().reset()
        self.current_offsets = np.zeros((self.num_drones, 2))
        self.last_avoid_dirs = np.zeros((self.num_drones, 2))
        self.symmetry_dirs = None
        self.offset_velocity = np.zeros((self.num_drones, 2))

    def _update_symmetry_directions(
        self,
        states: np.ndarray,
        formation_targets: np.ndarray,
        formation_velocities: Optional[np.ndarray],
    ) -> None:
        """Assign direction-aware symmetry-breaking angles with guaranteed 180° separation.

        For scenarios where drones have opposing vertical travel directions (e.g. vertical_swap),
        drones traveling in opposite vertical directions are explicitly paired and assigned
        antipodal angles (180° apart) with alternating polarity to maximize inter-drone clearance.
        For non-crossing formations (hover_column_3d, crossing_pair, grid), symmetry_dirs is set
        to None and default radial spread is preserved.
        """
        if formation_velocities is not None:
            vz = formation_velocities[:, 2]
        else:
            vz = states[:, 12]

        # Require intentional vertical swap velocity (> 0.04 m/s)
        # Prevents sensor noise in horizontal flight from falsely activating vertical swap corridors
        desc_idx = [i for i in range(self.num_drones) if vz[i] < -0.04]
        asc_idx = [i for i in range(self.num_drones) if vz[i] > 0.04]
        stat_idx = [i for i in range(self.num_drones) if abs(vz[i]) <= 0.04]

        if len(desc_idx) > 0 and len(asc_idx) > 0:
            desc_idx.sort(key=lambda i: vz[i])   # fastest descending first
            asc_idx.sort(key=lambda i: -vz[i])   # fastest ascending first
            ordered_drones = desc_idx + stat_idx + asc_idx

            # Interleave slots around the full 2*pi circle:
            # Even slots: 0, 2, 4, ...
            # Odd slots:  1, 3, 5, ...
            # This guarantees that adjacent drones in the vertical column receive
            # alternating sectors around the circle, and every single drone has
            # an optimal angular separation of at least 2*pi / N from all neighbors
            # with zero clustering or empty gaps.
            N = self.num_drones
            evens = list(range(0, N, 2))
            odds = list(range(1, N, 2))
            slots = evens + odds

            angles = np.zeros(N)
            for idx, drone_id in enumerate(ordered_drones):
                angles[drone_id] = slots[idx] * (2.0 * np.pi / N)

            self.symmetry_dirs = np.column_stack([np.cos(angles), np.sin(angles)])
            self.last_avoid_dirs = self.symmetry_dirs.copy()
        else:
            self.symmetry_dirs = None

    def compute_avoidance_offset(
        self,
        pos_upper: np.ndarray,
        vel_upper: np.ndarray,
        pos_lower: np.ndarray,
        drone_id_lower: int,
    ) -> np.ndarray:
        """Compute smooth lateral displacement for a lower drone in an upper drone's wake.

        Args:
            pos_upper: Position [x, y, z] of upper drone [m].
            vel_upper: Velocity [vx, vy, vz] of upper drone [m/s].
            pos_lower: Position [x, y, z] of lower drone [m].
            drone_id_lower: Index of lower drone for symmetry breaking.

        Returns:
            2D avoidance displacement vector [dx, dy] [m].
        """
        delta_p = pos_lower - pos_upper
        delta_z = delta_p[2]  # < 0 when lower is below upper

        # If lower drone is above or outside vertical reach, no avoidance needed
        if delta_z >= 0.0:
            return np.zeros(2)

        depth = abs(delta_z)
        if depth > self.radius_z:
            return np.zeros(2)

        if self.symmetry_dirs is not None:
            repel_dir = self.symmetry_dirs[drone_id_lower]
            weight_r = 1.0
        else:
            # Wake centerline deflection caused by upper drone velocity
            wake_skew = self.wake_skew_coeff * depth * vel_upper[0:2]
            rel_xy = delta_p[0:2] - wake_skew
            radial_dist = float(np.linalg.norm(rel_xy))

            # Outside assumed horizontal wake radius -> zero avoidance
            if radial_dist >= self.radius_xy:
                return np.zeros(2)

            if radial_dist > self.direction_lock_radius:
                repel_dir = rel_xy / radial_dist
                self.last_avoid_dirs[drone_id_lower] = repel_dir
            else:
                # Inside the wake core the direction of rel_xy is set by
                # centimetre-level wobble of the upper drone and flips sign
                # from step to step.  Following it made the offset (and its
                # velocity feedforward) chatter, which a 2-step command delay
                # turned into a growing oscillation and a fall.  Keep the
                # previous direction instead.  On first entry use the
                # per-drone spread angle: stacked drones then escape in
                # different directions, so a lower drone does not follow the
                # drone above it into that drone's shifted wake.
                if np.linalg.norm(self.last_avoid_dirs[drone_id_lower]) > 0.5:
                    repel_dir = self.last_avoid_dirs[drone_id_lower]
                else:
                    angle = (2.0 * np.pi * drone_id_lower) / max(1, self.num_drones)
                    repel_dir = np.array([np.cos(angle), np.sin(angle)])
                    self.last_avoid_dirs[drone_id_lower] = repel_dir

            # Radial weighting: smooth transition from core to boundary
            r_norm = radial_dist / self.radius_xy
            weight_r = 0.5 * (1.0 + np.cos(np.pi * r_norm))

        # Vertical weighting: active across wake column, smoothly tapering near bottom boundary
        z_norm = depth / self.radius_z
        if z_norm <= 0.85:
            weight_z = 1.0
        else:
            weight_z = 0.5 * (1.0 + np.cos(np.pi * (z_norm - 0.85) / 0.15))

        offset_magnitude = self.max_displacement * weight_r * weight_z
        return offset_magnitude * repel_dir

    def compute_3d_safety_repulsion(
        self,
        pos_self: np.ndarray,
        pos_other: np.ndarray,
        drone_id: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute emergency 3D Euclidean safety-net repulsion force and lateral offset.

        Activates based on true 3D Euclidean distance between any two drones,
        regardless of relative altitude, as a last-resort hardening layer.
        Scales smoothly to zero at the boundary using a C1 continuous cosine taper.

        Args:
            pos_self: Position [x, y, z] of this drone [m].
            pos_other: Position [x, y, z] of neighbor drone [m].
            drone_id: Drone index for deterministic tie-breaking.

        Returns:
            Tuple of:
                - force_3d: Repulsive force vector [Fx, Fy, Fz] [N].
                - offset_2d: Additive lateral displacement [dx, dy] [m].
        """
        delta = pos_self - pos_other
        dist_3d = float(np.linalg.norm(delta))

        # At safe separation, contributes ~zero force
        if dist_3d >= self.safe_distance_threshold:
            return np.zeros(3), np.zeros(2)

        if dist_3d > 1e-4:
            u_3d = delta / dist_3d
        else:
            # Deterministic symmetry breaking if coincident
            angle = (2.0 * np.pi * drone_id) / max(1, self.num_drones)
            u_3d = np.array([np.cos(angle), np.sin(angle), 0.0])

        # Smooth cosine taper: 1.0 at d=0 down to 0.0 at d=safe_distance_threshold
        s = dist_3d / max(self.safe_distance_threshold, 1e-6)
        weight = 0.5 * (1.0 + np.cos(np.pi * s))

        force_3d = self.safety_repulsion_max_force * weight * u_3d
        offset_2d = self.safety_repulsion_displacement * weight * u_3d[0:2]
        return force_3d, offset_2d

    def compute_commands(
        self,
        states: np.ndarray,
        formation_targets: np.ndarray,
        timestep: float,
        formation_velocities: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute setpoints with smooth lateral avoidance of vertical stacking and 3D safety net.

        Args:
            states: Current states array of shape (N, 20).
            formation_targets: Desired formation positions array of shape (N, 3).
            timestep: Control timestep [s].
            formation_velocities: Optional desired velocities array of shape (N, 3).

        Returns:
            Tuple of:
                - cmd_positions: Shape (N, 3), adapted formation positions.
                - cmd_velocities: Shape (N, 3), adapted target velocities.
                - feedforward_forces: Shape (N, 3), emergency 3D safety-net repulsion forces.
        """
        N = self.num_drones
        if self.symmetry_dirs is None:
            self._update_symmetry_directions(states, formation_targets, formation_velocities)

        cmd_positions = np.asarray(formation_targets, dtype=float).copy()

        if formation_velocities is not None:
            cmd_velocities = np.asarray(formation_velocities, dtype=float).copy()
        else:
            cmd_velocities = np.zeros_like(cmd_positions)

        positions = states[:, 0:3]
        velocities = states[:, 10:13]

        # Calculate target lateral avoidance offsets and 3D safety net forces
        target_offsets = np.zeros((N, 2))
        safety_forces = np.zeros((N, 3))
        safety_offsets = np.zeros((N, 2))

        for j in range(N):
            for i in range(N):
                if i == j:
                    continue

                # 1. Existing vertical-wake avoidance (upper drone on lower drone)
                # Evaluated relative to formation_targets[j] to prevent hunting/limit-cycle chatter
                if positions[i, 2] > formation_targets[j, 2]:
                    offset_ij = self.compute_avoidance_offset(
                        pos_upper=positions[i],
                        vel_upper=velocities[i],
                        pos_lower=formation_targets[j],
                        drone_id_lower=j,
                    )
                    target_offsets[j] += offset_ij

                # 2. General 3D minimum-distance safety net (independent of relative altitude)
                f_ji, d_ji = self.compute_3d_safety_repulsion(
                    pos_self=positions[j],
                    pos_other=positions[i],
                    drone_id=j,
                )
                safety_forces[j] += f_ji
                safety_offsets[j] += d_ji

            # Combine vertical-wake offset with additive 3D safety offset
            target_offsets[j] += safety_offsets[j]

            # Clamp net target offset to max_displacement
            norm_offset = np.linalg.norm(target_offsets[j])
            if norm_offset > self.max_displacement:
                target_offsets[j] = target_offsets[j] * (self.max_displacement / norm_offset)

        # Rate-limit offset changes to prevent velocity/acceleration spikes
        max_delta = self.max_velocity * timestep
        for j in range(N):
            diff = target_offsets[j] - self.current_offsets[j]
            diff_norm = np.linalg.norm(diff)
            if diff_norm > max_delta and diff_norm > 1e-6:
                diff = diff * (max_delta / diff_norm)

            new_offset = self.current_offsets[j] + diff
            rate_of_change = (new_offset - self.current_offsets[j]) / max(timestep, 1e-6)
            self.current_offsets[j] = new_offset

            # Apply lateral offset to commanded target position and velocity
            cmd_positions[j, 0:2] += self.current_offsets[j]
            if self.offset_velocity_mode == "raw":
                self.offset_velocity[j] = rate_of_change
            elif self.offset_velocity_mode == "filtered":
                k = timestep / max(self.offset_velocity_tau + timestep, 1e-6)
                self.offset_velocity[j] += k * (rate_of_change - self.offset_velocity[j])
            else:
                self.offset_velocity[j] = 0.0
            cmd_velocities[j, 0:2] += self.offset_velocity[j]

        feedforward_forces = safety_forces
        return cmd_positions, cmd_velocities, feedforward_forces
