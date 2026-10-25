"""Controller D: Uncertainty-Aware Adaptive Control (UA-ADC).

The primary novel contribution of this project.  UA-ADC does NOT pre-commit
to either soft avoidance OR feedforward compensation.  Instead, it:

  1. Estimates the downwash disturbance online with a per-drone RLS estimator.
  2. Tracks how confident that estimate is (posterior covariance → confidence).
  3. Blends continuously between soft-avoidance (high uncertainty) and
     feedforward compensation (high confidence) according to a sigmoid-shaped
     blend weight derived from the confidence.

Concretely, for drone j at every control step:

    α_j = sigmoid((c_j - blend_threshold) / blend_sharpness)
    cmd_pos_j  = (1 - α_j) * avoid_pos_j  +  α_j * nominal_pos_j
    cmd_vel_j  = (1 - α_j) * avoid_vel_j  +  α_j * nominal_vel_j
    ff_force_j = (1 - α_j) * safety_force_j  +  α_j * (-θ_j * F̂_assumed_j)

where c_j ∈ [0,1] is the drone's current confidence, α_j ∈ [0,1] is the
feedforward weight (α=0 → pure avoidance, α=1 → pure feedforward).

The 3D safety net from soft_avoid is always active (additive) and is never
blended away, even at full confidence.

Conflict override (closing-speed rule):
Confidence describes how well the wake model fits; it says nothing about
whether two drones are about to meet.  In a head-on vertical swap the
confidence can be high while the drones close on each other, which used to
blend the avoidance offset away.  For every pair we therefore predict the
closest point of approach (CPA) from both the measured states and the nominal
references.  If a pair will pass closer than ``conflict_radius`` within
``horizon_s``, the pair's conflict risk r_j rises toward 1 and

    alpha_j <- alpha_j * (1 - r_j)

forcing avoidance regardless of confidence.  r_j decays over ``hold_s``
after the conflict clears, so the drones do not snap back mid-crossing.

CRITICAL DESIGN RULE:
Uses only ASSUMED model parameters. Never inspects TRUE simulation configs.
"""
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union
import yaml
import numpy as np

from src.controllers.base import BaseSwarmController
from src.controllers.soft_avoid import SoftAvoidSwarmController
from src.controllers.feedforward import FeedforwardSwarmController
from src.models.downwash import DownwashModel
from src.models.disturbance_estimator import SwarmDisturbanceEstimator
from src.paths import config_path


def load_default_ua_adc_config() -> Dict[str, Any]:
    """Load ``configs/ua_adc.yaml`` (empty dict if the file is missing)."""
    path = config_path("ua_adc.yaml")
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


class UAAdaptiveController(BaseSwarmController):
    """Swarm controller that adaptively blends avoidance and feedforward
    based on online disturbance estimation confidence.

    Parameters (all config-driven, see ``configs/ua_adc.yaml``):
        blend_threshold (float): Confidence midpoint of sigmoid. Default 0.5.
        blend_sharpness (float): Steepness of sigmoid transition. Default 8.0.
        max_compensation_force (float): Clamp on feedforward force [N]. Default 0.25.
        drone_mass (float): Single-drone mass [kg]. Default 0.027.
        estimator: Sub-dict of RLS hyperparameters forwarded to
            SwarmDisturbanceEstimator.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        num_drones: int = 2,
        assumed_config: Optional[Union[Dict[str, Any], str, Path]] = None,
    ) -> None:
        """Initialise UA-ADC controller.

        Args:
            config: UA-ADC specific configuration dictionary.
            num_drones: Number of drones in the swarm.
            assumed_config: Assumed downwash parameters or path to assumed YAML.

        When ``config`` is None, ``configs/ua_adc.yaml`` is loaded so every
        entry point (demo, sweeps, tools) runs the same settings.
        """
        if config is None:
            config = load_default_ua_adc_config()
        super().__init__(name="Controller_D_UA_ADC", config=config, num_drones=num_drones)

        # ── 1. Load assumed downwash model parameters ──────────────────────
        self.assumed_params: Dict[str, Any] = self._load_assumed_config(assumed_config)
        self.assumed_model = DownwashModel(config=self.assumed_params)

        # ── 2. Sub-controllers (avoidance arm + safety net) ─────────────────
        # SoftAvoid provides avoidance setpoints AND the 3D safety repulsion.
        self._avoid_ctrl = SoftAvoidSwarmController(
            config=self.config.get("soft_avoid", None),
            num_drones=num_drones,
            assumed_config=self.assumed_params,
        )
        # Feedforward provides pure compensation setpoints (no position shift).
        self._ff_ctrl = FeedforwardSwarmController(
            config=self.config.get("feedforward", None),
            num_drones=num_drones,
            assumed_config=self.assumed_model,
        )

        # ── 3. Online disturbance estimator ─────────────────────────────────
        cfg = self.config or {}
        est_cfg = cfg.get("estimator", {})
        drone_mass: float = float(cfg.get("drone_mass", 0.027))
        self.estimator = SwarmDisturbanceEstimator(
            num_drones=num_drones,
            mass=drone_mass,
            config=est_cfg,
        )

        # ── 4. Blending parameters ──────────────────────────────────────────
        self.blend_threshold: float = float(cfg.get("blend_threshold", 0.5))
        self.blend_sharpness: float = float(cfg.get("blend_sharpness", 8.0))
        self.max_compensation_force: float = float(
            cfg.get("max_compensation_force",
                    1.5 * float(self.assumed_params.get("max_force_z", 0.15)))
        )

        # ── 5. Telemetry and state buffers ──────────────────────────────────
        self.drone_mass: float = drone_mass
        self.gravity: float = float(est_cfg.get("gravity", 9.81))
        self.last_confidences: np.ndarray = np.zeros(num_drones)
        self.last_blend_weights: np.ndarray = np.zeros(num_drones)
        self.last_commanded_thrusts: np.ndarray = np.full(
            num_drones, self.drone_mass * self.gravity
        )

        # ── 6. Conflict (closing-speed) override ────────────────────────────
        co_cfg = cfg.get("conflict_override", {}) or {}
        self.conflict_enabled: bool = bool(co_cfg.get("enabled", True))
        self.conflict_horizon_s: float = float(co_cfg.get("horizon_s", 2.5))
        self.conflict_radius: float = float(co_cfg.get("conflict_radius", 0.30))
        self.conflict_hold_s: float = float(co_cfg.get("hold_s", 1.0))
        self.conflict_use_reference: bool = bool(co_cfg.get("use_reference", True))
        self.last_conflict_risk: np.ndarray = np.zeros(num_drones)
        self.last_raw_blend_weights: np.ndarray = np.zeros(num_drones)

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _load_assumed_config(
        self,
        assumed_config: Optional[Union[Dict[str, Any], str, Path]] = None,
    ) -> Dict[str, Any]:
        """Load assumed downwash parameters from dict or YAML file."""
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

        # Minimal fallback defaults
        return {
            "radius_xy": 0.25,
            "radius_z": 0.80,
            "max_force_z": 0.15,
            "radial_falloff": "gaussian",
            "vertical_falloff": "exponential",
            "wake_skew_coefficient": 0.1,
            "safe_distance_threshold": 0.12,
            "safety_repulsion_max_force": 0.10,
            "safety_repulsion_displacement": 0.08,
        }

    @staticmethod
    def _sigmoid_blend(confidence: np.ndarray, threshold: float, sharpness: float) -> np.ndarray:
        """Compute per-drone feedforward blend weight α ∈ [0, 1].

        α → 0  when confidence < threshold  (prefer avoidance)
        α → 1  when confidence > threshold  (prefer feedforward)

        Args:
            confidence: Per-drone confidence values, shape (N,).
            threshold:  Sigmoid midpoint.
            sharpness:  Steepness k in σ(k*(c - t)).

        Returns:
            Blend weights α, shape (N,).
        """
        return 1.0 / (1.0 + np.exp(-sharpness * (confidence - threshold)))

    @staticmethod
    def _pair_cpa_risk(
        dp: np.ndarray, dv: np.ndarray, radius: float, horizon: float
    ) -> float:
        """Conflict risk in [0, 1] for one pair from relative position/velocity.

        Uses the closest point of approach (CPA) of a constant-velocity
        prediction.  Risk is 1 for a predicted miss distance of 0 and tapers
        smoothly (cosine) to 0 at ``radius``.  Pairs that are separating, or
        whose CPA lies beyond ``horizon`` seconds, carry no risk unless they
        are already inside ``radius``.

        Args:
            dp: Relative position p_j - p_i, shape (3,).
            dv: Relative velocity v_j - v_i, shape (3,).
            radius: Miss distance at which risk reaches 0 [m].
            horizon: Look-ahead time [s].
        """
        dist = float(np.linalg.norm(dp))
        closing = -float(np.dot(dp, dv))  # > 0 when the pair is approaching
        dv2 = float(np.dot(dv, dv))

        d_min = dist
        if closing > 0.0 and dv2 > 1e-9:
            t_cpa = closing / dv2
            if t_cpa <= horizon:
                d_min = float(np.linalg.norm(dp + dv * t_cpa))

        if d_min >= radius:
            return 0.0
        s = max(d_min / max(radius, 1e-6), 0.0)
        return float(0.5 * (1.0 + np.cos(np.pi * s)))

    def _compute_conflict_risk(
        self,
        states: np.ndarray,
        formation_targets: np.ndarray,
        formation_velocities: Optional[np.ndarray],
    ) -> np.ndarray:
        """Per-drone conflict risk: the largest CPA risk over all neighbours."""
        N = self.num_drones
        risk = np.zeros(N)
        pos = states[:, 0:3]
        vel = states[:, 10:13]
        ref_pos = np.asarray(formation_targets, dtype=float)
        ref_vel = (
            np.asarray(formation_velocities, dtype=float)
            if formation_velocities is not None
            else np.zeros_like(ref_pos)
        )
        for j in range(N):
            for i in range(j + 1, N):
                r = self._pair_cpa_risk(
                    pos[j] - pos[i], vel[j] - vel[i],
                    self.conflict_radius, self.conflict_horizon_s,
                )
                if self.conflict_use_reference:
                    r = max(r, self._pair_cpa_risk(
                        ref_pos[j] - ref_pos[i], ref_vel[j] - ref_vel[i],
                        self.conflict_radius, self.conflict_horizon_s,
                    ))
                risk[i] = max(risk[i], r)
                risk[j] = max(risk[j], r)
        return risk

    # ─────────────────────────────────────────────────────────────────────────
    # Public interface
    # ─────────────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset internal state for all sub-controllers and the estimator."""
        super().reset()
        self._avoid_ctrl.reset()
        self._ff_ctrl.reset()
        self.estimator.reset()
        self.last_confidences = np.zeros(self.num_drones)
        self.last_blend_weights = np.zeros(self.num_drones)
        self.last_commanded_thrusts = np.full(
            self.num_drones, self.drone_mass * self.gravity
        )
        self.last_conflict_risk = np.zeros(self.num_drones)
        self.last_raw_blend_weights = np.zeros(self.num_drones)

    def compute_commands(
        self,
        states: np.ndarray,
        formation_targets: np.ndarray,
        timestep: float,
        formation_velocities: Optional[np.ndarray] = None,
        commanded_thrusts: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute blended setpoints and feedforward forces.

        Algorithm per step:
          1. Get avoidance setpoints from SoftAvoid sub-controller.
          2. Get nominal (no-shift) setpoints from Feedforward sub-controller.
          3. Evaluate assumed-model disturbances on current positions.
          4. Update RLS estimators with commanded thrusts → confidences.
          5. Compute sigmoid blend weight α per drone.
          6. Blend position, velocity, force: X = (1-α)*avoid + α*ff.
          7. Add 3D safety-net force (always active, not blended away).

        Args:
            states: Drone states, shape (N, 20).
            formation_targets: Nominal formation positions, shape (N, 3).
            timestep: Control timestep [s].
            formation_velocities: Optional nominal velocities, shape (N, 3).
            commanded_thrusts: Optional commanded thrusts [N], shape (N,).

        Returns:
            Tuple of:
                - cmd_positions: shape (N, 3).
                - cmd_velocities: shape (N, 3).
                - feedforward_forces: shape (N, 3) [N].
        """
        N = self.num_drones

        # ── Step 1: Avoidance arm ────────────────────────────────────────────
        # SoftAvoid also returns safety net forces in its ff_forces output.
        avoid_pos, avoid_vel, safety_forces = self._avoid_ctrl.compute_commands(
            states=states,
            formation_targets=formation_targets,
            timestep=timestep,
            formation_velocities=formation_velocities,
        )

        # ── Step 2: Feedforward arm ──────────────────────────────────────────
        nominal_pos, nominal_vel, _ = self._ff_ctrl.compute_commands(
            states=states,
            formation_targets=formation_targets,
            timestep=timestep,
            formation_velocities=formation_velocities,
        )

        # ── Step 3: Assumed-model disturbances for estimator feature ─────────
        positions = states[:, 0:3]
        velocities = states[:, 10:13]
        assumed_forces = self.assumed_model.compute_swarm_forces(
            positions=positions,
            velocities=velocities,
        )

        # ── Step 4: RLS estimator update → per-drone confidences ────────────
        cmd_thrusts = (
            np.asarray(commanded_thrusts, dtype=float)
            if commanded_thrusts is not None
            else self.last_commanded_thrusts
        )
        estimated_forces, confidences = self.estimator.step(
            states=states,
            assumed_forces=assumed_forces,
            timestep=timestep,
            commanded_thrusts=cmd_thrusts,
        )

        # ── Step 5: Sigmoid blend weights α ∈ [0, 1] per drone ──────────────
        raw_alpha = self._sigmoid_blend(confidences, self.blend_threshold, self.blend_sharpness)

        # ── Step 5b: Conflict override (forces avoidance on approach) ───────
        if self.conflict_enabled and N > 1:
            risk_now = self._compute_conflict_risk(states, formation_targets, formation_velocities)
            decay = timestep / max(self.conflict_hold_s, 1e-6)
            self.last_conflict_risk = np.maximum(risk_now, self.last_conflict_risk - decay)
        else:
            self.last_conflict_risk = np.zeros(N)
        alpha = raw_alpha * (1.0 - self.last_conflict_risk)

        # Store telemetry for visualisation
        self.last_confidences = confidences.copy()
        self.last_raw_blend_weights = raw_alpha.copy()
        self.last_blend_weights = alpha.copy()

        # ── Step 6: Blend position, velocity, and compensation force ─────────
        cmd_positions = np.zeros((N, 3))
        cmd_velocities = np.zeros((N, 3))
        comp_forces = np.zeros((N, 3))

        for j in range(N):
            a = float(alpha[j])
            # avoid + a * (nominal - avoid) equals (1 - a) * avoid + a * nominal,
            # but is exact when both arms agree (no rounding drift where no
            # avoidance is active, e.g. coplanar formations).
            cmd_positions[j] = avoid_pos[j] + a * (nominal_pos[j] - avoid_pos[j])
            cmd_velocities[j] = avoid_vel[j] + a * (nominal_vel[j] - avoid_vel[j])

            # Adaptive compensation force scaled by blend weight
            # estimated_forces[j] = theta[j] * assumed_forces[j]
            comp_forces[j] = a * (-estimated_forces[j])

        # Clamp compensation forces
        comp_forces = np.clip(comp_forces, -self.max_compensation_force, self.max_compensation_force)

        # ── Step 7: Always-active 3D safety-net force (additive) ────────────
        feedforward_forces = comp_forces + safety_forces

        return cmd_positions, cmd_velocities, feedforward_forces

    def compute_actions(
        self,
        control_timestep: float,
        states: np.ndarray,
        formation_targets: np.ndarray,
        formation_velocities: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Full end-to-end swarm step: coordination + low-level motor commands.

        Overrides BaseSwarmController to record commanded thrusts for causal,
        closed-loop online disturbance estimation.
        """
        cmd_positions, cmd_velocities, ff_forces = self.compute_commands(
            states=states,
            formation_targets=formation_targets,
            timestep=control_timestep,
            formation_velocities=formation_velocities,
            commanded_thrusts=self.last_commanded_thrusts,
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

            # Vertical component of the commanded thrust from commanded RPMs.
            # Thrust acts along the body z-axis; its world-z share is
            # R[2, 2] = 1 - 2(qx^2 + qy^2): ~1 in hover, smaller under tilt.
            kf = getattr(self.low_level_controllers[i]._pid, "KF", 3.16e-10)
            t_cmd = float(kf * np.sum(rpm ** 2))
            qx, qy = float(states[i, 3]), float(states[i, 4])
            self.last_commanded_thrusts[i] = t_cmd * (1.0 - 2.0 * (qx * qx + qy * qy))

        return rpms, pos_errors, yaw_errors
