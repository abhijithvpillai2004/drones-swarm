"""Online disturbance estimator with confidence/uncertainty tracking.

Implements a per-drone Recursive Least Squares (RLS) estimator that fits a
parametric downwash model to the observed acceleration residuals in real-time.
A scalar confidence value is derived from the posterior covariance trace, and
this confidence drives the UA-ADC blending weight.

DESIGN RULES:
- Uses only ASSUMED model parameters as a prior — never peeks at TRUE config.
- Confidence is in [0, 1] where 1 = estimator is certain (low covariance).
- All hyperparameters are config-driven; no magic numbers inside the class.
"""
from __future__ import annotations

from typing import Dict, Any, Optional
import numpy as np


class DroneDisturbanceEstimator:
    """Per-drone recursive least-squares disturbance estimator with uncertainty.

    Maintains a running estimate of the downwash force acting on one drone by
    fitting a scalar gain on a Gaussian basis function evaluated from the
    assumed model prediction.  The RLS update rule is:

        θ_k = θ_{k-1} + K_k (y_k − φ_k θ_{k-1})
        P_k = (I − K_k φ_k) P_{k-1} λ^{-1}
        K_k = P_{k-1} φ_k / (λ + φ_k^T P_{k-1} φ_k)

    where
        y_k   = observed z-acceleration residual [m/s²]
        φ_k   = normalised assumed-model prediction (scalar feature)
        θ_k   = estimated gain scalar
        P_k   = posterior variance (scalar, 1×1 system for speed)
        λ     = forgetting factor in (0, 1]

    Confidence is defined as a sigmoid of the relative posterior standard
    deviation saturation, clipped to [0, 1].
    """

    def __init__(
        self,
        drone_id: int,
        mass: float,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialise a single-drone disturbance estimator.

        Args:
            drone_id: Index of this drone (for logging/debug only).
            mass: Drone mass [kg], used to convert acceleration → force.
            config: Hyperparameter dictionary. Recognised keys:
                - forgetting_factor: RLS λ in (0, 1]. Default 0.98.
                - prior_variance: Initial P_0 (uncertainty). Default 1.0.
                - max_covariance: Covariance upper bound. Default 5.0.
                - min_covariance: Covariance lower bound. Default 1e-4.
                - min_excitation: Normalized regressor threshold to freeze RLS. Default 0.01.
                - innovation_scale: Normalised innovation scale for confidence. Default 0.04.
                - confidence_smoothing: EMA coefficient for confidence signal. Default 0.90.
                - max_gain: Hard cap on |θ| to prevent runaway. Default 3.0.
                - gravity: Gravitational acceleration. Default 9.81 m/s².
        """
        cfg = config or {}
        self.drone_id: int = drone_id
        self.mass: float = float(mass)

        # RLS hyperparameters
        self.lam: float = float(cfg.get("forgetting_factor", 0.98))
        self.prior_variance: float = float(cfg.get("prior_variance", 1.0))
        self.max_gain: float = float(cfg.get("max_gain", 3.0))

        # Covariance bounds
        self.max_covariance: float = float(cfg.get("max_covariance", 5.0))
        self.min_covariance: float = float(cfg.get("min_covariance", 1e-4))

        # Excitation threshold (normalized regressor magnitude)
        self.min_excitation: float = float(cfg.get("min_excitation", 0.01))

        # Normalized innovation scale for confidence
        self.innovation_scale: float = float(cfg.get("innovation_scale", 0.04))

        # Legacy thresholds (retained for backward compatibility)
        self.P_uncertain: float = float(cfg.get("min_confidence_variance", 1.0))
        self.P_confident: float = float(cfg.get("max_confidence_variance", 1e-3))

        # Confidence EMA smoothing
        self.conf_smooth: float = float(cfg.get("confidence_smoothing", 0.90))

        # Misc
        self.gravity: float = float(cfg.get("gravity", 9.81))

        # Internal state
        self.theta: float = 1.0    # gain estimate (starts at 1 = trust assumed model fully)
        self.P: float = self.prior_variance     # posterior variance
        self.confidence: float = 0.0            # smoothed confidence in [0, 1]
        self._prev_vel: Optional[np.ndarray] = None

    def reset(self) -> None:
        """Reset estimator state (call at episode start)."""
        self.theta = 1.0
        self.P = self.prior_variance
        self.confidence = 0.0
        self._prev_vel = None

    def _compute_confidence(self, normalized_innovation: float) -> float:
        """Map normalized innovation (y - phi*theta)/(m*g) to confidence in [0, 1].

        High consistency (|innovation| ≈ 0) → confidence ≈ 1.
        Large mismatch (|innovation| >> innovation_scale) → confidence ≈ 0.
        """
        scale = max(self.innovation_scale, 1e-4)
        raw = float(np.exp(-0.5 * (normalized_innovation / scale) ** 2))
        return float(np.clip(raw, 0.0, 1.0))

    def update(
        self,
        assumed_model_force: np.ndarray,
        observed_accel: np.ndarray,
        timestep: float,
        commanded_thrust: Optional[float] = None,
    ) -> None:
        """Perform one RLS update step.

        Args:
            assumed_model_force: 3D force predicted by the assumed downwash model
                acting on this drone [N]. Shape (3,).
            observed_accel: Measured/estimated linear acceleration of this drone
                [m/s²]. Shape (3,).
            timestep: Control timestep Δt [s].
            commanded_thrust: Commanded vertical thrust [N] from low-level controller.
                If None, defaults to hover thrust m*g.
        """
        weight = self.mass * self.gravity  # nominal hover weight [N]

        # Commanded thrust (defaults to m*g if not supplied)
        t_cmd = float(commanded_thrust) if commanded_thrust is not None else weight

        # Measurement y = m * a_obs - (T_cmd - m * g) in [N]
        # In hover, closed-loop PID commands extra thrust to oppose downwash:
        # T_cmd - mg ≈ -F_downwash, a_obs ≈ 0 => y ≈ F_downwash (e.g. negative for downward).
        observed_az = float(observed_accel[2])
        y = self.mass * observed_az - (t_cmd - weight)

        # Regressor phi = assumed downwash signal [N] (negative = downward force)
        phi = float(assumed_model_force[2])

        # Normalize regressor and measurement by m*g (dimensionless)
        phi_norm = phi / max(weight, 1e-6)
        y_norm = y / max(weight, 1e-6)

        # Freeze updates when excitation is below threshold (e.g. not in wake)
        if abs(phi_norm) < self.min_excitation:
            return

        # RLS update with normalized regressor (1-D system)
        denom = self.lam + phi_norm * self.P * phi_norm
        if abs(denom) < 1e-9:
            return

        K = (self.P * phi_norm) / denom
        norm_innovation = y_norm - phi_norm * self.theta

        # Update gain estimate theta and clip
        self.theta = float(np.clip(self.theta + K * norm_innovation, -self.max_gain, self.max_gain))

        # Update posterior variance P with forgetting factor and enforce bounds
        p_updated = (1.0 / self.lam) * (1.0 - K * phi_norm) * self.P
        self.P = float(np.clip(p_updated, self.min_covariance, self.max_covariance))

        # Compute confidence from normalized innovation and update smoothed confidence
        raw_conf = self._compute_confidence(norm_innovation)
        self.confidence = float(self.conf_smooth * self.confidence + (1.0 - self.conf_smooth) * raw_conf)

    def estimate_disturbance(self, assumed_model_force: np.ndarray) -> np.ndarray:
        """Compute the online disturbance estimate: θ * assumed_model_force.

        Args:
            assumed_model_force: 3D force from assumed downwash model [N].

        Returns:
            Estimated true disturbance force [N], shape (3,).
        """
        return float(self.theta) * np.asarray(assumed_model_force, dtype=float)

    @property
    def posterior_variance(self) -> float:
        """Current posterior variance P (raw uncertainty scalar)."""
        return self.P


class SwarmDisturbanceEstimator:
    """Wrapper that maintains one DroneDisturbanceEstimator per drone.

    Provides a single compute step that updates all estimators and returns
    per-drone estimated disturbances and confidences.
    """

    def __init__(
        self,
        num_drones: int,
        mass: float,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialise per-drone estimators.

        Args:
            num_drones: Number of drones N.
            mass: Single-drone mass [kg].
            config: Shared hyperparameter dict passed to each DroneDisturbanceEstimator.
        """
        self.num_drones = num_drones
        self.estimators = [
            DroneDisturbanceEstimator(drone_id=i, mass=mass, config=config)
            for i in range(num_drones)
        ]
        self._prev_velocities: Optional[np.ndarray] = None

    def reset(self) -> None:
        """Reset all estimators (call at episode start)."""
        for est in self.estimators:
            est.reset()
        self._prev_velocities = None

    def step(
        self,
        states: np.ndarray,
        assumed_forces: np.ndarray,
        timestep: float,
        commanded_thrusts: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Perform one estimation step for all N drones.

        Args:
            states: Current drone states, shape (N, 20).
                    Positions = states[:, 0:3], velocities = states[:, 10:13].
            assumed_forces: Assumed-model disturbance forces, shape (N, 3) [N].
            timestep: Control timestep Δt [s].
            commanded_thrusts: Optional array of commanded vertical thrusts [N], shape (N,).

        Returns:
            Tuple of:
                - estimated_disturbances: shape (N, 3) [N].
                - confidences: shape (N,) in [0, 1].
        """
        velocities = states[:, 10:13]

        # Finite-difference acceleration estimate
        if self._prev_velocities is None:
            accels = np.zeros((self.num_drones, 3))
        else:
            accels = (velocities - self._prev_velocities) / max(timestep, 1e-6)
        self._prev_velocities = velocities.copy()

        estimated_disturbances = np.zeros((self.num_drones, 3))
        confidences = np.zeros(self.num_drones)

        for i, est in enumerate(self.estimators):
            cmd_t = float(commanded_thrusts[i]) if commanded_thrusts is not None else None
            est.update(
                assumed_model_force=assumed_forces[i],
                observed_accel=accels[i],
                timestep=timestep,
                commanded_thrust=cmd_t,
            )
            estimated_disturbances[i] = est.estimate_disturbance(assumed_forces[i])
            confidences[i] = est.confidence

        return estimated_disturbances, confidences
