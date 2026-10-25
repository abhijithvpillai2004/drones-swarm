"""
src/models/downwash_base.py
===========================
Abstract base class for downwash disturbance models.

Both TrueDownwash and AssumedDownwash inherit from this class,
ensuring a consistent interface while keeping their parameters
strictly separate.
"""

from __future__ import annotations

import abc
from typing import Optional

import numpy as np


class DownwashBase(abc.ABC):
    """Abstract base for all downwash models.

    Sub-classes implement :meth:`force_on_lower` which computes
    the 3-D disturbance force (N) applied to the *lower* drone
    given positions of an *upper* drone and the *lower* drone.

    Parameters are always loaded from a config dict passed at
    construction time — never from another model's config.
    """

    def __init__(self, cfg: dict) -> None:
        """Initialise common parameters from *cfg* dict.

        Expected keys (with types and units):
            peak_force_N  : float  — peak downwash force magnitude [N]
            sigma_r_m     : float  — horizontal wake radius 1σ [m]
            sigma_z_m     : float  — vertical wake extent 1σ [m]
            min_dz_m      : float  — minimum vertical separation to apply [m]
            max_range_m   : float  — beyond this, no interaction [m]
        """
        self.peak_force_N: float = float(cfg["peak_force_N"])
        self.sigma_r_m: float = float(cfg["sigma_r_m"])
        self.sigma_z_m: float = float(cfg["sigma_z_m"])
        self.min_dz_m: float = float(cfg.get("min_dz_m", 0.05))
        self.max_range_m: float = float(cfg.get("max_range_m", 3.0))

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def force_on_lower(
        self,
        pos_upper: np.ndarray,
        pos_lower: np.ndarray,
        rpm_upper: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute downwash force vector applied to the lower drone.

        Parameters
        ----------
        pos_upper : ndarray, shape (3,)
            World-frame position [x, y, z] of the upper (source) drone [m].
        pos_lower : ndarray, shape (3,)
            World-frame position [x, y, z] of the lower (victim) drone [m].
        rpm_upper : ndarray or None, shape (4,)
            Motor RPMs of the upper drone (unused in base model; available
            for RPM-dependent extensions).

        Returns
        -------
        ndarray, shape (3,)
            Force vector [Fx, Fy, Fz] in world frame [N].
            Fz is negative (downward) when in wake.
            Returns zero vector if geometry or range conditions not met.
        """
        pos_upper = np.asarray(pos_upper, dtype=float)
        pos_lower = np.asarray(pos_lower, dtype=float)

        dz = pos_upper[2] - pos_lower[2]   # positive = upper is above lower
        if dz < self.min_dz_m:
            return np.zeros(3)

        dxy = np.linalg.norm(pos_upper[:2] - pos_lower[:2])
        r = np.sqrt((dxy / self.sigma_r_m) ** 2 + (dz / self.sigma_z_m) ** 2)

        if r * max(self.sigma_r_m, self.sigma_z_m) > self.max_range_m:
            return np.zeros(3)

        magnitude = self._compute_magnitude(r, rpm_upper=rpm_upper)
        return np.array([0.0, 0.0, -magnitude])   # downward

    # ------------------------------------------------------------------
    # Protected helpers — sub-classes override _compute_magnitude
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def _compute_magnitude(
        self,
        r: float,
        rpm_upper: Optional[np.ndarray] = None,
    ) -> float:
        """Return the unsigned magnitude of the downwash force [N].

        Parameters
        ----------
        r : float
            Normalised radial distance in the ellipsoidal metric.
        rpm_upper : ndarray or None
            Upper drone RPMs (for RPM-aware extensions).
        """
        ...

    # ------------------------------------------------------------------
    # Utility: influence scalar (0–1), public for visualisation
    # ------------------------------------------------------------------

    def influence_at(self, pos_upper: np.ndarray, pos_lower: np.ndarray) -> float:
        """Return the normalised influence scalar ∈ [0, 1] at given positions.

        Useful for rendering wake ellipsoids in the visualiser.
        Returns 0 if conditions not met.
        """
        pos_upper = np.asarray(pos_upper, dtype=float)
        pos_lower = np.asarray(pos_lower, dtype=float)

        dz = pos_upper[2] - pos_lower[2]
        if dz < self.min_dz_m:
            return 0.0

        dxy = np.linalg.norm(pos_upper[:2] - pos_lower[:2])
        r = np.sqrt((dxy / self.sigma_r_m) ** 2 + (dz / self.sigma_z_m) ** 2)
        return float(self._compute_magnitude(r) / max(self.peak_force_N, 1e-9))
