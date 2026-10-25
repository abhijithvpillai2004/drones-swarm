"""
src/models/true_downwash.py
============================
TRUE downwash disturbance model.

This module is loaded ONLY by the simulator (src/sim/).
Controllers must NEVER import or read this module directly.
That separation prevents circular "controller grades its own exam" results
(see docs/decisions.md D-002).

Model: Ellipsoidal Gaussian, after Hönig et al. (IEEE T-RO 2018) and
       Neural-Swarm (Shi et al., ICRA 2020).

⚠️ All parameters come from configs/downwash_true.yaml and require
   literature verification before formal results can be cited.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import yaml

from .downwash_base import DownwashBase


class TrueDownwash(DownwashBase):
    """TRUE downwash model — Gaussian ellipsoidal wake.

    Parameters loaded exclusively from ``configs/downwash_true.yaml``.
    The ``falloff_shape`` is always Gaussian for the TRUE model (the
    analytical ground truth we simulate against).

    References
    ----------
    Hönig et al., "Trajectory Planning for Quadrotor Swarms",
    IEEE Transactions on Robotics, 2018.

    Shi et al., "Neural-Swarm: Decentralized Close-Proximity Multirotor
    Control Using Learned Interactions", ICRA 2020.
    """

    def __init__(self, cfg: dict) -> None:
        """Construct from a config dict (typically loaded from
        ``configs/downwash_true.yaml``).
        """
        super().__init__(cfg)
        # TRUE model always uses Gaussian decay — no knob here
        self._falloff = "gaussian"

    @classmethod
    def from_yaml(cls, path: str) -> "TrueDownwash":
        """Load from a YAML file path and construct."""
        with open(path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        return cls(cfg)

    # ------------------------------------------------------------------
    # DownwashBase interface
    # ------------------------------------------------------------------

    def _compute_magnitude(
        self,
        r: float,
        rpm_upper: Optional[np.ndarray] = None,
    ) -> float:
        """Gaussian decay: magnitude = peak * exp(-0.5 * r²)."""
        return self.peak_force_N * math.exp(-0.5 * r * r)

    # ------------------------------------------------------------------
    # Multi-drone convenience: compute all downwash forces on every drone
    # ------------------------------------------------------------------

    def compute_all_forces(self, positions: np.ndarray) -> np.ndarray:
        """Compute the net TRUE downwash force on each drone.

        Parameters
        ----------
        positions : ndarray, shape (N, 3)
            Current world-frame positions of all N drones [m].

        Returns
        -------
        forces : ndarray, shape (N, 3)
            Net downwash force [N] on each drone from ALL other drones
            above it.
        """
        n = positions.shape[0]
        forces = np.zeros((n, 3))
        for victim in range(n):
            for source in range(n):
                if source == victim:
                    continue
                forces[victim] += self.force_on_lower(
                    pos_upper=positions[source],
                    pos_lower=positions[victim],
                )
        return forces
