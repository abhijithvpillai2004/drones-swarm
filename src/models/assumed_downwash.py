"""
src/models/assumed_downwash.py
===============================
ASSUMED downwash model — used by controllers B (soft avoidance)
and C (feedforward compensation).

This model may differ from the TRUE model to simulate model mismatch.
Parameters come exclusively from ``configs/downwash_assumed.yaml``.

Controllers that import this module do NOT have access to the TRUE model
parameters, ensuring a clean separation (docs/decisions.md D-002).

Falloff shapes supported:
- gaussian : exp(-0.5 * r²)          [default, matches true model]
- linear   : max(0, 1 - r)           [abrupt linear cutoff]
- power4   : exp(-0.5 * r⁴)          [steeper decay near centre]

References
----------
Hönig et al., "Trajectory Planning for Quadrotor Swarms",
IEEE Transactions on Robotics, 2018.

Shi et al., "Neural-Swarm: Decentralized Close-Proximity Multirotor
Control Using Learned Interactions", ICRA 2020.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import yaml

from .downwash_base import DownwashBase


class AssumedDownwash(DownwashBase):
    """ASSUMED downwash model loaded from ``configs/downwash_assumed.yaml``.

    Supports three falloff shapes and a global scale_factor for
    robustness-study parameter sweeps.
    """

    FALLOFF_SHAPES = ("gaussian", "linear", "power4")

    def __init__(self, cfg: dict) -> None:
        """Construct from config dict.

        The ``scale_factor`` key (default 1.0) multiplies peak_force_N,
        sigma_r_m, and sigma_z_m simultaneously — representing a uniform
        over/under-estimation of wake strength and extent.
        """
        # Apply scale factor before passing to base class
        scale = float(cfg.get("scale_factor", 1.0))
        scaled_cfg = dict(cfg)
        scaled_cfg["peak_force_N"] = float(cfg["peak_force_N"]) * scale
        scaled_cfg["sigma_r_m"] = float(cfg["sigma_r_m"]) * scale
        scaled_cfg["sigma_z_m"] = float(cfg["sigma_z_m"]) * scale
        super().__init__(scaled_cfg)

        self._falloff: str = str(cfg.get("falloff_shape", "gaussian"))
        if self._falloff not in self.FALLOFF_SHAPES:
            raise ValueError(
                f"Unknown falloff_shape '{self._falloff}'. "
                f"Choose from {self.FALLOFF_SHAPES}"
            )

    @classmethod
    def from_yaml(cls, path: str) -> "AssumedDownwash":
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
        """Return force magnitude using the selected falloff shape."""
        if self._falloff == "gaussian":
            return self.peak_force_N * math.exp(-0.5 * r * r)
        elif self._falloff == "linear":
            return self.peak_force_N * max(0.0, 1.0 - r)
        elif self._falloff == "power4":
            return self.peak_force_N * math.exp(-0.5 * r ** 4)
        else:  # pragma: no cover — validated in __init__
            return 0.0

    # ------------------------------------------------------------------
    # Convenience: compute assumed forces for controller use
    # ------------------------------------------------------------------

    def compute_all_forces(self, positions: np.ndarray) -> np.ndarray:
        """Compute the net ASSUMED downwash force on each drone.

        Parameters
        ----------
        positions : ndarray, shape (N, 3)
            Current world-frame positions of all N drones [m].

        Returns
        -------
        forces : ndarray, shape (N, 3)
            Net assumed downwash force [N] per drone.
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

    def is_in_wake(self, pos_upper: np.ndarray, pos_lower: np.ndarray,
                   threshold: float = 0.5) -> bool:
        """Return True if lower drone is in the assumed wake.

        ``threshold`` is the normalised influence level (0–1) above which
        the drone is considered "in the wake" (default 50%).
        """
        return self.influence_at(pos_upper, pos_lower) >= threshold
