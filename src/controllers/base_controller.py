"""
src/controllers/base_controller.py
====================================
Abstract base class for all formation controllers.

A controller receives:
  - ``obs``     : (N, 20) array of all drone states (position, velocity, etc.)
  - ``targets`` : (N, 3)  array of formation target positions
  - ``ctrl_timestep``: float, time per control step [s]

And returns:
  - ``rpms``    : (N, 4)  array of motor RPMs for each drone

All controllers ultimately delegate low-level position tracking to a
per-drone PID inner loop (:class:`PIDInner`).

The controller may hold a reference to an :class:`AssumedDownwash` model,
but must NEVER hold a reference to :class:`TrueDownwash`. This separation
is enforced by construction — see docs/decisions.md D-002.
"""

from __future__ import annotations

import abc
from typing import List, Optional

import numpy as np

from .pid_inner import PIDInner
from src.sim.mock_physics import CF2X_PARAMS


class BaseController(abc.ABC):
    """Abstract base for all N-drone formation controllers.

    Concrete sub-classes implement :meth:`_modify_targets`, which
    optionally adjusts the raw formation setpoints before the inner
    PID loop tracks them.

    Parameters
    ----------
    n_drones : int
        Number of drones in the formation.
    hover_rpm : float
        Hover RPM of the drone model (used to initialise PID state).
    max_rpm : float
        Maximum motor RPM.
    """

    def __init__(
        self,
        n_drones: int,
        hover_rpm: float = CF2X_PARAMS["hover_rpm"],
        max_rpm: float = CF2X_PARAMS["max_rpm"],
    ) -> None:
        self.n_drones = n_drones
        self.hover_rpm = hover_rpm
        self.max_rpm = max_rpm

        # One PID inner loop per drone
        self.pids: List[PIDInner] = [PIDInner() for _ in range(n_drones)]

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def compute(
        self,
        obs: np.ndarray,
        targets: np.ndarray,
        ctrl_timestep: float,
    ) -> np.ndarray:
        """Compute motor RPMs for all drones.

        Parameters
        ----------
        obs : ndarray, shape (N, 20)
            Full state of all drones.
        targets : ndarray, shape (N, 3)
            Formation target positions [m].
        ctrl_timestep : float
            Control step duration [s].

        Returns
        -------
        rpms : ndarray, shape (N, 4)
            Motor RPMs for each drone.
        """
        # Extract positions and velocities from state vector
        positions = obs[:, 0:3]
        velocities = obs[:, 10:13]
        quats = obs[:, 3:7]

        # Sub-class modifies targets (e.g. repulsion, offset)
        modified_targets = self._modify_targets(
            positions=positions,
            velocities=velocities,
            targets=targets.copy(),
            obs=obs,
        )

        # Per-drone PID
        rpms = np.zeros((self.n_drones, 4))
        for i in range(self.n_drones):
            rpms[i] = self.pids[i].compute(
                dt=ctrl_timestep,
                pos=positions[i],
                quat=quats[i],
                vel=velocities[i],
                target_pos=modified_targets[i],
                hover_rpm=self.hover_rpm,
                max_rpm=self.max_rpm,
            )
        return rpms

    def reset(self) -> None:
        """Reset all PID integrators."""
        for pid in self.pids:
            pid.reset()

    # ------------------------------------------------------------------
    # Sub-class hook
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def _modify_targets(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        targets: np.ndarray,
        obs: np.ndarray,
    ) -> np.ndarray:
        """Optionally modify formation targets before PID tracking.

        Parameters
        ----------
        positions : ndarray (N,3)  — current positions
        velocities : ndarray (N,3) — current velocities
        targets : ndarray (N,3)    — raw formation setpoints (modifiable copy)
        obs : ndarray (N,20)       — full state

        Returns
        -------
        modified_targets : ndarray (N,3)
        """
        ...
