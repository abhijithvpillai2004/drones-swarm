"""
src/noise/noise_wrapper.py
===========================
Optional sensor noise and command delay wrapper.

Disabled by default (``enabled=False``). Toggled via config:
  noise:
    enabled: true
    position_sigma_m: 0.005
    velocity_sigma_ms: 0.01
    attitude_sigma_rad: 0.002
    command_delay_steps: 2

This module wraps the state observations and RPM commands to
simulate real-world imperfections for the robustness study (Phase 9).

State noise model (Gaussian, independent per axis):
  - Position:  x + N(0, σ_pos²)
  - Velocity:  v + N(0, σ_vel²)
  - Attitude:  rpy + N(0, σ_att²)

Command delay: circular buffer of past commands; outputs the command
from ``delay`` steps ago.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Optional

import numpy as np


class NoiseWrapper:
    """Applies Gaussian noise to states and delay to commands.

    Parameters
    ----------
    enabled : bool
        When False, all methods are no-ops (zero overhead).
    pos_sigma : float  — position noise std [m]
    vel_sigma : float  — velocity noise std [m/s]
    att_sigma : float  — attitude noise std [rad]
    cmd_delay_steps : int — number of control steps to delay commands
    n_drones : int     — number of drones
    seed : int         — random seed
    """

    def __init__(
        self,
        enabled: bool = False,
        pos_sigma: float = 0.005,
        vel_sigma: float = 0.01,
        att_sigma: float = 0.002,
        cmd_delay_steps: int = 0,
        n_drones: int = 1,
        seed: int = 42,
    ) -> None:
        self.enabled = enabled
        self.pos_sigma = pos_sigma
        self.vel_sigma = vel_sigma
        self.att_sigma = att_sigma
        self.delay = max(0, cmd_delay_steps)
        self.n_drones = n_drones
        self._rng = np.random.RandomState(seed)

        # Command delay buffer (FIFO)
        # Pre-fill with zeros so the buffer is always full
        self._cmd_buffer: Deque[np.ndarray] = deque(
            [np.zeros((n_drones, 4))] * max(1, self.delay),
            maxlen=max(1, self.delay),
        )

    # ------------------------------------------------------------------
    # State noise
    # ------------------------------------------------------------------

    def add_state_noise(self, obs: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to the observation state vectors.

        Parameters
        ----------
        obs : ndarray (N, 20)
            Clean state observations from the physics engine.

        Returns
        -------
        noisy_obs : ndarray (N, 20)
            Noise-corrupted observations. Returns ``obs`` unchanged
            if ``enabled=False``.
        """
        if not self.enabled:
            return obs

        noisy = obs.copy()
        # Position indices 0:3
        noisy[:, 0:3] += self._rng.normal(0.0, self.pos_sigma, (self.n_drones, 3))
        # Velocity indices 10:13
        noisy[:, 10:13] += self._rng.normal(0.0, self.vel_sigma, (self.n_drones, 3))
        # Attitude (rpy) indices 7:10
        noisy[:, 7:10] += self._rng.normal(0.0, self.att_sigma, (self.n_drones, 3))
        return noisy

    # ------------------------------------------------------------------
    # Command delay
    # ------------------------------------------------------------------

    def delay_command(self, rpms: np.ndarray) -> np.ndarray:
        """Apply command delay to the RPM output.

        Inserts *rpms* into a FIFO buffer and returns the oldest command
        (``delay`` steps ago). When ``delay=0``, returns *rpms* immediately.

        Parameters
        ----------
        rpms : ndarray (N, 4)

        Returns
        -------
        delayed_rpms : ndarray (N, 4)
        """
        if not self.enabled or self.delay == 0:
            return rpms

        # leftmost entry is the oldest (to be output now)
        oldest = self._cmd_buffer[0].copy()
        self._cmd_buffer.append(rpms.copy())
        return oldest
