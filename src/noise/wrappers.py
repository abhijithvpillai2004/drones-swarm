"""Sensor noise and command delay wrappers for robust control evaluation.

Provides:
1. GaussianObservationNoise: Inject zero-mean Gaussian noise on state estimates.
2. CommandDelayBuffer: FIFO buffer simulating transmission/processing delay.
"""
from typing import Dict, Any, Optional
from collections import deque
import numpy as np


class GaussianObservationNoise:
    """Additive zero-mean Gaussian noise model for drone state observations."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize observation noise model.

        Args:
            config: Optional config dict. Looks for keys:
                - enabled: bool (default False)
                - pos_std: float standard deviation for position [m] (default 0.005)
                - vel_std: float standard deviation for linear velocity [m/s] (default 0.01)
                - rpy_std: float standard deviation for Euler angles [rad] (default 0.005)
            seed: Optional integer random seed for reproducibility.
        """
        cfg = config or {}
        self.enabled: bool = bool(cfg.get("enabled", False))
        self.pos_std: float = float(cfg.get("pos_std", 0.005))
        self.vel_std: float = float(cfg.get("vel_std", 0.01))
        self.rpy_std: float = float(cfg.get("rpy_std", 0.005))
        self.rng: np.random.Generator = np.random.default_rng(seed)

    def set_seed(self, seed: int) -> None:
        """Re-seed the noise generator."""
        self.rng = np.random.default_rng(seed)

    def apply(self, state: np.ndarray) -> np.ndarray:
        """Apply noise to a single drone 20-element state vector or (N, 20) batch.

        State indices:
            0:3   - Position [x, y, z]
            3:7   - Quaternion [qx, qy, qz, qw]
            7:10  - Euler [roll, pitch, yaw]
            10:13 - Velocity [vx, vy, vz]
            13:16 - Angular velocity
            16:20 - Motor RPMs

        Args:
            state: Array of shape (20,) or (N, 20).

        Returns:
            noisy_state: State vector with perturbed position, velocity, and orientation.
        """
        if not self.enabled:
            return state.copy()

        noisy_state = state.copy()
        if noisy_state.ndim == 1:
            # Single drone (20,)
            noisy_state[0:3] += self.rng.normal(0.0, self.pos_std, size=3)
            noisy_state[7:10] += self.rng.normal(0.0, self.rpy_std, size=3)
            noisy_state[10:13] += self.rng.normal(0.0, self.vel_std, size=3)
        elif noisy_state.ndim == 2:
            # Multi-drone (N, 20)
            N = noisy_state.shape[0]
            noisy_state[:, 0:3] += self.rng.normal(0.0, self.pos_std, size=(N, 3))
            noisy_state[:, 7:10] += self.rng.normal(0.0, self.rpy_std, size=(N, 3))
            noisy_state[:, 10:13] += self.rng.normal(0.0, self.vel_std, size=(N, 3))

        return noisy_state


class CommandDelayBuffer:
    """FIFO queue that delays commanded control actions by a fixed number of steps."""

    def __init__(
        self,
        delay_steps: int = 0,
        enabled: bool = False,
    ) -> None:
        """Initialize command delay buffer.

        Args:
            delay_steps: Number of control cycles to delay.
            enabled: If False, delay is bypassed (passthrough).
        """
        self.delay_steps: int = max(0, int(delay_steps))
        self.enabled: bool = bool(enabled) and (self.delay_steps > 0)
        self.buffer: deque = deque()

    def reset(self) -> None:
        """Clear the delay buffer."""
        self.buffer.clear()

    def step(self, command: np.ndarray) -> np.ndarray:
        """Push new command into buffer and return delayed command.

        Args:
            command: Command array of any shape.

        Returns:
            delayed_command: Command from delay_steps ago.
        """
        if not self.enabled:
            return command.copy()

        self.buffer.append(command.copy())

        # If buffer hasn't filled to delay_steps yet, repeat the oldest available command
        if len(self.buffer) <= self.delay_steps:
            return self.buffer[0].copy()

        return self.buffer.popleft()
