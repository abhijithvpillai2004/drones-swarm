"""
src/viz/console_status.py
==========================
Console-based status printing for headless runs (no GUI).

Prints which drones are in each other's wake and their tracking error,
formatted as a coloured status line on stdout.
"""

from __future__ import annotations

import numpy as np
from typing import Optional


def print_wake_status(
    positions: np.ndarray,
    targets: np.ndarray,
    assumed_dw,
    step: int,
    ctrl_freq: float,
    wake_threshold: float = 0.3,
) -> None:
    """Print a one-line status summary to stdout.

    Parameters
    ----------
    positions : ndarray (N, 3)
    targets : ndarray (N, 3)
    assumed_dw : AssumedDownwash or None
    step : int — current control step
    ctrl_freq : float — for time display
    wake_threshold : float — influence fraction above which "in wake"
    """
    t = step / ctrl_freq
    N = positions.shape[0]
    errors = np.linalg.norm(positions - targets, axis=1)

    parts = [f"t={t:5.2f}s"]
    for j in range(N):
        in_wake = False
        if assumed_dw is not None:
            for src in range(N):
                if src == j:
                    continue
                if assumed_dw.influence_at(positions[src], positions[j]) >= wake_threshold:
                    in_wake = True
                    break
        flag = "⚠WAKE " if in_wake else "      "
        parts.append(f"D{j}:{flag}e={errors[j]:.3f}m")

    print("  ".join(parts), end="\r", flush=True)
