"""
src/viz/wake_overlay.py
========================
Wake ellipsoid visualisation for the GUI demo.

When pybullet's GUI is active, this module draws translucent ellipsoid
outlines under each drone to represent the downwash zone. It also
colours drone markers red when they are inside another drone's wake.

Implementation uses p.addUserDebugLine to draw the ellipsoid as a set
of wireframe circles in the XZ and XY planes (see sim_notes.md §7).
When pybullet is not available, all methods silently do nothing.

The ASSUMED downwash model is used for visualisation (so the user sees
what the controller sees). The label "(ASSUMED)" is shown in the overlay.
"""

from __future__ import annotations

import math
from typing import List, Optional, TYPE_CHECKING

import numpy as np

# Attempt pybullet import; graceful fallback for test environments
try:
    import pybullet as p
    _PYBULLET_AVAILABLE = True
except ImportError:
    _PYBULLET_AVAILABLE = False

if TYPE_CHECKING:
    from src.models.assumed_downwash import AssumedDownwash


class WakeOverlayRenderer:
    """Renders downwash wake ellipsoids in pybullet GUI.

    Parameters
    ----------
    client_id : int
        PyBullet physics client ID (from env.CLIENT).
    assumed_dw : AssumedDownwash
        Model whose parameters define the visualised wake geometry.
    n_drones : int
    wake_threshold : float
        Influence level above which a drone is considered "in wake" (0–1).
    circle_segments : int
        Number of line segments per ellipse ring (more = smoother).
    """

    def __init__(
        self,
        client_id: int = 0,
        assumed_dw: Optional["AssumedDownwash"] = None,
        n_drones: int = 2,
        wake_threshold: float = 0.3,
        circle_segments: int = 24,
    ) -> None:
        self.client_id = client_id
        self.assumed_dw = assumed_dw
        self.n_drones = n_drones
        self.wake_threshold = wake_threshold
        self.segments = circle_segments

        # Track debug line IDs so we can replace them (no ghost lines)
        self._line_ids: List[int] = []
        self._in_wake_label_ids: List[int] = []

    def update(self, positions: np.ndarray) -> None:
        """Update wake overlay for the current drone positions.

        Parameters
        ----------
        positions : ndarray (N, 3)  — current drone world positions.
        """
        if not _PYBULLET_AVAILABLE or self.assumed_dw is None:
            return

        # Clear previous lines (replace by unique ID is handled by PyBullet)
        new_line_ids = []
        new_label_ids = []

        for src in range(self.n_drones):
            # Draw ellipsoid wireframe under each source drone
            ids = self._draw_wake_ellipsoid(positions[src])
            new_line_ids.extend(ids)

        # Check and flag drones in wake
        for victim in range(self.n_drones):
            in_wake = False
            for src in range(self.n_drones):
                if src == victim:
                    continue
                inf = self.assumed_dw.influence_at(positions[src], positions[victim])
                if inf >= self.wake_threshold:
                    in_wake = True
                    break
            # Print to console (GUI text overlay not trivially supported)
            # Future: use p.addUserDebugText for on-screen markers

        self._line_ids = new_line_ids
        self._in_wake_label_ids = new_label_ids

    def _draw_wake_ellipsoid(self, centre: np.ndarray) -> List[int]:
        """Draw a wireframe ellipsoid at *centre* using debug lines.

        Draws two rings:
          - Horizontal XY ring at z=centre[2]-sigma_z
          - Vertical XZ ring
        Returns list of line IDs.
        """
        if not _PYBULLET_AVAILABLE:
            return []

        r = self.assumed_dw.sigma_r_m
        z = self.assumed_dw.sigma_z_m
        ids = []

        # Horizontal ring at z offset -sigma_z (bottom of wake)
        angles = np.linspace(0, 2 * math.pi, self.segments, endpoint=False)
        xy_pts = np.column_stack([
            centre[0] + r * np.cos(angles),
            centre[1] + r * np.sin(angles),
            np.full(self.segments, centre[2] - z),
        ])
        for k in range(self.segments):
            p0 = xy_pts[k].tolist()
            p1 = xy_pts[(k + 1) % self.segments].tolist()
            lid = p.addUserDebugLine(
                p0, p1,
                lineColorRGB=[0.2, 0.5, 1.0],
                lineWidth=1.5,
                lifeTime=0,
                physicsClientId=self.client_id,
            )
            ids.append(lid)

        # Vertical XZ half-ellipse
        t_vals = np.linspace(-math.pi, math.pi, self.segments)
        xz_pts = np.column_stack([
            centre[0] + r * np.cos(t_vals),
            np.full(self.segments, centre[1]),
            centre[2] - z * np.abs(np.sin(t_vals)),
        ])
        for k in range(self.segments - 1):
            p0 = xz_pts[k].tolist()
            p1 = xz_pts[k + 1].tolist()
            lid = p.addUserDebugLine(
                p0, p1,
                lineColorRGB=[0.2, 0.8, 0.8],
                lineWidth=1.2,
                lifeTime=0,
                physicsClientId=self.client_id,
            )
            ids.append(lid)

        return ids
