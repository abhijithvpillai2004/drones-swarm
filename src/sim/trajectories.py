"""Trajectory reference generators for single-drone and multi-drone tracking."""
from typing import Tuple
import numpy as np


class CircleTrajectory:
    """Parametric circular trajectory in 3D space with analytical velocity feedforward."""

    def __init__(
        self,
        radius: float = 0.5,
        period: float = 6.0,
        altitude: float = 1.0,
        center: Tuple[float, float] = (0.0, 0.0),
    ) -> None:
        """Initialize circle trajectory.

        Args:
            radius: Radius of the circular orbit in meters.
            period: Duration of one full orbit in seconds.
            altitude: Height above ground in meters.
            center: (xc, yc) center coordinates in meters.
        """
        self.radius: float = radius
        self.period: float = period
        self.altitude: float = altitude
        self.center: Tuple[float, float] = center
        self.omega: float = 2.0 * np.pi / self.period

    def evaluate(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate reference position and velocity at time t.

        Args:
            t: Simulation time in seconds.

        Returns:
            Tuple of:
                - target_pos: Desired position [x, y, z] in meters.
                - target_vel: Desired linear velocity [vx, vy, vz] in m/s.
        """
        theta = self.omega * t
        x = self.center[0] + self.radius * np.cos(theta)
        y = self.center[1] + self.radius * np.sin(theta)
        z = self.altitude

        vx = -self.radius * self.omega * np.sin(theta)
        vy = self.radius * self.omega * np.cos(theta)
        vz = 0.0

        target_pos = np.array([x, y, z])
        target_vel = np.array([vx, vy, vz])
        return target_pos, target_vel


class WaypointTrajectory:
    """Piecewise-linear or step-based 3D waypoint trajectory."""

    def __init__(self, waypoints: np.ndarray, dwell_time: float = 2.0) -> None:
        """Initialize waypoint trajectory.

        Args:
            waypoints: Array of shape (M, 3) containing 3D waypoints.
            dwell_time: Time spent per waypoint in seconds.
        """
        self.waypoints: np.ndarray = np.array(waypoints, dtype=float)
        self.dwell_time: float = dwell_time
        self.num_waypoints: int = len(self.waypoints)

    def evaluate(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate reference position and zero velocity at time t.

        Args:
            t: Current time in seconds.

        Returns:
            Tuple of (target_pos, target_vel).
        """
        idx = min(int(t // self.dwell_time), self.num_waypoints - 1)
        target_pos = self.waypoints[idx].copy()
        target_vel = np.zeros(3)
        return target_pos, target_vel
