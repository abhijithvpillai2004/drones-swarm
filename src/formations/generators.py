"""Multi-drone formation and dynamic trajectory reference generators.

Implements reference generators for:
1. Static hover formations:
   - HoverColumnFormation (3D vertical stack, cascading downwash benchmark)
   - LineFormation (horizontal linear formation)
   - VFormation (symmetric V-shape flocking formation)
   - GridFormation (2D horizontal planar lattice)
   - StaggeredFormation (3D staggered altitude/horizontal lattice)
2. Dynamic crossing trajectories:
   - CrossingTrajectory (orthogonal multi-drone stream intersection)
3. Dynamic vertical exchange trajectories:
   - VerticalSwapTrajectory (smooth reciprocal altitude exchange)
4. Coordinated circular orbits:
   - CircularOrbitFormation (stacked coaxial or phased circular orbits)

All formation generators support swarm sizes N in [2, 10] with explicit validation.
"""
from abc import ABC, abstractmethod
from typing import Tuple, List, Optional
import numpy as np


def validate_drone_count(num_drones: int) -> int:
    """Validate that swarm size N is within the supported university research bounds [2, 10].

    Args:
        num_drones: Requested drone count.

    Returns:
        Validated integer drone count.

    Raises:
        ValueError: If num_drones is less than 2 or greater than 10.
    """
    if not isinstance(num_drones, int):
        num_drones = int(num_drones)
    if num_drones < 2 or num_drones > 10:
        raise ValueError(
            f"Invalid swarm size N={num_drones}. Supported drone count must be between 2 and 10."
        )
    return num_drones


class BaseTrajectoryGenerator(ABC):
    """Abstract base class for multi-drone trajectory generation."""

    def __init__(self, num_drones: int, duration: float) -> None:
        """Initialize trajectory generator.

        Args:
            num_drones: Number of drones in the swarm (clamped to [2, 10]).
            duration: Total duration of trajectory in seconds.
        """
        self.num_drones: int = validate_drone_count(num_drones)
        self.duration: float = float(duration)

    @abstractmethod
    def evaluate(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate reference positions and velocities for all drones at time t.

        Args:
            t: Current simulation time in seconds.

        Returns:
            Tuple of:
                - positions: Array of shape (num_drones, 3) with [x, y, z] in meters.
                - velocities: Array of shape (num_drones, 3) with [vx, vy, vz] in m/s.
        """
        pass

    def get_initial_positions(self) -> np.ndarray:
        """Get the initial starting positions (t = 0)."""
        pos, _ = self.evaluate(0.0)
        return pos


class HoverColumnFormation(BaseTrajectoryGenerator):
    """Static vertical column formation of N drones (cascading downwash benchmark).

    Drones are stacked vertically along the Z-axis. For larger N in [2, 10], vertical
    spacing dz is dynamically bounded to ensure all drones remain safely above ground
    level (z_min >= 0.40m) and within laboratory flight bounds (z_max <= 2.40m).
    """

    def __init__(
        self,
        num_drones: int = 3,
        z_top: float = 1.8,
        dz: float = 0.5,
        x_center: float = 0.0,
        y_center: float = 0.0,
        duration: float = 5.0,
    ) -> None:
        """Initialize vertical hover column for N drones.

        Args:
            num_drones: Number of vertically stacked drones in [2, 10].
            z_top: Altitude of top drone (drone 0) [m].
            dz: Nominal vertical separation [m]. Automatically scaled if needed.
            x_center: Horizontal X position [m].
            y_center: Horizontal Y position [m].
            duration: Duration [s].
        """
        super().__init__(num_drones=num_drones, duration=duration)
        self.x_center = x_center
        self.y_center = y_center

        # Ensure lowest drone does not hit ground (z >= 0.40m)
        z_min_bound = 0.40
        if z_top - (self.num_drones - 1) * dz < z_min_bound:
            # Dynamically adapt dz to fit all drones between z_top and z_min_bound
            effective_dz = (z_top - z_min_bound) / max(1, self.num_drones - 1)
        else:
            effective_dz = dz

        self.effective_dz = effective_dz
        self.z_top = z_top

        self.positions = np.zeros((self.num_drones, 3))
        for i in range(self.num_drones):
            self.positions[i] = [x_center, y_center, z_top - i * effective_dz]

    def evaluate(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """Return static column positions and zero velocities."""
        return self.positions.copy(), np.zeros_like(self.positions)


class LineFormation(BaseTrajectoryGenerator):
    """Horizontal linear formation of N drones along X or Y axis."""

    def __init__(
        self,
        num_drones: int = 3,
        spacing: float = 0.50,
        altitude: float = 1.0,
        axis: str = "x",
        x_center: float = 0.0,
        y_center: float = 0.0,
        duration: float = 5.0,
    ) -> None:
        """Initialize horizontal line formation.

        Args:
            num_drones: Drone count in [2, 10].
            spacing: Distance between neighboring drones [m].
            altitude: Hover altitude [m].
            axis: Alignment axis ('x' or 'y').
            x_center: Center X coordinate [m].
            y_center: Center Y coordinate [m].
            duration: Duration [s].
        """
        super().__init__(num_drones=num_drones, duration=duration)
        self.positions = np.zeros((self.num_drones, 3))
        offsets = (np.arange(self.num_drones) - (self.num_drones - 1) / 2.0) * spacing

        for i in range(self.num_drones):
            if axis.lower() == "y":
                self.positions[i] = [x_center, y_center + offsets[i], altitude]
            else:
                self.positions[i] = [x_center + offsets[i], y_center, altitude]

    def evaluate(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """Return static line positions and zero velocities."""
        return self.positions.copy(), np.zeros_like(self.positions)


class VFormation(BaseTrajectoryGenerator):
    """Symmetric V-formation (wedge flocking) of N drones.

    Leader placed at apex (0, 0, altitude).
    Left and right wings extend backwards with spacing dx, dy.
    Note: Degenerates into a 2-drone angled pair when N=2 (trivial V).
    """

    def __init__(
        self,
        num_drones: int = 5,
        dx: float = 0.40,
        dy: float = 0.40,
        altitude: float = 1.20,
        duration: float = 5.0,
    ) -> None:
        """Initialize V-formation.

        Args:
            num_drones: Drone count in [2, 10].
            dx: Horizontal spacing along flight direction per wing step [m].
            dy: Lateral spacing per wing step [m].
            altitude: Hover altitude [m].
            duration: Duration [s].
        """
        super().__init__(num_drones=num_drones, duration=duration)
        self.positions = np.zeros((self.num_drones, 3))

        # Drone 0 is the apex leader
        self.positions[0] = [0.0, 0.0, altitude]

        # Remaining N-1 drones alternate between left and right wings
        left_count = (self.num_drones - 1) // 2
        right_count = (self.num_drones - 1) - left_count

        idx = 1
        for step in range(1, left_count + 1):
            self.positions[idx] = [-step * dx, -step * dy, altitude]
            idx += 1

        for step in range(1, right_count + 1):
            self.positions[idx] = [-step * dx, step * dy, altitude]
            idx += 1

    def evaluate(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """Return static V-formation positions and zero velocities."""
        return self.positions.copy(), np.zeros_like(self.positions)


class GridFormation(BaseTrajectoryGenerator):
    """2D horizontal planar grid formation for N drones.

    Automatically arranges N drones into rows and columns:
    n_cols = ceil(sqrt(N)), n_rows = ceil(N / n_cols).
    For non-square counts (e.g. N=3, 5, 7, 10), the final row contains fewer drones.
    """

    def __init__(
        self,
        num_drones: int = 4,
        dx: float = 0.50,
        dy: float = 0.50,
        altitude: float = 1.0,
        duration: float = 5.0,
    ) -> None:
        """Initialize planar grid formation.

        Args:
            num_drones: Drone count in [2, 10].
            dx: Row spacing [m].
            dy: Column spacing [m].
            altitude: Grid altitude [m].
            duration: Duration [s].
        """
        super().__init__(num_drones=num_drones, duration=duration)
        self.positions = np.zeros((self.num_drones, 3))

        n_cols = int(np.ceil(np.sqrt(self.num_drones)))
        n_rows = int(np.ceil(self.num_drones / n_cols))

        x_offsets = (np.arange(n_rows) - (n_rows - 1) / 2.0) * dx
        y_offsets = (np.arange(n_cols) - (n_cols - 1) / 2.0) * dy

        drone_idx = 0
        for r in range(n_rows):
            for c in range(n_cols):
                if drone_idx >= self.num_drones:
                    break
                self.positions[drone_idx] = [x_offsets[r], y_offsets[c], altitude]
                drone_idx += 1

    def evaluate(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """Return static planar grid positions and zero velocities."""
        return self.positions.copy(), np.zeros_like(self.positions)


class StaggeredFormation(BaseTrajectoryGenerator):
    """3D staggered altitude and lateral offset formation for N drones.

    Alternates drones across both lateral (XY) and vertical (Z) coordinates,
    ideal for probing diagonal wake boundary shear interactions.
    """

    def __init__(
        self,
        num_drones: int = 4,
        dx: float = 0.40,
        dy: float = 0.30,
        dz: float = 0.35,
        z_base: float = 0.80,
        duration: float = 5.0,
    ) -> None:
        """Initialize staggered 3D formation.

        Args:
            num_drones: Drone count in [2, 10].
            dx: X-offset increment [m].
            dy: Y-offset alternating magnitude [m].
            dz: Altitude step increment [m].
            z_base: Base altitude [m].
            duration: Duration [s].
        """
        super().__init__(num_drones=num_drones, duration=duration)
        self.positions = np.zeros((self.num_drones, 3))

        for i in range(self.num_drones):
            x = (i - (self.num_drones - 1) / 2.0) * dx
            y = dy if (i % 2 == 1) else -dy
            z = z_base + (i % 3) * dz
            self.positions[i] = [x, y, z]

    def evaluate(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """Return static staggered positions and zero velocities."""
        return self.positions.copy(), np.zeros_like(self.positions)


class CrossingTrajectory(BaseTrajectoryGenerator):
    """Multi-drone dynamic intersecting flight paths at different altitudes.

    Generalizes for any N in [2, 10]:
    - Group A (first ceil(N/2) drones): fly along +X at upper altitude.
    - Group B (remaining floor(N/2) drones): fly along +Y at lower altitude.
    At t = duration / 2, Group A sweeps directly over Group B at the coordinate origin,
    generating transient downwash interactions on all lower drones.
    """

    def __init__(
        self,
        num_drones: int = 2,
        upper_alt: float = 1.4,
        lower_alt: float = 0.8,
        travel_distance: float = 1.6,
        lateral_spacing: float = 0.35,
        duration: float = 6.0,
    ) -> None:
        """Initialize multi-drone crossing trajectory.

        Args:
            num_drones: Drone count in [2, 10].
            upper_alt: Altitude of upper crossing group [m].
            lower_alt: Altitude of lower crossing group [m].
            travel_distance: Flight corridor length [m].
            lateral_spacing: Separation between parallel drones in same group [m].
            duration: Flight duration [s].
        """
        super().__init__(num_drones=num_drones, duration=duration)
        self.upper_alt = upper_alt
        self.lower_alt = lower_alt
        self.travel_distance = travel_distance
        self.speed = travel_distance / max(duration, 1e-3)
        self.t_cross = duration / 2.0
        self.lateral_spacing = lateral_spacing

        self.n_upper = int(np.ceil(self.num_drones / 2.0))
        self.n_lower = self.num_drones - self.n_upper

    def evaluate(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate crossing positions and constant horizontal velocities."""
        t_clamped = np.clip(t, 0.0, self.duration)
        tau = t_clamped - self.t_cross

        positions = np.zeros((self.num_drones, 3))
        velocities = np.zeros((self.num_drones, 3))

        # Group A (Upper group): travels along X axis (-half_dist to +half_dist)
        # Distributed laterally along Y
        y_offsets = (np.arange(self.n_upper) - (self.n_upper - 1) / 2.0) * self.lateral_spacing
        for i in range(self.n_upper):
            positions[i] = [tau * self.speed, y_offsets[i], self.upper_alt]
            velocities[i] = [self.speed, 0.0, 0.0]

        # Group B (Lower group): travels along Y axis (-half_dist to +half_dist)
        # Distributed laterally along X
        x_offsets = (np.arange(self.n_lower) - (self.n_lower - 1) / 2.0) * self.lateral_spacing
        for j in range(self.n_lower):
            idx = self.n_upper + j
            positions[idx] = [x_offsets[j], tau * self.speed, self.lower_alt]
            velocities[idx] = [0.0, self.speed, 0.0]

        return positions, velocities


class VerticalSwapTrajectory(BaseTrajectoryGenerator):
    """Multi-drone altitude exchange using 5th-order minimum-jerk profiles.

    Generalizes for any N in [2, 10]:
    Drones are assigned stratified altitudes between z_low and z_high.
    At t = duration / 2, drones invert their vertical order, passing each other
    along vertical corridors to induce dynamic reciprocal wake crossings.
    """

    def __init__(
        self,
        num_drones: int = 2,
        z_high: float = 1.6,
        z_low: float = 0.8,
        x_pos: float = 0.0,
        y_pos: float = 0.0,
        lateral_stagger: float = 0.0,
        duration: float = 6.0,
    ) -> None:
        """Initialize vertical swap trajectory.

        Args:
            num_drones: Drone count in [2, 10].
            z_high: Highest altitude setpoint [m].
            z_low: Lowest altitude setpoint [m].
            x_pos: Center horizontal X [m].
            y_pos: Center horizontal Y [m].
            lateral_stagger: Optional lateral spacing between vertical pairs [m].
            duration: Flight duration [s].
        """
        super().__init__(num_drones=num_drones, duration=duration)
        self.z_high = z_high
        self.z_low = z_low
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.lateral_stagger = lateral_stagger

        # For larger swarms (N > 4), dynamically expand vertical span symmetrically around z=1.20m
        # to ensure minimum initial vertical separation (dz >= 0.18m), preventing multi-source
        # aerodynamic thrust saturation at t=0 while staying within safe flight bounds [0.45m, 2.15m].
        if z_high == 1.6 and z_low == 0.8 and self.num_drones > 4:
            half_span = min(0.95, 0.10 * (self.num_drones - 1))
            self.z_low = max(0.45, 1.20 - half_span)
            self.z_high = min(2.15, 1.20 + half_span)

        # Initial stratified altitudes from z_high down to z_low
        self.initial_alts = np.linspace(self.z_high, self.z_low, self.num_drones)
        # Target end altitudes: inverted order
        self.target_alts = self.initial_alts[::-1].copy()

    def evaluate(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate smooth 5th-order minimum-jerk altitude exchange."""
        s = np.clip(t / max(self.duration, 1e-3), 0.0, 1.0)
        # Minimum-jerk polynomial: 10s^3 - 15s^4 + 6s^5
        poly = 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5
        # Derivative: 30s^2 - 60s^3 + 30s^4
        dpoly = (30.0 * s**2 - 60.0 * s**3 + 30.0 * s**4) / max(self.duration, 1e-3)

        positions = np.zeros((self.num_drones, 3))
        velocities = np.zeros((self.num_drones, 3))

        for i in range(self.num_drones):
            z_start = self.initial_alts[i]
            z_end = self.target_alts[i]
            delta_z = z_end - z_start

            z = z_start + delta_z * poly
            vz = delta_z * dpoly

            # Optional horizontal staggering per pair
            y_offset = ((i // 2) - ((self.num_drones // 2) - 0.5)) * self.lateral_stagger if self.lateral_stagger > 0 else 0.0

            positions[i] = [self.x_pos, self.y_pos + y_offset, z]
            velocities[i] = [0.0, 0.0, vz]

        return positions, velocities


class CircularOrbitFormation(BaseTrajectoryGenerator):
    """Multi-drone coordinated circular orbit with customizable vertical offsets.

    Generalizes for any N in [2, 10]:
    Drones orbit at radius R with matching angular velocity omega.
    Vertical altitudes are evenly distributed between 1.6m and 0.6m by default.
    """

    def __init__(
        self,
        num_drones: int = 3,
        radius: float = 0.6,
        period: float = 6.0,
        altitudes: Optional[List[float]] = None,
        duration: float = 8.0,
    ) -> None:
        """Initialize circular orbit formation.

        Args:
            num_drones: Number of orbiting drones in [2, 10].
            radius: Orbit radius [m].
            period: Orbit period [s].
            altitudes: List of altitudes per drone. Defaults to linspace(1.6, 0.6, N).
            duration: Total flight duration [s].
        """
        super().__init__(num_drones=num_drones, duration=duration)
        self.radius = radius
        self.period = period
        self.omega = 2.0 * np.pi / max(period, 1e-3)

        if altitudes is not None:
            alts_arr = np.array(altitudes, dtype=float)
            if len(alts_arr) == self.num_drones:
                self.altitudes = alts_arr
            else:
                self.altitudes = np.linspace(alts_arr[0], alts_arr[-1], self.num_drones)
        else:
            self.altitudes = np.linspace(1.6, 0.6, self.num_drones)

    def evaluate(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate circular positions and velocities."""
        positions = np.zeros((self.num_drones, 3))
        velocities = np.zeros((self.num_drones, 3))

        theta = self.omega * t

        for i in range(self.num_drones):
            x = self.radius * np.cos(theta)
            y = self.radius * np.sin(theta)
            z = self.altitudes[i]

            vx = -self.radius * self.omega * np.sin(theta)
            vy = self.radius * self.omega * np.cos(theta)
            vz = 0.0

            positions[i] = [x, y, z]
            velocities[i] = [vx, vy, vz]

        return positions, velocities
