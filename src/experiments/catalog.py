"""One place that names every scenario and controller.

Used by the demo, the strict regression, the repeat runs and the studies, so
"vertical_swap" or "ua_adc" means the same thing everywhere.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from src.controllers.feedforward import FeedforwardSwarmController
from src.controllers.naive import NaiveSwarmController
from src.controllers.soft_avoid import SoftAvoidSwarmController
from src.controllers.ua_adc import UAAdaptiveController, load_default_ua_adc_config
from src.formations.generators import (
    BaseTrajectoryGenerator,
    CircularOrbitFormation,
    CrossingTrajectory,
    GridFormation,
    HoverColumnFormation,
    LineFormation,
    StaggeredFormation,
    VerticalSwapTrajectory,
    VFormation,
    validate_drone_count,
)

# Scenario name -> (factory(n, duration), default N, default experiment duration [s])
SCENARIOS: Dict[str, tuple] = {
    "hover_column_3d": (lambda n, d: HoverColumnFormation(num_drones=n, z_top=1.8, dz=0.45, duration=d), 3, 5.0),
    "crossing_pair": (lambda n, d: CrossingTrajectory(num_drones=n, upper_alt=1.4, lower_alt=0.8,
                                                      travel_distance=1.6, duration=d), 2, 6.0),
    "vertical_swap": (lambda n, d: VerticalSwapTrajectory(num_drones=n, z_high=1.6, z_low=0.8, duration=d), 2, 6.0),
    "orbit_stack": (lambda n, d: CircularOrbitFormation(num_drones=n, radius=0.6, period=6.0, duration=d), 3, 6.0),
    "line": (lambda n, d: LineFormation(num_drones=n, spacing=0.45, altitude=1.2, duration=d), 4, 5.0),
    "v_formation": (lambda n, d: VFormation(num_drones=n, dx=0.35, dy=0.35, altitude=1.2, duration=d), 5, 5.0),
    "grid": (lambda n, d: GridFormation(num_drones=n, dx=0.45, dy=0.45, altitude=1.2, duration=d), 4, 5.0),
    "staggered": (lambda n, d: StaggeredFormation(num_drones=n, duration=d), 4, 5.0),
}

SCENARIO_NAMES = list(SCENARIOS.keys())
CONTROLLER_NAMES = ["naive", "soft_avoid", "feedforward", "ua_adc"]
CONTROLLER_LABELS = {
    "naive": "A  Naive",
    "soft_avoid": "B  Soft-Avoid",
    "feedforward": "C  Feedforward",
    "ua_adc": "D  UA-ADC",
}


def make_scenario(name: str, num_drones: Optional[int] = None,
                  duration: Optional[float] = None) -> BaseTrajectoryGenerator:
    """Build a scenario generator by name."""
    if name not in SCENARIOS:
        raise ValueError(f"Unknown scenario '{name}'. Choose from {SCENARIO_NAMES}")
    factory, n_default, d_default = SCENARIOS[name]
    n = validate_drone_count(num_drones if num_drones is not None else n_default)
    return factory(n, float(duration if duration is not None else d_default))


def make_controller(name: str, num_drones: int,
                    assumed_config: Optional[Dict[str, Any]] = None,
                    ua_config: Optional[Dict[str, Any]] = None) -> Any:
    """Build a controller by name.

    ``assumed_config`` is the ASSUMED wake (never the TRUE one); None loads
    ``configs/downwash_assumed.yaml``.  ``ua_config`` overrides
    ``configs/ua_adc.yaml`` for UA-ADC.
    """
    if name == "naive":
        return NaiveSwarmController(num_drones=num_drones)
    if name == "soft_avoid":
        return SoftAvoidSwarmController(num_drones=num_drones, assumed_config=assumed_config)
    if name == "feedforward":
        return FeedforwardSwarmController(num_drones=num_drones, assumed_config=assumed_config)
    if name == "ua_adc":
        return UAAdaptiveController(config=ua_config if ua_config is not None else load_default_ua_adc_config(),
                                    num_drones=num_drones, assumed_config=assumed_config)
    raise ValueError(f"Unknown controller '{name}'. Choose from {CONTROLLER_NAMES}")


CONTROLLER_FACTORIES: Dict[str, Callable[..., Any]] = {
    name: (lambda n, _name=name, **kw: make_controller(_name, n, **kw)) for name in CONTROLLER_NAMES
}
