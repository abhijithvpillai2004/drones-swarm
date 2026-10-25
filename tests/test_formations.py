"""Unit and integration tests for Phase 5 formation generators and multi-drone benchmarks."""
import os
import pytest
import numpy as np
import yaml

from src.formations.generators import (
    HoverColumnFormation,
    CrossingTrajectory,
    VerticalSwapTrajectory,
    CircularOrbitFormation,
)
from src.models.downwash import DownwashModel
from src.controllers.naive import NaiveSwarmController
from src.controllers.soft_avoid import SoftAvoidSwarmController
from src.controllers.feedforward import FeedforwardSwarmController
from src.experiments.multi_drone_comparison import run_single_trial


def test_hover_column_formation():
    """Verify HoverColumnFormation outputs vertically stacked positions and zero velocities."""
    gen = HoverColumnFormation(num_drones=3, z_top=1.8, dz=0.6, duration=5.0)
    init_pos = gen.get_initial_positions()
    assert init_pos.shape == (3, 3)
    np.testing.assert_allclose(init_pos[0], [0.0, 0.0, 1.8])
    np.testing.assert_allclose(init_pos[1], [0.0, 0.0, 1.2])
    np.testing.assert_allclose(init_pos[2], [0.0, 0.0, 0.6])

    p, v = gen.evaluate(2.5)
    np.testing.assert_allclose(p, init_pos)
    np.testing.assert_allclose(v, np.zeros((3, 3)))


def test_crossing_trajectory():
    """Verify CrossingTrajectory evaluates correct start, mid-crossing, and velocity vectors."""
    gen = CrossingTrajectory(upper_alt=1.4, lower_alt=0.8, travel_distance=1.6, duration=4.0)
    # Mid-flight is crossing point (t = 2.0s)
    p_cross, v_cross = gen.evaluate(2.0)

    # At crossing: both should be at x=0, y=0 horizontally
    assert abs(p_cross[0, 0]) < 1e-4
    assert abs(p_cross[0, 1]) < 1e-4
    assert abs(p_cross[1, 0]) < 1e-4
    assert abs(p_cross[1, 1]) < 1e-4

    # Altitude difference should be maintained
    assert p_cross[0, 2] == pytest.approx(1.4)
    assert p_cross[1, 2] == pytest.approx(0.8)

    # Velocities: Drone 0 along X, Drone 1 along Y
    assert v_cross[0, 0] > 0.0 and v_cross[0, 1] == 0.0
    assert v_cross[1, 0] == 0.0 and v_cross[1, 1] > 0.0


def test_vertical_swap_trajectory():
    """Verify VerticalSwapTrajectory endpoints and smooth continuous altitude transition."""
    gen = VerticalSwapTrajectory(z_high=1.6, z_low=0.8, duration=4.0)
    p_start, v_start = gen.evaluate(0.0)
    p_mid, v_mid = gen.evaluate(2.0)
    p_end, v_end = gen.evaluate(4.0)

    # Start: Drone 0 high, Drone 1 low; zero start velocity
    np.testing.assert_allclose(p_start[0], [0.0, 0.0, 1.6])
    np.testing.assert_allclose(p_start[1], [0.0, 0.0, 0.8])
    np.testing.assert_allclose(v_start, np.zeros((2, 3)), atol=1e-5)

    # Midpoint: both at mid altitude (1.2m)
    assert p_mid[0, 2] == pytest.approx(1.2, abs=1e-3)
    assert p_mid[1, 2] == pytest.approx(1.2, abs=1e-3)

    # End: Drone 0 low, Drone 1 high; zero final velocity
    np.testing.assert_allclose(p_end[0], [0.0, 0.0, 0.8])
    np.testing.assert_allclose(p_end[1], [0.0, 0.0, 1.6])
    np.testing.assert_allclose(v_end, np.zeros((2, 3)), atol=1e-5)


def test_circular_orbit_formation():
    """Verify CircularOrbitFormation geometry, periodicity, and velocity vectors."""
    radius = 0.6
    period = 6.0
    gen = CircularOrbitFormation(num_drones=3, radius=radius, period=period, duration=6.0)

    p0, v0 = gen.evaluate(0.0)
    # At t=0, all drones at x = radius, y = 0
    for d in range(3):
        assert p0[d, 0] == pytest.approx(radius, abs=1e-4)
        assert p0[d, 1] == pytest.approx(0.0, abs=1e-4)
        # Velocity should be purely tangential along Y
        assert v0[d, 0] == pytest.approx(0.0, abs=1e-4)
        assert v0[d, 1] > 0.0

    # Periodicity check: p(period) == p(0)
    p_end, _ = gen.evaluate(period)
    np.testing.assert_allclose(p_end, p0, atol=1e-4)


def test_scenario_yaml_configs_exist_and_valid():
    """Verify all 4 scenario YAML configuration files exist and have required fields."""
    scenarios = ["hover_column_3d", "crossing_pair", "vertical_swap", "orbit_stack"]
    for sc in scenarios:
        yaml_path = os.path.join("configs", "scenarios", f"{sc}.yaml")
        assert os.path.exists(yaml_path), f"Missing scenario config: {yaml_path}"
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert "name" in data
        assert "drone_count" in data
        assert "initial_positions" in data
        assert len(data["initial_positions"]) == data["drone_count"]


def test_multi_drone_trial_execution(tmp_path):
    """Integration test: execute a short 1-second 3-drone run and verify outputs."""
    with open("configs/downwash_true.yaml", "r", encoding="utf-8") as f:
        true_cfg = yaml.safe_load(f)
    true_dw = DownwashModel(true_cfg)

    gen = HoverColumnFormation(num_drones=3, z_top=1.8, dz=0.6, duration=1.0)
    trial = run_single_trial(
        scenario_name="test_col",
        scenario_gen=gen,
        controller_name="Feedforward",
        controller_class=FeedforwardSwarmController,
        ctrl_kwargs={},
        true_downwash_model=true_dw,
        output_dir=str(tmp_path),
        gui=False,
    )

    assert "summary" in trial
    assert "actual_positions" in trial
    assert trial["actual_positions"].shape == (48, 3, 3)
    assert trial["summary"]["rmse_mean"] < 0.02


def test_drone_count_validation():
    """Verify that drone count N is strictly clamped/validated in [2, 10]."""
    from src.formations.generators import validate_drone_count

    assert validate_drone_count(2) == 2
    assert validate_drone_count(10) == 10
    assert validate_drone_count(5) == 5

    with pytest.raises(ValueError):
        validate_drone_count(1)

    with pytest.raises(ValueError):
        validate_drone_count(0)

    with pytest.raises(ValueError):
        validate_drone_count(11)

    with pytest.raises(ValueError):
        validate_drone_count(100)


def test_new_formation_generators_scaling():
    """Verify Line, V, Grid, and Staggered formations produce valid layouts across N in [2, 10]."""
    from src.formations.generators import LineFormation, VFormation, GridFormation, StaggeredFormation

    for n in [2, 3, 5, 7, 10]:
        # 1. Line
        line = LineFormation(num_drones=n, spacing=0.4, altitude=1.0)
        p, v = line.evaluate(0.0)
        assert p.shape == (n, 3)
        assert v.shape == (n, 3)
        assert np.all(p[:, 2] == 1.0)

        # 2. V Formation
        v_form = VFormation(num_drones=n, altitude=1.2)
        p_v, _ = v_form.evaluate(0.0)
        assert p_v.shape == (n, 3)
        assert np.all(p_v[:, 2] == 1.2)

        # 3. Grid Formation
        grid = GridFormation(num_drones=n, altitude=1.0)
        p_g, _ = grid.evaluate(0.0)
        assert p_g.shape == (n, 3)
        # Check all drone positions in grid are unique
        unique_pos = np.unique(np.round(p_g, 3), axis=0)
        assert len(unique_pos) == n

        # 4. Staggered Formation
        stag = StaggeredFormation(num_drones=n)
        p_s, _ = stag.evaluate(0.0)
        assert p_s.shape == (n, 3)


def test_existing_generators_generalize_to_arbitrary_n():
    """Verify HoverColumn, Crossing, VerticalSwap, and CircularOrbit for N=2, 5, 10."""
    for n in [2, 5, 10]:
        col = HoverColumnFormation(num_drones=n, z_top=1.8, dz=0.4)
        p_col, _ = col.evaluate(1.0)
        assert p_col.shape == (n, 3)
        assert np.all(p_col[:, 2] >= 0.39)  # Above ground bound

        cross = CrossingTrajectory(num_drones=n, duration=4.0)
        p_cross, v_cross = cross.evaluate(2.0)
        assert p_cross.shape == (n, 3)
        assert v_cross.shape == (n, 3)

        swap = VerticalSwapTrajectory(num_drones=n, duration=4.0)
        p_swap, v_swap = swap.evaluate(2.0)
        assert p_swap.shape == (n, 3)
        assert v_swap.shape == (n, 3)

        orbit = CircularOrbitFormation(num_drones=n, duration=4.0)
        p_orb, v_orb = orbit.evaluate(1.0)
        assert p_orb.shape == (n, 3)
        assert v_orb.shape == (n, 3)

