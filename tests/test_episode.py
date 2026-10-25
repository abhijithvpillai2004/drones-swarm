"""Tests for the shared episode runner (seed handling and logged telemetry)."""
import numpy as np

from src.experiments.catalog import make_controller, make_scenario
from src.sim.episode import EpisodeConfig, run_episode


def _run(controller="naive", seed=None, duration=1.0):
    gen = make_scenario("hover_column_3d", 2, duration)
    return run_episode(gen, make_controller(controller, 2), cfg=EpisodeConfig(seed=seed))


def test_no_seed_is_deterministic():
    a, b = _run(), _run()
    assert np.allclose(a.pos, b.pos)


def test_same_seed_repeats_and_different_seeds_differ():
    a, b, c = _run(seed=3), _run(seed=3), _run(seed=4)
    assert np.allclose(a.pos, b.pos)
    assert not np.allclose(a.pos, c.pos)


def test_trace_shapes_and_metrics():
    r = _run("ua_adc", duration=1.0)
    assert r.pos.shape == (48, 2, 3)
    assert r.confidence.shape == (48, 2)
    for k in ("rmse_mean_m", "min_sep_m", "min_z_m", "ground_contact", "conf_rise_time_max_s"):
        assert k in r.metrics
    # Lower drone sits in the TRUE wake from the first step; the top drone does not.
    assert r.true_force[0, 1, 2] < 0.0 and r.true_force[0, 0, 2] == 0.0
    assert r.metrics["in_wake_drones"] == 1


def test_demo_csv_matches_trace(tmp_path):
    """The demo CSV must hold exactly the values the terminal HUD prints."""
    import pandas as pd
    from demo_gui import save_trace_csv
    r = _run("ua_adc", duration=1.0)
    path = tmp_path / "trace.csv"
    save_trace_csv(r, path)
    d = pd.read_csv(path)
    row = d[(d.step == 24) & (d.drone == 1)].iloc[0]
    assert row.z == r.pos[24, 1, 2]
    assert row.confidence == r.confidence[24, 1]
    assert row.blend == r.blend[24, 1]


def test_ua_adc_learns_wake_in_hover():
    r = _run("ua_adc", duration=2.0)
    assert r.metrics["conf_rise_time_max_s"] is not None
    assert r.metrics["conf_rise_time_max_s"] < 1.0
    assert not r.metrics["ground_contact"]
