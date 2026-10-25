"""Unit tests for swarm size scaling, multi-source downwash, and scaling experiment config."""
import os
import pytest
import yaml
import numpy as np

from src.formations.generators import validate_drone_count, HoverColumnFormation
from src.models.downwash import DownwashModel
from src.controllers.feedforward import FeedforwardSwarmController
from src.experiments.scaling_sweep import run_scaling_trial


def test_scaling_config_parsing():
    """Verify configs/scenarios/scaling_column.yaml exists and contains valid parameters."""
    cfg_path = "configs/scenarios/scaling_column.yaml"
    assert os.path.exists(cfg_path), f"Missing scaling scenario config: {cfg_path}"

    with open(cfg_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    assert data["name"] == "scaling_column"
    assert "drone_counts_swept" in data
    assert isinstance(data["drone_counts_swept"], list)
    for n in data["drone_counts_swept"]:
        assert 2 <= n <= 10, f"Drone count {n} out of bounds [2, 10]"
    assert data["seeds_count"] >= 20


def test_drone_count_bounds_validation():
    """Verify validate_drone_count strictly enforces [2, 10] range."""
    for valid_n in [2, 3, 5, 7, 10]:
        assert validate_drone_count(valid_n) == valid_n

    for invalid_n in [-5, 0, 1, 11, 15, 100]:
        with pytest.raises(ValueError):
            validate_drone_count(invalid_n)


def test_scaling_sweep_single_trial(tmp_path):
    """Integration test: execute a fast 0.5s trial with N=5 and verify runtime and metrics."""
    with open("configs/downwash_true.yaml", "r", encoding="utf-8") as f:
        true_cfg = yaml.safe_load(f)
    true_model = DownwashModel(true_cfg)

    record = run_scaling_trial(
        num_drones=5,
        controller_name="Controller_C_Feedforward",
        controller_class=FeedforwardSwarmController,
        ctrl_kwargs={},
        true_downwash_model=true_model,
        seed=42,
        duration=0.5,
        ctrl_freq_hz=48,
        output_dir=str(tmp_path),
    )

    assert record["num_drones"] == 5
    assert record["controller"] == "Controller_C_Feedforward"
    assert record["seed"] == 42
    assert "rmse_mean" in record
    assert "min_interdrone_dist" in record
    assert "near_collision_steps" in record
    assert "sim_runtime_sec" in record
    assert record["sim_runtime_sec"] > 0.0
