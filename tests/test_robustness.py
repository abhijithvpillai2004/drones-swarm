"""Unit tests for Phase 6 model robustness, parameter mismatches, and noise/delay handling."""
import pytest
import numpy as np
import yaml

from src.models.downwash import DownwashModel
from src.controllers.feedforward import FeedforwardSwarmController
from src.controllers.soft_avoid import SoftAvoidSwarmController
from src.formations.generators import HoverColumnFormation
from src.noise.wrappers import GaussianObservationNoise, CommandDelayBuffer
from src.experiments.robustness_sweep import run_robustness_trial


@pytest.fixture
def true_downwash() -> DownwashModel:
    """Fixture providing true downwash model."""
    with open("configs/downwash_true.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return DownwashModel(cfg)


@pytest.fixture
def assumed_config() -> dict:
    """Fixture providing nominal assumed downwash configuration."""
    with open("configs/downwash_assumed.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_feedforward_under_and_over_compensation(true_downwash, assumed_config):
    """Verify Feedforward response under 50% under-compensation and 150% over-compensation."""
    scenario = HoverColumnFormation(num_drones=2, z_top=1.5, dz=0.7, duration=2.0)
    nom_f = float(assumed_config["max_force_z"])

    # 1. Under-compensation (alpha = 0.5)
    cfg_under = assumed_config.copy()
    cfg_under["max_force_z"] = nom_f * 0.5
    rmse_under = run_robustness_trial(scenario, FeedforwardSwarmController, cfg_under, true_downwash)

    # 2. Perfect nominal model (alpha = 1.0)
    rmse_nominal = run_robustness_trial(scenario, FeedforwardSwarmController, assumed_config, true_downwash)

    # 3. Over-compensation (alpha = 1.5)
    cfg_over = assumed_config.copy()
    cfg_over["max_force_z"] = nom_f * 1.5
    rmse_over = run_robustness_trial(scenario, FeedforwardSwarmController, cfg_over, true_downwash)

    # Nominal model must achieve lower tracking error than both under- and over-compensation
    assert rmse_nominal < rmse_under, f"Nominal ({rmse_nominal:.4f}) not better than under-comp ({rmse_under:.4f})"
    assert rmse_nominal < rmse_over, f"Nominal ({rmse_nominal:.4f}) not better than over-comp ({rmse_over:.4f})"


def test_soft_avoid_bounded_under_radius_mismatch(true_downwash, assumed_config):
    """Verify Soft-Avoid remains bounded and stable even when wake radius is overestimated by 50%."""
    scenario = HoverColumnFormation(num_drones=2, z_top=1.5, dz=0.7, duration=2.0)
    nom_r = float(assumed_config["radius_xy"])

    cfg_wide = assumed_config.copy()
    cfg_wide["radius_xy"] = nom_r * 1.5

    rmse_wide = run_robustness_trial(
        scenario,
        SoftAvoidSwarmController,
        cfg_wide,
        true_downwash,
        ctrl_kwargs={"config": {"max_displacement": 0.20, "max_velocity": 0.15}},
    )

    # Must stay within safe bounds without diverging (< 0.25m error)
    assert rmse_wide < 0.25, f"Soft avoid error diverged under wide radius: {rmse_wide:.4f}m"


def test_controller_under_noise_and_latency(true_downwash, assumed_config):
    """Verify controllers handle active observation noise and 2-step command latency."""
    scenario = HoverColumnFormation(num_drones=2, z_top=1.5, dz=0.7, duration=2.0)
    noise = GaussianObservationNoise(config={"enabled": True, "pos_std": 0.005, "vel_std": 0.01}, seed=123)
    delay = CommandDelayBuffer(delay_steps=2, enabled=True)

    rmse = run_robustness_trial(
        scenario,
        FeedforwardSwarmController,
        assumed_config,
        true_downwash,
        noise_wrapper=noise,
        delay_buffer=delay,
    )

    # Tracking error should remain bounded (< 5 cm)
    assert np.isfinite(rmse)
    assert rmse < 0.05, f"Error excessively high under noise & delay: {rmse:.4f}m"
