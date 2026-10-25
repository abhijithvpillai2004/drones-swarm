"""Tests for the committed pass criteria checker."""
import pandas as pd

from src.evaluation.criteria import apply_criteria, load_criteria


def _row(controller, seed=0, rmse=0.02, min_sep=0.4, min_z=0.5, in_wake=2, rise=0.2, nan=False):
    return {"scenario": "hover_column_3d", "N": 3, "seed": seed, "controller": controller,
            "rmse_mean_m": rmse, "min_sep_m": min_sep, "min_z_m": min_z, "has_nan": nan,
            "in_wake_drones": in_wake, "conf_rise_time_max_s": rise}


def test_criteria_file_has_the_four_agreed_rules():
    c = load_criteria()
    assert c["no_ground_contact"]["min_z_m"] == 0.05
    assert c["min_separation"]["min_sep_m"] == 0.12
    assert c["confidence_rise"]["level"] == 0.5 and c["confidence_rise"]["within_s"] == 1.0
    assert c["tracking_vs_baseline"]["baseline"] == "soft_avoid"


def test_good_runs_pass():
    df = pd.DataFrame([_row("soft_avoid", rmse=0.07), _row("ua_adc", rmse=0.02)])
    out = apply_criteria(df)
    assert out["passed"].all()
    assert (out["fail_reason"] == "OK").all()


def test_ground_contact_and_separation_fail():
    df = pd.DataFrame([_row("naive", min_z=0.01), _row("feedforward", min_sep=0.05)])
    out = apply_criteria(df).set_index("controller")
    assert out.loc["naive", "fail_reason"] == "no_ground_contact"
    assert out.loc["feedforward", "fail_reason"] == "min_separation"


def test_ua_adc_must_not_be_worse_than_soft_avoid():
    df = pd.DataFrame([_row("soft_avoid", rmse=0.05), _row("ua_adc", rmse=0.08)])
    out = apply_criteria(df).set_index("controller")
    assert not out.loc["ua_adc", "passed"]
    assert "tracking_vs_baseline" in out.loc["ua_adc", "fail_reason"]


def test_confidence_rise_only_applies_when_a_drone_starts_in_a_wake():
    df = pd.DataFrame([_row("soft_avoid"), _row("ua_adc", in_wake=0, rise=None)])
    out = apply_criteria(df).set_index("controller")
    assert pd.isna(out.loc["ua_adc", "crit_confidence_rise"])
    assert out.loc["ua_adc", "passed"]


def test_slow_or_missing_confidence_rise_fails():
    df = pd.DataFrame([_row("soft_avoid", seed=0), _row("ua_adc", seed=0, rise=None),
                       _row("soft_avoid", seed=1), _row("ua_adc", seed=1, rise=1.5)])
    out = apply_criteria(df)
    assert not out[out["controller"] == "ua_adc"]["passed"].any()
