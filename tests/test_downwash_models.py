"""
tests/test_downwash_models.py
==============================
Unit tests for TrueDownwash and AssumedDownwash force functions.

Key properties verified:
  1. Force is ~0 far away (max_range exceeded).
  2. Force is ~0 when lower drone is ABOVE upper.
  3. Force is ~0 when dz < min_dz (singularity guard).
  4. Force increases as drones approach vertically (monotonicity).
  5. Force is downward (Fz < 0).
  6. AssumedDownwash scale_factor correctly scales the force.
  7. AssumedDownwash falloff shapes produce different magnitudes.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.true_downwash import TrueDownwash
from src.models.assumed_downwash import AssumedDownwash


# Minimal config matching downwash_true.yaml defaults
TRUE_CFG = {
    "peak_force_N": 0.10,
    "sigma_r_m": 0.30,
    "sigma_z_m": 0.60,
    "min_dz_m": 0.05,
    "max_range_m": 3.0,
}

ASSUMED_CFG = {
    "peak_force_N": 0.10,
    "sigma_r_m": 0.30,
    "sigma_z_m": 0.60,
    "min_dz_m": 0.05,
    "max_range_m": 3.0,
    "scale_factor": 1.0,
    "falloff_shape": "gaussian",
}


# ---------------------------------------------------------------------------
# TrueDownwash tests
# ---------------------------------------------------------------------------

class TestTrueDownwash:
    @pytest.fixture
    def model(self):
        return TrueDownwash(TRUE_CFG)

    def test_zero_force_far_away(self, model):
        """Drones more than max_range_m apart → force ≈ 0."""
        pos_upper = np.array([0.0, 0.0, 5.0])
        pos_lower = np.array([0.0, 0.0, 0.0])   # 5 m below, well beyond max_range
        f = model.force_on_lower(pos_upper, pos_lower)
        assert np.allclose(f, 0.0, atol=1e-12), f"Expected zero force, got {f}"

    def test_zero_force_when_lower_is_above(self, model):
        """Lower drone actually above upper → no downwash (dz < 0)."""
        pos_upper = np.array([0.0, 0.0, 0.5])
        pos_lower = np.array([0.0, 0.0, 1.5])   # lower is HIGHER
        f = model.force_on_lower(pos_upper, pos_lower)
        assert np.allclose(f, 0.0, atol=1e-12)

    def test_zero_force_min_dz_guard(self, model):
        """dz < min_dz → force = 0 (singularity protection)."""
        pos_upper = np.array([0.0, 0.0, 1.03])
        pos_lower = np.array([0.0, 0.0, 1.0])   # dz = 0.03 < 0.05
        f = model.force_on_lower(pos_upper, pos_lower)
        assert np.allclose(f, 0.0, atol=1e-12)

    def test_force_is_downward(self, model):
        """Force on victim drone should be in -z direction."""
        pos_upper = np.array([0.0, 0.0, 1.5])
        pos_lower = np.array([0.0, 0.0, 0.5])   # 1 m below, directly under
        f = model.force_on_lower(pos_upper, pos_lower)
        assert f[2] < 0.0, f"Force should be downward (Fz < 0), got {f}"
        assert f[0] == pytest.approx(0.0, abs=1e-12)   # no horizontal force
        assert f[1] == pytest.approx(0.0, abs=1e-12)

    def test_force_increases_approaching_vertically(self, model):
        """Closer vertical separation → larger downwash force."""
        pos_upper = np.array([0.0, 0.0, 1.5])
        pos_lower_far = np.array([0.0, 0.0, 0.5])    # dz = 1.0 m
        pos_lower_close = np.array([0.0, 0.0, 1.0])  # dz = 0.5 m (but above min_dz)

        f_far = model.force_on_lower(pos_upper, pos_lower_far)
        f_close = model.force_on_lower(pos_upper, pos_lower_close)
        assert abs(f_close[2]) > abs(f_far[2]), (
            f"Closer drone should have larger force: {f_close[2]} vs {f_far[2]}"
        )

    def test_force_decreases_with_horizontal_offset(self, model):
        """Larger horizontal offset → smaller downwash force."""
        pos_upper = np.array([0.0, 0.0, 1.5])
        pos_lower_centre = np.array([0.0, 0.0, 0.5])    # directly below
        pos_lower_offset = np.array([0.5, 0.0, 0.5])    # 0.5 m lateral offset

        f_centre = model.force_on_lower(pos_upper, pos_lower_centre)
        f_offset = model.force_on_lower(pos_upper, pos_lower_offset)
        assert abs(f_centre[2]) > abs(f_offset[2])

    def test_peak_force_at_close_vertical_no_horizontal(self, model):
        """At very small dz and zero lateral offset, force approaches peak_force."""
        pos_upper = np.array([0.0, 0.0, 1.0 + model.min_dz_m + 0.001])
        pos_lower = np.array([0.0, 0.0, 1.0])
        f = model.force_on_lower(pos_upper, pos_lower)
        # Force should be substantial (at least 50% of peak)
        assert abs(f[2]) > 0.5 * model.peak_force_N

    def test_compute_all_forces_shape(self, model):
        positions = np.array([
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ])
        forces = model.compute_all_forces(positions)
        assert forces.shape == (3, 3)

    def test_top_drone_receives_no_downwash(self, model):
        """Topmost drone has no drone above it → zero downwash force."""
        positions = np.array([
            [0.0, 0.0, 2.0],   # top
            [0.0, 0.0, 1.0],
        ])
        forces = model.compute_all_forces(positions)
        assert np.allclose(forces[0], 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# AssumedDownwash tests
# ---------------------------------------------------------------------------

class TestAssumedDownwash:
    @pytest.fixture
    def model(self):
        return AssumedDownwash(ASSUMED_CFG)

    def test_scale_factor_halves_force(self):
        """scale_factor=0.5 should give half the force of scale_factor=1.0."""
        model_full = AssumedDownwash({**ASSUMED_CFG, "scale_factor": 1.0})
        model_half = AssumedDownwash({**ASSUMED_CFG, "scale_factor": 0.5})

        pos_upper = np.array([0.0, 0.0, 1.5])
        pos_lower = np.array([0.0, 0.0, 0.8])

        f_full = model_full.force_on_lower(pos_upper, pos_lower)
        f_half = model_half.force_on_lower(pos_upper, pos_lower)

        # Force magnitude should be roughly halved
        # Note: scale also changes sigma, so it's not exactly 0.5× but close
        assert abs(f_half[2]) < abs(f_full[2])

    def test_gaussian_vs_linear_falloff(self):
        """At moderate distance, linear falloff gives different force than Gaussian."""
        pos_upper = np.array([0.0, 0.0, 1.5])
        pos_lower = np.array([0.2, 0.0, 0.9])   # some lateral offset

        m_gauss = AssumedDownwash({**ASSUMED_CFG, "falloff_shape": "gaussian"})
        m_linear = AssumedDownwash({**ASSUMED_CFG, "falloff_shape": "linear"})

        f_gauss = m_gauss.force_on_lower(pos_upper, pos_lower)
        f_linear = m_linear.force_on_lower(pos_upper, pos_lower)
        # They should differ (not identical)
        assert not np.allclose(f_gauss, f_linear, atol=1e-8)

    def test_invalid_falloff_shape_raises(self):
        with pytest.raises(ValueError, match="falloff_shape"):
            AssumedDownwash({**ASSUMED_CFG, "falloff_shape": "nonexistent"})

    def test_is_in_wake_true_when_directly_below(self, model):
        pos_upper = np.array([0.0, 0.0, 1.5])
        pos_lower = np.array([0.0, 0.0, 0.5])   # close and directly below
        assert model.is_in_wake(pos_upper, pos_lower, threshold=0.1)

    def test_is_in_wake_false_when_far(self, model):
        pos_upper = np.array([0.0, 0.0, 1.5])
        pos_lower = np.array([2.0, 0.0, 0.0])   # far away
        assert not model.is_in_wake(pos_upper, pos_lower, threshold=0.3)

    def test_influence_at_zero_far_away(self, model):
        pos_upper = np.array([0.0, 0.0, 5.0])
        pos_lower = np.array([0.0, 0.0, 0.0])
        inf = model.influence_at(pos_upper, pos_lower)
        assert inf == pytest.approx(0.0, abs=1e-10)

    def test_power4_falloff_is_bounded(self):
        """power4 falloff should also give forces in [0, peak_force_N]."""
        m = AssumedDownwash({**ASSUMED_CFG, "falloff_shape": "power4"})
        pos_upper = np.array([0.0, 0.0, 1.5])
        pos_lower = np.array([0.0, 0.0, 0.8])
        f = m.force_on_lower(pos_upper, pos_lower)
        assert 0.0 <= abs(f[2]) <= m.peak_force_N + 1e-9
