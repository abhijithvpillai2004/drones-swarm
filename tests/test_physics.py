"""
tests/test_physics.py
======================
Unit tests for the MockPhysics simulator.

Tests:
  1. Hover at correct RPM → minimal drift.
  2. External force increases z-velocity downward.
  3. Multiple drones don't interfere (independent integration).
  4. State vector has correct length and layout.
  5. reset() restores initial state.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sim.mock_physics import MockPhysics, CF2X_PARAMS


class TestMockPhysics:
    @pytest.fixture
    def sim(self):
        """2-drone sim starting at [0,0,1] and [0.5,0,1]."""
        init_pos = np.array([[0.0, 0.0, 1.0], [0.5, 0.0, 1.0]])
        return MockPhysics(num_drones=2, initial_xyzs=init_pos, pyb_freq=240, ctrl_freq=48)

    def test_state_vector_length(self, sim):
        obs = sim.get_obs()
        assert obs.shape == (2, 20), f"Expected (2,20), got {obs.shape}"

    def test_hover_rpm_minimal_drift(self):
        """One drone at hover RPM → z should barely change over 1s."""
        sim = MockPhysics(
            num_drones=1,
            initial_xyzs=np.array([[0.0, 0.0, 1.0]]),
            pyb_freq=240, ctrl_freq=48
        )
        hover = CF2X_PARAMS["hover_rpm"]
        action = np.full((1, 4), hover)

        # Run for 1 second (48 ctrl steps)
        for _ in range(48):
            obs = sim.step(action)

        z_final = obs[0, 2]
        assert abs(z_final - 1.0) < 0.2, (
            f"Hover drift too large: {z_final - 1.0:.4f} m"
        )

    def test_zero_rpm_drone_falls(self):
        """Zero RPM → drone should fall (z decreases)."""
        sim = MockPhysics(
            num_drones=1,
            initial_xyzs=np.array([[0.0, 0.0, 2.0]]),
            pyb_freq=240, ctrl_freq=48,
        )
        action = np.zeros((1, 4))
        for _ in range(24):  # 0.5 s
            obs = sim.step(action)
        assert obs[0, 2] < 2.0, "Drone should fall with zero RPM"

    def test_external_downward_force(self):
        """Applying a strong downward force → drone z decreases faster."""
        hover = CF2X_PARAMS["hover_rpm"]
        sim_with = MockPhysics(
            num_drones=1,
            initial_xyzs=np.array([[0.0, 0.0, 1.0]]),
            pyb_freq=240, ctrl_freq=48,
        )
        sim_without = MockPhysics(
            num_drones=1,
            initial_xyzs=np.array([[0.0, 0.0, 1.0]]),
            pyb_freq=240, ctrl_freq=48,
        )
        action = np.full((1, 4), hover)

        for _ in range(10):
            sim_with.apply_external_force(0, np.array([0.0, 0.0, -0.5]))  # 0.5 N down
            sim_with.step(action)
            sim_without.step(action)

        obs_with = sim_with.get_obs()
        obs_without = sim_without.get_obs()
        assert obs_with[0, 2] < obs_without[0, 2], (
            "External downward force should lower the drone"
        )

    def test_two_drones_independent(self, sim):
        """Drones should move independently — different initial x."""
        hover = CF2X_PARAMS["hover_rpm"]
        action = np.full((2, 4), hover)
        for _ in range(10):
            obs = sim.step(action)
        # Drone 0 at x≈0, Drone 1 at x≈0.5 (positions don't swap)
        assert obs[0, 0] < 0.2
        assert abs(obs[1, 0] - 0.5) < 0.2

    def test_reset_restores_positions(self, sim):
        """After reset, positions should match the new initial positions."""
        hover = CF2X_PARAMS["hover_rpm"]
        action = np.full((2, 4), hover)
        for _ in range(20):
            sim.step(action)

        new_pos = np.array([[1.0, 1.0, 2.0], [2.0, 1.0, 2.0]])
        obs = sim.reset(initial_xyzs=new_pos)
        assert np.allclose(obs[0, 0:3], [1.0, 1.0, 2.0], atol=1e-10)
        assert np.allclose(obs[1, 0:3], [2.0, 1.0, 2.0], atol=1e-10)

    def test_freq_validation(self):
        """pyb_freq not divisible by ctrl_freq → ValueError."""
        with pytest.raises(ValueError):
            MockPhysics(num_drones=1, pyb_freq=240, ctrl_freq=7)

    def test_step_counter_increments(self, sim):
        action = np.full((2, 4), CF2X_PARAMS["hover_rpm"])
        for i in range(5):
            sim.step(action)
        assert sim.step_counter == 5
