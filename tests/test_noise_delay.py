"""Unit tests for sensor noise model and command delay buffer."""
import numpy as np
import pytest

from src.noise.wrappers import GaussianObservationNoise, CommandDelayBuffer


def test_observation_noise_disabled() -> None:
    """Verify that disabled noise acts as exact identity passthrough."""
    noise = GaussianObservationNoise(config={"enabled": False})
    state = np.ones(20)
    noisy = noise.apply(state)
    assert np.array_equal(state, noisy)


def test_observation_noise_statistics() -> None:
    """Verify Gaussian noise statistics (zero-mean and matching standard deviations)."""
    std_pos = 0.02
    std_vel = 0.05
    noise = GaussianObservationNoise(
        config={"enabled": True, "pos_std": std_pos, "vel_std": std_vel, "rpy_std": 0.01},
        seed=123,
    )

    N_samples = 10000
    state = np.zeros(20)
    pos_samples = []
    vel_samples = []

    for _ in range(N_samples):
        noisy = noise.apply(state)
        pos_samples.append(noisy[0:3])
        vel_samples.append(noisy[10:13])

    pos_samples = np.array(pos_samples)
    vel_samples = np.array(vel_samples)

    # Check zero-mean (< 3 std errors)
    assert np.all(np.abs(np.mean(pos_samples, axis=0)) < 0.003)
    # Check sample standard deviation matches requested std
    assert np.all(np.isclose(np.std(pos_samples, axis=0), std_pos, rtol=0.10))
    assert np.all(np.isclose(np.std(vel_samples, axis=0), std_vel, rtol=0.10))


def test_command_delay_disabled() -> None:
    """Verify that disabled delay outputs commands immediately."""
    delay = CommandDelayBuffer(delay_steps=3, enabled=False)
    cmd = np.array([1000.0, 1000.0, 1000.0, 1000.0])
    out = delay.step(cmd)
    assert np.array_equal(cmd, out)


def test_command_delay_steps() -> None:
    """Verify that enabled delay holds commands by exact number of steps."""
    delay = CommandDelayBuffer(delay_steps=2, enabled=True)

    c0 = np.array([0.0, 0.0])
    c1 = np.array([1.0, 1.0])
    c2 = np.array([2.0, 2.0])
    c3 = np.array([3.0, 3.0])

    # Step 0: buffer has [c0] -> output is c0
    out0 = delay.step(c0)
    assert np.array_equal(out0, c0)

    # Step 1: buffer has [c0, c1] -> output is oldest c0
    out1 = delay.step(c1)
    assert np.array_equal(out1, c0)

    # Step 2: buffer has [c0, c1, c2] -> output is c0
    out2 = delay.step(c2)
    assert np.array_equal(out2, c0)

    # Step 3: buffer pops oldest -> output is c1 (delayed by 2 steps from c3)
    out3 = delay.step(c3)
    assert np.array_equal(out3, c1)
