"""
src/sim/sim_runner.py
======================
Main simulation orchestrator: wires together physics, controllers,
TRUE downwash injection, noise, and logging.

This is the single entry-point for running a complete episode.
It operates in "mock" mode (pure Python) by default, and can be
switched to "pybullet" mode (aviary_wrapper) when pybullet is available.

Design:
  1. sim_runner loads its OWN copy of downwash_true.yaml → injects forces.
  2. Controllers receive only downwash_assumed.yaml (via the controller
     constructor). They never interact with the true model directly.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

from src.models.true_downwash import TrueDownwash
from src.noise.noise_wrapper import NoiseWrapper
from src.sim.mock_physics import MockPhysics


class SimRunner:
    """Run a complete simulation episode and return a log dict.

    Parameters
    ----------
    cfg : dict
        Merged simulation config (default.yaml + scenario overrides).
    true_dw_cfg : dict
        TRUE downwash config (loaded from downwash_true.yaml).
    controller : BaseController
        Instantiated formation controller (Naive / SoftAvoid / Feedforward).
    scenario : dict
        Scenario-specific config (initial positions, targets, duration).
    seed : int
        Random seed for reproducibility.
    gui : bool
        Whether to enable PyBullet GUI (ignored in mock mode).
    """

    def __init__(
        self,
        cfg: dict,
        true_dw_cfg: dict,
        controller: Any,  # BaseController
        scenario: dict,
        seed: int = 0,
        gui: bool = False,
    ) -> None:
        self.cfg = cfg
        self.controller = controller
        self.scenario = scenario
        self.seed = seed
        self.gui = gui

        np.random.seed(seed)

        # Simulator frequencies
        sim_cfg = cfg.get("sim", {})
        self.pyb_freq: int = int(sim_cfg.get("pyb_freq_hz", 240))
        self.ctrl_freq: int = int(sim_cfg.get("ctrl_freq_hz", 48))
        self.duration_s: float = float(
            scenario.get("duration_s", sim_cfg.get("duration_s", 10.0))
        )
        self.n_ctrl_steps: int = int(self.duration_s * self.ctrl_freq)
        self.ctrl_timestep: float = 1.0 / self.ctrl_freq

        # Drone count and positions
        n_drones: int = scenario.get(
            "n_drones", cfg.get("formation", {}).get("n_drones", 2)
        )
        raw_pos = scenario.get("initial_positions", None)
        if raw_pos is not None:
            init_xyzs = np.array(raw_pos, dtype=float)
        else:
            init_xyzs = np.array(
                [[i * 0.5, 0.0, 1.0] for i in range(n_drones)], dtype=float
            )
        self.n_drones = n_drones

        # Add init jitter
        jitter = cfg.get("env", {}).get("init_jitter_m", 0.0)
        if jitter > 0:
            init_xyzs += np.random.uniform(-jitter, jitter, init_xyzs.shape)

        # Physics backend (mock only for now)
        self.physics = MockPhysics(
            num_drones=n_drones,
            initial_xyzs=init_xyzs,
            pyb_freq=self.pyb_freq,
            ctrl_freq=self.ctrl_freq,
        )

        # TRUE downwash model (simulator-only, never passed to controller)
        self.true_dw = TrueDownwash(true_dw_cfg)

        # Noise wrapper
        noise_cfg = cfg.get("noise", {})
        self.noise = NoiseWrapper(
            enabled=noise_cfg.get("enabled", False),
            pos_sigma=noise_cfg.get("position_sigma_m", 0.005),
            vel_sigma=noise_cfg.get("velocity_sigma_ms", 0.01),
            att_sigma=noise_cfg.get("attitude_sigma_rad", 0.002),
            cmd_delay_steps=int(noise_cfg.get("command_delay_steps", 0)),
            n_drones=n_drones,
            seed=seed + 1000,
        )

        # Target positions
        raw_tgt = scenario.get("target_positions", None)
        if raw_tgt is not None:
            self.targets = np.array(raw_tgt, dtype=float)
        else:
            self.targets = init_xyzs.copy()

    # ------------------------------------------------------------------
    # Main episode runner
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        """Execute the full episode and return a log dict.

        Returns
        -------
        log : dict
            Keys:
              "positions"    : ndarray (T, N, 3) — position history
              "targets"      : ndarray (N, 3)    — target positions
              "rpms"         : ndarray (T, N, 4) — RPM history
              "dw_forces"    : ndarray (T, N, 3) — TRUE downwash forces
              "ctrl_time_s"  : float — wall-clock time in controller
              "n_steps"      : int — number of control steps
              "ctrl_freq_hz" : float
              "seed"         : int
        """
        pos_log = np.zeros((self.n_ctrl_steps, self.n_drones, 3))
        rpm_log = np.zeros((self.n_ctrl_steps, self.n_drones, 4))
        dw_log = np.zeros((self.n_ctrl_steps, self.n_drones, 3))

        obs = self.physics.get_obs()   # (N, 20)
        ctrl_time = 0.0

        for step in range(self.n_ctrl_steps):
            # Apply noise to observation (state estimation error)
            noisy_obs = self.noise.add_state_noise(obs)

            # Compute downwash forces from TRUE model
            positions = obs[:, 0:3]
            dw_forces = self.true_dw.compute_all_forces(positions)

            # Inject TRUE disturbances into physics engine
            for i in range(self.n_drones):
                self.physics.apply_external_force(i, dw_forces[i])

            # Update targets if scenario has time-varying targets
            current_targets = self._get_targets(step)

            # Run controller (has NO access to true_dw)
            t0 = time.perf_counter()
            rpms = self.controller.compute(
                obs=noisy_obs,
                targets=current_targets,
                ctrl_timestep=self.ctrl_timestep,
            )
            ctrl_time += time.perf_counter() - t0

            # Apply command delay
            rpms = self.noise.delay_command(rpms)

            # Physics step
            obs = self.physics.step(rpms)

            # Log
            pos_log[step] = obs[:, 0:3]
            rpm_log[step] = obs[:, 16:20]
            dw_log[step] = dw_forces

        return {
            "positions": pos_log,
            "targets": self._get_targets(0),
            "rpms": rpm_log,
            "dw_forces": dw_log,
            "ctrl_time_s": ctrl_time,
            "n_steps": self.n_ctrl_steps,
            "ctrl_freq_hz": float(self.ctrl_freq),
            "seed": self.seed,
            "duration_s": self.duration_s,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_targets(self, step: int) -> np.ndarray:
        """Return target positions at control step *step*.

        Supports time-varying targets (position_swap scenario).
        """
        # Position swap scenario
        if "target_positions_phase2" in self.scenario:
            swap_step = int(
                self.scenario.get("swap_start_s", 3.0) * self.ctrl_freq
            )
            if step >= swap_step:
                return np.array(
                    self.scenario["target_positions_phase2"], dtype=float
                )
            return np.array(
                self.scenario.get("target_positions_phase1", self.targets),
                dtype=float,
            )

        # Formation transition scenario
        if "formation_end" in self.scenario and "transition_time_s" in self.scenario:
            trans_step = int(
                self.scenario["transition_time_s"] * self.ctrl_freq
            )
            if step >= trans_step:
                # Re-generate end formation targets lazily
                if not hasattr(self, "_end_targets"):
                    from src.formations.generators import generate_formation

                    self._end_targets = generate_formation(
                        formation_cfg=self.scenario["formation_end"],
                        n_drones=self.n_drones,
                        base_height=self.targets[0, 2],
                    )
                return self._end_targets

        return self.targets.copy()
