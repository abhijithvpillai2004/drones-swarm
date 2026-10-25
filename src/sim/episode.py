"""Single shared episode loop for every PyBullet experiment.

Before this module, the same "reset aviary → step controller → step physics"
loop was copied into the demo, the sweeps and about ten tools, and the copies
drifted apart (for example, one of them accepted a ``seed`` and ignored it).
All new runners use :func:`run_episode`, so a run is defined in one place.

Randomness is explicit:
    * ``seed=None``  → fully deterministic run (no jitter, no noise).
    * ``seed=int``   → initial-position jitter (``init_jitter_std``) and,
      if enabled, Gaussian observation noise, both drawn from that seed.

The TRUE downwash model is handed to the simulator only.  Controllers receive
observed states; the TRUE disturbance is logged for evaluation, never fed back.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import numpy as np

from src.models.downwash import DownwashModel
from src.noise.wrappers import CommandDelayBuffer, GaussianObservationNoise
from src.paths import config_path, load_yaml

GROUND_CONTACT_Z = 0.05      # [m] any drone below this counts as ground contact
NEAR_COLLISION_DIST = 0.12   # [m] same value as the 3D safety net threshold


@dataclass
class EpisodeConfig:
    """Settings for one episode (all optional except the essentials)."""

    seed: Optional[int] = None
    init_jitter_std: float = 0.003            # [m] used only when seed is set
    observation_noise: bool = False
    pos_noise_std: float = 0.002              # [m]
    vel_noise_std: float = 0.005              # [m/s]
    delay_steps: int = 0                      # command latency in control steps
    ctrl_freq_hz: int = 48
    pyb_freq_hz: int = 240
    gui: bool = False
    realtime_speed: Optional[float] = None    # wall-clock pacing (GUI only)
    stop_on_ground_contact: bool = False


@dataclass
class EpisodeResult:
    """Full trace of one episode plus summary metrics."""

    t: np.ndarray                 # (T,)
    pos: np.ndarray               # (T, N, 3) true positions
    ref: np.ndarray               # (T, N, 3) nominal references
    vel: np.ndarray               # (T, N, 3) true velocities
    rpms: np.ndarray              # (T, N, 4)
    true_force: np.ndarray        # (T, N, 3) TRUE downwash applied (evaluation only)
    confidence: np.ndarray        # (T, N)  zeros for non-UA-ADC controllers
    blend: np.ndarray             # (T, N)  alpha after conflict override
    blend_raw: np.ndarray         # (T, N)  alpha from confidence only
    conflict_risk: np.ndarray     # (T, N)
    theta: np.ndarray             # (T, N)  learned wake gain
    metrics: Dict[str, Any] = field(default_factory=dict)
    wall_time_s: float = 0.0


def load_true_model(path: Optional[str] = None, force_scale: float = 1.0) -> DownwashModel:
    """Load the TRUE wake model (simulator side only), optionally scaled."""
    cfg = dict(load_yaml(path or config_path("downwash_true.yaml")))
    if force_scale != 1.0:
        cfg["max_force_z"] = float(cfg.get("max_force_z", 0.15)) * force_scale
        if "max_combined_force_z" in cfg:
            cfg["max_combined_force_z"] = float(cfg["max_combined_force_z"]) * force_scale
    return DownwashModel(cfg)


def load_assumed_config(force_scale: float = 1.0) -> Dict[str, Any]:
    """Load the ASSUMED wake parameters, optionally scaled (model mismatch)."""
    cfg = dict(load_yaml(config_path("downwash_assumed.yaml")))
    if force_scale != 1.0:
        cfg["max_force_z"] = float(cfg.get("max_force_z", 0.15)) * force_scale
        if "max_combined_force_z" in cfg:
            cfg["max_combined_force_z"] = float(cfg["max_combined_force_z"]) * force_scale
    return cfg


def _pairwise_min_distance(pos: np.ndarray) -> float:
    n = pos.shape[0]
    if n < 2:
        return float("inf")
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
    d[np.arange(n), np.arange(n)] = np.inf
    return float(d.min())


def run_episode(
    scenario_gen: Any,
    controller: Any,
    true_model: Optional[DownwashModel] = None,
    cfg: Optional[EpisodeConfig] = None,
    on_step: Optional[Callable[[int, float, Dict[str, Any]], None]] = None,
    on_finish: Optional[Callable[[Any, np.ndarray], None]] = None,
) -> EpisodeResult:
    """Run one closed-loop episode in PyBullet and return the full trace.

    Args:
        scenario_gen: Trajectory generator (``evaluate(t)``, ``duration``,
            ``num_drones``, ``get_initial_positions()``).
        controller: Swarm controller instance with ``compute_actions``.
        true_model: TRUE downwash model. Defaults to ``configs/downwash_true.yaml``.
        cfg: Episode settings.
        on_step: Optional callback ``on_step(step, t, info)`` for live display;
            ``info`` holds the current positions, min distance and telemetry.
        on_finish: Optional callback ``on_finish(aviary, obs)`` called after the
            last step and before the simulator closes (the demo uses it to
            keep the drones hovering in the GUI window).
    """
    # Imported here so that pure-numpy users of this module do not need pybullet.
    from src.sim.downwash_aviary import DownwashAviary

    cfg = cfg or EpisodeConfig()
    true_model = true_model if true_model is not None else load_true_model()

    n = scenario_gen.num_drones
    dt = 1.0 / cfg.ctrl_freq_hz
    steps = int(round(scenario_gen.duration * cfg.ctrl_freq_hz))

    init_xyzs = np.asarray(scenario_gen.get_initial_positions(), dtype=float).copy()
    noise = None
    if cfg.seed is not None:
        rng = np.random.default_rng(cfg.seed)
        init_xyzs = init_xyzs + rng.normal(0.0, cfg.init_jitter_std, size=init_xyzs.shape)
        if cfg.observation_noise:
            noise = GaussianObservationNoise(
                config={"enabled": True, "pos_std": cfg.pos_noise_std,
                        "vel_std": cfg.vel_noise_std, "rpy_std": 0.0},
                seed=cfg.seed + 1,
            )
    delay = CommandDelayBuffer(delay_steps=cfg.delay_steps, enabled=cfg.delay_steps > 0)

    aviary = DownwashAviary(
        num_drones=n,
        initial_xyzs=init_xyzs,
        ctrl_freq=cfg.ctrl_freq_hz,
        pyb_freq=cfg.pyb_freq_hz,
        gui=cfg.gui,
        record=False,
        downwash_model=true_model,
    )

    T = steps
    res = EpisodeResult(
        t=np.zeros(T), pos=np.zeros((T, n, 3)), ref=np.zeros((T, n, 3)),
        vel=np.zeros((T, n, 3)), rpms=np.zeros((T, n, 4)),
        true_force=np.zeros((T, n, 3)), confidence=np.zeros((T, n)),
        blend=np.zeros((T, n)), blend_raw=np.zeros((T, n)),
        conflict_risk=np.zeros((T, n)), theta=np.zeros((T, n)),
    )

    is_ua = hasattr(controller, "estimator") and hasattr(controller, "last_blend_weights")
    wall_start = time.perf_counter()
    last_step = T
    try:
        obs, _ = aviary.reset()
        controller.reset()
        delay.reset()
        for step in range(T):
            t = step * dt
            ref_p, ref_v = scenario_gen.evaluate(t)
            observed = noise.apply(obs) if noise is not None else obs

            rpms, _, _ = controller.compute_actions(
                control_timestep=dt,
                states=observed,
                formation_targets=ref_p,
                formation_velocities=ref_v,
            )
            applied = delay.step(rpms)

            res.t[step] = t
            res.pos[step] = obs[:, 0:3]
            res.vel[step] = obs[:, 10:13]
            res.ref[step] = ref_p
            res.rpms[step] = rpms
            if is_ua:
                res.confidence[step] = controller.last_confidences
                res.blend[step] = controller.last_blend_weights
                res.blend_raw[step] = getattr(controller, "last_raw_blend_weights",
                                              controller.last_blend_weights)
                res.conflict_risk[step] = getattr(controller, "last_conflict_risk", np.zeros(n))
                res.theta[step] = [e.theta for e in controller.estimator.estimators]

            obs, _, _, _, _ = aviary.step(applied)
            res.true_force[step] = aviary.last_applied_disturbances

            if on_step is not None:
                on_step(step, t, {
                    "positions": res.pos[step],
                    "min_dist": _pairwise_min_distance(res.pos[step]),
                    "confidence": res.confidence[step],
                    "blend": res.blend[step],
                    "conflict_risk": res.conflict_risk[step],
                    "theta": res.theta[step],
                    "is_ua": is_ua,
                    "client": aviary.CLIENT,
                })

            if cfg.gui and cfg.realtime_speed:
                target_wall = wall_start + (step + 1) * dt / max(cfg.realtime_speed, 0.05)
                sleep_s = target_wall - time.perf_counter()
                if sleep_s > 0:
                    time.sleep(sleep_s)

            if cfg.stop_on_ground_contact and np.any(res.pos[step, :, 2] < GROUND_CONTACT_Z):
                last_step = step + 1
                break
        if on_finish is not None:
            on_finish(aviary, obs)
    finally:
        aviary.close()

    if last_step < T:
        for name in ("t", "pos", "ref", "vel", "rpms", "true_force", "confidence",
                     "blend", "blend_raw", "conflict_risk", "theta"):
            setattr(res, name, getattr(res, name)[:last_step])

    res.wall_time_s = time.perf_counter() - wall_start
    res.metrics = summarise_episode(res, dt)
    return res


def confidence_rise_times(res: EpisodeResult, level: float = 0.5,
                          wake_force_frac: float = 0.01, mass: float = 0.027,
                          g: float = 9.81) -> Dict[int, Optional[float]]:
    """Time at which each in-wake drone's confidence first exceeds ``level``.

    A drone counts as "in the wake" if the TRUE disturbance on it at the first
    step exceeds ``wake_force_frac`` of its weight.  Evaluation-side only.
    Returns {drone_id: time or None if it never rose}.
    """
    out: Dict[int, Optional[float]] = {}
    if res.t.size == 0:
        return out
    start_force = np.abs(res.true_force[0, :, 2])
    for d in np.where(start_force > wake_force_frac * mass * g)[0]:
        above = np.where(res.confidence[:, d] > level)[0]
        out[int(d)] = float(res.t[above[0]]) if above.size else None
    return out


def summarise_episode(res: EpisodeResult, dt: float) -> Dict[str, Any]:
    """Compute the summary metrics used by the pass criteria and the tables."""
    if res.t.size == 0:
        return {}
    err = np.linalg.norm(res.pos - res.ref, axis=2)                 # (T, N)
    per_drone_rmse = np.sqrt(np.mean(err ** 2, axis=0))
    n = res.pos.shape[1]
    min_d = np.array([_pairwise_min_distance(p) for p in res.pos])
    rise = confidence_rise_times(res)
    rise_vals = [v for v in rise.values()]
    return {
        "rmse_mean_m": float(np.mean(per_drone_rmse)),
        "rmse_max_m": float(np.max(per_drone_rmse)),
        "min_sep_m": float(np.min(min_d)) if n > 1 else float("nan"),
        "t_min_sep_s": float(res.t[int(np.argmin(min_d))]) if n > 1 else float("nan"),
        "near_collision_steps": int(np.sum(min_d < NEAR_COLLISION_DIST)),
        "min_z_m": float(np.min(res.pos[:, :, 2])),
        "ground_contact": bool(np.any(res.pos[:, :, 2] < GROUND_CONTACT_Z)),
        "has_nan": bool(np.isnan(res.pos).any()),
        "confidence_mean": float(np.mean(res.confidence)),
        "blend_mean": float(np.mean(res.blend)),
        "conflict_steps": int(np.sum(np.max(res.conflict_risk, axis=1) > 0.5)),
        "in_wake_drones": len(rise),
        "conf_rise_time_max_s": (None if not rise_vals or any(v is None for v in rise_vals)
                                 else float(max(rise_vals))),
        "energy_proxy": float(np.sum(res.rpms.astype(float) ** 2) * dt),
        "duration_s": float(res.t[-1] + dt),
        "steps_run": int(res.t.size),
    }
