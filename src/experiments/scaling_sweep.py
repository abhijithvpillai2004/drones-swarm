"""Swarm Size Scaling Sweep Experiment Runner.

Evaluates Controllers A (Naive), B (Soft-Avoidance), and C (Feedforward)
across scaling drone counts N in [2, 3, 5, 7, 10] under cascading multi-source
downwash disturbance in PyBullet.

Measures tracking RMSE, minimum separation distance, near-collisions, energy proxy,
settling time, and wall-clock simulation execution runtime across >=20 seeds.
"""
from src.paths import resolve
import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import yaml
import numpy as np
import pandas as pd

from src.models.downwash import DownwashModel
from src.sim.downwash_aviary import DownwashAviary
from src.controllers.naive import NaiveSwarmController
from src.controllers.soft_avoid import SoftAvoidSwarmController
from src.controllers.feedforward import FeedforwardSwarmController
from src.controllers.ua_adc import UAAdaptiveController
from src.noise.wrappers import GaussianObservationNoise
from src.formations.generators import HoverColumnFormation, validate_drone_count
from src.metrics.swarm_metrics import compute_all_metrics
from src.experiments.logger import ExperimentLogger


def _load_ua_config(path: str = "configs/ua_adc.yaml") -> Dict[str, Any]:
    """Load UA-ADC configuration dictionary."""
    p = Path(path)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


CONTROLLERS = [
    ("Controller_A_Naive", NaiveSwarmController, {}),
    ("Controller_B_SoftAvoid", SoftAvoidSwarmController, {"config": {"max_displacement": 0.20, "max_velocity": 0.15}}),
    ("Controller_C_Feedforward", FeedforwardSwarmController, {}),
    ("Controller_D_UA_ADC", UAAdaptiveController, {"config": _load_ua_config()}),
]


def load_true_downwash_model(config_path: str = "configs/downwash_true.yaml") -> DownwashModel:
    """Load the ground-truth aerodynamic downwash model."""
    with open(resolve(config_path), "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return DownwashModel(config=cfg)


def run_scaling_trial(
    num_drones: int,
    controller_name: str,
    controller_class: type,
    ctrl_kwargs: Dict[str, Any],
    true_downwash_model: DownwashModel,
    seed: int,
    duration: float = 4.0,
    ctrl_freq_hz: int = 48,
    output_dir: str = "results",
) -> Dict[str, Any]:
    """Execute a single scaling trial recording metrics and wall-clock execution time.

    Args:
        num_drones: Swarm size N in [2, 10].
        controller_name: Name of controller.
        controller_class: Controller class.
        ctrl_kwargs: Extra args for controller initialization.
        true_downwash_model: Physical downwash model.
        seed: Random seed for repeatability.
        duration: Flight duration [s].
        ctrl_freq_hz: Control frequency [Hz].
        output_dir: Results output directory.

    Returns:
        Dictionary of recorded metrics including wall-clock runtime.
    """
    n = validate_drone_count(num_drones)
    scenario_gen = HoverColumnFormation(
        num_drones=n,
        z_top=2.0,
        dz=0.35,
        x_center=0.0,
        y_center=0.0,
        duration=duration,
    )
    dt = 1.0 / ctrl_freq_hz
    total_steps = int(duration * ctrl_freq_hz)

    # Documented randomness: initial position placement jitter (3 mm) and state observation noise
    rng = np.random.default_rng(seed)
    init_xyzs = scenario_gen.get_initial_positions()
    jitter = rng.normal(0.0, 0.003, size=init_xyzs.shape)
    actual_init_xyzs = init_xyzs + jitter

    noise_wrapper = GaussianObservationNoise(
        config={"enabled": True, "pos_std": 0.002, "vel_std": 0.005},
        seed=seed,
    )

    # Measure wall-clock simulation execution runtime
    wall_start = time.perf_counter()

    aviary = DownwashAviary(
        num_drones=n,
        initial_xyzs=actual_init_xyzs,
        ctrl_freq=ctrl_freq_hz,
        pyb_freq=240,
        gui=False,
        record=False,
        downwash_model=true_downwash_model,
    )

    controller = controller_class(num_drones=n, **ctrl_kwargs)
    obs, _ = aviary.reset()
    if hasattr(controller, "reset"):
        controller.reset()

    # Per-run logger with N in identifier
    scenario_tag = f"scaling_column_N{n}"
    run_config = {
        "scenario": scenario_tag,
        "controller": controller_name,
        "num_drones": n,
        "duration": duration,
        "seed": seed,
        "ctrl_freq": ctrl_freq_hz,
    }
    logger = ExperimentLogger(
        scenario_name=scenario_tag,
        controller_name=controller_name,
        seed=seed,
        config=run_config,
        output_dir=os.path.join(output_dir, "scaling_runs"),
    )

    actual_positions = np.zeros((total_steps, n, 3))
    ref_positions = np.zeros((total_steps, n, 3))
    motor_rpms = np.zeros((total_steps, n, 4))
    timestamps = np.zeros(total_steps)

    conf_history: List[np.ndarray] = []
    alpha_history: List[np.ndarray] = []

    for step in range(total_steps):
        t = step * dt
        timestamps[step] = t

        target_pos, target_vel = scenario_gen.evaluate(t)
        ref_positions[step] = target_pos

        states = obs
        actual_positions[step] = states[:, 0:3]

        observed_states = noise_wrapper.apply(states) if noise_wrapper is not None and noise_wrapper.enabled else states

        rpms, _, _ = controller.compute_actions(
            control_timestep=dt,
            states=observed_states,
            formation_targets=target_pos,
            formation_velocities=target_vel,
        )
        motor_rpms[step] = rpms

        if isinstance(controller, UAAdaptiveController):
            conf_history.append(controller.last_confidences.copy())
            alpha_history.append(controller.last_blend_weights.copy())

        for d in range(n):
            logger.log_step(
                timestamp=t,
                step=step,
                drone_id=d,
                pos=states[d, 0:3],
                target_pos=target_pos[d],
                rpy=states[d, 7:10],
                vel=states[d, 10:13],
                rpms=rpms[d],
                disturb_force=aviary.last_applied_disturbances[d],
            )

        obs, _, _, _, _ = aviary.step(rpms)

    aviary.close()
    wall_duration = time.perf_counter() - wall_start

    # Compute standard swarm metrics
    summary_metrics = compute_all_metrics(
        actual_pos=actual_positions,
        target_pos=ref_positions,
        timestamps=timestamps,
        motor_rpms=motor_rpms,
        dt=dt,
        safety_threshold=0.15,
        settling_tolerance=0.05,
    )

    # Average settling time across drones
    settling_times = summary_metrics.get("settling_times", [])
    valid_settling = [st for st in settling_times if st is not None]
    avg_settling = float(np.mean(valid_settling)) if valid_settling else duration

    conf_mean = float(np.mean(conf_history)) if conf_history else 0.0
    alpha_mean = float(np.mean(alpha_history)) if alpha_history else (1.0 if isinstance(controller, FeedforwardSwarmController) else 0.0)

    summary_metrics["settling_time_mean"] = avg_settling
    summary_metrics["sim_runtime_sec"] = wall_duration
    summary_metrics["confidence_mean"] = conf_mean
    summary_metrics["blend_weight_mean"] = alpha_mean

    logger.save(summary_metrics=summary_metrics)

    return {
        "num_drones": n,
        "controller": controller_name,
        "seed": seed,
        "rmse_mean": summary_metrics["rmse_mean"],
        "rmse_max": summary_metrics["rmse_max"],
        "min_interdrone_dist": summary_metrics["min_interdrone_dist"],
        "near_collision_steps": summary_metrics["near_collision_steps"],
        "total_energy_proxy": summary_metrics["total_energy_proxy"],
        "settling_time_mean": avg_settling,
        "sim_runtime_sec": wall_duration,
        "confidence_mean": conf_mean,
        "blend_weight_mean": alpha_mean,
    }


def run_scaling_sweep(
    drone_counts: List[int] = [2, 3, 5, 7, 10],
    num_seeds: int = 20,
    base_seed: int = 42,
    selected_controller: str = "all",
    output_dir: str = "results",
) -> pd.DataFrame:
    """Execute complete scaling sweep across N and controllers for >= 20 seeds."""
    if selected_controller != "all":
        controllers = [
            c for c in CONTROLLERS
            if selected_controller.lower().replace("_", "").replace("-", "") in c[0].lower().replace("_", "")
        ]
        if not controllers:
            raise ValueError(f"No controller matching '{selected_controller}'. Available: {[c[0] for c in CONTROLLERS]}")
    else:
        controllers = CONTROLLERS

    print("\n" + "=" * 75)
    print("  LAUNCHING SWARM SIZE SCALING SWEEP (N in [2, 3, 5, 7, 10])")
    print(f"  Total Drone Counts: {drone_counts}")
    print(f"  Controllers:        {[c[0] for c in controllers]}")
    print(f"  Seeds per combo:    {num_seeds} (base={base_seed})")
    print(f"  Total Simulations:  {len(drone_counts) * len(controllers) * num_seeds}")
    print("=" * 75)

    true_model = load_true_downwash_model("configs/downwash_true.yaml")
    seeds = [base_seed + i for i in range(num_seeds)]

    all_records: List[Dict[str, Any]] = []
    total_runs = len(drone_counts) * len(controllers) * num_seeds
    run_idx = 0

    sweep_start = time.time()

    for n in drone_counts:
        print(f"\n--- Swarm Size: N = {n} Drones ---")
        for ctrl_name, ctrl_class, ctrl_kwargs in controllers:
            for s in seeds:
                run_idx += 1
                record = run_scaling_trial(
                    num_drones=n,
                    controller_name=ctrl_name,
                    controller_class=ctrl_class,
                    ctrl_kwargs=ctrl_kwargs,
                    true_downwash_model=true_model,
                    seed=s,
                    duration=4.0,
                    ctrl_freq_hz=48,
                    output_dir=output_dir,
                )
                all_records.append(record)
                if run_idx % 15 == 0 or run_idx == total_runs:
                    elapsed = time.time() - sweep_start
                    print(f"  [{run_idx:3d}/{total_runs}] N={n:2d} | {ctrl_name:<26} | Seed {s} | Elapsed: {elapsed:.1f}s")

    df = pd.DataFrame(all_records)

    # Save consolidated dataset
    os.makedirs(output_dir, exist_ok=True)
    out_csv = os.path.join(output_dir, "scaling_sweep_results.csv")
    out_all4 = os.path.join(output_dir, "scaling_sweep_results_all4.csv")
    df.to_csv(out_csv, index=False)
    df.to_csv(out_all4, index=False)
    print(f"\n[SUCCESS] Scaling sweep complete! {len(df)} trials saved to {out_csv} and {out_all4}")
    return df


def main() -> None:
    """CLI entrypoint for scaling sweep."""
    parser = argparse.ArgumentParser(description="Swarm Size Scaling Sweep Runner")
    parser.add_argument(
        "--num-drones",
        type=int,
        nargs="+",
        default=[2, 3, 5, 7, 10],
        help="Drone counts to sweep (each in [2, 10]).",
    )
    parser.add_argument(
        "--controller",
        type=str,
        default="all",
        choices=["all", "naive", "soft_avoid", "feedforward", "ua_adc"],
        help="Controller to run (default: 'all').",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=20,
        help="Number of random seeds per (controller, N) combination (default: 20).",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=42,
        help="Base random seed integer (default: 42).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for results CSV and run logs.",
    )
    args = parser.parse_args()

    # Validate all requested drone counts
    validated_counts = []
    for count in args.num_drones:
        if count < 2 or count > 10:
            parser.error(f"Invalid drone count {count}. Must be between 2 and 10.")
        validated_counts.append(count)

    run_scaling_sweep(
        drone_counts=validated_counts,
        num_seeds=args.seeds,
        base_seed=args.base_seed,
        selected_controller=args.controller,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
