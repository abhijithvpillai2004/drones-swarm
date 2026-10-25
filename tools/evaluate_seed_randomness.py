"""Step 2 Verification: Evaluate seed randomness and 20-seed statistics.

Demonstrates what seed controls:
1. Initial position jitter: delta_p ~ N(0, (0.003 m)^2) per drone.
2. State observation noise: position std = 2 mm, velocity std = 5 mm/s.
Logs raw per-seed RMSE across 20 seeds for all 4 controllers on hover_column_3d (N=3, 5s).
Saves results to results/diagnostics/seed_statistics_20seeds.csv.
"""
from pathlib import Path
import os
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml
import numpy as np
import pandas as pd
from scipy import stats

from src.models.downwash import DownwashModel
from src.sim.downwash_aviary import DownwashAviary
from src.controllers.naive import NaiveSwarmController
from src.controllers.soft_avoid import SoftAvoidSwarmController
from src.controllers.feedforward import FeedforwardSwarmController
from src.controllers.ua_adc import UAAdaptiveController
from src.formations.generators import HoverColumnFormation
from src.metrics.swarm_metrics import compute_tracking_rmse
from src.noise.wrappers import GaussianObservationNoise


def run_seeded_trial(
    seed: int,
    controller_class: type,
    ctrl_kwargs: dict,
    true_model: DownwashModel,
    duration: float = 5.0,
    ctrl_freq: int = 48,
    num_drones: int = 3,
    with_randomness: bool = True,
):
    dt = 1.0 / ctrl_freq
    total_steps = int(duration * ctrl_freq)
    scenario_gen = HoverColumnFormation(num_drones=num_drones, z_top=1.8, dz=0.45, duration=duration)
    init_xyzs = scenario_gen.get_initial_positions()

    rng = np.random.default_rng(seed)

    if with_randomness:
        # 1. Documented physical randomness: initial placement jitter (std = 3 mm)
        jitter = rng.normal(0.0, 0.003, size=init_xyzs.shape)
        actual_init_xyzs = init_xyzs + jitter
        # 2. Documented sensor noise (pos_std = 2 mm, vel_std = 5 mm/s)
        noise_wrapper = GaussianObservationNoise(
            config={"enabled": True, "pos_std": 0.002, "vel_std": 0.005},
            seed=seed,
        )
    else:
        actual_init_xyzs = init_xyzs
        noise_wrapper = None

    aviary = DownwashAviary(
        num_drones=num_drones,
        initial_xyzs=actual_init_xyzs,
        ctrl_freq=ctrl_freq,
        pyb_freq=240,
        gui=False,
        record=False,
        downwash_model=true_model,
    )

    controller = controller_class(num_drones=num_drones, **ctrl_kwargs)
    obs, _ = aviary.reset()
    controller.reset()

    actual_positions = np.zeros((total_steps, num_drones, 3))
    ref_positions = np.zeros((total_steps, num_drones, 3))

    for step in range(total_steps):
        t = step * dt
        target_pos, target_vel = scenario_gen.evaluate(t)
        ref_positions[step] = target_pos

        states = obs
        actual_positions[step] = states[:, 0:3]

        if noise_wrapper is not None and noise_wrapper.enabled:
            observed_states = noise_wrapper.apply(states)
        else:
            observed_states = states

        rpms, _, _ = controller.compute_actions(
            control_timestep=dt,
            states=observed_states,
            formation_targets=target_pos,
            formation_velocities=target_vel,
        )

        obs, _, _, _, _ = aviary.step(rpms)

    aviary.close()
    metrics = compute_tracking_rmse(actual_positions, ref_positions)
    return float(metrics["swarm_mean_rmse"])


def main():
    with open("configs/downwash_true.yaml", "r", encoding="utf-8") as f:
        true_cfg = yaml.safe_load(f)
    true_model = DownwashModel(config=true_cfg)

    with open("configs/downwash_assumed.yaml", "r", encoding="utf-8") as f:
        assumed_cfg = yaml.safe_load(f)

    with open("configs/ua_adc.yaml", "r", encoding="utf-8") as f:
        ua_cfg = yaml.safe_load(f)

    seeds = list(range(20))
    records = []

    os.makedirs("results/diagnostics", exist_ok=True)

    print("\n" + "=" * 95)
    print("STEP 2: EVALUATING SEED RANDOMNESS & 20-SEED STATISTICS")
    print("=" * 95)

    for s in seeds:
        rmse_naive = run_seeded_trial(s, NaiveSwarmController, {}, true_model)
        rmse_soft = run_seeded_trial(
            s, SoftAvoidSwarmController,
            {"assumed_config": assumed_cfg, "config": ua_cfg.get("soft_avoid", {})},
            true_model
        )
        rmse_ff = run_seeded_trial(
            s, FeedforwardSwarmController,
            {"assumed_config": assumed_cfg},
            true_model
        )
        rmse_ua = run_seeded_trial(
            s, UAAdaptiveController,
            {"config": ua_cfg, "assumed_config": assumed_cfg},
            true_model
        )

        records.append({
            "seed": s,
            "rmse_naive_cm": rmse_naive * 100,
            "rmse_soft_avoid_cm": rmse_soft * 100,
            "rmse_feedforward_cm": rmse_ff * 100,
            "rmse_ua_adc_cm": rmse_ua * 100,
        })
        print(f"Seed {s:2d}: Naive={rmse_naive*100:.3f}cm, SoftAvoid={rmse_soft*100:.3f}cm, Feedforward={rmse_ff*100:.3f}cm, UA-ADC={rmse_ua*100:.3f}cm")

    df = pd.DataFrame(records)
    csv_path = "results/diagnostics/seed_statistics_20seeds.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[SAVED] 20-seed raw RMSE log to {csv_path}")

    # Compute summary statistics
    print("\n" + "=" * 95)
    print("20-SEED SUMMARY STATISTICS (Mean +/- Std [cm])")
    print("=" * 95)
    for col, name in [
        ("rmse_naive_cm", "Controller A (Naive)"),
        ("rmse_soft_avoid_cm", "Controller B (Soft-Avoid)"),
        ("rmse_feedforward_cm", "Controller C (Feedforward)"),
        ("rmse_ua_adc_cm", "Controller D (UA-ADC)"),
    ]:
        mean_val = df[col].mean()
        std_val = df[col].std()
        print(f"  {name:28s}: {mean_val:.4f} +/- {std_val:.4f} cm (min: {df[col].min():.4f}, max: {df[col].max():.4f})")

    # Wilcoxon signed-rank tests against UA-ADC
    print("\n" + "=" * 95)
    print("WILCOXON SIGNED-RANK TESTS (Paired vs UA-ADC, N=20 seeds)")
    print("=" * 95)
    for col, name in [
        ("rmse_naive_cm", "Naive vs UA-ADC"),
        ("rmse_soft_avoid_cm", "Soft-Avoid vs UA-ADC"),
        ("rmse_feedforward_cm", "Feedforward vs UA-ADC"),
    ]:
        stat, pval = stats.wilcoxon(df[col], df["rmse_ua_adc_cm"])
        print(f"  {name:25s}: statistic = {stat:.1f}, p-value = {pval:.4e} {'(significant p < 0.05)' if pval < 0.05 else ''}")


if __name__ == "__main__":
    main()
