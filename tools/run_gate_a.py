"""Gate A Verification Script: Nominal hover_column_3d trace with UA-ADC.

Runs closed-loop PyBullet simulation of hover_column_3d (N=3, duration=10.0s).
Saves detailed time series to results/diagnostics/ua_adc_trace_nominal.csv
and plots to results/diagnostics/ua_adc_trace_nominal.png.
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
import matplotlib.pyplot as plt

from src.models.downwash import DownwashModel
from src.sim.downwash_aviary import DownwashAviary
from src.controllers.ua_adc import UAAdaptiveController
from src.formations.generators import HoverColumnFormation


def run_gate_a():
    num_drones = 3
    duration = 10.0
    ctrl_freq = 48
    dt = 1.0 / ctrl_freq
    total_steps = int(duration * ctrl_freq)

    # Assumed and True downwash models
    with open("configs/downwash_assumed.yaml", "r", encoding="utf-8") as f:
        assumed_cfg = yaml.safe_load(f)
    with open("configs/downwash_true.yaml", "r", encoding="utf-8") as f:
        true_cfg = yaml.safe_load(f)

    with open("configs/ua_adc.yaml", "r", encoding="utf-8") as f:
        ua_cfg = yaml.safe_load(f)

    true_downwash_model = DownwashModel(config=true_cfg)
    scenario_gen = HoverColumnFormation(num_drones=num_drones, z_top=1.8, dz=0.45, duration=duration)
    init_xyzs = scenario_gen.get_initial_positions()

    aviary = DownwashAviary(
        num_drones=num_drones,
        initial_xyzs=init_xyzs,
        ctrl_freq=ctrl_freq,
        pyb_freq=240,
        gui=False,
        record=False,
        downwash_model=true_downwash_model,
    )

    controller = UAAdaptiveController(
        config=ua_cfg,
        num_drones=num_drones,
        assumed_config=assumed_cfg,
    )
    obs, _ = aviary.reset()
    controller.reset()

    records = []

    for step in range(total_steps):
        t = step * dt
        target_pos, target_vel = scenario_gen.evaluate(t)
        states = obs

        # Step controller
        rpms, pos_errs, _ = controller.compute_actions(
            control_timestep=dt,
            states=states,
            formation_targets=target_pos,
            formation_velocities=target_vel,
        )

        # Assumed model forces for estimator
        assumed_forces = controller.assumed_model.compute_swarm_forces(
            positions=states[:, 0:3],
            velocities=states[:, 10:13],
        )

        # Record telemetry for each drone
        row = {"step": step, "t": t}
        for j in range(num_drones):
            est = controller.estimator.estimators[j]
            f_hat = float(est.estimate_disturbance(assumed_forces[j])[2])
            f_true = float(aviary.last_applied_disturbances[j, 2])
            row[f"drone_{j}_pos_z"] = float(states[j, 2])
            row[f"drone_{j}_target_z"] = float(target_pos[j, 2])
            row[f"drone_{j}_F_hat"] = f_hat
            row[f"drone_{j}_F_true"] = f_true
            row[f"drone_{j}_theta"] = float(est.theta)
            row[f"drone_{j}_P"] = float(est.P)
            row[f"drone_{j}_c"] = float(est.confidence)
            row[f"drone_{j}_alpha"] = float(controller.last_blend_weights[j])
            row[f"drone_{j}_T_cmd"] = float(controller.last_commanded_thrusts[j])

        records.append(row)

        # Environment step
        obs, _, _, _, _ = aviary.step(rpms)

    aviary.close()

    df = pd.DataFrame(records)

    # Ensure output directories exist
    os.makedirs("results/diagnostics/ua_adc", exist_ok=True)
    csv_path = "results/diagnostics/ua_adc_trace_nominal.csv"
    csv_path2 = "results/diagnostics/ua_adc/ua_adc_trace_nominal.csv"
    df.to_csv(csv_path, index=False)
    df.to_csv(csv_path2, index=False)
    print(f"[SAVED] Telemetry CSV to {csv_path} and {csv_path2}")

    # Plot
    fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

    # 1. Disturbance force F_hat vs F_true
    for j in range(num_drones):
        line = axs[0].plot(df["t"], df[f"drone_{j}_F_true"], label=f"Drone {j} F_true", linestyle="--")
        axs[0].plot(df["t"], df[f"drone_{j}_F_hat"], label=f"Drone {j} F_hat", color=line[0].get_color())
    axs[0].set_ylabel("Vertical Force [N]")
    axs[0].set_title("Gate A: Disturbance Estimate F_hat vs True Downwash F_true")
    axs[0].grid(True)
    axs[0].legend(loc="upper right", ncol=3)

    # 2. Gain theta
    for j in range(num_drones):
        axs[1].plot(df["t"], df[f"drone_{j}_theta"], label=f"Drone {j} theta")
    axs[1].set_ylabel("Gain theta")
    axs[1].set_title("Gain Estimate theta (Nominal ratio = 1.0)")
    axs[1].grid(True)
    axs[1].legend(loc="upper right")

    # 3. Covariance P
    for j in range(num_drones):
        axs[2].plot(df["t"], df[f"drone_{j}_P"], label=f"Drone {j} P")
    axs[2].set_ylabel("Covariance P")
    axs[2].set_title("Posterior Variance P (Bounded)")
    axs[2].grid(True)
    axs[2].legend(loc="upper right")

    # 4. Confidence c and Blend Weight alpha
    for j in range(num_drones):
        line = axs[3].plot(df["t"], df[f"drone_{j}_c"], label=f"Drone {j} confidence c")
        axs[3].plot(df["t"], df[f"drone_{j}_alpha"], label=f"Drone {j} blend weight alpha", linestyle=":", color=line[0].get_color())
    axs[3].axhline(0.5, color="gray", linestyle="-.", alpha=0.5, label="c = 0.5 threshold")
    axs[3].set_ylabel("Confidence / Weight")
    axs[3].set_xlabel("Time [s]")
    axs[3].set_title("Confidence c(t) and Blend Weight alpha(t)")
    axs[3].grid(True)
    axs[3].legend(loc="lower right", ncol=2)

    plt.tight_layout()
    plot_path = "results/diagnostics/ua_adc_trace_nominal.png"
    plot_path2 = "results/diagnostics/ua_adc/ua_adc_trace_nominal.png"
    plt.savefig(plot_path, dpi=300)
    plt.savefig(plot_path2, dpi=300)
    plt.close()
    print(f"[SAVED] Plot to {plot_path} and {plot_path2}")

    # Print requested sample times
    target_times = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]
    print("\n" + "=" * 95)
    print("GATE A NOMINAL TRACE SUMMARY (N=3, duration=10.0s)")
    print("=" * 95)

    for tt in target_times:
        step_idx = int(round(tt * ctrl_freq))
        if step_idx >= len(df):
            step_idx = len(df) - 1
        row_t = df.iloc[step_idx]
        actual_t = row_t["t"]
        print(f"\n--- At t = {actual_t:.2f} s (step {step_idx}) ---")
        for j in range(num_drones):
            print(
                f"  Drone {j}: F_hat = {row_t[f'drone_{j}_F_hat']:+.4f} N, "
                f"F_true = {row_t[f'drone_{j}_F_true']:+.4f} N, "
                f"theta = {row_t[f'drone_{j}_theta']:.4f}, "
                f"P = {row_t[f'drone_{j}_P']:.4f}, "
                f"c = {row_t[f'drone_{j}_c']:.4f}, "
                f"alpha = {row_t[f'drone_{j}_alpha']:.4f}"
            )

    print("\n--- Confidence Threshold Check (c > 0.5) ---")
    for j in range(num_drones):
        c_series = df[f"drone_{j}_c"]
        exceed_mask = c_series > 0.5
        if exceed_mask.any():
            t_first = df.loc[exceed_mask, "t"].iloc[0]
            print(f"  Drone {j}: confidence first exceeds 0.5 at t = {t_first:.3f} s")
        else:
            print(f"  Drone {j}: confidence never exceeds 0.5 (max c = {c_series.max():.4f})")

    # Gate verification logic
    # Drone 1 and 2 are under downwash. Check:
    # 1. F_hat does not decay to 0
    # 2. P stays bounded
    # 3. c rises above 0.5 at some point
    f_hat_d1_end = abs(df.iloc[-1]["drone_1_F_hat"])
    f_hat_d2_end = abs(df.iloc[-1]["drone_2_F_hat"])
    p_bounded = (df[[f"drone_{j}_P" for j in range(num_drones)]].max().max() <= 5.0) and (df[[f"drone_{j}_P" for j in range(num_drones)]].min().min() >= 1e-4)
    c_d1_exceeded = (df["drone_1_c"] > 0.5).any()
    c_d2_exceeded = (df["drone_2_c"] > 0.5).any()

    print("\n--- Gate A Pass/Fail Assessment ---")
    print(f"  1. F_hat does not decay to 0: Drone 1 final |F_hat| = {f_hat_d1_end:.4f} N, Drone 2 final |F_hat| = {f_hat_d2_end:.4f} N -> {'PASS' if (f_hat_d1_end > 0.01 and f_hat_d2_end > 0.01) else 'FAIL'}")
    print(f"  2. P stays bounded: max P = {df[[f'drone_{j}_P' for j in range(num_drones)]].max().max():.4f}, min P = {df[[f'drone_{j}_P' for j in range(num_drones)]].min().min():.4f} -> {'PASS' if p_bounded else 'FAIL'}")
    print(f"  3. c rises above 0.5: Drone 1 exceeded = {c_d1_exceeded}, Drone 2 exceeded = {c_d2_exceeded} -> {'PASS' if (c_d1_exceeded and c_d2_exceeded) else 'FAIL'}")

    passed = (f_hat_d1_end > 0.01) and (f_hat_d2_end > 0.01) and p_bounded and c_d1_exceeded and c_d2_exceeded
    print(f"\nOVERALL GATE A STATUS: {'PASS' if passed else 'FAIL'}")
    return passed


if __name__ == "__main__":
    run_gate_a()
