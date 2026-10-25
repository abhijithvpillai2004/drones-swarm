"""Gate B Verification Script: Mismatch traces (alpha_force in {0.5, 1.5, 3.0}).

Runs all 4 controllers (naive, soft_avoid, feedforward, ua_adc) across the three
mismatch conditions on hover_column_3d (N=3, duration=6.0s).
Saves detailed log to results/diagnostics/gate_b_mismatch_traces.csv.
Reports theta at t = 1, 2, 5 s, confidence c, blend weight alpha, and RMSE.
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

from src.models.downwash import DownwashModel
from src.sim.downwash_aviary import DownwashAviary
from src.controllers.naive import NaiveSwarmController
from src.controllers.soft_avoid import SoftAvoidSwarmController
from src.controllers.feedforward import FeedforwardSwarmController
from src.controllers.ua_adc import UAAdaptiveController
from src.formations.generators import HoverColumnFormation
from src.metrics.swarm_metrics import compute_tracking_rmse


def run_single_simulation(
    controller_class: type,
    ctrl_kwargs: dict,
    true_downwash_model: DownwashModel,
    duration: float = 6.0,
    ctrl_freq: int = 48,
    num_drones: int = 3,
    dz: float = 0.45,
):
    dt = 1.0 / ctrl_freq
    total_steps = int(duration * ctrl_freq)
    scenario_gen = HoverColumnFormation(num_drones=num_drones, z_top=1.8, dz=dz, duration=duration)
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

    controller = controller_class(num_drones=num_drones, **ctrl_kwargs)
    obs, _ = aviary.reset()
    controller.reset()

    actual_positions = np.zeros((total_steps, num_drones, 3))
    ref_positions = np.zeros((total_steps, num_drones, 3))

    step_telemetry = []

    for step in range(total_steps):
        t = step * dt
        target_pos, target_vel = scenario_gen.evaluate(t)
        ref_positions[step] = target_pos

        states = obs
        actual_positions[step] = states[:, 0:3]

        rpms, _, _ = controller.compute_actions(
            control_timestep=dt,
            states=states,
            formation_targets=target_pos,
            formation_velocities=target_vel,
        )

        entry = {"step": step, "t": t}
        if isinstance(controller, UAAdaptiveController):
            # record theta, P, c, alpha for all drones
            for j in range(num_drones):
                est = controller.estimator.estimators[j]
                entry[f"drone_{j}_theta"] = float(est.theta)
                entry[f"drone_{j}_P"] = float(est.P)
                entry[f"drone_{j}_c"] = float(est.confidence)
                entry[f"drone_{j}_alpha"] = float(controller.last_blend_weights[j])
        step_telemetry.append(entry)

        obs, _, _, _, _ = aviary.step(rpms)

    aviary.close()

    metrics = compute_tracking_rmse(actual_positions, ref_positions)
    rmse = float(metrics["swarm_mean_rmse"])
    return rmse, pd.DataFrame(step_telemetry)


def run_gate_b():
    duration = 6.0
    ctrl_freq = 48
    num_drones = 3
    alpha_forces = [0.5, 1.5, 3.0]

    with open("configs/downwash_true.yaml", "r", encoding="utf-8") as f:
        true_cfg = yaml.safe_load(f)
    true_model = DownwashModel(config=true_cfg)

    with open("configs/downwash_assumed.yaml", "r", encoding="utf-8") as f:
        nominal_assumed_cfg = yaml.safe_load(f)
    nom_max_f = float(nominal_assumed_cfg["max_force_z"])

    with open("configs/ua_adc.yaml", "r", encoding="utf-8") as f:
        ua_cfg = yaml.safe_load(f)

    results_summary = []
    telemetry_dfs = {}

    os.makedirs("results/diagnostics", exist_ok=True)

    print("\n" + "=" * 95)
    print("GATE B: MODEL MISMATCH TRACES (alpha_force in {0.5, 1.5, 3.0})")
    print("=" * 95)

    for alpha in alpha_forces:
        print(f"\n>>> Running alpha_force = {alpha:.2f} (max_force_z = {nom_max_f * alpha:.4f} N) <<<")
        assumed_perturbed = nominal_assumed_cfg.copy()
        assumed_perturbed["max_force_z"] = nom_max_f * alpha

        # 1. Naive (ignores downwash, independent of assumed config)
        rmse_naive, _ = run_single_simulation(
            controller_class=NaiveSwarmController,
            ctrl_kwargs={},
            true_downwash_model=true_model,
            duration=duration,
        )

        # 2. Soft-Avoid
        rmse_soft, _ = run_single_simulation(
            controller_class=SoftAvoidSwarmController,
            ctrl_kwargs={
                "assumed_config": assumed_perturbed,
                "config": ua_cfg.get("soft_avoid", {}),
            },
            true_downwash_model=true_model,
            duration=duration,
        )

        # 3. Feedforward
        rmse_ff, _ = run_single_simulation(
            controller_class=FeedforwardSwarmController,
            ctrl_kwargs={"assumed_config": assumed_perturbed},
            true_downwash_model=true_model,
            duration=duration,
        )

        # 4. UA-ADC
        rmse_ua, df_ua = run_single_simulation(
            controller_class=UAAdaptiveController,
            ctrl_kwargs={
                "config": ua_cfg,
                "assumed_config": assumed_perturbed,
            },
            true_downwash_model=true_model,
            duration=duration,
        )
        telemetry_dfs[alpha] = df_ua

        # Extract values at t = 1, 2, 5 s
        step_1s = int(round(1.0 * ctrl_freq))
        step_2s = int(round(2.0 * ctrl_freq))
        step_5s = int(round(5.0 * ctrl_freq))

        row_1s = df_ua.iloc[step_1s]
        row_2s = df_ua.iloc[step_2s]
        row_5s = df_ua.iloc[step_5s]

        # Drones 1 and 2 are under wake
        theta_d1 = [row_1s["drone_1_theta"], row_2s["drone_1_theta"], row_5s["drone_1_theta"]]
        theta_d2 = [row_1s["drone_2_theta"], row_2s["drone_2_theta"], row_5s["drone_2_theta"]]

        c_d1 = [row_1s["drone_1_c"], row_2s["drone_1_c"], row_5s["drone_1_c"]]
        c_d2 = [row_1s["drone_2_c"], row_2s["drone_2_c"], row_5s["drone_2_c"]]

        a_d1 = [row_1s["drone_1_alpha"], row_2s["drone_1_alpha"], row_5s["drone_1_alpha"]]
        a_d2 = [row_1s["drone_2_alpha"], row_2s["drone_2_alpha"], row_5s["drone_2_alpha"]]

        print(f"  RMSE Comparison:")
        print(f"    Naive:       {rmse_naive * 100:6.2f} cm")
        print(f"    Soft-Avoid:  {rmse_soft * 100:6.2f} cm")
        print(f"    Feedforward: {rmse_ff * 100:6.2f} cm")
        print(f"    UA-ADC:      {rmse_ua * 100:6.2f} cm")

        print(f"  UA-ADC Drone 1 (under wake):")
        print(f"    theta: t=1s: {theta_d1[0]:.4f}, t=2s: {theta_d1[1]:.4f}, t=5s: {theta_d1[2]:.4f}")
        print(f"    c:     t=1s: {c_d1[0]:.4f}, t=2s: {c_d1[1]:.4f}, t=5s: {c_d1[2]:.4f}")
        print(f"    alpha: t=1s: {a_d1[0]:.4f}, t=2s: {a_d1[1]:.4f}, t=5s: {a_d1[2]:.4f}")

        print(f"  UA-ADC Drone 2 (under wake):")
        print(f"    theta: t=1s: {theta_d2[0]:.4f}, t=2s: {theta_d2[1]:.4f}, t=5s: {theta_d2[2]:.4f}")
        print(f"    c:     t=1s: {c_d2[0]:.4f}, t=2s: {c_d2[1]:.4f}, t=5s: {c_d2[2]:.4f}")
        print(f"    alpha: t=1s: {a_d2[0]:.4f}, t=2s: {a_d2[1]:.4f}, t=5s: {a_d2[2]:.4f}")

        results_summary.append({
            "alpha_force": alpha,
            "rmse_naive_m": rmse_naive,
            "rmse_soft_avoid_m": rmse_soft,
            "rmse_feedforward_m": rmse_ff,
            "rmse_ua_adc_m": rmse_ua,
            "d1_theta_1s": theta_d1[0],
            "d1_theta_2s": theta_d1[1],
            "d1_theta_5s": theta_d1[2],
            "d1_c_1s": c_d1[0],
            "d1_c_2s": c_d1[1],
            "d1_c_5s": c_d1[2],
            "d1_alpha_1s": a_d1[0],
            "d1_alpha_2s": a_d1[1],
            "d1_alpha_5s": a_d1[2],
            "d2_theta_1s": theta_d2[0],
            "d2_theta_2s": theta_d2[1],
            "d2_theta_5s": theta_d2[2],
            "d2_c_1s": c_d2[0],
            "d2_c_2s": c_d2[1],
            "d2_c_5s": c_d2[2],
            "d2_alpha_1s": a_d2[0],
            "d2_alpha_2s": a_d2[1],
            "d2_alpha_5s": a_d2[2],
        })

    summary_df = pd.DataFrame(results_summary)
    summary_csv = "results/diagnostics/gate_b_mismatch_traces.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\n[SAVED] Gate B summary to {summary_csv}")


if __name__ == "__main__":
    run_gate_b()
