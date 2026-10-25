"""Phase 6: Robustness, Sensitivity, and Model Error Sweep Experiment.

Quantifies the degradation of downwash formation controllers under:
1. Aerodynamic parameter mismatch (force magnitude and influence radius)
2. Additive state observation noise
3. Control command latency delay
4. Identifies the break-even / crossover envelope where avoidance outperforms feedforward.
"""
from src.paths import config_path
import os
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
from src.formations.generators import HoverColumnFormation, BaseTrajectoryGenerator
from src.metrics.swarm_metrics import compute_tracking_rmse
from src.noise.wrappers import GaussianObservationNoise, CommandDelayBuffer
from src.analysis.robustness_plots import (
    plot_force_mismatch_sensitivity,
    plot_radius_mismatch_sensitivity,
    plot_noise_delay_comparison,
    plot_2d_robustness_heatmap,
)


def _load_ua_config() -> Dict[str, Any]:
    p = config_path("ua_adc.yaml")
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def run_robustness_trial(
    scenario_gen: BaseTrajectoryGenerator,
    controller_class: type,
    assumed_config: Dict[str, Any],
    true_downwash_model: DownwashModel,
    noise_wrapper: Optional[GaussianObservationNoise] = None,
    delay_buffer: Optional[CommandDelayBuffer] = None,
    ctrl_freq_hz: int = 48,
    ctrl_kwargs: Optional[Dict[str, Any]] = None,
    return_details: bool = False,
):
    """Run a single robustness trial with perturbed models, noise, and latency.

    Returns:
        Swarm mean RMSE in meters (if return_details=False),
        or dict of {rmse, mean_confidence, mean_blend_weight, any_fell, any_saturated}.
    """
    num_drones = scenario_gen.num_drones
    duration = scenario_gen.duration
    dt = 1.0 / ctrl_freq_hz
    total_steps = int(duration * ctrl_freq_hz)

    init_xyzs = scenario_gen.get_initial_positions()

    aviary = DownwashAviary(
        num_drones=num_drones,
        initial_xyzs=init_xyzs,
        ctrl_freq=ctrl_freq_hz,
        pyb_freq=240,
        gui=False,
        record=False,
        downwash_model=true_downwash_model,
    )

    kwargs = ctrl_kwargs.copy() if ctrl_kwargs else {}
    if controller_class in [SoftAvoidSwarmController, FeedforwardSwarmController, UAAdaptiveController]:
        kwargs["assumed_config"] = assumed_config
    if controller_class == UAAdaptiveController and "config" not in kwargs:
        kwargs["config"] = _load_ua_config()

    controller = controller_class(num_drones=num_drones, **kwargs)
    obs, _ = aviary.reset()
    controller.reset()

    if delay_buffer is not None:
        delay_buffer.reset()

    actual_positions = np.zeros((total_steps, num_drones, 3))
    ref_positions = np.zeros((total_steps, num_drones, 3))

    conf_history: List[np.ndarray] = []
    alpha_history: List[np.ndarray] = []
    any_fell = False
    any_saturated = False

    for step in range(total_steps):
        t = step * dt
        target_pos, target_vel = scenario_gen.evaluate(t)
        ref_positions[step] = target_pos

        # Ground truth states
        states = obs
        actual_positions[step] = states[:, 0:3]

        # Apply observation noise if active
        if noise_wrapper is not None and noise_wrapper.enabled:
            observed_states = noise_wrapper.apply(states)
        else:
            observed_states = states

        # Compute controller commands
        rpms, _, _ = controller.compute_actions(
            control_timestep=dt,
            states=observed_states,
            formation_targets=target_pos,
            formation_velocities=target_vel,
        )

        # Check for motor saturation
        if np.any(rpms >= 21600.0):
            any_saturated = True

        # Telemetry for UA-ADC
        if isinstance(controller, UAAdaptiveController):
            conf_history.append(controller.last_confidences.copy())
            alpha_history.append(controller.last_blend_weights.copy())

        # Apply command latency buffer if active
        if delay_buffer is not None and delay_buffer.enabled:
            effective_rpms = delay_buffer.step(rpms)
        else:
            effective_rpms = rpms

        obs, _, _, _, _ = aviary.step(effective_rpms)

        # Check for falls (altitude < 0.10m)
        if np.any(obs[:, 2] < 0.10):
            any_fell = True

    aviary.close()

    metrics = compute_tracking_rmse(actual_positions, ref_positions)
    rmse = float(metrics["swarm_mean_rmse"])
    mean_conf = float(np.mean(conf_history)) if conf_history else 0.0
    mean_alpha = (
        float(np.mean(alpha_history))
        if alpha_history
        else (1.0 if controller_class == FeedforwardSwarmController else 0.0)
    )

    if return_details:
        return {
            "rmse": rmse,
            "mean_confidence": mean_conf,
            "mean_blend_weight": mean_alpha,
            "any_fell": any_fell,
            "any_saturated": any_saturated,
        }
    return rmse


def execute_robustness_study(
    output_dir: str = "results",
    scenario_name: str = "hover_column_3d",
) -> pd.DataFrame:
    """Execute the complete Phase 6 robustness sweep."""
    os.makedirs(output_dir, exist_ok=True)
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    with open("configs/downwash_true.yaml", "r", encoding="utf-8") as f:
        true_cfg = yaml.safe_load(f)
    true_model = DownwashModel(true_cfg)

    with open("configs/downwash_assumed.yaml", "r", encoding="utf-8") as f:
        nominal_assumed_cfg = yaml.safe_load(f)

    scenario_gen = HoverColumnFormation(num_drones=3, z_top=1.8, dz=0.6, duration=4.0)

    sweep_records: List[Dict[str, Any]] = []

    # ==============================================================================
    # 1. Sweep 1: Force Magnitude Mismatch (0.0x to 2.0x)
    # ==============================================================================
    print("\n[PHASE 6] --- 1. Running Force Mismatch Sweep ---")
    force_ratios = np.array([0.0, 0.25, 0.50, 0.75, 1.0, 1.25, 1.50, 1.75, 2.0])
    nom_f = float(nominal_assumed_cfg["max_force_z"])

    rmse_force_sweep = {
        "Controller_A_Naive": np.zeros(len(force_ratios)),
        "Controller_B_SoftAvoid": np.zeros(len(force_ratios)),
        "Controller_C_Feedforward": np.zeros(len(force_ratios)),
        "Controller_D_UA_ADC": np.zeros(len(force_ratios)),
    }

    # Naive baseline (constant across sweep)
    rmse_naive = run_robustness_trial(
        scenario_gen=scenario_gen,
        controller_class=NaiveSwarmController,
        assumed_config=nominal_assumed_cfg,
        true_downwash_model=true_model,
    )
    rmse_force_sweep["Controller_A_Naive"][:] = rmse_naive

    # Soft Avoidance baseline
    rmse_soft = run_robustness_trial(
        scenario_gen=scenario_gen,
        controller_class=SoftAvoidSwarmController,
        assumed_config=nominal_assumed_cfg,
        true_downwash_model=true_model,
        ctrl_kwargs={"config": {"max_displacement": 0.20, "max_velocity": 0.15}},
    )
    rmse_force_sweep["Controller_B_SoftAvoid"][:] = rmse_soft

    crossover_ratio = None
    for idx, alpha in enumerate(force_ratios):
        cfg_perturbed = nominal_assumed_cfg.copy()
        cfg_perturbed["max_force_z"] = nom_f * alpha

        rmse_ff = run_robustness_trial(
            scenario_gen=scenario_gen,
            controller_class=FeedforwardSwarmController,
            assumed_config=cfg_perturbed,
            true_downwash_model=true_model,
        )
        rmse_force_sweep["Controller_C_Feedforward"][idx] = rmse_ff

        res_ua = run_robustness_trial(
            scenario_gen=scenario_gen,
            controller_class=UAAdaptiveController,
            assumed_config=cfg_perturbed,
            true_downwash_model=true_model,
            return_details=True,
        )
        rmse_force_sweep["Controller_D_UA_ADC"][idx] = res_ua["rmse"]

        print(
            f"  alpha_force={alpha:.2f}: Naive={rmse_naive*100:.2f}cm, "
            f"SoftAvoid={rmse_soft*100:.2f}cm, Feedforward={rmse_ff*100:.2f}cm, "
            f"UA-ADC={res_ua['rmse']*100:.2f}cm (c={res_ua['mean_confidence']:.2f}, blend={res_ua['mean_blend_weight']:.2f})"
        )

        sweep_records.append({
            "sweep_type": "force_mismatch",
            "alpha_force": alpha,
            "alpha_radius": 1.0,
            "noise_enabled": False,
            "delay_steps": 0,
            "rmse_naive": rmse_naive,
            "rmse_soft_avoid": rmse_soft,
            "rmse_feedforward": rmse_ff,
            "rmse_ua_adc": res_ua["rmse"],
            "conf_mean_ua_adc": res_ua["mean_confidence"],
            "blend_weight_mean_ua_adc": res_ua["mean_blend_weight"],
        })

        # Check for break-even crossover (where Feedforward degrades worse than Soft-Avoid)
        if alpha > 1.0 and rmse_ff >= rmse_soft and crossover_ratio is None:
            crossover_ratio = alpha

    plot_force_mismatch_sensitivity(
        mismatch_ratios=force_ratios,
        rmse_data=rmse_force_sweep,
        crossover_ratio=crossover_ratio,
        output_dir=figures_dir,
        scenario_name=scenario_name,
    )

    # ==============================================================================
    # 2. Sweep 2: Wake Radius Mismatch (0.5x to 1.5x)
    # ==============================================================================
    print("\n[PHASE 6] --- 2. Running Radius Mismatch Sweep ---")
    radius_ratios = np.array([0.50, 0.75, 1.0, 1.25, 1.50])
    nom_r = float(nominal_assumed_cfg["radius_xy"])

    rmse_radius_sweep = {
        "Controller_A_Naive": np.full(len(radius_ratios), rmse_naive),
        "Controller_B_SoftAvoid": np.zeros(len(radius_ratios)),
        "Controller_C_Feedforward": np.zeros(len(radius_ratios)),
        "Controller_D_UA_ADC": np.zeros(len(radius_ratios)),
    }

    for idx, beta in enumerate(radius_ratios):
        cfg_perturbed = nominal_assumed_cfg.copy()
        cfg_perturbed["radius_xy"] = nom_r * beta

        rmse_soft_r = run_robustness_trial(
            scenario_gen=scenario_gen,
            controller_class=SoftAvoidSwarmController,
            assumed_config=cfg_perturbed,
            true_downwash_model=true_model,
            ctrl_kwargs={"config": {"max_displacement": 0.20, "max_velocity": 0.15}},
        )
        rmse_ff_r = run_robustness_trial(
            scenario_gen=scenario_gen,
            controller_class=FeedforwardSwarmController,
            assumed_config=cfg_perturbed,
            true_downwash_model=true_model,
        )
        res_ua_r = run_robustness_trial(
            scenario_gen=scenario_gen,
            controller_class=UAAdaptiveController,
            assumed_config=cfg_perturbed,
            true_downwash_model=true_model,
            return_details=True,
        )

        rmse_radius_sweep["Controller_B_SoftAvoid"][idx] = rmse_soft_r
        rmse_radius_sweep["Controller_C_Feedforward"][idx] = rmse_ff_r
        rmse_radius_sweep["Controller_D_UA_ADC"][idx] = res_ua_r["rmse"]

        print(
            f"  alpha_radius={beta:.2f}: SoftAvoid={rmse_soft_r*100:.2f}cm, "
            f"Feedforward={rmse_ff_r*100:.2f}cm, UA-ADC={res_ua_r['rmse']*100:.2f}cm "
            f"(c={res_ua_r['mean_confidence']:.2f}, blend={res_ua_r['mean_blend_weight']:.2f})"
        )

        sweep_records.append({
            "sweep_type": "radius_mismatch",
            "alpha_force": 1.0,
            "alpha_radius": beta,
            "noise_enabled": False,
            "delay_steps": 0,
            "rmse_naive": rmse_naive,
            "rmse_soft_avoid": rmse_soft_r,
            "rmse_feedforward": rmse_ff_r,
            "rmse_ua_adc": res_ua_r["rmse"],
            "conf_mean_ua_adc": res_ua_r["mean_confidence"],
            "blend_weight_mean_ua_adc": res_ua_r["mean_blend_weight"],
        })

    plot_radius_mismatch_sensitivity(
        radius_ratios=radius_ratios,
        rmse_data=rmse_radius_sweep,
        output_dir=figures_dir,
        scenario_name=scenario_name,
    )

    # ==============================================================================
    # 3. Sweep 3: Sensor Noise & Command Latency Delay
    # ==============================================================================
    print("\n[PHASE 6] --- 3. Running Noise & Delay Environmental Sweep ---")
    conditions = ["Clean", "Sensor Noise", "Latency 21ms", "Latency 42ms", "Stress Test"]
    cond_results: Dict[str, List[float]] = {
        "Controller_A_Naive": [],
        "Controller_B_SoftAvoid": [],
        "Controller_C_Feedforward": [],
        "Controller_D_UA_ADC": [],
    }

    test_cases = [
        # (noise_enabled, delay_steps, alpha_force, label)
        (False, 0, 1.0, "Clean"),
        (True, 0, 1.0, "Sensor Noise"),
        (False, 1, 1.0, "Latency 21ms"),
        (False, 2, 1.0, "Latency 42ms"),
        (True, 2, 1.3, "Stress Test"),
    ]

    for noise_on, delay_s, alpha_f, cond_label in test_cases:
        noise = GaussianObservationNoise(config={"enabled": noise_on}, seed=42)
        delay = CommandDelayBuffer(delay_steps=delay_s, enabled=(delay_s > 0))

        cfg_p = nominal_assumed_cfg.copy()
        cfg_p["max_force_z"] = nom_f * alpha_f

        r_naive = run_robustness_trial(
            scenario_gen, NaiveSwarmController, nominal_assumed_cfg, true_model,
            noise_wrapper=noise, delay_buffer=delay,
        )
        r_soft = run_robustness_trial(
            scenario_gen, SoftAvoidSwarmController, nominal_assumed_cfg, true_model,
            noise_wrapper=noise, delay_buffer=delay,
            ctrl_kwargs={"config": {"max_displacement": 0.20, "max_velocity": 0.15}},
        )
        r_ff = run_robustness_trial(
            scenario_gen, FeedforwardSwarmController, cfg_p, true_model,
            noise_wrapper=noise, delay_buffer=delay,
        )
        res_ua_env = run_robustness_trial(
            scenario_gen, UAAdaptiveController, cfg_p, true_model,
            noise_wrapper=noise, delay_buffer=delay,
            return_details=True,
        )

        cond_results["Controller_A_Naive"].append(r_naive)
        cond_results["Controller_B_SoftAvoid"].append(r_soft)
        cond_results["Controller_C_Feedforward"].append(r_ff)
        cond_results["Controller_D_UA_ADC"].append(res_ua_env["rmse"])

        print(
            f"  {cond_label:<14}: Naive={r_naive*100:.2f}cm, SoftAvoid={r_soft*100:.2f}cm, "
            f"Feedforward={r_ff*100:.2f}cm, UA-ADC={res_ua_env['rmse']*100:.2f}cm "
            f"(c={res_ua_env['mean_confidence']:.2f}, blend={res_ua_env['mean_blend_weight']:.2f})"
        )

        if cond_label == "Stress Test":
            print(
                f"\n  [STRESS CASE ASSESSMENT] UA-ADC under 1.3x force + noise + 2-step delay: "
                f"Any drone fell: {res_ua_env['any_fell']} | Any drone saturated: {res_ua_env['any_saturated']}\n"
            )

        sweep_records.append({
            "sweep_type": "environmental_degradation",
            "alpha_force": alpha_f,
            "alpha_radius": 1.0,
            "noise_enabled": noise_on,
            "delay_steps": delay_s,
            "rmse_naive": r_naive,
            "rmse_soft_avoid": r_soft,
            "rmse_feedforward": r_ff,
            "rmse_ua_adc": res_ua_env["rmse"],
            "conf_mean_ua_adc": res_ua_env["mean_confidence"],
            "blend_weight_mean_ua_adc": res_ua_env["mean_blend_weight"],
            "any_fell": res_ua_env["any_fell"],
            "any_saturated": res_ua_env["any_saturated"],
        })

    plot_noise_delay_comparison(
        conditions=conditions,
        results_by_ctrl=cond_results,
        output_dir=figures_dir,
        scenario_name=scenario_name,
    )

    # ==============================================================================
    # 4. Sweep 4: 2D Sensitivity Heatmap (Force vs Radius Mismatch for Feedforward)
    # ==============================================================================
    print("\n[PHASE 6] --- 4. Running 2D Parameter Grid Sweep for Feedforward ---")
    grid_force = np.linspace(0.5, 1.5, 5)
    grid_radius = np.linspace(0.6, 1.4, 5)
    grid_rmse = np.zeros((len(grid_radius), len(grid_force)))

    for r_idx, b in enumerate(grid_radius):
        for f_idx, a in enumerate(grid_force):
            cfg_p = nominal_assumed_cfg.copy()
            cfg_p["max_force_z"] = nom_f * a
            cfg_p["radius_xy"] = nom_r * b

            err = run_robustness_trial(
                scenario_gen=scenario_gen,
                controller_class=FeedforwardSwarmController,
                assumed_config=cfg_p,
                true_downwash_model=true_model,
            )
            grid_rmse[r_idx, f_idx] = err

    plot_2d_robustness_heatmap(
        force_ratios=grid_force,
        radius_ratios=grid_radius,
        rmse_grid=grid_rmse,
        controller_name="Feedforward",
        output_dir=figures_dir,
        scenario_name=scenario_name,
    )

    # ==============================================================================
    # Save Consolidated Sweep Results to CSV
    # ==============================================================================
    df_sweep = pd.DataFrame(sweep_records)
    csv_path = os.path.join(output_dir, "robustness_sweep.csv")
    csv_path_all4 = os.path.join(output_dir, "robustness_sweep_all4.csv")
    df_sweep.to_csv(csv_path, index=False)
    df_sweep.to_csv(csv_path_all4, index=False)
    print(f"\n[SUCCESS] Phase 6 Robustness Sweep complete! Results saved to {csv_path} and {csv_path_all4}")

    return df_sweep


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 6 Robustness Sweep Runner")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--scenario", type=str, default="hover_column_3d")
    parser.add_argument(
        "--controller",
        type=str,
        default="all",
        choices=["all", "naive", "soft_avoid", "feedforward", "ua_adc"],
        help="Controller to evaluate (default: 'all').",
    )
    args = parser.parse_args()

    execute_robustness_study(output_dir=args.output_dir, scenario_name=args.scenario)


if __name__ == "__main__":
    main()
