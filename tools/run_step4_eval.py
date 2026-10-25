import sys, os, time, yaml
from pathlib import Path
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from src.models.downwash import DownwashModel
from src.sim.downwash_aviary import DownwashAviary
from src.controllers.naive import NaiveSwarmController
from src.controllers.soft_avoid import SoftAvoidSwarmController
from src.controllers.feedforward import FeedforwardSwarmController
from src.controllers.ua_adc import UAAdaptiveController
from src.formations.generators import HoverColumnFormation, validate_drone_count
from src.experiments.multi_drone_comparison import load_true_downwash_model
from src.experiments.robustness_sweep import run_robustness_trial
from src.experiments.scaling_sweep import run_scaling_trial
from src.noise.wrappers import GaussianObservationNoise, CommandDelayBuffer

true_model = load_true_downwash_model("configs/downwash_true.yaml")
with open("configs/downwash_assumed.yaml", "r", encoding="utf-8") as f:
    assumed_cfg = yaml.safe_load(f)

print("=" * 80)
print("PART 2: ROBUSTNESS / MODEL-MISMATCH SWEEP WITH UA-ADC INCLUDED")
print("=" * 80)

df_rob = pd.read_csv("results/robustness_sweep.csv")
rob_scenario = HoverColumnFormation(num_drones=3, z_top=1.8, dz=0.6, duration=4.0)

ua_adc_rmses = []
nom_f = float(assumed_cfg["max_force_z"])
nom_r = float(assumed_cfg["radius_xy"])

for idx, row in df_rob.iterrows():
    sw_type = row["sweep_type"]
    a_f = float(row["alpha_force"])
    a_r = float(row["alpha_radius"])
    noise_on = bool(row["noise_enabled"])
    del_steps = int(row["delay_steps"])

    cfg_pert = assumed_cfg.copy()
    cfg_pert["max_force_z"] = nom_f * a_f
    cfg_pert["radius_xy"] = nom_r * a_r

    nw = GaussianObservationNoise(config={"enabled": noise_on}, seed=42)
    db = CommandDelayBuffer(delay_steps=del_steps, enabled=(del_steps > 0))

    rmse_val = run_robustness_trial(
        scenario_gen=rob_scenario,
        controller_class=UAAdaptiveController,
        assumed_config=cfg_pert,
        true_downwash_model=true_model,
        noise_wrapper=nw,
        delay_buffer=db,
        ctrl_kwargs={"assumed_config": cfg_pert},
    )
    ua_adc_rmses.append(rmse_val)
    print(f"Row {idx:02d} [{sw_type}] a_f={a_f:.2f}, a_r={a_r:.2f}, noise={noise_on}, delay={del_steps} -> UA-ADC RMSE = {rmse_val*100:.3f} cm")

df_rob["rmse_ua_adc"] = ua_adc_rmses
df_rob.to_csv("results/robustness_sweep_all4.csv", index=False)
print("\n--- FULL ROBUSTNESS SWEEP TABLE ---")
print(df_rob.to_string())

print("\n" + "=" * 80)
print("PART 3: SCALING SWEEP (N=2, 3, 5, 7, 10) WITH UA-ADC INCLUDED")
print("=" * 80)

df_scale_existing = pd.read_csv("results/scaling_sweep_results.csv")
drone_counts = [2, 3, 5, 7, 10]
seeds = df_scale_existing["seed"].unique().tolist()[:20]

ua_scaling_rows = []
for n in drone_counts:
    print(f"Running scaling N={n} for 20 seeds with UA-ADC ...")
    for seed in seeds:
        res = run_scaling_trial(
            num_drones=n,
            controller_name="Controller_D_UAADC",
            controller_class=UAAdaptiveController,
            ctrl_kwargs={"assumed_config": "configs/downwash_assumed.yaml"},
            true_downwash_model=true_model,
            seed=seed,
            duration=4.0,
            output_dir="results",
        )
        ua_scaling_rows.append({
            "num_drones": n,
            "controller": "Controller_D_UAADC",
            "seed": seed,
            "rmse_mean": res["rmse_mean"],
            "rmse_max": res["rmse_max"],
            "min_interdrone_dist": res["min_interdrone_dist"],
            "near_collision_steps": res["near_collision_steps"],
            "total_energy_proxy": res["total_energy_proxy"],
            "wall_clock_time": res["wall_clock_time"],
        })

df_ua_scale = pd.DataFrame(ua_scaling_rows)
df_scale_all = pd.concat([df_scale_existing, df_ua_scale], ignore_index=True)
df_scale_all.to_csv("results/scaling_sweep_results_all4.csv", index=False)

agg = df_scale_all.groupby(["num_drones", "controller"])[["rmse_mean", "rmse_max", "min_interdrone_dist", "near_collision_steps", "total_energy_proxy"]].mean().reset_index()
print("\n--- AGGREGATE SCALING SWEEP TABLE (ALL 4 CONTROLLERS) ---")
print(agg.to_string())

print("\n>>> ALL STEP 4 BENCHMARKS SUCCESSFULLY EXECUTED AND SAVED <<<")
