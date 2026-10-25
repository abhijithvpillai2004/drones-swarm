import sys, os, time, yaml
from pathlib import Path
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from src.models.downwash import DownwashModel
from src.controllers.ua_adc import UAAdaptiveController
from src.experiments.multi_drone_comparison import load_true_downwash_model
from src.experiments.scaling_sweep import run_scaling_trial

true_model = load_true_downwash_model("configs/downwash_true.yaml")

print("=" * 80)
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
            "sim_runtime_sec": res.get("sim_runtime_sec", 0.0),
        })

df_ua_scale = pd.DataFrame(ua_scaling_rows)
df_scale_all = pd.concat([df_scale_existing, df_ua_scale], ignore_index=True)
df_scale_all.to_csv("results/scaling_sweep_results_all4.csv", index=False)

agg = df_scale_all.groupby(["num_drones", "controller"])[["rmse_mean", "rmse_max", "min_interdrone_dist", "near_collision_steps", "total_energy_proxy"]].mean().reset_index()
print("\n--- AGGREGATE SCALING SWEEP TABLE (ALL 4 CONTROLLERS) ---")
print(agg.to_string())

print("\n>>> ALL STEP 4 BENCHMARKS SUCCESSFULLY EXECUTED AND SAVED <<<")
