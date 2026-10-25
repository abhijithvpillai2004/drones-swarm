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
from src.experiments.multi_drone_comparison import (
    load_true_downwash_model,
    create_scenario_generator,
    run_single_trial,
    SCENARIO_NAMES,
)

true_model = load_true_downwash_model("configs/downwash_true.yaml")

print("=" * 80)
print("UA-ADC FULL REGRESSION SWEEP: EVERY SCENARIO x N in [2..10]")
print("=" * 80)

scenarios = [
    "hover_column_3d",
    "crossing_pair",
    "vertical_swap",
    "orbit_stack",
    "line",
    "v_formation",
    "grid",
    "staggered",
]
drone_counts = list(range(2, 11))

results_list = []

for sc in scenarios:
    print(f"\n>>> Running Scenario: {sc.upper()} across N=2..10")
    for n in drone_counts:
        try:
            gen = create_scenario_generator(sc, num_drones=n)
            # Short 2.0s dynamic simulation to evaluate stability and collision avoidance
            gen.duration = 2.5
            trial = run_single_trial(
                scenario_name=f"reg_{sc}_N{n}",
                scenario_gen=gen,
                controller_name="UA_ADC",
                controller_class=UAAdaptiveController,
                ctrl_kwargs={"assumed_config": "configs/downwash_assumed.yaml"},
                true_downwash_model=true_model,
                output_dir="results",
                gui=False,
            )
            s = trial["summary"]
            actual = trial["actual_positions"]

            # Validation criteria
            no_nan = bool(not np.any(np.isnan(actual)))
            no_inf = bool(not np.any(np.isinf(actual)))
            above_ground = bool(np.all(actual[:, :, 2] > 0.05))
            safe_separation = bool(s["min_interdrone_dist"] > 0.04) # physical collision bound
            bounded_rmse = bool(s["rmse_mean"] < 0.40)

            passed = no_nan and no_inf and above_ground and safe_separation and bounded_rmse
            status = "PASS" if passed else "FAIL"

            results_list.append({
                "scenario": sc,
                "N": n,
                "status": status,
                "rmse_mean_m": s["rmse_mean"],
                "rmse_max_m": s["rmse_max"],
                "min_dist_m": s["min_interdrone_dist"],
                "near_col_steps": s["near_collision_steps"],
                "no_nan": no_nan,
                "above_ground": above_ground,
                "reason": "OK" if passed else (
                    "NaN" if not (no_nan and no_inf) else
                    "Ground crash" if not above_ground else
                    "Collision (<4cm)" if not safe_separation else
                    f"High RMSE ({s['rmse_mean']:.3f}m)"
                ),
            })
            print(f"  [{status}] {sc} (N={n:2d}): Mean RMSE = {s['rmse_mean']*100:5.2f} cm | Min Dist = {s['min_interdrone_dist']:5.3f} m | Near-Col = {s['near_collision_steps']}")
        except Exception as e:
            results_list.append({
                "scenario": sc,
                "N": n,
                "status": "FAIL",
                "rmse_mean_m": 999.0,
                "rmse_max_m": 999.0,
                "min_dist_m": 0.0,
                "near_col_steps": 999,
                "no_nan": False,
                "above_ground": False,
                "reason": f"Exception: {str(e)[:40]}",
            })
            print(f"  [FAIL] {sc} (N={n:2d}): Exception: {e}")

df_results = pd.DataFrame(results_list)
df_results.to_csv("results/ua_adc_full_regression_sweep.csv", index=False)

print("\n" + "=" * 80)
print("REGRESSION SWEEP SUMMARY TABLE")
print("=" * 80)
pivot_status = df_results.pivot(index="scenario", columns="N", values="status")
print(pivot_status.to_string())

total_runs = len(df_results)
passed_runs = (df_results["status"] == "PASS").sum()
print(f"\nTOTAL RUNS: {total_runs} | PASSED: {passed_runs} | FAILED: {total_runs - passed_runs}")
print("Saved complete regression log to results/ua_adc_full_regression_sweep.csv")
