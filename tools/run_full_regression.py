"""Full Regression Sweep Runner for Controller D (UA-ADC).

Runs all 8 scenarios across swarm sizes N in [2, 10].
Checks no_nan, above_ground (> 0.05m), min_interdrone_dist, tracking RMSE,
and records mean confidence and mean blend weight.
Saves results to results/ua_adc_full_regression_sweep.csv.
"""
from pathlib import Path
import os
import sys
import yaml
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.downwash import DownwashModel
from src.sim.downwash_aviary import DownwashAviary
from src.controllers.ua_adc import UAAdaptiveController
from src.experiments.multi_drone_comparison import create_scenario_generator, SCENARIO_NAMES
from src.metrics.swarm_metrics import compute_all_metrics


def run_full_regression(output_csv: str = "results/ua_adc_full_regression_sweep.csv") -> pd.DataFrame:
    with open("configs/downwash_true.yaml", "r", encoding="utf-8") as f:
        true_cfg = yaml.safe_load(f)
    with open("configs/downwash_assumed.yaml", "r", encoding="utf-8") as f:
        assumed_cfg = yaml.safe_load(f)
    with open("configs/ua_adc.yaml", "r", encoding="utf-8") as f:
        ua_cfg = yaml.safe_load(f)

    true_model = DownwashModel(true_cfg)
    results = []

    print("\n" + "=" * 80)
    print("  LAUNCHING UA-ADC FULL REGRESSION SWEEP (8 Scenarios x N in [2..10])")
    print("=" * 80)

    for sc in SCENARIO_NAMES:
        for n in range(2, 11):
            try:
                gen = create_scenario_generator(sc, num_drones=n)
                init_xyzs = gen.get_initial_positions()
                dt = 1.0 / 48
                steps = int(gen.duration * 48)

                aviary = DownwashAviary(
                    num_drones=n,
                    initial_xyzs=init_xyzs,
                    ctrl_freq=48,
                    pyb_freq=240,
                    gui=False,
                    record=False,
                    downwash_model=true_model,
                )
                ctrl = UAAdaptiveController(config=ua_cfg, num_drones=n, assumed_config=assumed_cfg)
                obs, _ = aviary.reset()
                ctrl.reset()

                act_pos = np.zeros((steps, n, 3))
                ref_pos = np.zeros((steps, n, 3))
                rpms = np.zeros((steps, n, 4))
                ts = np.zeros(steps)
                conf_history = []
                blend_history = []

                for step in range(steps):
                    t = step * dt
                    ts[step] = t
                    tgt_p, tgt_v = gen.evaluate(t)
                    ref_pos[step] = tgt_p
                    act_pos[step] = obs[:, 0:3]

                    rpm, _, _ = ctrl.compute_actions(dt, obs, tgt_p, tgt_v)
                    rpms[step] = rpm
                    conf_history.append(ctrl.last_confidences.copy())
                    blend_history.append(ctrl.last_blend_weights.copy())

                    obs, _, _, _, _ = aviary.step(rpm)
                aviary.close()

                m = compute_all_metrics(act_pos, ref_pos, ts, rpms, dt)
                no_nan = not np.isnan(act_pos).any()
                above_ground = bool((act_pos[:, :, 2] > 0.05).all())
                passed = no_nan and above_ground

                conf_mean = float(np.mean(conf_history)) if conf_history else 0.0
                blend_mean = float(np.mean(blend_history)) if blend_history else 0.0

                reason = "OK" if passed else ("Ground crash" if not above_ground else "NaN")
                results.append({
                    "scenario": sc,
                    "N": n,
                    "status": "PASS" if passed else "FAIL",
                    "rmse_mean_m": m["rmse_mean"],
                    "rmse_max_m": m["rmse_max"],
                    "min_dist_m": m["min_interdrone_dist"],
                    "near_col_steps": m["near_collision_steps"],
                    "confidence_mean": conf_mean,
                    "blend_weight_mean": blend_mean,
                    "no_nan": no_nan,
                    "above_ground": above_ground,
                    "reason": reason,
                })
                print(f"  {sc:18s} N={n:2d} -> {'PASS' if passed else 'FAIL'} | RMSE={m['rmse_mean']*100:.2f}cm | c={conf_mean:.2f} | blend={blend_mean:.2f} | {reason}")
            except Exception as e:
                results.append({
                    "scenario": sc,
                    "N": n,
                    "status": "ERROR",
                    "rmse_mean_m": float("nan"),
                    "rmse_max_m": float("nan"),
                    "min_dist_m": float("nan"),
                    "near_col_steps": -1,
                    "confidence_mean": 0.0,
                    "blend_weight_mean": 0.0,
                    "no_nan": False,
                    "above_ground": False,
                    "reason": str(e),
                })
                print(f"  {sc:18s} N={n:2d} -> ERROR: {e}")

    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\n[SUCCESS] Full regression complete! Saved {len(df)} records to {output_csv}")
    return df


if __name__ == "__main__":
    run_full_regression()
