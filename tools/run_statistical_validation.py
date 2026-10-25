import sys, os, time, yaml
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

sys.stdout.reconfigure(encoding='utf-8')
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from src.models.downwash import DownwashModel
from src.controllers.soft_avoid import SoftAvoidSwarmController
from src.controllers.feedforward import FeedforwardSwarmController
from src.controllers.ua_adc import UAAdaptiveController
from src.formations.generators import VerticalSwapTrajectory, HoverColumnFormation
from src.experiments.multi_drone_comparison import load_true_downwash_model, run_single_trial
from src.experiments.robustness_sweep import run_robustness_trial
from src.noise.wrappers import CommandDelayBuffer

true_model = load_true_downwash_model("configs/downwash_true.yaml")
with open("configs/downwash_assumed.yaml", "r", encoding="utf-8") as f:
    assumed_cfg = yaml.safe_load(f)

def safe_wilcoxon(x, y):
    diff = np.array(x) - np.array(y)
    if np.all(diff == 0):
        return 1.0
    try:
        res = stats.wilcoxon(x, y, zero_method="pratt")
        return float(res.pvalue)
    except Exception:
        return 1.0

seeds = list(range(42, 62)) # 20 seeds

print("=" * 80)
print("EXPERIMENT 1: VERTICAL SWAP (20 SEEDS)")
print("=" * 80)

vswap_results = {"SoftAvoid": {"rmse": [], "min_dist": []},
                 "Feedforward": {"rmse": [], "min_dist": []},
                 "UA_ADC": {"rmse": [], "min_dist": []}}

for s in seeds:
    gen = VerticalSwapTrajectory(num_drones=2, z_high=1.6, z_low=0.8, duration=6.0)
    
    # SoftAvoid
    res_sa = run_single_trial("vswap", gen, "SoftAvoid", SoftAvoidSwarmController,
                              {"config": {"max_displacement": 0.20, "max_velocity": 0.15}},
                              true_model, output_dir="results", seed=s, gui=False, randomize=True)
    vswap_results["SoftAvoid"]["rmse"].append(res_sa["summary"]["rmse_mean"])
    vswap_results["SoftAvoid"]["min_dist"].append(res_sa["summary"]["min_interdrone_dist"])
    
    # Feedforward
    res_ff = run_single_trial("vswap", gen, "Feedforward", FeedforwardSwarmController,
                              {}, true_model, output_dir="results", seed=s, gui=False, randomize=True)
    vswap_results["Feedforward"]["rmse"].append(res_ff["summary"]["rmse_mean"])
    vswap_results["Feedforward"]["min_dist"].append(res_ff["summary"]["min_interdrone_dist"])
    
    # UA-ADC
    res_ua = run_single_trial("vswap", gen, "UA_ADC", UAAdaptiveController,
                              {"assumed_config": "configs/downwash_assumed.yaml"},
                              true_model, output_dir="results", seed=s, gui=False, randomize=True)
    vswap_results["UA_ADC"]["rmse"].append(res_ua["summary"]["rmse_mean"])
    vswap_results["UA_ADC"]["min_dist"].append(res_ua["summary"]["min_interdrone_dist"])

print("\n--- VERTICAL SWAP 20-SEED RESULTS ---")
for ctrl in ["SoftAvoid", "Feedforward", "UA_ADC"]:
    r = vswap_results[ctrl]["rmse"]
    d = vswap_results[ctrl]["min_dist"]
    print(f"{ctrl:12s}: RMSE = {np.mean(r):.4f} +/- {np.std(r):.4f} m | Min Dist = {np.mean(d):.4f} +/- {np.std(d):.4f} m")

p_vswap_sa_rmse = safe_wilcoxon(vswap_results["UA_ADC"]["rmse"], vswap_results["SoftAvoid"]["rmse"])
p_vswap_ff_rmse = safe_wilcoxon(vswap_results["UA_ADC"]["rmse"], vswap_results["Feedforward"]["rmse"])
p_vswap_sa_dist = safe_wilcoxon(vswap_results["UA_ADC"]["min_dist"], vswap_results["SoftAvoid"]["min_dist"])
p_vswap_ff_dist = safe_wilcoxon(vswap_results["UA_ADC"]["min_dist"], vswap_results["Feedforward"]["min_dist"])

print(f"\nWilcoxon p-values for UA-ADC vs Soft-Avoid:   RMSE p = {p_vswap_sa_rmse:.4e}, MinDist p = {p_vswap_sa_dist:.4e}")
print(f"Wilcoxon p-values for UA-ADC vs Feedforward:  RMSE p = {p_vswap_ff_rmse:.4e}, MinDist p = {p_vswap_ff_dist:.4e}")

print("\n" + "=" * 80)
print("EXPERIMENT 2: LATENCY CASE (2-STEP DELAY, 20 SEEDS)")
print("=" * 80)

lat_results = {"SoftAvoid": [], "Feedforward": [], "UA_ADC": []}
lat_gen = HoverColumnFormation(num_drones=3, z_top=1.8, dz=0.6, duration=4.0)

for s in seeds:
    np.random.seed(s)
    db = CommandDelayBuffer(delay_steps=2, enabled=True)
    
    r_sa = run_robustness_trial(lat_gen, SoftAvoidSwarmController, assumed_cfg, true_model,
                                delay_buffer=db, ctrl_kwargs={"config": {"max_displacement": 0.20, "max_velocity": 0.15}})
    lat_results["SoftAvoid"].append(r_sa)
    
    db.reset()
    r_ff = run_robustness_trial(lat_gen, FeedforwardSwarmController, assumed_cfg, true_model,
                                delay_buffer=db)
    lat_results["Feedforward"].append(r_ff)
    
    db.reset()
    r_ua = run_robustness_trial(lat_gen, UAAdaptiveController, assumed_cfg, true_model,
                                delay_buffer=db, ctrl_kwargs={"assumed_config": assumed_cfg})
    lat_results["UA_ADC"].append(r_ua)

print("\n--- LATENCY (2-STEP DELAY) 20-SEED RESULTS ---")
for ctrl in ["SoftAvoid", "Feedforward", "UA_ADC"]:
    r = lat_results[ctrl]
    print(f"{ctrl:12s}: RMSE = {np.mean(r):.4f} +/- {np.std(r):.4f} m")

p_lat_sa = safe_wilcoxon(lat_results["UA_ADC"], lat_results["SoftAvoid"])
p_lat_ff = safe_wilcoxon(lat_results["UA_ADC"], lat_results["Feedforward"])
print(f"\nWilcoxon p-values for UA-ADC vs Soft-Avoid:   p = {p_lat_sa:.4e}")
print(f"Wilcoxon p-values for UA-ADC vs Feedforward:  p = {p_lat_ff:.4e}")

print("\n" + "=" * 80)
print("EXPERIMENT 3: SCALING SWEEP (N in [2, 3, 5, 7, 10], 20 SEEDS)")
print("=" * 80)

df_scale = pd.read_csv("results/scaling_sweep_results_all4.csv")
for n in [2, 3, 5, 7, 10]:
    sub = df_scale[df_scale["num_drones"] == n]
    sa_sub = sub[sub["controller"] == "Controller_B_SoftAvoid"].sort_values("seed")
    ff_sub = sub[sub["controller"] == "Controller_C_Feedforward"].sort_values("seed")
    ua_sub = sub[sub["controller"] == "Controller_D_UAADC"].sort_values("seed")
    
    r_sa = sa_sub["rmse_mean"].values
    r_ff = ff_sub["rmse_mean"].values
    r_ua = ua_sub["rmse_mean"].values
    
    p_sa = safe_wilcoxon(r_ua, r_sa)
    p_ff = safe_wilcoxon(r_ua, r_ff)
    
    print(f"N = {n:2d}:")
    print(f"  SoftAvoid:   {np.mean(r_sa):.4f} +/- {np.std(r_sa):.4f} m")
    print(f"  Feedforward: {np.mean(r_ff):.4f} +/- {np.std(r_ff):.4f} m")
    print(f"  UA-ADC:      {np.mean(r_ua):.4f} +/- {np.std(r_ua):.4f} m")
    print(f"  p(UA-ADC vs SoftAvoid) = {p_sa:.4e} | p(UA-ADC vs Feedforward) = {p_ff:.4e}")
