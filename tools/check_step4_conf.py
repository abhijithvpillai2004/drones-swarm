import sys, os, yaml
from pathlib import Path
import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from src.models.downwash import DownwashModel
from src.sim.downwash_aviary import DownwashAviary
from src.controllers.ua_adc import UAAdaptiveController
from src.formations.generators import HoverColumnFormation
from src.experiments.multi_drone_comparison import load_true_downwash_model

true_model = load_true_downwash_model("configs/downwash_true.yaml")
with open("configs/downwash_assumed.yaml", "r", encoding="utf-8") as f:
    assumed_cfg = yaml.safe_load(f)

# The exact Step 4 robustness scenario from tools/run_step4_eval.py line 32:
rob_scenario = HoverColumnFormation(num_drones=3, z_top=1.8, dz=0.6, duration=4.0)

dt = 1.0 / 48
total_steps = int(4.0 * 48)

aviary = DownwashAviary(
    num_drones=3,
    initial_xyzs=rob_scenario.get_initial_positions(),
    ctrl_freq=48,
    pyb_freq=240,
    gui=False,
    downwash_model=true_model,
)

ctrl = UAAdaptiveController(num_drones=3, assumed_config=assumed_cfg)
obs, _ = aviary.reset()

logged_confidences = []
logged_blend_weights = []

for step in range(total_steps):
    t = step * dt
    target_pos, target_vel = rob_scenario.evaluate(t)
    rpms, _, _ = ctrl.compute_actions(dt, obs, target_pos, target_vel)
    logged_confidences.append(ctrl.last_confidences.copy())
    logged_blend_weights.append(ctrl.last_blend_weights.copy())
    obs, _, _, _, _ = aviary.step(rpms)

aviary.close()

conf_arr = np.array(logged_confidences)
blend_arr = np.array(logged_blend_weights)

print("="*80)
print("RECONCILING STEP 4 TABLE CLAIM (alpha_force = 1.0)")
print("="*80)
print(f"Max confidence across all drones and all steps: {conf_arr.max():.6f}")
print(f"Mean confidence across all drones: {conf_arr.mean():.6f}")
print(f"Final confidence at t=4.0s: {conf_arr[-1]}")
print(f"Max blend weight across all drones: {blend_arr.max():.6f}")
print(f"Mean blend weight across all drones: {blend_arr.mean():.6f}")
print(f"Final blend weight at t=4.0s: {blend_arr[-1]}")
