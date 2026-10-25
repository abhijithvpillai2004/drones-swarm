import sys, os
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from src.models.downwash import DownwashModel
from src.sim.downwash_aviary import DownwashAviary
from src.controllers.ua_adc import UAAdaptiveController
from src.formations.generators import HoverColumnFormation
from src.experiments.multi_drone_comparison import load_true_downwash_model

# 1. Setup
duration = 10.0
ctrl_freq = 48
dt = 1.0 / ctrl_freq
total_steps = int(duration * ctrl_freq) # 480 steps
num_drones = 3

true_model = load_true_downwash_model("configs/downwash_true.yaml")
with open("configs/downwash_assumed.yaml", "r", encoding="utf-8") as f:
    assumed_cfg = yaml.safe_load(f)
assumed_model = DownwashModel(config=assumed_cfg)

with open("configs/ua_adc.yaml", "r", encoding="utf-8") as f:
    ua_cfg = yaml.safe_load(f)

# Hover column scenario: N=3, z_top=1.8, dz=0.45
scenario_gen = HoverColumnFormation(num_drones=num_drones, z_top=1.8, dz=0.45, duration=duration)
init_xyzs = scenario_gen.get_initial_positions()

aviary = DownwashAviary(
    num_drones=num_drones,
    initial_xyzs=init_xyzs,
    ctrl_freq=ctrl_freq,
    pyb_freq=240,
    gui=False,
    record=False,
    downwash_model=true_model,
)

ctrl = UAAdaptiveController(
    config=ua_cfg,
    num_drones=num_drones,
    assumed_config=assumed_cfg,
)
ctrl.reset()
obs, _ = aviary.reset()

# Drone parameters
mass_phys = aviary.M # PyBullet mass from URDF
kf = aviary.KF # thrust coefficient
g = 9.81 # gravity

records = []
prev_vel = None

for step in range(total_steps):
    t = round(step * dt, 5)
    target_pos, target_vel = scenario_gen.evaluate(t)
    states = obs.copy()
    
    positions = states[:, 0:3]
    velocities = states[:, 10:13]
    
    # Finite-difference acceleration (matching SwarmDisturbanceEstimator exactly)
    if prev_vel is None:
        az_obs = np.zeros(num_drones)
    else:
        az_obs = (velocities[:, 2] - prev_vel[:, 2]) / dt
    prev_vel = velocities.copy()
    
    # Assumed forces before step
    assumed_forces = assumed_model.compute_swarm_forces(positions, velocities)
    
    # UA-ADC compute actions
    rpms, pos_err, yaw_err = ctrl.compute_actions(
        control_timestep=dt,
        states=states,
        formation_targets=target_pos,
        formation_velocities=target_vel,
    )
    
    # Commanded thrust per drone
    # Total thrust = kf * sum(rpm^2)
    T_cmd = kf * np.sum(rpms**2, axis=1)
    
    # a_no_wash = (T_cmd - m*g)/m
    m_calc = ctrl.estimator.estimators[0].mass # 0.027
    a_no_wash = (T_cmd - m_calc * g) / m_calc
    
    # a_predicted = a_no_wash + F_assumed / m
    F_assumed_z = assumed_forces[:, 2]
    a_predicted = a_no_wash + (F_assumed_z / m_calc)
    
    residual = az_obs - a_predicted
    
    # Estimator outputs
    confidences = ctrl.last_confidences.copy()
    blend_weights = ctrl.last_blend_weights.copy()
    
    F_hat_z = np.zeros(num_drones)
    P_var = np.zeros(num_drones)
    theta_val = np.zeros(num_drones)
    for j in range(num_drones):
        est = ctrl.estimator.estimators[j]
        F_hat_z[j] = est.estimate_disturbance(assumed_forces[j])[2]
        P_var[j] = est.P
        theta_val[j] = est.theta
        
    # True force applied in aviary
    # Note: in step, aviary applies true downwash
    obs, _, _, _, _ = aviary.step(rpms)
    F_true_z = aviary.last_applied_disturbances[:, 2].copy()
    
    for j in range(num_drones):
        records.append({
            "step": step,
            "t": t,
            "drone_id": j,
            "z": positions[j, 2],
            "vz": velocities[j, 2],
            "az_obs": az_obs[j],
            "T_cmd": T_cmd[j],
            "m": m_calc,
            "a_no_wash": a_no_wash[j],
            "F_assumed": F_assumed_z[j],
            "a_predicted": a_predicted[j],
            "residual": residual[j],
            "F_hat": F_hat_z[j],
            "theta": theta_val[j],
            "P": P_var[j],
            "confidence": confidences[j],
            "blend_weight": blend_weights[j],
            "F_true": F_true_z[j],
        })

aviary.close()

df = pd.DataFrame(records)
out_csv1 = Path("results/diagnostics/ua_adc_trace_nominal.csv")
out_csv2 = Path("results/diagnostics/ua_adc/ua_adc_trace_nominal.csv")
out_csv1.parent.mkdir(parents=True, exist_ok=True)
out_csv2.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out_csv1, index=False)
df.to_csv(out_csv2, index=False)

print(f"Saved {len(df)} rows to {out_csv1} and {out_csv2}")
print(f"Physics mass: {mass_phys:.6f} kg, Controller mass: {m_calc:.6f} kg")
