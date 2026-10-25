"""Comprehensive root-cause diagnosis of soft-avoidance multi-drone failure.

Investigates:
1. Numerical instability (NaN/Inf, unbounded growth)
2. Thrust/motor saturation under multi-source downwash
3. Potential-field local minima, vector cancellation, and deadlock
4. Workspace and inter-drone spacing constraints
5. Comparison across formations (hover_column_3d, vertical_swap, crossing_pair, grid) across N in [3, 4, 5, 6, 7]
6. Verification of the overlapping-wake threshold vs. structural geometry changes
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml

from src.models.downwash import DownwashModel
from src.sim.downwash_aviary import DownwashAviary
from src.controllers.soft_avoid import SoftAvoidSwarmController
from src.formations.generators import (
    HoverColumnFormation,
    VerticalSwapTrajectory,
    CrossingTrajectory,
    GridFormation,
)


def run_detailed_diagnostic(
    scenario_name: str,
    n_drones: int,
    duration: float = 8.0,
    ctrl_freq_hz: int = 48,
    output_dir: str = "results/diagnostics",
):
    os.makedirs(output_dir, exist_ok=True)
    dt = 1.0 / ctrl_freq_hz
    total_steps = int(duration * ctrl_freq_hz)

    # 1. Setup Scenario
    if scenario_name == "hover_column_3d":
        gen = HoverColumnFormation(num_drones=n_drones, z_top=1.8, dz=0.45, duration=duration)
    elif scenario_name == "vertical_swap":
        gen = VerticalSwapTrajectory(num_drones=n_drones, z_high=1.6, z_low=0.8, duration=duration)
    elif scenario_name == "crossing_pair":
        gen = CrossingTrajectory(num_drones=n_drones, upper_alt=1.4, lower_alt=0.8, travel_distance=1.6, duration=duration)
    elif scenario_name == "grid":
        gen = GridFormation(num_drones=n_drones, dx=0.45, dy=0.45, altitude=1.2, duration=duration)
    else:
        raise ValueError(scenario_name)

    with open("configs/downwash_true.yaml", "r") as f:
        true_cfg = yaml.safe_load(f)
    true_model = DownwashModel(true_cfg)

    init_xyzs = gen.get_initial_positions()
    aviary = DownwashAviary(
        num_drones=n_drones,
        initial_xyzs=init_xyzs,
        ctrl_freq=ctrl_freq_hz,
        pyb_freq=240,
        gui=False,
        record=False,
        downwash_model=true_model,
    )

    ctrl = SoftAvoidSwarmController(num_drones=n_drones, config={"max_displacement": 0.20, "max_velocity": 0.15})
    obs, _ = aviary.reset()

    # Telemetry logging arrays
    timestamps = []
    positions_history = []
    velocities_history = []
    rpy_history = []
    rpms_history = []
    downwash_forces_history = []
    target_offsets_history = []
    applied_offsets_history = []
    num_overlapping_disturbers = np.zeros((total_steps, n_drones), dtype=int)

    has_nan_inf = False
    first_nan_step = None
    unbounded_growth = False

    first_fall_step = None
    failing_drones = set()

    for step in range(total_steps):
        t = step * dt
        timestamps.append(t)
        target_pos, target_vel = gen.evaluate(t)
        states = obs

        # Check for NaN / Inf
        if np.any(np.isnan(states)) or np.any(np.isinf(states)):
            has_nan_inf = True
            if first_nan_step is None:
                first_nan_step = step

        # Check for unbounded velocity growth (> 20 m/s)
        vels = states[:, 10:13]
        if np.any(np.abs(vels) > 20.0):
            unbounded_growth = True

        positions = states[:, 0:3]
        rpms, _, _ = ctrl.compute_actions(
            control_timestep=dt,
            states=states,
            formation_targets=target_pos,
            formation_velocities=target_vel,
        )

        # Detect falling drones (altitude dropping below 0.15m)
        for d in range(n_drones):
            if positions[d, 2] < 0.15:
                failing_drones.add(d)
                if first_fall_step is None:
                    first_fall_step = step

        # Check multi-source overlapping downwash count for each drone
        # Count how many upper drones are within influence range of drone j
        for j in range(n_drones):
            upper_count = 0
            for i in range(n_drones):
                if i == j:
                    continue
                if positions[i, 2] > positions[j, 2]:
                    # Check if inside horizontal radius and vertical reach
                    delta_p = positions[j] - positions[i]
                    depth = abs(delta_p[2])
                    if depth <= true_model.radius_z:
                        rel_xy = delta_p[0:2] - (true_model.wake_skew_coeff * depth * states[i, 10:12])
                        if np.linalg.norm(rel_xy) <= true_model.radius_xy:
                            upper_count += 1
            num_overlapping_disturbers[step, j] = upper_count

        positions_history.append(positions.copy())
        velocities_history.append(vels.copy())
        rpy_history.append(states[:, 7:10].copy())
        rpms_history.append(rpms.copy())
        downwash_forces_history.append(aviary.last_applied_disturbances.copy())
        applied_offsets_history.append(ctrl.current_offsets.copy())

        obs, _, _, _, _ = aviary.step(rpms)

    aviary.close()

    positions_arr = np.array(positions_history)            # (T, N, 3)
    velocities_arr = np.array(velocities_history)          # (T, N, 3)
    rpy_arr = np.array(rpy_history)                        # (T, N, 3)
    rpms_arr = np.array(rpms_history)                      # (T, N, 4)
    dw_arr = np.array(downwash_forces_history)             # (T, N, 3)
    offsets_arr = np.array(applied_offsets_history)        # (T, N, 2)

    # Check motor saturation (Crazyflie max RPM ~ 21702)
    max_rpm = 21702
    is_saturated = (rpms_arr >= max_rpm - 50)              # (T, N, 4)
    drone_saturated_steps = np.sum(np.any(is_saturated, axis=2), axis=0) # per drone count of saturated timesteps

    # Maximum overlapping disturbers seen by any drone
    max_overlap_per_drone = np.max(num_overlapping_disturbers, axis=0)
    has_overlapping_wakes = bool(np.any(max_overlap_per_drone >= 2))

    summary = {
        "scenario": scenario_name,
        "n_drones": n_drones,
        "has_nan_inf": has_nan_inf,
        "first_nan_step": first_nan_step,
        "unbounded_growth": unbounded_growth,
        "failing_drones": sorted(list(failing_drones)),
        "first_fall_step": first_fall_step,
        "has_overlapping_wakes": has_overlapping_wakes,
        "max_overlap_count": int(np.max(max_overlap_per_drone)),
        "max_rpm_recorded": float(np.max(rpms_arr)),
        "drone_saturated_steps": drone_saturated_steps.tolist(),
        "max_dw_force_z": float(np.max(np.abs(dw_arr[:, :, 2]))),
    }

    # Save detailed telemetry CSV
    telemetry_records = []
    for step in range(total_steps):
        for d in range(n_drones):
            telemetry_records.append({
                "step": step,
                "t": timestamps[step],
                "drone_id": d,
                "x": positions_arr[step, d, 0],
                "y": positions_arr[step, d, 1],
                "z": positions_arr[step, d, 2],
                "vx": velocities_arr[step, d, 0],
                "vy": velocities_arr[step, d, 1],
                "vz": velocities_arr[step, d, 2],
                "roll": rpy_arr[step, d, 0],
                "pitch": rpy_arr[step, d, 1],
                "yaw": rpy_arr[step, d, 2],
                "rpm_mean": float(np.mean(rpms_arr[step, d])),
                "rpm_max": float(np.max(rpms_arr[step, d])),
                "saturated": bool(np.any(is_saturated[step, d])),
                "dw_fz": dw_arr[step, d, 2],
                "offset_x": offsets_arr[step, d, 0],
                "offset_y": offsets_arr[step, d, 1],
                "offset_norm": float(np.linalg.norm(offsets_arr[step, d])),
                "overlap_disturbers": int(num_overlapping_disturbers[step, d]),
            })

    df_tel = pd.DataFrame(telemetry_records)
    csv_file = os.path.join(output_dir, f"diagnostic_{scenario_name}_N{n_drones}.csv")
    df_tel.to_csv(csv_file, index=False)

    # Plot Altitude Time Series
    fig, axes = plt.subplots(3, 1, figsize=(9.0, 8.5), sharex=True)

    # 1. Altitudes z(t)
    for d in range(n_drones):
        axes[0].plot(timestamps, positions_arr[:, d, 2], label=f"Drone {d}", linewidth=1.8)
    axes[0].axhline(y=0.15, color="red", linestyle="--", label="Ground / Crash Boundary (0.15m)")
    axes[0].set_ylabel("Altitude Z [m]")
    axes[0].set_title(f"Diagnostic: {scenario_name} (N={n_drones}) - Altitudes & Motor Saturation")
    axes[0].legend(loc="upper right", ncol=min(n_drones, 5))
    axes[0].grid(True, linestyle="--", alpha=0.4)

    # 2. Maximum Motor RPM
    for d in range(n_drones):
        axes[1].plot(timestamps, np.max(rpms_arr[:, d], axis=1), label=f"Drone {d}", linewidth=1.8)
    axes[1].axhline(y=max_rpm, color="black", linestyle="--", label=f"Max RPM Limit ({max_rpm})")
    axes[1].set_ylabel("Peak Motor RPM")
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend(loc="lower right", ncol=min(n_drones, 5))

    # 3. Downward Downwash Force F_z
    for d in range(n_drones):
        axes[2].plot(timestamps, dw_arr[:, d, 2], label=f"Drone {d}", linewidth=1.8)
    axes[2].set_ylabel("Downwash F_z [N]")
    axes[2].set_xlabel("Time [s]")
    axes[2].grid(True, linestyle="--", alpha=0.4)
    axes[2].legend(loc="lower right", ncol=min(n_drones, 5))

    fig.tight_layout()
    fig_path = os.path.join(output_dir, f"diagnostic_{scenario_name}_N{n_drones}.png")
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    return summary


def main():
    output_dir = "results/diagnostics"
    os.makedirs(output_dir, exist_ok=True)

    scenarios = ["hover_column_3d", "vertical_swap", "crossing_pair", "grid"]
    drone_counts = [3, 4, 5, 6, 7]

    all_summaries = []

    print("\n" + "=" * 80)
    print("  RUNNING SYSTEMATIC DIAGNOSTIC SUITE FOR SOFT-AVOIDANCE")
    print("=" * 80)

    for sc in scenarios:
        print(f"\n>>> Scenario: {sc}")
        for n in drone_counts:
            res = run_detailed_diagnostic(scenario_name=sc, n_drones=n, duration=8.0, output_dir=output_dir)
            all_summaries.append(res)

            status_str = f"FAILED: Drones {res['failing_drones']} fell at step {res['first_fall_step']}" if res['failing_drones'] else "PASSED (No fall)"
            overlap_str = f"Overlap: YES (max {res['max_overlap_count']})" if res['has_overlapping_wakes'] else "Overlap: NO (single-wake)"
            print(f"  N={n:2d} | {status_str:<42} | {overlap_str:<26} | NaN/Inf: {res['has_nan_inf']}")

    df_sum = pd.DataFrame(all_summaries)
    df_sum.to_csv(os.path.join(output_dir, "diagnostic_summary_matrix.csv"), index=False)
    print("\n" + "=" * 80)
    print(f"  DIAGNOSTICS COMPLETE! Saved summary to {output_dir}/diagnostic_summary_matrix.csv")
    print("=" * 80)


if __name__ == "__main__":
    main()
