"""Two-drone downwash aerodynamic characterisation experiment.

Sweeps vertical and horizontal relative offsets between an upper and lower drone,
quantifies position destabilisation, attitude tilt, and thrust increase,
and generates spatial heatmaps and CSV logs.
"""
import argparse
import os
from typing import Dict, Any, List, Tuple
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from gym_pybullet_drones.utils.enums import DroneModel, Physics

from src.models.downwash import DownwashModel
from src.sim.downwash_aviary import DownwashAviary
from src.controllers.single_drone_pid import SingleDronePIDController


def load_yaml(path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run_grid_sweep(
    dx_values: np.ndarray,
    dz_values: np.ndarray,
    duration_sec: float = 3.0,
    config_default: str = "configs/default.yaml",
    config_downwash: str = "configs/downwash_true.yaml",
    output_dir: str = "results",
    gui: bool = False,
) -> pd.DataFrame:
    """Execute grid sweep across horizontal (dx) and vertical (dz) relative offsets.

    Args:
        dx_values: 1D array of horizontal offsets in meters (x_lower - x_upper).
        dz_values: 1D array of vertical offsets in meters (z_upper - z_lower).
        duration_sec: Simulation duration per grid point in seconds.
        config_default: Path to default simulation config.
        config_downwash: Path to TRUE downwash disturbance config.
        output_dir: Output directory for results and figures.
        gui: Whether to render GUI.

    Returns:
        DataFrame containing all recorded metrics for each (dx, dz) test condition.
    """
    cfg_sim = load_yaml(config_default)
    cfg_dw = load_yaml(config_downwash)

    sim_freq = int(cfg_sim["simulation"]["sim_freq_hz"])
    ctrl_freq = int(cfg_sim["simulation"]["ctrl_freq_hz"])
    drone_model = DroneModel.CF2X

    downwash_model = DownwashModel(cfg_dw)

    total_conditions = len(dx_values) * len(dz_values)
    print(f"[INFO] Starting Downwash Grid Sweep: {len(dx_values)} dx x {len(dz_values)} dz = {total_conditions} conditions")
    print(f"       Horizontal offsets (dx): {np.round(dx_values, 3)} m")
    print(f"       Vertical offsets   (dz): {np.round(dz_values, 3)} m")

    results: List[Dict[str, float]] = []
    z_upper = 1.5  # Upper drone fixed altitude [m]

    condition_idx = 0
    for dz in dz_values:
        z_lower = z_upper - dz
        for dx in dx_values:
            condition_idx += 1
            # Initial positions: Upper at [0, 0, z_upper], Lower at [dx, 0, z_lower]
            initial_xyzs = np.array([
                [0.0, 0.0, z_upper],
                [dx,  0.0, z_lower],
            ])
            target_upper = np.array([0.0, 0.0, z_upper])
            target_lower = np.array([dx,  0.0, z_lower])

            env = DownwashAviary(
                drone_model=drone_model,
                num_drones=2,
                initial_xyzs=initial_xyzs,
                physics=Physics.PYB,
                pyb_freq=sim_freq,
                ctrl_freq=ctrl_freq,
                gui=gui,
                obstacles=False,
                user_debug_gui=False,
                downwash_model=downwash_model,
            )

            ctrl_upper = SingleDronePIDController(drone_id=0, drone_model=drone_model)
            ctrl_lower = SingleDronePIDController(drone_id=1, drone_model=drone_model)
            ctrl_upper.reset()
            ctrl_lower.reset()

            action = np.zeros((2, 4))
            obs, _, _, _, _ = env.step(action)

            nominal_hover_rpm = env.HOVER_RPM

            num_steps = int(duration_sec * ctrl_freq)
            # Evaluate metrics after initial stabilization (discard first 0.5s)
            burn_in_steps = int(0.5 * ctrl_freq)

            lower_pos_errors = []
            lower_z_errors = []
            lower_att_deg = []
            lower_rpms = []
            applied_disturb_z = []

            for step in range(num_steps):
                # Upper drone control
                action[0, :], _, _ = ctrl_upper.compute_action(
                    control_timestep=env.CTRL_TIMESTEP,
                    state=obs[0],
                    target_pos=target_upper,
                )
                # Lower drone control
                action[1, :], _, _ = ctrl_lower.compute_action(
                    control_timestep=env.CTRL_TIMESTEP,
                    state=obs[1],
                    target_pos=target_lower,
                )

                obs, reward, terminated, truncated, info = env.step(action)

                if step >= burn_in_steps:
                    lower_pos = obs[1][0:3]
                    lower_rpy = obs[1][7:10]
                    lower_rpm = action[1, :]
                    disturb_force = env.last_applied_disturbances[1]

                    err_3d = np.linalg.norm(lower_pos - target_lower)
                    err_z = abs(lower_pos[2] - target_lower[2])
                    att_norm_deg = np.rad2deg(np.linalg.norm(lower_rpy[0:2]))  # roll & pitch tilt

                    lower_pos_errors.append(err_3d)
                    lower_z_errors.append(err_z)
                    lower_att_deg.append(att_norm_deg)
                    lower_rpms.append(np.mean(lower_rpm))
                    applied_disturb_z.append(abs(disturb_force[2]))

            env.close()

            # Quantitative metrics for condition
            rmse_pos = float(np.sqrt(np.mean(np.array(lower_pos_errors) ** 2)))
            peak_pos_err = float(np.max(lower_pos_errors))
            rmse_z = float(np.sqrt(np.mean(np.array(lower_z_errors) ** 2)))
            peak_z_err = float(np.max(lower_z_errors))
            mean_att_deg = float(np.mean(lower_att_deg))
            peak_att_deg = float(np.max(lower_att_deg))
            mean_rpm = float(np.mean(lower_rpms))
            rpm_increase = float(mean_rpm - nominal_hover_rpm)
            mean_disturb_z = float(np.mean(applied_disturb_z))

            record = {
                "dx": float(dx),
                "dz": float(dz),
                "rmse_pos": rmse_pos,
                "peak_pos_err": peak_pos_err,
                "rmse_z": rmse_z,
                "peak_z_err": peak_z_err,
                "mean_att_deg": mean_att_deg,
                "peak_att_deg": peak_att_deg,
                "mean_rpm": mean_rpm,
                "rpm_increase": rpm_increase,
                "nominal_hover_rpm": float(nominal_hover_rpm),
                "mean_disturb_force_z": mean_disturb_z,
            }
            results.append(record)
            if condition_idx % 5 == 0 or condition_idx == total_conditions:
                print(f"  [{condition_idx:2d}/{total_conditions}] dx={dx:.2f}m, dz={dz:.2f}m -> RMSE={rmse_pos:.4f}m, PeakErr={peak_pos_err:.4f}m, dRPM={rpm_increase:+.1f}")

    df = pd.DataFrame(results)

    # Save CSV
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "downwash_grid_sweep.csv")
    version = 1
    while os.path.exists(csv_path):
        csv_path = os.path.join(output_dir, f"downwash_grid_sweep_v{version}.csv")
        version += 1
    df.to_csv(csv_path, index=False)
    print(f"\n[INFO] Grid sweep saved to {csv_path}")

    # Generate heatmaps
    plot_downwash_heatmaps(df, output_dir=os.path.join(output_dir, "figures"))

    return df


def plot_downwash_heatmaps(df: pd.DataFrame, output_dir: str = "results/figures") -> None:
    """Generate publication-ready 2D spatial heatmaps from grid sweep logs."""
    os.makedirs(output_dir, exist_ok=True)

    dx_vals = np.sort(df["dx"].unique())
    dz_vals = np.sort(df["dz"].unique())

    # Helper function to render a 2D matrix
    def create_grid(column_name: str) -> np.ndarray:
        grid = np.zeros((len(dz_vals), len(dx_vals)))
        for i, dz in enumerate(dz_vals):
            for j, dx in enumerate(dx_vals):
                val = df[(df["dx"] == dx) & (df["dz"] == dz)][column_name].values
                grid[i, j] = val[0] if len(val) > 0 else 0.0
        return grid

    # 1. Heatmap: Position Tracking RMSE [m]
    rmse_grid = create_grid("rmse_pos")
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(
        rmse_grid,
        origin="upper",
        cmap="plasma",
        extent=[dx_vals[0], dx_vals[-1], dz_vals[-1], dz_vals[0]],
        aspect="auto",
    )
    plt.colorbar(im, ax=ax, label="Lower Drone Position RMSE [m]")
    ax.set_xlabel("Horizontal Offset $\\Delta x$ [m]")
    ax.set_ylabel("Vertical Separation $\\Delta z$ [m]")
    ax.set_title("Downwash Interaction: Position Tracking RMSE")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "downwash_heatmap_rmse.png"), dpi=300)
    plt.savefig(os.path.join(output_dir, "downwash_heatmap_rmse.pdf"))
    plt.close()

    # 2. Heatmap: Peak Position Error [m]
    peak_grid = create_grid("peak_pos_err")
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(
        peak_grid,
        origin="upper",
        cmap="inferno",
        extent=[dx_vals[0], dx_vals[-1], dz_vals[-1], dz_vals[0]],
        aspect="auto",
    )
    plt.colorbar(im, ax=ax, label="Peak Position Error [m]")
    ax.set_xlabel("Horizontal Offset $\\Delta x$ [m]")
    ax.set_ylabel("Vertical Separation $\\Delta z$ [m]")
    ax.set_title("Downwash Interaction: Peak Position Deviation")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "downwash_heatmap_peak_error.png"), dpi=300)
    plt.savefig(os.path.join(output_dir, "downwash_heatmap_peak_error.pdf"))
    plt.close()

    # 3. Heatmap: Motor RPM / Thrust Increase [RPM]
    rpm_grid = create_grid("rpm_increase")
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(
        rpm_grid,
        origin="upper",
        cmap="viridis",
        extent=[dx_vals[0], dx_vals[-1], dz_vals[-1], dz_vals[0]],
        aspect="auto",
    )
    plt.colorbar(im, ax=ax, label="Motor RPM Increase $\\Delta RPM$")
    ax.set_xlabel("Horizontal Offset $\\Delta x$ [m]")
    ax.set_ylabel("Vertical Separation $\\Delta z$ [m]")
    ax.set_title("Downwash Interaction: Motor Thrust Increase")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "downwash_heatmap_thrust.png"), dpi=300)
    plt.savefig(os.path.join(output_dir, "downwash_heatmap_thrust.pdf"))
    plt.close()

    # 4. Heatmap: Peak Attitude Deviation [deg]
    att_grid = create_grid("peak_att_deg")
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(
        att_grid,
        origin="upper",
        cmap="magma",
        extent=[dx_vals[0], dx_vals[-1], dz_vals[-1], dz_vals[0]],
        aspect="auto",
    )
    plt.colorbar(im, ax=ax, label="Peak Tilt / Attitude Deviation [deg]")
    ax.set_xlabel("Horizontal Offset $\\Delta x$ [m]")
    ax.set_ylabel("Vertical Separation $\\Delta z$ [m]")
    ax.set_title("Downwash Interaction: Peak Attitude Deviation")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "downwash_heatmap_attitude.png"), dpi=300)
    plt.savefig(os.path.join(output_dir, "downwash_heatmap_attitude.pdf"))
    plt.close()

    print(f"[INFO] Generated 4 heatmaps in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Two-drone downwash grid sweep characterisation")
    parser.add_argument("--fast", action="store_true", default=False, help="Run reduced grid for rapid validation")
    parser.add_argument("--duration", type=float, default=2.5, help="Simulation seconds per grid point")
    parser.add_argument("--gui", action="store_true", default=False)
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()

    if args.fast:
        dx_sweep = np.array([0.0, 0.15, 0.35])
        dz_sweep = np.array([0.25, 0.50, 0.80])
    else:
        dx_sweep = np.array([0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50])
        dz_sweep = np.array([0.20, 0.35, 0.50, 0.65, 0.80, 1.00])

    df = run_grid_sweep(
        dx_values=dx_sweep,
        dz_values=dz_sweep,
        duration_sec=args.duration,
        output_dir=args.output_dir,
        gui=args.gui,
    )
