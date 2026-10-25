"""Single-drone baseline experiment runner.

Executes circle or waypoint tracking in PyBullet headless (or optional GUI),
logs full state per timestep, computes tracking metrics, and generates figures.
"""
from src.paths import resolve
import argparse
import os
from typing import Dict, Any, Tuple
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary

from src.controllers.single_drone_pid import SingleDronePIDController
from src.sim.trajectories import CircleTrajectory, WaypointTrajectory


def load_config(config_path: str = "configs/default.yaml") -> Dict[str, Any]:
    """Load default simulation configuration."""
    with open(resolve(config_path), "r") as f:
        return yaml.safe_load(f)


def run_single_drone_experiment(
    trajectory_type: str = "circle",
    duration_sec: float = 12.0,
    gui: bool = False,
    output_dir: str = "results",
    config_path: str = "configs/default.yaml",
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Run single drone trajectory tracking experiment.

    Args:
        trajectory_type: "circle" or "waypoints".
        duration_sec: Simulation duration in seconds.
        gui: Whether to enable PyBullet visualizer.
        output_dir: Directory to save CSV logs and figures.
        config_path: Path to default YAML configuration.

    Returns:
        Tuple of (dataframe_log, metrics_dict).
    """
    cfg = load_config(config_path)
    sim_freq = int(cfg["simulation"]["sim_freq_hz"])
    ctrl_freq = int(cfg["simulation"]["ctrl_freq_hz"])
    drone_model_str = cfg["drone"]["model"]
    drone_model = DroneModel.CF2X if drone_model_str == "cf2x" else DroneModel.CF2P

    # Initialize trajectory
    if trajectory_type == "circle":
        trajectory = CircleTrajectory(radius=0.5, period=6.0, altitude=1.0, center=(0.0, 0.0))
        init_pos, _ = trajectory.evaluate(0.0)
    else:
        waypoints = np.array([
            [0.0, 0.0, 0.5],
            [0.5, 0.0, 1.0],
            [0.5, 0.5, 1.0],
            [0.0, 0.5, 1.0],
            [0.0, 0.0, 1.0]
        ])
        trajectory = WaypointTrajectory(waypoints=waypoints, dwell_time=2.5)
        init_pos, _ = trajectory.evaluate(0.0)

    # Initialize environment with single drone
    initial_xyzs = np.array([init_pos])
    env = CtrlAviary(
        drone_model=drone_model,
        num_drones=1,
        initial_xyzs=initial_xyzs,
        physics=Physics.PYB,
        pyb_freq=sim_freq,
        ctrl_freq=ctrl_freq,
        gui=gui,
        obstacles=False,
        user_debug_gui=False,
    )

    ctrl = SingleDronePIDController(drone_id=0, drone_model=drone_model)
    ctrl.reset()

    # Step environment once to obtain initial observation
    action = np.zeros((1, 4))
    obs, _, _, _, _ = env.step(action)

    num_steps = int(duration_sec * ctrl_freq)
    records = []

    for step in range(num_steps):
        t = step * env.CTRL_TIMESTEP
        target_pos, target_vel = trajectory.evaluate(t)

        # Compute PID motor action
        drone_state = obs[0]
        motor_rpms, pos_e, yaw_e = ctrl.compute_action(
            control_timestep=env.CTRL_TIMESTEP,
            state=drone_state,
            target_pos=target_pos,
            target_vel=target_vel,
        )
        action[0, :] = motor_rpms

        # Step physics
        obs, reward, terminated, truncated, info = env.step(action)

        # Extract current state
        cur_pos = drone_state[0:3]
        cur_rpy = drone_state[7:10]
        cur_vel = drone_state[10:13]
        error_xyz = cur_pos - target_pos
        error_norm = float(np.linalg.norm(error_xyz))

        record = {
            "timestamp": t,
            "step": step,
            "pos_x": cur_pos[0],
            "pos_y": cur_pos[1],
            "pos_z": cur_pos[2],
            "target_x": target_pos[0],
            "target_y": target_pos[1],
            "target_z": target_pos[2],
            "error_x": error_xyz[0],
            "error_y": error_xyz[1],
            "error_z": error_xyz[2],
            "error_norm": error_norm,
            "roll": cur_rpy[0],
            "pitch": cur_rpy[1],
            "yaw": cur_rpy[2],
            "vel_x": cur_vel[0],
            "vel_y": cur_vel[1],
            "vel_z": cur_vel[2],
            "rpm_0": motor_rpms[0],
            "rpm_1": motor_rpms[1],
            "rpm_2": motor_rpms[2],
            "rpm_3": motor_rpms[3],
        }
        records.append(record)

    env.close()

    df = pd.DataFrame(records)

    # Compute key quantitative metrics
    rmse_x = float(np.sqrt(np.mean(df["error_x"] ** 2)))
    rmse_y = float(np.sqrt(np.mean(df["error_y"] ** 2)))
    rmse_z = float(np.sqrt(np.mean(df["error_z"] ** 2)))
    rmse_total = float(np.sqrt(np.mean(df["error_norm"] ** 2)))
    max_error = float(np.max(df["error_norm"]))
    # Steady state error (last 25% of trajectory)
    steady_state_slice = df.iloc[int(0.75 * len(df)) :]
    ss_rmse = float(np.sqrt(np.mean(steady_state_slice["error_norm"] ** 2)))

    metrics = {
        "rmse_x": rmse_x,
        "rmse_y": rmse_y,
        "rmse_z": rmse_z,
        "rmse_total": rmse_total,
        "max_error": max_error,
        "steady_state_rmse": ss_rmse,
    }

    # Save CSV
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "single_drone_baseline.csv")
    df.to_csv(csv_path, index=False)
    print(f"[INFO] Simulation logs saved to {csv_path}")

    # Generate plots
    plot_results(df, metrics, output_dir=os.path.join(output_dir, "figures"))

    return df, metrics


def plot_results(df: pd.DataFrame, metrics: Dict[str, float], output_dir: str = "results/figures") -> None:
    """Generate publication-ready diagnostic tracking plots."""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    # Subplot 1: Position Tracking vs Time
    t = df["timestamp"]
    axes[0].plot(t, df["target_x"], "r--", label="Target X", alpha=0.8)
    axes[0].plot(t, df["pos_x"], "r-", label="Actual X")
    axes[0].plot(t, df["target_y"], "g--", label="Target Y", alpha=0.8)
    axes[0].plot(t, df["pos_y"], "g-", label="Actual Y")
    axes[0].plot(t, df["target_z"], "b--", label="Target Z", alpha=0.8)
    axes[0].plot(t, df["pos_z"], "b-", label="Actual Z")
    axes[0].set_ylabel("Position [m]")
    axes[0].set_title(
        f"Single Drone Baseline Trajectory Tracking (RMSE: {metrics['rmse_total']:.4f} m, SS-RMSE: {metrics['steady_state_rmse']:.4f} m)"
    )
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(loc="upper right", ncol=3)

    # Subplot 2: Tracking Errors
    axes[1].plot(t, df["error_x"], "r-", label=f"Error X (RMSE: {metrics['rmse_x']:.3f}m)", alpha=0.7)
    axes[1].plot(t, df["error_y"], "g-", label=f"Error Y (RMSE: {metrics['rmse_y']:.3f}m)", alpha=0.7)
    axes[1].plot(t, df["error_z"], "b-", label=f"Error Z (RMSE: {metrics['rmse_z']:.3f}m)", alpha=0.7)
    axes[1].plot(t, df["error_norm"], "k-", linewidth=1.5, label="Euclidean Error Norm")
    axes[1].set_ylabel("Error [m]")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(loc="upper right", ncol=2)

    # Subplot 3: Motor RPM commands
    axes[2].plot(t, df["rpm_0"], label="Motor 0", alpha=0.8)
    axes[2].plot(t, df["rpm_1"], label="Motor 1", alpha=0.8)
    axes[2].plot(t, df["rpm_2"], label="Motor 2", alpha=0.8)
    axes[2].plot(t, df["rpm_3"], label="Motor 3", alpha=0.8)
    axes[2].set_ylabel("Rotor Speed [RPM]")
    axes[2].set_xlabel("Time [s]")
    axes[2].grid(True, linestyle="--", alpha=0.6)
    axes[2].legend(loc="upper right", ncol=4)

    plt.tight_layout()
    png_path = os.path.join(output_dir, "single_drone_tracking.png")
    pdf_path = os.path.join(output_dir, "single_drone_tracking.pdf")
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()
    print(f"[INFO] Diagnostic figures saved to {png_path} and {pdf_path}")

    # Generate 3D trajectory plot
    fig_3d = plt.figure(figsize=(8, 7))
    ax_3d = fig_3d.add_subplot(111, projection="3d")
    ax_3d.plot(df["target_x"], df["target_y"], df["target_z"], "k--", label="Reference Path", linewidth=2)
    ax_3d.plot(df["pos_x"], df["pos_y"], df["pos_z"], "b-", label="Actual Trajectory", linewidth=1.8)
    ax_3d.scatter([df["pos_x"].iloc[0]], [df["pos_y"].iloc[0]], [df["pos_z"].iloc[0]], color="green", s=50, label="Start")
    ax_3d.scatter([df["pos_x"].iloc[-1]], [df["pos_y"].iloc[-1]], [df["pos_z"].iloc[-1]], color="red", s=50, label="End")
    ax_3d.set_xlabel("X [m]")
    ax_3d.set_ylabel("Y [m]")
    ax_3d.set_zlabel("Z [m]")
    ax_3d.set_title("3D Trajectory Profile")
    ax_3d.legend()
    fig_3d_path = os.path.join(output_dir, "single_drone_3d_orbit.png")
    plt.savefig(fig_3d_path, dpi=300)
    plt.close()
    print(f"[INFO] 3D orbit figure saved to {fig_3d_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single drone baseline tracking experiment")
    parser.add_argument("--trajectory", type=str, default="circle", choices=["circle", "waypoints"])
    parser.add_argument("--duration", type=float, default=12.0)
    parser.add_argument("--gui", action="store_true", default=False)
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()

    df, metrics = run_single_drone_experiment(
        trajectory_type=args.trajectory,
        duration_sec=args.duration,
        gui=args.gui,
        output_dir=args.output_dir,
    )
    print("\n=== Single Drone Tracking Baseline Metrics ===")
    for k, v in metrics.items():
        print(f"  {k:20s}: {v:.5f}")
