"""Publication-quality plotting and visualization routines for multi-drone experiments."""
import os
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Modern publication styling configuration
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 14,
    "lines.linewidth": 1.8,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

# Curated accessible color palette
CTRL_COLORS = {
    "Naive": "#D9534F",          # Coral / Crimson
    "Controller_A_Naive": "#D9534F",
    "SoftAvoid": "#F0AD4E",      # Amber / Gold
    "Controller_B_SoftAvoid": "#F0AD4E",
    "Feedforward": "#0275D8",    # Sapphire / Cobalt
    "Controller_C_Feedforward": "#0275D8",
    "UAADC": "#2ECC71",          # Emerald / Teal
    "Controller_D_UAADC": "#2ECC71",
    "Controller_D_UA_ADC": "#2ECC71",
    "Reference": "#333333",      # Dark Charcoal
}

DRONE_COLORS = ["#2E7D32", "#1565C0", "#C62828", "#6A1B9A", "#EF6C00"]


def save_figure(fig: plt.Figure, base_filename: str, output_dir: str = "results/figures") -> Tuple[str, str]:
    """Save figure as both high-resolution PNG and publication PDF."""
    os.makedirs(output_dir, exist_ok=True)
    png_path = os.path.join(output_dir, f"{base_filename}.png")
    pdf_path = os.path.join(output_dir, f"{base_filename}.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    print(f"[PLOT] Saved: {png_path} and {pdf_path}")
    return png_path, pdf_path


def plot_3d_swarm_trajectories(
    actual_trajectories: Dict[str, np.ndarray],
    ref_trajectory: np.ndarray,
    scenario_name: str,
    output_dir: str = "results/figures",
) -> None:
    """Plot 3D spatial trajectories comparing Controllers A, B, and C side-by-side.

    Args:
        actual_trajectories: Dict mapping controller_name -> (T, N, 3) position array.
        ref_trajectory: Array of shape (T, N, 3) reference trajectories.
        scenario_name: Human-readable scenario name for plot title.
        output_dir: Directory where figures are saved.
    """
    ctrl_names = list(actual_trajectories.keys())
    num_ctrls = len(ctrl_names)
    num_drones = ref_trajectory.shape[1]

    fig = plt.figure(figsize=(6.0 * num_ctrls, 6.0))

    for idx, ctrl_name in enumerate(ctrl_names):
        ax = fig.add_subplot(1, num_ctrls, idx + 1, projection="3d")
        traj = actual_trajectories[ctrl_name]

        # Plot reference paths
        for d in range(num_drones):
            ax.plot(
                ref_trajectory[:, d, 0],
                ref_trajectory[:, d, 1],
                ref_trajectory[:, d, 2],
                color="black",
                linestyle=":",
                alpha=0.6,
                label=f"Ref Drone {d}" if idx == 0 else "",
            )

        # Plot actual paths per drone
        for d in range(num_drones):
            color = DRONE_COLORS[d % len(DRONE_COLORS)]
            ax.plot(
                traj[:, d, 0],
                traj[:, d, 1],
                traj[:, d, 2],
                color=color,
                label=f"Drone {d} (Actual)" if idx == 0 else "",
                linewidth=2.0,
            )
            # Start and End markers
            ax.scatter(traj[0, d, 0], traj[0, d, 1], traj[0, d, 2], color=color, marker="o", s=40)
            ax.scatter(traj[-1, d, 0], traj[-1, d, 1], traj[-1, d, 2], color=color, marker="^", s=50)

        clean_name = ctrl_name.replace("Controller_", "").replace("_", " ")
        ax.set_title(clean_name, fontweight="bold", pad=10)
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_zlabel("Z [m]")
        ax.view_init(elev=25, azim=45)

    if num_drones > 0:
        fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=num_drones * 2, frameon=True)

    fig.suptitle(f"3D Flight Trajectories: {scenario_name}", y=1.03, fontweight="bold")
    save_figure(fig, f"traj3d_{scenario_name}", output_dir)
    plt.close(fig)


def plot_tracking_error_timeseries(
    time_series_data: Dict[str, Dict[str, np.ndarray]],
    scenario_name: str,
    output_dir: str = "results/figures",
) -> None:
    """Plot time series of tracking error ||e_p(t)|| and altitude z(t) for all controllers.

    Args:
        time_series_data: Dict mapping ctrl_name -> {
            'time': (T,),
            'errors': (T, N),
            'actual_pos': (T, N, 3),
            'ref_pos': (T, N, 3)
        }
        scenario_name: Scenario name string.
        output_dir: Output folder.
    """
    first_ctrl = next(iter(time_series_data.keys()))
    t = time_series_data[first_ctrl]["time"]
    num_drones = time_series_data[first_ctrl]["errors"].shape[1]

    fig, axes = plt.subplots(num_drones, 2, figsize=(12.0, 3.2 * num_drones), sharex=True, squeeze=False)

    for d in range(num_drones):
        ax_err = axes[d, 0]
        ax_alt = axes[d, 1]

        # Plot reference altitude
        ref_z = time_series_data[first_ctrl]["ref_pos"][:, d, 2]
        ax_alt.plot(t, ref_z, "k--", alpha=0.7, label="Reference Setpoint")

        for ctrl_name, data in time_series_data.items():
            color = CTRL_COLORS.get(ctrl_name, "#555555")
            clean_name = ctrl_name.replace("Controller_", "").replace("_", " ")

            # Column 1: Euclidean position error
            err_d = data["errors"][:, d]
            ax_err.plot(t, err_d * 100.0, label=clean_name, color=color, alpha=0.85)

            # Column 2: Actual altitude z
            act_z = data["actual_pos"][:, d, 2]
            ax_alt.plot(t, act_z, label=clean_name, color=color, alpha=0.85)

        ax_err.set_ylabel(f"Drone {d}\nPos Error [cm]")
        ax_alt.set_ylabel(f"Drone {d}\nAltitude Z [m]")
        if d == 0:
            ax_err.set_title("Tracking Error Magnitude ||e(t)||", fontweight="bold")
            ax_alt.set_title("Altitude Z Tracking Comparison", fontweight="bold")
            ax_err.legend(loc="upper right", frameon=True)
            ax_alt.legend(loc="upper right", frameon=True)

    axes[-1, 0].set_xlabel("Time [s]")
    axes[-1, 1].set_xlabel("Time [s]")
    fig.suptitle(f"Dynamic Tracking Performance: {scenario_name}", y=1.01, fontweight="bold")
    plt.tight_layout()
    save_figure(fig, f"timeseries_{scenario_name}", output_dir)
    plt.close(fig)


def plot_comparative_metrics_barchart(
    summary_metrics: Dict[str, Dict[str, Any]],
    scenario_name: str,
    output_dir: str = "results/figures",
) -> None:
    """Plot summary bar chart of RMSE, Max Error, and Energy Proxy across controllers.

    Args:
        summary_metrics: Dict mapping ctrl_name -> metrics_dict from compute_all_metrics.
        scenario_name: Name of the scenario.
        output_dir: Output folder.
    """
    ctrl_names = list(summary_metrics.keys())
    clean_names = [c.replace("Controller_", "").replace("_", " ") for c in ctrl_names]

    mean_rmse = [summary_metrics[c]["rmse_mean"] * 100.0 for c in ctrl_names]  # cm
    max_rmse = [summary_metrics[c]["rmse_max"] * 100.0 for c in ctrl_names]   # cm
    energy = [summary_metrics[c]["total_energy_proxy"] / 1e8 for c in ctrl_names]    # scaled energy
    min_dist = [summary_metrics[c]["min_interdrone_dist"] for c in ctrl_names]   # m

    fig, axes = plt.subplots(1, 4, figsize=(15.0, 4.0))
    bar_colors = [CTRL_COLORS.get(c, "#555555") for c in ctrl_names]

    # 1. Mean RMSE
    axes[0].bar(clean_names, mean_rmse, color=bar_colors, alpha=0.85, edgecolor="black", linewidth=1.0)
    axes[0].set_title("Mean RMSE [cm]", fontweight="bold")
    axes[0].set_ylabel("Error [cm]")

    # 2. Peak Error
    axes[1].bar(clean_names, max_rmse, color=bar_colors, alpha=0.85, edgecolor="black", linewidth=1.0)
    axes[1].set_title("Peak Drone Error [cm]", fontweight="bold")
    axes[1].set_ylabel("Error [cm]")

    # 3. Energy Proxy
    axes[2].bar(clean_names, energy, color=bar_colors, alpha=0.85, edgecolor="black", linewidth=1.0)
    axes[2].set_title("Energy Proxy [10⁸ RPM²·s]", fontweight="bold")
    axes[2].set_ylabel("Energy Proxy")

    # 4. Min Inter-Drone Distance
    axes[3].bar(clean_names, min_dist, color=bar_colors, alpha=0.85, edgecolor="black", linewidth=1.0)
    axes[3].axhline(y=0.15, color="red", linestyle="--", label="d_safe = 0.15m")
    axes[3].set_title("Min Inter-Drone Dist [m]", fontweight="bold")
    axes[3].set_ylabel("Distance [m]")
    axes[3].legend(loc="upper right")

    for ax in axes:
        ax.set_xticks(range(len(clean_names)))
        ax.set_xticklabels(clean_names, rotation=15, ha="right")

    fig.suptitle(f"Benchmark Summary Metrics: {scenario_name}", y=1.03, fontweight="bold")
    plt.tight_layout()
    save_figure(fig, f"metrics_bar_{scenario_name}", output_dir)
    plt.close(fig)
