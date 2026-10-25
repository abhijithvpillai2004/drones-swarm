"""Plotting routines for Phase 6 model robustness, sensitivity, and break-even analysis."""
import os
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from src.analysis.plots import save_figure, CTRL_COLORS

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
    "lines.linewidth": 2.0,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})


def plot_force_mismatch_sensitivity(
    mismatch_ratios: np.ndarray,
    rmse_data: Dict[str, np.ndarray],
    crossover_ratio: Optional[float] = None,
    output_dir: str = "results/figures",
    scenario_name: str = "hover_column_3d",
) -> Tuple[str, str]:
    """Plot tracking RMSE as a function of assumed-to-true force mismatch ratio.

    Args:
        mismatch_ratios: 1D array of ratios alpha = F_assumed / F_true.
        rmse_data: Dict mapping ctrl_name -> array of RMSE values in meters.
        crossover_ratio: Optional identified break-even point where B outperforms C.
        output_dir: Output folder.
        scenario_name: Scenario name string.

    Returns:
        Tuple of saved PNG and PDF filepaths.
    """
    fig, ax = plt.subplots(figsize=(8.0, 5.0))

    for ctrl_name, errors in rmse_data.items():
        color = CTRL_COLORS.get(ctrl_name, "#555555")
        clean_name = ctrl_name.replace("Controller_", "").replace("_", " ")

        if "Naive" in ctrl_name:
            # Controller A ignores downwash model, so its performance is constant across mismatch
            ax.axhline(
                y=errors[0] * 100.0,
                color=color,
                linestyle="--",
                label=f"{clean_name} (No Model)",
            )
        else:
            ax.plot(
                mismatch_ratios,
                errors * 100.0,
                marker="o",
                label=clean_name,
                color=color,
            )

    # Reference nominal line (1.0 = perfect model)
    ax.axvline(x=1.0, color="gray", linestyle=":", label="Nominal Model (Ratio = 1.0)")

    # Mark crossover if present
    if crossover_ratio is not None:
        ax.axvline(
            x=crossover_ratio,
            color="#C62828",
            linestyle="-.",
            alpha=0.8,
            label=f"Break-Even Crossover (~{crossover_ratio:.2f}x)",
        )

    ax.set_title(
        f"Robustness to Aerodynamic Force Model Uncertainty\n({scenario_name.replace('_', ' ')})",
        fontweight="bold",
        pad=10,
    )
    ax.set_xlabel(r"Assumed Force Scaling Factor $\alpha_{force} = \hat{F}_{max} / F_{max, true}$")
    ax.set_ylabel("Swarm Tracking RMSE [cm]")
    ax.set_xlim([mismatch_ratios[0] - 0.05, mismatch_ratios[-1] + 0.05])
    ax.legend(loc="upper left", frameon=True)
    plt.tight_layout()

    return save_figure(fig, f"robustness_force_sweep_{scenario_name}", output_dir)


def plot_radius_mismatch_sensitivity(
    radius_ratios: np.ndarray,
    rmse_data: Dict[str, np.ndarray],
    output_dir: str = "results/figures",
    scenario_name: str = "hover_column_3d",
) -> Tuple[str, str]:
    """Plot tracking RMSE as a function of wake influence radius mismatch ratio."""
    fig, ax = plt.subplots(figsize=(8.0, 5.0))

    for ctrl_name, errors in rmse_data.items():
        color = CTRL_COLORS.get(ctrl_name, "#555555")
        clean_name = ctrl_name.replace("Controller_", "").replace("_", " ")

        if "Naive" in ctrl_name:
            ax.axhline(
                y=errors[0] * 100.0,
                color=color,
                linestyle="--",
                label=f"{clean_name} (No Model)",
            )
        else:
            ax.plot(
                radius_ratios,
                errors * 100.0,
                marker="s",
                label=clean_name,
                color=color,
            )

    ax.axvline(x=1.0, color="gray", linestyle=":", label="Nominal Model (Ratio = 1.0)")
    ax.set_title(
        f"Robustness to Wake Radius Model Uncertainty\n({scenario_name.replace('_', ' ')})",
        fontweight="bold",
        pad=10,
    )
    ax.set_xlabel(r"Assumed Radius Scaling Factor $\alpha_{radius} = \hat{R}_{xy} / R_{xy, true}$")
    ax.set_ylabel("Swarm Tracking RMSE [cm]")
    ax.set_xlim([radius_ratios[0] - 0.05, radius_ratios[-1] + 0.05])
    ax.legend(loc="upper left", frameon=True)
    plt.tight_layout()

    return save_figure(fig, f"robustness_radius_sweep_{scenario_name}", output_dir)


def plot_noise_delay_comparison(
    conditions: List[str],
    results_by_ctrl: Dict[str, List[float]],
    output_dir: str = "results/figures",
    scenario_name: str = "hover_column_3d",
) -> Tuple[str, str]:
    """Plot grouped bar chart comparing controllers under sensor noise and latency delay."""
    fig, ax = plt.subplots(figsize=(10.0, 5.0))
    num_conds = len(conditions)
    ctrl_names = list(results_by_ctrl.keys())
    num_ctrls = len(ctrl_names)

    x = np.arange(num_conds)
    bar_width = 0.8 / num_ctrls

    for i, ctrl_name in enumerate(ctrl_names):
        color = CTRL_COLORS.get(ctrl_name, "#555555")
        clean_name = ctrl_name.replace("Controller_", "").replace("_", " ")
        values = [v * 100.0 for v in results_by_ctrl[ctrl_name]]
        offset = (i - num_ctrls / 2.0 + 0.5) * bar_width
        bars = ax.bar(x + offset, values, bar_width, label=clean_name, color=color, alpha=0.85, edgecolor="black")

        # Value labels above bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=10, ha="right")
    ax.set_ylabel("Swarm Tracking RMSE [cm]")
    ax.set_title(
        f"Controller Performance Under Noise, Latency & Model Mismatch\n({scenario_name.replace('_', ' ')})",
        fontweight="bold",
        pad=10,
    )
    ax.legend(loc="upper left", frameon=True)
    plt.tight_layout()

    return save_figure(fig, f"noise_delay_comparison_{scenario_name}", output_dir)


def plot_2d_robustness_heatmap(
    force_ratios: np.ndarray,
    radius_ratios: np.ndarray,
    rmse_grid: np.ndarray,
    controller_name: str = "Feedforward",
    output_dir: str = "results/figures",
    scenario_name: str = "hover_column_3d",
) -> Tuple[str, str]:
    """Plot 2D contour heatmap of RMSE over simultaneous force and radius mismatches."""
    fig, ax = plt.subplots(figsize=(8.0, 6.0))

    X, Y = np.meshgrid(force_ratios, radius_ratios)
    Z = rmse_grid * 100.0  # cm

    cp = ax.contourf(X, Y, Z, levels=15, cmap="viridis", alpha=0.9)
    fig.colorbar(cp, ax=ax, label="Swarm Tracking RMSE [cm]")

    # Overlay nominal point (1.0, 1.0)
    ax.scatter([1.0], [1.0], color="red", s=100, marker="*", label="Nominal Model (1.0, 1.0)", zorder=5)

    clean_name = controller_name.replace("Controller_", "").replace("_", " ")
    ax.set_title(
        f"2D Parameter Sensitivity Landscape: {clean_name}\n({scenario_name.replace('_', ' ')})",
        fontweight="bold",
        pad=10,
    )
    ax.set_xlabel(r"Assumed Force Scaling Factor $\alpha_{force}$")
    ax.set_ylabel(r"Assumed Radius Scaling Factor $\alpha_{radius}$")
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()

    return save_figure(fig, f"robustness_2d_heatmap_{controller_name}_{scenario_name}", output_dir)
