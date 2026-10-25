"""UA-ADC Signature Visualisation — Confidence & Blend Weight Over Time.

Runs UA-ADC on a synthetic hover-column scenario (N drones stacked vertically)
and plots three panels showing how the online estimator behaves across the episode:

  Panel 1: Per-drone confidence c_j(t) ∈ [0, 1]
  Panel 2: Per-drone blend weight α_j(t) ∈ [0, 1]  (sigmoid of confidence)
  Panel 3: Per-drone estimated-vs-assumed downwash Fz

Usage (from project root, conda env activated):
    python tools/ua_adc_signature.py                   # N=3 default
    python tools/ua_adc_signature.py --num-drones 5    # N=5
    python tools/ua_adc_signature.py --out results/figures/ua_adc_signature.png
"""
import argparse
from pathlib import Path
import sys
from typing import List
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import yaml

# Ensure project root is in sys.path so the script can be run directly:
# python tools/ua_adc_signature.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.controllers.ua_adc import UAAdaptiveController
from src.models.downwash import DownwashModel


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic scenario helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_hover_states(num_drones: int, z_top: float = 1.8, dz: float = 0.40) -> np.ndarray:
    """Create static hover column: drones stacked with dz vertical spacing."""
    states = np.zeros((num_drones, 20))
    for i in range(num_drones):
        states[i, 0] = 0.0
        states[i, 1] = 0.0
        states[i, 2] = z_top - i * dz
        states[i, 3:7] = [0.0, 0.0, 0.0, 1.0]   # identity quaternion
    return states


def _load_assumed(path: str = "configs/downwash_assumed.yaml"):
    p = Path(path)
    if p.exists():
        with open(p) as f:
            return yaml.safe_load(f) or {}
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# Main simulation loop
# ─────────────────────────────────────────────────────────────────────────────

def run_signature(
    num_drones: int = 3,
    duration: float = 8.0,
    control_hz: float = 48.0,
    ua_config_path: str = "configs/ua_adc.yaml",
    assumed_config_path: str = "configs/downwash_assumed.yaml",
    true_config_path: str = "configs/downwash_true.yaml",
) -> dict:
    """Run UA-ADC on a hover column with full PyBullet physics and collect telemetry.

    Returns a dict with keys: t, confidences, blend_weights, assumed_fz, estimated_fz.
    """
    from src.sim.downwash_aviary import DownwashAviary
    from src.formations.generators import HoverColumnFormation

    dt = 1.0 / control_hz
    steps = int(duration * control_hz)
    ua_cfg_path = Path(ua_config_path)
    ua_cfg = yaml.safe_load(ua_cfg_path.read_text()) if ua_cfg_path.exists() else {}

    assumed_params = _load_assumed(assumed_config_path)
    assumed_model = DownwashModel(config=assumed_params)

    true_cfg_p = Path(true_config_path)
    true_cfg = yaml.safe_load(true_cfg_p.read_text()) if true_cfg_p.exists() else {}
    true_model = DownwashModel(config=true_cfg)

    scenario_gen = HoverColumnFormation(num_drones=num_drones, z_top=1.8, dz=0.45, duration=duration)
    init_xyzs = scenario_gen.get_initial_positions()

    aviary = DownwashAviary(
        num_drones=num_drones,
        initial_xyzs=init_xyzs,
        ctrl_freq=int(control_hz),
        pyb_freq=240,
        gui=False,
        record=False,
        downwash_model=true_model,
    )

    ctrl = UAAdaptiveController(
        config=ua_cfg,
        num_drones=num_drones,
        assumed_config=assumed_params,
    )
    obs, _ = aviary.reset()
    ctrl.reset()

    # Telemetry arrays
    t_arr = np.arange(steps) * dt
    conf_arr = np.zeros((steps, num_drones))
    blend_arr = np.zeros((steps, num_drones))
    assumed_fz_arr = np.zeros((steps, num_drones))
    estimated_fz_arr = np.zeros((steps, num_drones))

    for step in range(steps):
        t = step * dt
        target_pos, target_vel = scenario_gen.evaluate(t)
        states = obs

        # Assumed-model forces (what the model predicts)
        assumed_forces = ctrl.assumed_model.compute_swarm_forces(states[:, 0:3], states[:, 10:13])
        assumed_fz_arr[step] = assumed_forces[:, 2]

        # UA-ADC control action
        rpms, _, _ = ctrl.compute_actions(dt, states, target_pos, target_vel)

        # Record telemetry
        conf_arr[step] = ctrl.last_confidences
        blend_arr[step] = ctrl.last_blend_weights

        # Estimated forces from RLS estimator
        for j, est in enumerate(ctrl.estimator.estimators):
            estimated_fz_arr[step, j] = est.estimate_disturbance(assumed_forces[j])[2]

        # Step simulation physics
        obs, _, _, _, _ = aviary.step(rpms)

    aviary.close()

    return {
        "t": t_arr,
        "confidences": conf_arr,
        "blend_weights": blend_arr,
        "assumed_fz": assumed_fz_arr,
        "estimated_fz": estimated_fz_arr,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

PALETTE = [
    "#4C9BE8",   # blue
    "#F57C51",   # orange
    "#57C278",   # green
    "#C678DD",   # purple
    "#E06C75",   # red
    "#56B6C2",   # cyan
    "#D19A66",   # yellow-orange
    "#98C379",   # light green
    "#E5C07B",   # gold
    "#ABB2BF",   # grey
]


def plot_signature(data: dict, num_drones: int, out_path: Path) -> None:
    """Create the 3-panel signature figure."""
    t = data["t"]
    colors = PALETTE[:num_drones]
    labels = [f"Drone {j}" for j in range(num_drones)]

    fig = plt.figure(figsize=(12, 8), facecolor="#1E1E2E")
    fig.patch.set_facecolor("#1E1E2E")
    gs = gridspec.GridSpec(3, 1, hspace=0.45, top=0.90, bottom=0.09, left=0.09, right=0.97)

    panel_bg = "#282838"
    title_color = "#CDD6F4"
    tick_color = "#BAC2DE"
    grid_color = "#45475A"

    ax_styles = dict(facecolor=panel_bg, tick_params=dict(colors=tick_color))

    def _style(ax, ylabel, ylim=None, yticks=None):
        ax.set_facecolor(panel_bg)
        ax.tick_params(colors=tick_color, labelsize=9)
        ax.spines[:].set_color(grid_color)
        ax.yaxis.label.set_color(tick_color)
        ax.xaxis.label.set_color(tick_color)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, color=grid_color, linewidth=0.5, linestyle="--", alpha=0.6)
        if ylim:
            ax.set_ylim(*ylim)
        if yticks is not None:
            ax.set_yticks(yticks)

    # Panel 1: Confidence
    ax1 = fig.add_subplot(gs[0])
    for j in range(num_drones):
        ax1.plot(t, data["confidences"][:, j], color=colors[j], lw=1.8, label=labels[j])
    ax1.axhline(0.5, color="#F38BA8", lw=1.0, linestyle=":", label="blend threshold")
    _style(ax1, "Confidence  c(t)", ylim=(-0.05, 1.05), yticks=[0, 0.25, 0.5, 0.75, 1.0])
    ax1.legend(fontsize=8, framealpha=0.3, labelcolor=tick_color, ncol=min(num_drones + 1, 6))
    ax1.set_title(
        "UA-ADC Signature: Online Confidence & Blend Weight Evolution",
        color=title_color, fontsize=11, pad=8,
    )

    # Panel 2: Blend weight α
    ax2 = fig.add_subplot(gs[1])
    for j in range(num_drones):
        ax2.plot(t, data["blend_weights"][:, j], color=colors[j], lw=1.8, label=labels[j])
    ax2.axhline(0.5, color="#F38BA8", lw=1.0, linestyle=":")
    ax2.fill_between(t, 0, 0.5, alpha=0.05, color="#4C9BE8", label="avoid dominant")
    ax2.fill_between(t, 0.5, 1, alpha=0.05, color="#57C278", label="ff dominant")
    _style(ax2, "Blend weight  α(t)", ylim=(-0.05, 1.05), yticks=[0, 0.25, 0.5, 0.75, 1.0])
    ax2.text(t[-1] * 0.02, 0.20, "← Avoid", fontsize=8, color="#4C9BE8", alpha=0.7)
    ax2.text(t[-1] * 0.02, 0.72, "← FeedFwd", fontsize=8, color="#57C278", alpha=0.7)
    ax2.legend(fontsize=8, framealpha=0.3, labelcolor=tick_color, ncol=min(num_drones, 5))

    # Panel 3: Assumed vs Estimated Fz
    ax3 = fig.add_subplot(gs[2])
    for j in range(num_drones):
        ax3.plot(t, data["assumed_fz"][:, j], color=colors[j], lw=1.2,
                 linestyle="--", alpha=0.6, label=f"D{j} assumed")
        ax3.plot(t, data["estimated_fz"][:, j], color=colors[j], lw=1.8,
                 label=f"D{j} estimated")
    _style(ax3, "Downwash  Fz  [N]")
    ax3.set_xlabel("Time  [s]", fontsize=10, color=tick_color)
    ax3.legend(fontsize=7, framealpha=0.3, labelcolor=tick_color, ncol=min(num_drones * 2, 8))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[ua_adc_signature] Saved -> {out_path}")


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="UA-ADC signature visualisation")
    parser.add_argument("--num-drones", type=int, default=3)
    parser.add_argument("--duration", type=float, default=8.0, help="Episode duration [s]")
    parser.add_argument("--out", type=str, default="results/figures/ua_adc_signature.png")
    args = parser.parse_args()

    print(f"[ua_adc_signature] Running N={args.num_drones} for {args.duration}s ...")
    data = run_signature(num_drones=args.num_drones, duration=args.duration)
    plot_signature(data, num_drones=args.num_drones, out_path=Path(args.out))


if __name__ == "__main__":
    main()
