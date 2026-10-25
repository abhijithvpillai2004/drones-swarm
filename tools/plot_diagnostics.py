import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

df = pd.read_csv("results/diagnostics/ua_adc/ua_adc_trace_nominal.csv")

# Filter for Drone 1 (middle drone, directly in wake)
d1 = df[df["drone_id"] == 1].copy()
d2 = df[df["drone_id"] == 2].copy()

t = d1["t"].values

fig = plt.figure(figsize=(14, 12), facecolor="#181825")
gs = gridspec.GridSpec(4, 1, hspace=0.35, top=0.94, bottom=0.06, left=0.08, right=0.96)

bg_color = "#1e1e2e"
grid_color = "#313244"
text_color = "#cdd6f4"

def style_ax(ax, title, ylabel):
    ax.set_facecolor(bg_color)
    ax.grid(True, color=grid_color, linestyle="--", linewidth=0.6, alpha=0.7)
    ax.tick_params(colors=text_color, labelsize=9)
    ax.spines["top"].set_color(grid_color)
    ax.spines["bottom"].set_color(grid_color)
    ax.spines["left"].set_color(grid_color)
    ax.spines["right"].set_color(grid_color)
    ax.set_title(title, color=text_color, fontsize=11, fontweight="bold", pad=8)
    ax.set_ylabel(ylabel, color=text_color, fontsize=10)
    ax.set_xlim(0, 10.0)

# Panel 1: Disturbance Forces (True, Assumed, Estimated)
ax1 = fig.add_subplot(gs[0])
ax1.plot(t, d1["F_true"], color="#a6e3a1", lw=2.0, label="True Downwash Force F_true (Drone 1)")
ax1.plot(t, d1["F_assumed"], color="#89b4fa", lw=1.8, linestyle="--", label="Assumed Downwash Force F_assumed (Drone 1)")
ax1.plot(t, d1["F_hat"], color="#f38ba8", lw=2.2, label="Estimated Downwash Force F_hat = theta * F_assumed (Decays to 0!)")
ax1.axhline(0, color="#6c7086", lw=0.8, linestyle=":")
style_ax(ax1, "Panel 1: Disturbance Forces Over Time (Drone 1) — Estimator Decays to 0 Despite Active Downwash", "Force [N]")
ax1.legend(loc="upper right", framealpha=0.4, facecolor=bg_color, labelcolor=text_color, fontsize=9)

# Panel 2: Acceleration Breakdown & Residual
ax2 = fig.add_subplot(gs[1])
ax2.plot(t, d1["az_obs"], color="#fab387", lw=1.5, alpha=0.8, label="Measured az_obs = dv_z / dt ~ 0 (Kinematic Acceleration)")
ax2.plot(t, d1["a_no_wash"], color="#cba6f7", lw=1.5, label="Thrust-Weight Accel a_no_wash = (T_cmd - mg)/m")
ax2.plot(t, d1["residual"], color="#f9e2af", lw=1.8, label="Residual az_obs - a_pred (Mean ~ 0.008 m/s^2)")
ax2.axhline(0, color="#6c7086", lw=0.8, linestyle=":")
style_ax(ax2, "Panel 2: Acceleration Signals — Controller Motor Thrust Cancels Wake, Driving Observed Acceleration to 0", "Accel [m/s^2]")
ax2.legend(loc="upper right", framealpha=0.4, facecolor=bg_color, labelcolor=text_color, fontsize=9)

# Panel 3: Estimator Gain theta and Covariance P
ax3 = fig.add_subplot(gs[2])
ax3_twin = ax3.twinx()
p1 = ax3.plot(t, d1["theta"], color="#f38ba8", lw=2.0, label="Gain Estimate theta(t) (decays from 1.0 -> 0.007)")
p2 = ax3_twin.plot(t, d1["P"], color="#89dceb", lw=2.0, linestyle="-.", label="Covariance P(t) (Windup: 1.0 -> 62.9)")
ax3.axhline(1.0, color="#6c7086", lw=0.8, linestyle=":")
style_ax(ax3, "Panel 3: RLS State Dynamics — Parameter Decay & Covariance Windup (Lack of Persistent Excitation)", "Gain theta")
ax3_twin.set_ylabel("Posterior Variance P", color="#89dceb", fontsize=10)
ax3_twin.tick_params(colors="#89dceb", labelsize=9)
ax3_twin.spines["top"].set_color(grid_color)
ax3_twin.spines["bottom"].set_color(grid_color)
ax3_twin.spines["left"].set_color(grid_color)
ax3_twin.spines["right"].set_color(grid_color)
lines = p1 + p2
labels = [l.get_label() for l in lines]
ax3.legend(lines, labels, loc="upper right", framealpha=0.4, facecolor=bg_color, labelcolor=text_color, fontsize=9)

# Panel 4: Confidence c(t) and Blend Weight alpha(t)
ax4 = fig.add_subplot(gs[3])
ax4.plot(t, d1["confidence"], color="#f38ba8", lw=2.5, label="Confidence c_1(t) = 0.000 (Locked at 0 because P >= 1.0)")
ax4.plot(t, d1["blend_weight"], color="#eba0ac", lw=2.5, linestyle="--", label="Blend Weight alpha_1(t) = 0.01799 (sigma(-4.0) pure avoid)")
ax4.axhline(0.5, color="#f9e2af", lw=1.0, linestyle=":", label="Sigmoid Midpoint Threshold (c = 0.5)")
style_ax(ax4, "Panel 4: Confidence & Blend Weight — Frozen at c=0.0 and alpha ~ 0.018 (Controller Never Blends to Feedforward)", "Weight [0, 1]")
ax4.set_ylim(-0.05, 1.05)
ax4.set_xlabel("Time [s]", color=text_color, fontsize=10)
ax4.legend(loc="upper right", framealpha=0.4, facecolor=bg_color, labelcolor=text_color, fontsize=9)

out_plot = Path("results/diagnostics/ua_adc/ua_adc_trace_nominal.png")
plt.savefig(out_plot, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
plt.close()
print(f"Saved diagnostic figure to {out_plot}")
