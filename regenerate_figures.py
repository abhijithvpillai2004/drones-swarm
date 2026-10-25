#!/usr/bin/env python3
"""
regenerate_figures.py
======================
Regenerate ALL analysis figures from saved results/ logs
WITHOUT re-running any simulation.

Usage:
  python regenerate_figures.py [--results-dir results] [--figures-dir results/figures]

This reads:
  - results/summary.csv           → bar charts, summary table, Wilcoxon test
  - results/downwash_sweep.csv    → heatmaps (if available)
  - results/robustness_scale_sweep.csv → mismatch curves (if available)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from src.analysis.plots import (
    plot_tracking_rmse_bars,
    plot_downwash_heatmap,
    plot_mismatch_curves,
)
from src.analysis.stats import wilcoxon_comparison, summarise_results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Regenerate all figures from saved logs")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--figures-dir", default="results/figures")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    import os; os.chdir(Path(__file__).parent)

    results_dir = Path(args.results_dir)
    figures_dir = args.figures_dir

    # 1. Bar charts from summary.csv
    summary_path = results_dir / "summary.csv"
    if summary_path.exists():
        df = pd.read_csv(summary_path)
        df["rmse_m"] = pd.to_numeric(df["rmse_m"], errors="coerce")

        for groupby in ["formation", "scenario", "n_drones"]:
            if groupby in df.columns and df[groupby].nunique() > 0:
                plot_tracking_rmse_bars(df, figures_dir=figures_dir, groupby=groupby)

        # Summary table
        summarise_results(df, figures_dir=figures_dir, metric="rmse_m")

        # Wilcoxon test
        wtest = wilcoxon_comparison(df, baseline="naive",
                                    alternatives=["soft_avoid", "feedforward"],
                                    metric="rmse_m")
        wtest_path = Path(figures_dir) / "wilcoxon_test.csv"
        wtest.to_csv(wtest_path, index=False)
        print(f"[regen] Wilcoxon results → {wtest_path}")
        print(wtest.to_string())
    else:
        print(f"[regen] No summary.csv found at {summary_path} — skipping bar charts.")

    # 2. Downwash heatmaps
    heatmap_path = results_dir / "downwash_sweep.csv"
    if heatmap_path.exists():
        df_hw = pd.read_csv(heatmap_path)
        dz_values = sorted(df_hw["dz_m"].unique())
        dxy_values = sorted(df_hw["dxy_m"].unique())
        rmse_grid = np.zeros((len(dz_values), len(dxy_values)))
        for i, dz in enumerate(dz_values):
            for j, dxy in enumerate(dxy_values):
                val = df_hw[(df_hw["dz_m"] == dz) & (df_hw["dxy_m"] == dxy)]["rmse_m"]
                rmse_grid[i, j] = val.values[0] if len(val) > 0 else 0.0
        plot_downwash_heatmap(
            np.array(dz_values), np.array(dxy_values), rmse_grid,
            metric_name="Tracking RMSE [m]",
            figures_dir=figures_dir, filename="heatmap_rmse",
        )
    else:
        print(f"[regen] No downwash_sweep.csv found — skipping heatmaps.")
        print(f"        Run: python characterise_downwash.py")

    # 3. Mismatch curves
    mismatch_path = results_dir / "robustness_scale_sweep.csv"
    if mismatch_path.exists():
        df_mm = pd.read_csv(mismatch_path)
        df_gauss = df_mm[df_mm["falloff_shape"] == "gaussian"]
        scale_factors = sorted(df_gauss["scale_factor"].unique())
        controllers = sorted(df_gauss["controller"].unique())
        rmse_by_ctrl = {}
        for ctrl in controllers:
            means = [
                df_gauss[(df_gauss["controller"] == ctrl) &
                         (df_gauss["scale_factor"] == sf)]["rmse_m"].mean()
                for sf in scale_factors
            ]
            rmse_by_ctrl[ctrl] = means
        plot_mismatch_curves(scale_factors, rmse_by_ctrl, figures_dir=figures_dir)
    else:
        print(f"[regen] No robustness_scale_sweep.csv found — skipping mismatch curves.")
        print(f"        Run: python robustness_study.py")

    print(f"\n[regen] All available figures saved to {figures_dir}/")


if __name__ == "__main__":
    main()
