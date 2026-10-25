"""
src/analysis/stats.py
======================
Statistical analysis: Wilcoxon signed-rank test and summary table generation.

Design decision D-012 (docs/decisions.md):
  Uses the non-parametric Wilcoxon signed-rank test (paired, two-sided) to
  compare controllers B and C against the Naive baseline A. This is more
  appropriate than a t-test because RMSE distributions from downwash
  simulations are unlikely to be Gaussian (heavy tails from disturbance
  episodes).

  Significance threshold: α = 0.05.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def wilcoxon_comparison(
    summary_df: pd.DataFrame,
    baseline: str = "naive",
    alternatives: Optional[List[str]] = None,
    metric: str = "rmse_m",
    groupby: Optional[str] = None,
) -> pd.DataFrame:
    """Wilcoxon signed-rank test comparing alternatives against baseline.

    For each unique value of *groupby* (if specified), tests whether
    each alternative controller's metric distribution differs significantly
    from the baseline.

    Parameters
    ----------
    summary_df : DataFrame — loaded from results/summary.csv
    baseline : str — baseline controller name ("naive")
    alternatives : list[str] — controllers to compare against baseline
    metric : str — column name of the metric to compare
    groupby : str or None — group variable (e.g. "formation"); if None,
        test over all data.

    Returns
    -------
    results_df : DataFrame
        Columns: group, controller, n_samples, statistic, p_value, significant
    """
    alternatives = alternatives or ["soft_avoid", "feedforward"]
    summary_df = summary_df.copy()
    summary_df[metric] = summary_df[metric].astype(float)

    rows = []

    def _test_group(df_group: pd.DataFrame, group_label: str) -> None:
        base_vals = df_group[df_group["controller"] == baseline][metric].values
        for alt in alternatives:
            alt_vals = df_group[df_group["controller"] == alt][metric].values

            # Match by seed for paired test
            df_base = df_group[df_group["controller"] == baseline][["seed", metric]]
            df_alt = df_group[df_group["controller"] == alt][["seed", metric]]
            merged = pd.merge(df_base, df_alt, on="seed", suffixes=("_base", "_alt"))

            if len(merged) < 3:
                rows.append({
                    "group": group_label, "controller": alt,
                    "n_samples": len(merged), "statistic": np.nan,
                    "p_value": np.nan, "significant": False,
                    "mean_baseline": float(np.mean(base_vals)) if len(base_vals) else np.nan,
                    "mean_alt": float(np.mean(alt_vals)) if len(alt_vals) else np.nan,
                })
                continue

            try:
                stat, p = stats.wilcoxon(
                    merged[f"{metric}_base"].values,
                    merged[f"{metric}_alt"].values,
                    alternative="two-sided",
                )
            except ValueError:
                stat, p = np.nan, np.nan

            rows.append({
                "group": group_label,
                "controller": alt,
                "n_samples": len(merged),
                "statistic": round(float(stat), 4) if not np.isnan(stat) else np.nan,
                "p_value": round(float(p), 6) if not np.isnan(p) else np.nan,
                "significant": bool(p < 0.05) if not np.isnan(p) else False,
                "mean_baseline": float(np.mean(merged[f"{metric}_base"])),
                "mean_alt": float(np.mean(merged[f"{metric}_alt"])),
            })

    if groupby is not None:
        for grp_val, grp_df in summary_df.groupby(groupby):
            _test_group(grp_df, str(grp_val))
    else:
        _test_group(summary_df, "all")

    return pd.DataFrame(rows)


def summarise_results(
    summary_df: pd.DataFrame,
    figures_dir: str = "results/figures",
    metric: str = "rmse_m",
) -> pd.DataFrame:
    """Generate and save a summary table (CSV + markdown).

    Pivot table: rows = controller × formation, columns = metric statistics.

    Parameters
    ----------
    summary_df : DataFrame
    figures_dir : str
    metric : str

    Returns
    -------
    pivot : DataFrame
    """
    summary_df = summary_df.copy()
    summary_df[metric] = summary_df[metric].astype(float)

    pivot = summary_df.groupby(["controller", "formation"])[metric].agg(
        ["mean", "std", "min", "max", "count"]
    ).round(4)

    Path(figures_dir).mkdir(parents=True, exist_ok=True)

    pivot.to_csv(Path(figures_dir) / "summary_table.csv")
    pivot.to_markdown(Path(figures_dir) / "summary_table.md")

    print(f"[stats] Summary table saved to {figures_dir}/summary_table.csv/.md")
    return pivot
