"""Apply the committed pass criteria (configs/pass_criteria.yaml) to run tables.

Input is a table with one row per run and at least the columns written by
``src.sim.episode.summarise_episode`` plus ``scenario``, ``N``, ``seed`` and
``controller``.  Output adds one boolean/NA column per criterion and an
overall ``passed`` column, so a reviewer can see exactly why a run failed.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.paths import config_path, load_yaml


def load_criteria(path: Optional[str] = None) -> Dict[str, Any]:
    """Load the pass criteria YAML."""
    return load_yaml(path or config_path("pass_criteria.yaml"))["criteria"]


def apply_criteria(df: pd.DataFrame, criteria: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Return a copy of ``df`` with per-criterion verdicts and ``passed``.

    Criterion columns hold True (pass), False (fail) or NA (not applicable).
    """
    crit = criteria or load_criteria()
    out = df.copy()

    c = crit["no_ground_contact"]
    out["crit_no_ground_contact"] = out["min_z_m"] > float(c["min_z_m"])

    c = crit["min_separation"]
    out["crit_min_separation"] = out["min_sep_m"] > float(c["min_sep_m"])

    # Confidence rise: UA-ADC only, and only when some drone starts in a wake.
    c = crit["confidence_rise"]
    applies = out["controller"].isin(c.get("applies_to", ["ua_adc"])) & (out["in_wake_drones"] > 0)
    rise = pd.to_numeric(out["conf_rise_time_max_s"], errors="coerce")
    verdict = pd.Series(pd.NA, index=out.index, dtype="boolean")
    verdict[applies] = (rise[applies] <= float(c["within_s"])).fillna(False)
    out["crit_confidence_rise"] = verdict

    # Tracking vs baseline (paired on scenario, N, seed).
    c = crit["tracking_vs_baseline"]
    key = ["scenario", "N", "seed"]
    base = out[out["controller"] == c["baseline"]][key + ["rmse_mean_m"]].rename(
        columns={"rmse_mean_m": "baseline_rmse_m"})
    out = out.merge(base, on=key, how="left")
    applies = out["controller"].isin(c.get("applies_to", ["ua_adc"])) & out["baseline_rmse_m"].notna()
    verdict = pd.Series(pd.NA, index=out.index, dtype="boolean")
    verdict[applies] = out.loc[applies, "rmse_mean_m"] <= out.loc[applies, "baseline_rmse_m"] * float(c["tolerance"])
    out["crit_tracking_vs_baseline"] = verdict

    crit_cols = [col for col in out.columns if col.startswith("crit_")]

    def _row_pass(row: pd.Series) -> bool:
        vals = [row[col] for col in crit_cols if not pd.isna(row[col])]
        return bool(all(vals)) and not bool(row.get("has_nan", False))

    def _row_reason(row: pd.Series) -> str:
        failed = [col[5:] for col in crit_cols if not pd.isna(row[col]) and not bool(row[col])]
        if bool(row.get("has_nan", False)):
            failed.insert(0, "nan_state")
        return "OK" if not failed else ";".join(failed)

    out["passed"] = out.apply(_row_pass, axis=1)
    out["fail_reason"] = out.apply(_row_reason, axis=1)
    return out


def summarise_pass_rates(checked: pd.DataFrame) -> pd.DataFrame:
    """Pass counts per controller (and overall), for the README/report."""
    g = checked.groupby("controller")["passed"].agg(["sum", "count"]).reset_index()
    g = g.rename(columns={"sum": "passed", "count": "runs"})
    g["pass_rate"] = np.where(g["runs"] > 0, g["passed"] / g["runs"], np.nan)
    return g
