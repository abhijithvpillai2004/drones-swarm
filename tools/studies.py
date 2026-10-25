"""Reproducible studies for the open items, written to results/studies/.

    python tools/studies.py regression     # item 4: strict regression, 8 scenarios x N=2..10 x 4 controllers
    python tools/studies.py vswap          # item 2: vertical-swap failure rate, 20 seeds
    python tools/studies.py gateb          # item 5: model mismatch, 20 seeds, paired test
    python tools/studies.py wake           # item 6: does the N=8 orbit collapse depend on wake strength?
    python tools/studies.py stress         # stress case (1.3x force, noise, 2-step delay), 20 seeds
    python tools/studies.py all            # everything above, then the report
    python tools/studies.py report         # rebuild docs/results_v2.md from the saved CSVs only

Every number in docs/results_v2.md is computed from these CSVs, so it can be
traced back to a file (project rule).  Pass/fail uses configs/pass_criteria.yaml.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.evaluation.criteria import apply_criteria, summarise_pass_rates  # noqa: E402
from src.experiments.batch import run_jobs  # noqa: E402
from src.experiments.catalog import CONTROLLER_NAMES, SCENARIO_NAMES  # noqa: E402

OUT = PROJECT_ROOT / "results" / "studies"
NO_OVERRIDE = {"conflict_override": {"enabled": False}}


def _save(rows: List[Dict[str, Any]], name: str) -> pd.DataFrame:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if "traceback" in df.columns:
        df = df.drop(columns=["traceback"])
    df.to_csv(OUT / f"{name}.csv", index=False)
    errors = df[df["error"].astype(str) != ""] if "error" in df.columns else df.iloc[0:0]
    print(f"[saved] {OUT / (name + '.csv')}  ({len(df)} rows, {len(errors)} errors)")
    for _, r in errors.head(5).iterrows():
        print(f"   ERROR {r.get('scenario')} N={r.get('N')} {r.get('controller')}: {r['error']}")
    return df


def _check(df: pd.DataFrame, name: str) -> pd.DataFrame:
    ok = df[df["error"].astype(str) == ""].copy()
    checked = apply_criteria(ok)
    checked.to_csv(OUT / f"{name}_checked.csv", index=False)
    print(summarise_pass_rates(checked).to_string(index=False))
    return checked


# ── Studies ──────────────────────────────────────────────────────────────────

def study_regression(workers, seed: int = 0) -> None:
    jobs = [{"study": "regression", "variant": c, "scenario": s, "N": n, "controller": c, "seed": seed}
            for s in SCENARIO_NAMES for n in range(2, 11) for c in CONTROLLER_NAMES]
    jobs += [{"study": "regression", "variant": "ua_adc_no_override", "scenario": s, "N": n,
              "controller": "ua_adc", "seed": seed, "ua_overrides": NO_OVERRIDE}
             for s in SCENARIO_NAMES for n in range(2, 11)]
    df = _save(run_jobs(jobs, workers), "strict_regression")
    main = df[df["variant"] != "ua_adc_no_override"]
    _check(main, "strict_regression")
    ablation = pd.concat([df[df["variant"] == "soft_avoid"], df[df["variant"] == "ua_adc_no_override"]])
    ablation.loc[ablation["variant"] == "ua_adc_no_override", "controller"] = "ua_adc"
    apply_criteria(ablation[ablation["error"].astype(str) == ""]).to_csv(
        OUT / "strict_regression_no_override_checked.csv", index=False)


def study_vswap(workers, seeds: int = 20) -> None:
    jobs = []
    for s in range(seeds):
        for c in CONTROLLER_NAMES:
            jobs.append({"study": "vswap", "variant": c, "scenario": "vertical_swap", "N": 2,
                         "controller": c, "seed": s})
        jobs.append({"study": "vswap", "variant": "ua_adc_no_override", "scenario": "vertical_swap",
                     "N": 2, "controller": "ua_adc", "seed": s, "ua_overrides": NO_OVERRIDE})
    df = _save(run_jobs(jobs, workers), "vswap_repeats")
    df = df[df["error"].astype(str) == ""]
    df["fail"] = (df["min_z_m"] <= 0.05) | (df["min_sep_m"] <= 0.12)
    print(df.groupby("variant").agg(runs=("fail", "size"), failures=("fail", "sum"),
                                    min_sep_mean=("min_sep_m", "mean")).to_string())


def study_gateb(workers, seeds: int = 20) -> None:
    jobs = [{"study": "gateb", "variant": c, "scenario": "hover_column_3d", "N": 3, "duration": 6.0,
             "controller": c, "seed": s, "assumed_scale": a}
            for a in (0.5, 1.0, 1.5, 3.0) for c in CONTROLLER_NAMES for s in range(seeds)]
    _save(run_jobs(jobs, workers), "gateb_mismatch_20seeds")


def study_wake(workers) -> None:
    jobs = [{"study": "wake", "variant": c, "scenario": "orbit_stack", "N": n, "controller": c,
             "seed": 0, "true_scale": k, "assumed_scale": k}
            for k in (0.25, 0.5, 0.75, 1.0) for n in (5, 6, 7, 8, 9, 10)
            for c in ("soft_avoid", "ua_adc", "feedforward")]
    _save(run_jobs(jobs, workers), "wake_strength_orbit")


def study_stress(workers, seeds: int = 20) -> None:
    jobs = [{"study": "stress", "variant": c, "scenario": "hover_column_3d", "N": 3, "duration": 6.0,
             "controller": c, "seed": s, "true_scale": 1.3, "observation_noise": True, "delay_steps": 2}
            for c in CONTROLLER_NAMES for s in range(seeds)]
    _save(run_jobs(jobs, workers), "stress_20seeds")


# ── Report (reads CSVs only) ─────────────────────────────────────────────────

def _cm(x: float) -> str:
    return "n/a" if pd.isna(x) else f"{100 * x:.2f}"


def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def _wilcoxon(x: np.ndarray, y: np.ndarray) -> str:
    from scipy import stats
    d = np.asarray(x) - np.asarray(y)
    if len(d) < 6 or np.allclose(d, 0):
        return "n/a"
    return f"{stats.wilcoxon(x, y).pvalue:.2e}"


def build_report() -> Path:
    parts: List[str] = [
        "# Results v2 (generated)",
        "",
        "Generated by `python tools/studies.py report` from the CSV files in `results/studies/`. "
        "Do not edit by hand; rerun the studies instead. Pass/fail uses `configs/pass_criteria.yaml`. "
        "Tracking error is mean RMSE over drones and time, in cm.",
        "",
    ]

    f = OUT / "strict_regression_checked.csv"
    if f.exists():
        d = pd.read_csv(f)
        pr = summarise_pass_rates(d)
        pr["pass_rate"] = (100 * pr["pass_rate"]).round(1).astype(str) + " %"
        parts += ["## 1. Strict regression (8 scenarios x N = 2..10, seed 0)", "",
                  _md_table(pr), ""]
        fails = d[~d["passed"]][["scenario", "N", "controller", "fail_reason", "min_sep_m", "min_z_m"]]
        fails = fails.assign(min_sep_m=fails["min_sep_m"].round(3), min_z_m=fails["min_z_m"].round(3))
        parts += ["Failed runs:", "", _md_table(fails) if len(fails) else "None.", ""]
        f2 = OUT / "strict_regression_no_override_checked.csv"
        if f2.exists():
            d2 = pd.read_csv(f2)
            u2 = d2[d2["controller"] == "ua_adc"]
            parts += [f"Ablation: UA-ADC with the conflict override disabled passes "
                      f"{int(u2['passed'].sum())} / {len(u2)} runs.", ""]

    f = OUT / "vswap_repeats.csv"
    if f.exists():
        d = pd.read_csv(f)
        d = d[d["error"].isna() | (d["error"].astype(str) == "")]
        d["fail"] = (d["min_z_m"] <= 0.05) | (d["min_sep_m"] <= 0.12)
        g = d.groupby("variant").agg(runs=("fail", "size"), failures=("fail", "sum"),
                                     min_sep_mean_m=("min_sep_m", "mean"), min_sep_worst_m=("min_sep_m", "min"),
                                     rmse_mean=("rmse_mean_m", "mean")).reset_index()
        g["min_sep_mean_m"] = g["min_sep_mean_m"].round(3)
        g["min_sep_worst_m"] = g["min_sep_worst_m"].round(3)
        g["rmse_mean"] = g["rmse_mean"].map(_cm)
        parts += ["## 2. Vertical swap, N = 2, repeated over seeds", "",
                  "A run fails on ground contact (z <= 0.05 m) or separation <= 0.12 m.", "",
                  _md_table(g), ""]

    f = OUT / "gateb_mismatch_20seeds.csv"
    if f.exists():
        d = pd.read_csv(f)
        rows = []
        for a, sub in d.groupby("assumed_scale"):
            row = {"assumed scale": a}
            for c in CONTROLLER_NAMES:
                v = sub[sub["controller"] == c]["rmse_mean_m"]
                row[c] = f"{_cm(v.mean())} +/- {_cm(v.std())}"
            ua = sub[sub["controller"] == "ua_adc"].sort_values("seed")["rmse_mean_m"].values
            ff = sub[sub["controller"] == "feedforward"].sort_values("seed")["rmse_mean_m"].values
            row["n"] = len(ua)
            if len(ua) == len(ff):
                row["UA better in"] = f"{int(np.sum(ua < ff))}/{len(ua)} seeds"
                row["p (UA vs FF, Wilcoxon)"] = _wilcoxon(ua, ff)
            th = sub[sub["controller"] == "ua_adc"]["final_theta_mean"]
            row["UA final theta (ideal)"] = f"{th.mean():.2f} ({1.0 / a:.2f})"
            rows.append(row)
        parts += ["## 3. Model mismatch (Gate B), hover column N = 3, 20 seeds", "",
                  "Assumed wake force = scale x true. Values are mean +/- std of RMSE in cm. "
                  "`UA final theta` is the learned gain at the end of the run, averaged over the "
                  "drones that start in a wake; the ideal value is 1/scale. "
                  "With 20 pairs, the smallest possible two-sided Wilcoxon p-value is 1.91e-06; it "
                  "appears whenever one controller is better in all 20 seeds, so read it together "
                  "with the `UA better in` column for the direction.", "",
                  _md_table(pd.DataFrame(rows)), ""]

    f = OUT / "wake_strength_orbit.csv"
    if f.exists():
        d = pd.read_csv(f)
        d["crash"] = d["min_z_m"] <= 0.05
        piv = d.pivot_table(index=["true_scale", "N"], columns="controller", values="crash",
                            aggfunc="first").reset_index()
        for c in piv.columns[2:]:
            piv[c] = piv[c].map({True: "ground", False: "ok"})
        parts += ["## 4. Orbit stack: ground contact vs wake strength", "",
                  "TRUE and ASSUMED wake force both scaled by `true_scale` (model kept matched).", "",
                  _md_table(piv), ""]

    f = OUT / "stress_20seeds.csv"
    if f.exists():
        d = pd.read_csv(f)
        g = d.groupby("controller").agg(rmse_mean=("rmse_mean_m", "mean"), rmse_std=("rmse_mean_m", "std"),
                                        ground=("ground_contact", "sum"), runs=("seed", "size")).reset_index()
        g["rmse_mean"] = g["rmse_mean"].map(_cm)
        g["rmse_std"] = g["rmse_std"].map(_cm)
        parts += ["## 5. Stress case (1.3x true force, observation noise, 2-step delay), 20 seeds", "",
                  _md_table(g), ""]

    out = PROJECT_ROOT / "docs" / "results_v2.md"
    out.write_text("\n".join(parts), encoding="utf-8")
    print(f"[report] {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("study", choices=["regression", "vswap", "gateb", "wake", "stress", "all", "report"])
    ap.add_argument("--workers", type=int, default=None, help="parallel processes (default: CPUs - 1)")
    ap.add_argument("--seeds", type=int, default=20)
    a = ap.parse_args()

    todo = ["regression", "vswap", "gateb", "wake", "stress"] if a.study == "all" else [a.study]
    for s in todo:
        print(f"\n=== {s} ===")
        if s == "regression":
            study_regression(a.workers)
        elif s == "vswap":
            study_vswap(a.workers, a.seeds)
        elif s == "gateb":
            study_gateb(a.workers, a.seeds)
        elif s == "wake":
            study_wake(a.workers)
        elif s == "stress":
            study_stress(a.workers, a.seeds)
    build_report()


if __name__ == "__main__":
    main()
