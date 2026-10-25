"""Parallel batch execution of episodes described as plain dicts.

A job is a dict such as::

    {"scenario": "vertical_swap", "N": 2, "controller": "ua_adc", "seed": 3,
     "duration": None, "true_scale": 1.0, "assumed_scale": 1.0,
     "observation_noise": False, "delay_steps": 0, "ua_overrides": {}}

``run_job`` returns one flat row (job fields + summary metrics), so results
of any study can be written straight to CSV and checked against the pass
criteria.  Jobs run in a process pool; each process has its own PyBullet
client, so runs do not interfere.
"""
from __future__ import annotations

import contextlib
import copy
import io
import multiprocessing as mp
import os
import sys
import time
import traceback
import warnings
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from src.experiments.catalog import make_controller, make_scenario
from src.sim.episode import EpisodeConfig, load_assumed_config, load_true_model, run_episode

JOB_DEFAULTS: Dict[str, Any] = {
    "duration": None,
    "seed": None,
    "true_scale": 1.0,
    "assumed_scale": 1.0,
    "observation_noise": False,
    "delay_steps": 0,
    "ua_overrides": None,
}


def _deep_update(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def run_job(job: Dict[str, Any]) -> Dict[str, Any]:
    """Run one job and return a flat result row (never raises)."""
    j = {**JOB_DEFAULTS, **job}
    row: Dict[str, Any] = {k: v for k, v in j.items() if k != "ua_overrides"}
    try:
        gen = make_scenario(j["scenario"], j["N"], j["duration"])
        true_model = load_true_model(force_scale=float(j["true_scale"]))
        assumed = load_assumed_config(force_scale=float(j["assumed_scale"]))
        ua_cfg = None
        if j["controller"] == "ua_adc" and j["ua_overrides"]:
            from src.controllers.ua_adc import load_default_ua_adc_config
            ua_cfg = _deep_update(copy.deepcopy(load_default_ua_adc_config()), j["ua_overrides"])
        ctrl = make_controller(j["controller"], gen.num_drones, assumed_config=assumed, ua_config=ua_cfg)
        # The simulator prints URDF parameters on every construction; keep batch output readable.
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = run_episode(gen, ctrl, true_model, EpisodeConfig(
                seed=j["seed"],
                observation_noise=bool(j["observation_noise"]),
                delay_steps=int(j["delay_steps"]),
            ))
        row.update(res.metrics)
        # Learned gain averaged over drones that start inside a TRUE wake
        # (drones never in a wake keep their initial value 1 and would bias it).
        in_wake = np.abs(res.true_force[0, :, 2]) > 0.01 * 0.027 * 9.81 if res.t.size else np.zeros(0, bool)
        row["final_theta_mean"] = float(res.theta[-1, in_wake].mean()) if in_wake.any() else float("nan")
        row["wall_time_s"] = res.wall_time_s
        row["error"] = ""
    except Exception as exc:  # recorded, not hidden: shows up as an ERROR row
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["traceback"] = traceback.format_exc(limit=3)
    return row


def run_jobs(jobs: Iterable[Dict[str, Any]], workers: Optional[int] = None,
             progress: bool = True) -> List[Dict[str, Any]]:
    """Run jobs in parallel and return rows in the original job order."""
    jobs = list(jobs)
    workers = workers if workers is not None else max(1, (os.cpu_count() or 2) - 1)
    rows: List[Optional[Dict[str, Any]]] = [None] * len(jobs)
    start = time.time()

    def _report(done: int) -> None:
        if progress:
            el = time.time() - start
            eta = el / done * (len(jobs) - done) if done else 0.0
            sys.stdout.write(f"\r  [{done:4d}/{len(jobs)}] elapsed {el:6.0f}s  ETA {eta:6.0f}s ")
            sys.stdout.flush()

    if workers <= 1 or len(jobs) <= 1:
        for i, job in enumerate(jobs):
            rows[i] = run_job(job)
            _report(i + 1)
    else:
        with mp.get_context("spawn").Pool(processes=workers) as pool:
            for done, (i, row) in enumerate(pool.imap_unordered(_indexed_run, list(enumerate(jobs))), 1):
                rows[i] = row
                _report(done)
    if progress:
        sys.stdout.write("\n")
    return [r for r in rows if r is not None]


def _indexed_run(item):
    i, job = item
    return i, run_job(job)
