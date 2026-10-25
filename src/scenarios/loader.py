"""
src/scenarios/loader.py
========================
Load and validate scenario YAML files.
"""

from __future__ import annotations

from pathlib import Path

import yaml


def load_scenario(path: str) -> dict:
    """Load a scenario YAML file and return its contents as a dict.

    Parameters
    ----------
    path : str
        Path to a YAML file under configs/scenarios/.

    Returns
    -------
    scenario : dict
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Scenario file not found: {p}")
    with open(p, "r", encoding="utf-8") as fh:
        scenario = yaml.safe_load(fh)
    if not isinstance(scenario, dict):
        raise ValueError(f"Scenario file must be a YAML mapping: {p}")
    return scenario
