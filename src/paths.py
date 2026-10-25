"""Project-relative paths.

Every config and results path is resolved against the repository root, so
scripts work no matter which directory they are launched from (and no matter
which drive the project was copied to).
"""
from pathlib import Path
from typing import Any, Dict, Union

import yaml

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
CONFIGS_DIR: Path = PROJECT_ROOT / "configs"
RESULTS_DIR: Path = PROJECT_ROOT / "results"


def config_path(name: str) -> Path:
    """Return the absolute path of a file inside ``configs/``."""
    return CONFIGS_DIR / name


def resolve(path: Union[str, Path]) -> Path:
    """Resolve a relative path against the project root (absolute paths pass through)."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a YAML file (relative paths are taken from the project root)."""
    with open(resolve(path), "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
