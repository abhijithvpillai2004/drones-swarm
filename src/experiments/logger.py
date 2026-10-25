"""Standardized reproducible experiment run logger.

Generates consistent per-run log files with config hash, random seed, controller name,
and scenario name. Guarantees that prior results are never silently overwritten.
"""
import os
import json
import hashlib
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd


def compute_config_hash(config: Dict[str, Any], length: int = 8) -> str:
    """Compute a deterministic hash of a configuration dictionary.

    Args:
        config: Nested dictionary of configuration parameters.
        length: Number of leading hexadecimal characters to return.

    Returns:
        hash_string: Truncated SHA-256 hash string.
    """
    serialized = json.dumps(config, sort_keys=True, default=str)
    full_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return full_hash[:length]


class ExperimentLogger:
    """Per-run structured logger for swarm flight trajectories and summary metrics."""

    def __init__(
        self,
        scenario_name: str,
        controller_name: str,
        seed: int,
        config: Dict[str, Any],
        output_dir: str = "results",
        allow_overwrite: bool = False,
    ) -> None:
        """Initialize logger and resolve non-colliding file paths.

        Args:
            scenario_name: Identifier of the test scenario.
            controller_name: Name of controller (e.g. 'naive', 'soft_avoid', 'feedforward').
            seed: Simulation random seed.
            config: Full experiment configuration dictionary.
            output_dir: Root directory for results.
            allow_overwrite: If False, appends version suffix if file exists.
        """
        self.scenario_name: str = scenario_name
        self.controller_name: str = controller_name
        self.seed: int = seed
        self.config: Dict[str, Any] = config
        self.config_hash: str = compute_config_hash(config)
        self.output_dir: str = output_dir
        self.allow_overwrite: bool = allow_overwrite
        self.records: List[Dict[str, Any]] = []
        self.start_time: str = datetime.now().isoformat()

        os.makedirs(self.output_dir, exist_ok=True)

        base_name = f"{self.scenario_name}_{self.controller_name}_seed{self.seed}_{self.config_hash}"
        csv_filename = f"{base_name}.csv"
        meta_filename = f"{base_name}_meta.json"

        # Prevent silent overwrite by finding a unique suffix if necessary
        if not self.allow_overwrite:
            counter = 1
            while os.path.exists(os.path.join(self.output_dir, csv_filename)):
                csv_filename = f"{base_name}_v{counter}.csv"
                meta_filename = f"{base_name}_v{counter}_meta.json"
                counter += 1

        self.csv_path: str = os.path.join(self.output_dir, csv_filename)
        self.meta_path: str = os.path.join(self.output_dir, meta_filename)

    def log_step(
        self,
        timestamp: float,
        step: int,
        drone_id: int,
        pos: Any,
        target_pos: Any,
        rpy: Any,
        vel: Any,
        rpms: Any,
        disturb_force: Optional[Any] = None,
    ) -> None:
        """Record state and commands for a single drone at the current timestep.

        Args:
            timestamp: Simulation time in seconds.
            step: Control step counter.
            drone_id: Drone index.
            pos: Array-like of length 3 [x, y, z].
            target_pos: Array-like of length 3 [tx, ty, tz].
            rpy: Array-like of length 3 [roll, pitch, yaw].
            vel: Array-like of length 3 [vx, vy, vz].
            rpms: Array-like of length 4 [rpm0, rpm1, rpm2, rpm3].
            disturb_force: Optional array-like of length 3 [fx, fy, fz].
        """
        err_xyz = [pos[i] - target_pos[i] for i in range(3)]
        err_norm = float((err_xyz[0]**2 + err_xyz[1]**2 + err_xyz[2]**2)**0.5)

        fx, fy, fz = (0.0, 0.0, 0.0) if disturb_force is None else (disturb_force[0], disturb_force[1], disturb_force[2])

        record = {
            "timestamp": float(timestamp),
            "step": int(step),
            "drone_id": int(drone_id),
            "x": float(pos[0]),
            "y": float(pos[1]),
            "z": float(pos[2]),
            "target_x": float(target_pos[0]),
            "target_y": float(target_pos[1]),
            "target_z": float(target_pos[2]),
            "err_x": float(err_xyz[0]),
            "err_y": float(err_xyz[1]),
            "err_z": float(err_xyz[2]),
            "err_norm": err_norm,
            "roll": float(rpy[0]),
            "pitch": float(rpy[1]),
            "yaw": float(rpy[2]),
            "vx": float(vel[0]),
            "vy": float(vel[1]),
            "vz": float(vel[2]),
            "rpm_0": float(rpms[0]),
            "rpm_1": float(rpms[1]),
            "rpm_2": float(rpms[2]),
            "rpm_3": float(rpms[3]),
            "disturb_fx": float(fx),
            "disturb_fy": float(fy),
            "disturb_fz": float(fz),
        }
        self.records.append(record)

    def save(self, summary_metrics: Optional[Dict[str, Any]] = None) -> Tuple[str, str]:
        """Save collected timeseries data to CSV and run metadata to JSON.

        Args:
            summary_metrics: Optional dictionary of aggregated run metrics.

        Returns:
            Tuple of (csv_path, meta_path).
        """
        df = pd.DataFrame(self.records)
        df.to_csv(self.csv_path, index=False)

        metadata = {
            "scenario_name": self.scenario_name,
            "controller_name": self.controller_name,
            "seed": self.seed,
            "config_hash": self.config_hash,
            "start_time": self.start_time,
            "end_time": datetime.now().isoformat(),
            "total_records": len(self.records),
            "summary_metrics": summary_metrics or {},
            "config": self.config,
            "files": {
                "csv": os.path.basename(self.csv_path),
                "meta": os.path.basename(self.meta_path),
            },
        }

        with open(self.meta_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        return self.csv_path, self.meta_path
