"""Multi-drone comparative benchmark experiment runner (Phase 5).

Evaluates and compares Controllers A (Naive), B (Soft-Avoidance), and C (Feedforward)
across 3+ drone scenarios under TRUE physical downwash disturbance in PyBullet.
Generates comprehensive metric tables and publication-ready figures.
Supports configurable swarm sizes N in [2, 10].
"""
from src.paths import resolve
import os
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import yaml
import numpy as np
import pandas as pd

from src.models.downwash import DownwashModel
from src.sim.downwash_aviary import DownwashAviary
from src.controllers.naive import NaiveSwarmController
from src.controllers.soft_avoid import SoftAvoidSwarmController
from src.controllers.feedforward import FeedforwardSwarmController
from src.controllers.ua_adc import UAAdaptiveController
from src.formations.generators import (
    BaseTrajectoryGenerator,
    validate_drone_count,
    HoverColumnFormation,
    CrossingTrajectory,
    VerticalSwapTrajectory,
    CircularOrbitFormation,
    LineFormation,
    VFormation,
    GridFormation,
    StaggeredFormation,
)
from src.metrics.swarm_metrics import compute_all_metrics
from src.experiments.logger import ExperimentLogger
from src.analysis.plots import (
    plot_3d_swarm_trajectories,
    plot_tracking_error_timeseries,
    plot_comparative_metrics_barchart,
)


SCENARIO_NAMES = [
    "hover_column_3d",
    "crossing_pair",
    "vertical_swap",
    "orbit_stack",
    "line",
    "v_formation",
    "grid",
    "staggered",
]

DEFAULT_DRONE_COUNTS = {
    "hover_column_3d": 3,
    "crossing_pair": 2,
    "vertical_swap": 2,
    "orbit_stack": 3,
    "line": 4,
    "v_formation": 5,
    "grid": 4,
    "staggered": 4,
}


def create_scenario_generator(scenario_name: str, num_drones: Optional[int] = None) -> BaseTrajectoryGenerator:
    """Instantiate a trajectory generator with validated swarm size N in [2, 10]."""
    if scenario_name not in SCENARIO_NAMES:
        raise ValueError(f"Unknown scenario '{scenario_name}'. Available: {SCENARIO_NAMES}")

    n = num_drones if num_drones is not None else DEFAULT_DRONE_COUNTS.get(scenario_name, 3)
    n = validate_drone_count(n)

    if scenario_name == "hover_column_3d":
        return HoverColumnFormation(num_drones=n, z_top=1.8, dz=0.45, duration=5.0)
    elif scenario_name == "crossing_pair":
        return CrossingTrajectory(num_drones=n, upper_alt=1.4, lower_alt=0.8, travel_distance=1.6, duration=6.0)
    elif scenario_name == "vertical_swap":
        return VerticalSwapTrajectory(num_drones=n, z_high=1.6, z_low=0.8, duration=6.0)
    elif scenario_name == "orbit_stack":
        return CircularOrbitFormation(num_drones=n, radius=0.6, period=6.0, duration=6.0)
    elif scenario_name == "line":
        return LineFormation(num_drones=n, spacing=0.45, altitude=1.2, duration=5.0)
    elif scenario_name == "v_formation":
        return VFormation(num_drones=n, dx=0.35, dy=0.35, altitude=1.2, duration=5.0)
    elif scenario_name == "grid":
        return GridFormation(num_drones=n, dx=0.45, dy=0.45, altitude=1.2, duration=5.0)
    elif scenario_name == "staggered":
        return StaggeredFormation(num_drones=n, duration=5.0)
    else:
        raise ValueError(f"Unhandled scenario generator '{scenario_name}'")


def load_true_downwash_model(config_path: str = "configs/downwash_true.yaml") -> DownwashModel:
    """Load the ground-truth aerodynamic downwash model."""
    with open(resolve(config_path), "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return DownwashModel(config=cfg)


def run_single_trial(
    scenario_name: str,
    scenario_gen: BaseTrajectoryGenerator,
    controller_name: str,
    controller_class: type,
    ctrl_kwargs: Dict[str, Any],
    true_downwash_model: DownwashModel,
    output_dir: str = "results",
    seed: int = 42,
    ctrl_freq_hz: int = 48,
    gui: bool = False,
    randomize: bool = False,
) -> Dict[str, Any]:
    """Execute a single simulation trial for one controller on a given scenario.

    Args:
        scenario_name: Scenario name string.
        scenario_gen: Trajectory generator defining setpoints.
        controller_name: Human-readable controller name.
        controller_class: Swarm controller class (A, B, or C).
        ctrl_kwargs: Extra keyword args for controller init.
        true_downwash_model: Physical downwash disturbance model.
        output_dir: Directory for results.
        seed: Random seed.
        ctrl_freq_hz: Control update frequency in Hz.
        gui: Whether to render PyBullet GUI.
        randomize: If True, ``seed`` draws a 3 mm initial-position jitter.
            If False (default, used for the published single-seed benchmark)
            the run is deterministic and ``seed`` is only recorded.  Before
            this flag existed the seed was silently ignored, so "20-seed"
            runs through this function were 20 identical runs.

    Returns:
        Dictionary containing trajectory history, time-series metrics, and summary stats.
    """
    num_drones = scenario_gen.num_drones
    duration = scenario_gen.duration
    dt = 1.0 / ctrl_freq_hz
    total_steps = int(duration * ctrl_freq_hz)

    init_xyzs = scenario_gen.get_initial_positions()
    if randomize:
        init_xyzs = init_xyzs + np.random.default_rng(seed).normal(0.0, 0.003, size=init_xyzs.shape)

    aviary = DownwashAviary(
        num_drones=num_drones,
        initial_xyzs=init_xyzs,
        ctrl_freq=ctrl_freq_hz,
        pyb_freq=240,
        gui=gui,
        record=False,
        downwash_model=true_downwash_model,
    )

    controller = controller_class(num_drones=num_drones, **ctrl_kwargs)
    obs, _ = aviary.reset()

    # Initialize per-run logger
    run_config = {
        "scenario": scenario_name,
        "controller": controller_name,
        "duration": duration,
        "ctrl_freq": ctrl_freq_hz,
        "num_drones": num_drones,
        "seed": seed,
    }
    logger = ExperimentLogger(
        scenario_name=scenario_name,
        controller_name=controller_name,
        seed=seed,
        config=run_config,
        output_dir=os.path.join(output_dir, "runs"),
    )

    # Time series storage
    actual_positions = np.zeros((total_steps, num_drones, 3))
    ref_positions = np.zeros((total_steps, num_drones, 3))
    ref_velocities = np.zeros((total_steps, num_drones, 3))
    motor_rpms = np.zeros((total_steps, num_drones, 4))
    timestamps = np.zeros(total_steps)

    conf_history: List[np.ndarray] = []
    alpha_history: List[np.ndarray] = []

    for step in range(total_steps):
        t = step * dt
        timestamps[step] = t

        # Query nominal formation setpoint
        target_pos, target_vel = scenario_gen.evaluate(t)
        ref_positions[step] = target_pos
        ref_velocities[step] = target_vel

        # Extract current states
        states = obs
        actual_positions[step] = states[:, 0:3]

        # Compute swarm actions (motor RPMs)
        rpms, _, _ = controller.compute_actions(
            control_timestep=dt,
            states=states,
            formation_targets=target_pos,
            formation_velocities=target_vel,
        )
        motor_rpms[step] = rpms

        # Record UA-ADC confidence and blend weights if available
        if isinstance(controller, UAAdaptiveController):
            conf_history.append(controller.last_confidences.copy())
            alpha_history.append(controller.last_blend_weights.copy())

        # Log per-drone states to logger
        for d in range(num_drones):
            logger.log_step(
                timestamp=t,
                step=step,
                drone_id=d,
                pos=states[d, 0:3],
                target_pos=target_pos[d],
                rpy=states[d, 7:10],
                vel=states[d, 10:13],
                rpms=rpms[d],
                disturb_force=aviary.last_applied_disturbances[d],
            )

        # Step simulation environment by 1 control timestep
        obs, _, _, _, _ = aviary.step(rpms)

        # Sync with wall-clock time if viewing in GUI
        if gui:
            import time
            from gym_pybullet_drones.utils.utils import sync
            if step == 0:
                trial_start_time = time.time()
            sync(step, trial_start_time, dt)

    aviary.close()

    # Compute Euclidean position errors over time (T, N)
    pos_errors = np.linalg.norm(actual_positions - ref_positions, axis=2)

    # Compute comprehensive evaluation metrics
    summary_metrics = compute_all_metrics(
        actual_pos=actual_positions,
        target_pos=ref_positions,
        timestamps=timestamps,
        motor_rpms=motor_rpms,
        dt=dt,
        safety_threshold=0.15,
        settling_tolerance=0.05,
    )

    if conf_history:
        summary_metrics["confidence_mean"] = float(np.mean(conf_history))
        summary_metrics["blend_weight_mean"] = float(np.mean(alpha_history))
    else:
        summary_metrics["confidence_mean"] = 0.0
        summary_metrics["blend_weight_mean"] = 1.0 if isinstance(controller, FeedforwardSwarmController) else 0.0

    # Save detailed run log and metadata
    logger.save(summary_metrics=summary_metrics)

    return {
        "timestamps": timestamps,
        "actual_positions": actual_positions,
        "ref_positions": ref_positions,
        "ref_velocities": ref_velocities,
        "motor_rpms": motor_rpms,
        "pos_errors": pos_errors,
        "summary": summary_metrics,
    }


def _load_ua_config(path: str = "configs/ua_adc.yaml") -> Dict[str, Any]:
    """Load UA-ADC configuration dictionary."""
    p = Path(path)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def run_scenario_benchmark(
    scenario_name: str,
    true_downwash_model: DownwashModel,
    num_drones: Optional[int] = None,
    output_dir: str = "results",
    selected_controller: str = "all",
    gui: bool = False,
) -> Dict[str, Any]:
    """Run full comparative benchmark across Controllers A, B, C, and D for a scenario.

    Args:
        scenario_name: Scenario key (e.g. 'hover_column_3d', 'crossing_pair').
        true_downwash_model: Physical downwash model.
        num_drones: Swarm size N in [2, 10].
        output_dir: Output base directory.
        selected_controller: Controller name filter ('all', 'naive', 'soft_avoid', 'feedforward', 'ua_adc').
        gui: Whether to enable GUI.

    Returns:
        Dictionary of results per controller.
    """
    scenario_gen = create_scenario_generator(scenario_name, num_drones=num_drones)
    n = scenario_gen.num_drones

    print(f"\n====================================================================")
    print(f"  RUNNING BENCHMARK SCENARIO: {scenario_name} (N={n} Drones)")
    print(f"====================================================================")

    ua_cfg = _load_ua_config()
    all_controllers = [
        ("Controller_A_Naive", NaiveSwarmController, {}),
        ("Controller_B_SoftAvoid", SoftAvoidSwarmController, {"config": {"max_displacement": 0.20, "max_velocity": 0.15}}),
        ("Controller_C_Feedforward", FeedforwardSwarmController, {}),
        ("Controller_D_UAADC", UAAdaptiveController, {"config": ua_cfg}),
    ]

    if selected_controller != "all":
        controllers = [
            c for c in all_controllers
            if selected_controller.lower().replace("_", "").replace("-", "") in c[0].lower().replace("_", "")
        ]
        if not controllers:
            raise ValueError(f"No controller matching '{selected_controller}'. Available: {[c[0] for c in all_controllers]}")
    else:
        controllers = all_controllers

    scenario_results: Dict[str, Any] = {}
    time_series_plot_data: Dict[str, Dict[str, np.ndarray]] = {}
    actual_trajectories_plot_data: Dict[str, np.ndarray] = {}
    summary_metrics_plot_data: Dict[str, Dict[str, Any]] = {}

    for ctrl_name, ctrl_class, ctrl_kwargs in controllers:
        print(f"\n>>> Running {ctrl_name} (N={n})...")
        trial = run_single_trial(
            scenario_name=scenario_name,
            scenario_gen=scenario_gen,
            controller_name=ctrl_name,
            controller_class=ctrl_class,
            ctrl_kwargs=ctrl_kwargs,
            true_downwash_model=true_downwash_model,
            output_dir=output_dir,
            gui=gui,
        )
        scenario_results[ctrl_name] = trial

        # Prepare plotting dicts
        time_series_plot_data[ctrl_name] = {
            "time": trial["timestamps"],
            "errors": trial["pos_errors"],
            "actual_pos": trial["actual_positions"],
            "ref_pos": trial["ref_positions"],
        }
        actual_trajectories_plot_data[ctrl_name] = trial["actual_positions"]
        summary_metrics_plot_data[ctrl_name] = trial["summary"]

        s = trial["summary"]
        print(f"    Swarm Mean RMSE:          {s['rmse_mean']*100:.2f} cm")
        print(f"    Swarm Max RMSE:           {s['rmse_max']*100:.2f} cm")
        print(f"    Min Inter-drone Distance: {s['min_interdrone_dist']:.3f} m")
        print(f"    Near-Collisions (<0.15m): {s['near_collision_steps']}")
        print(f"    Energy Proxy:             {s['total_energy_proxy']/1e6:.2f} M (RPM^2*s)")

    # Generate Figures
    figures_dir = os.path.join(output_dir, "figures")
    ref_pos = scenario_results["Controller_A_Naive"]["ref_positions"]

    print(f"\n[INFO] Generating publication figures for {scenario_name} (N={n})...")
    plot_3d_swarm_trajectories(
        actual_trajectories=actual_trajectories_plot_data,
        ref_trajectory=ref_pos,
        scenario_name=f"{scenario_name}_N{n}",
        output_dir=figures_dir,
    )
    plot_tracking_error_timeseries(
        time_series_data=time_series_plot_data,
        scenario_name=f"{scenario_name}_N{n}",
        output_dir=figures_dir,
    )
    plot_comparative_metrics_barchart(
        summary_metrics=summary_metrics_plot_data,
        scenario_name=f"{scenario_name}_N{n}",
        output_dir=figures_dir,
    )

    return scenario_results


def main() -> None:
    """CLI Entrypoint for multi-drone benchmark experiments."""
    parser = argparse.ArgumentParser(description="Multi-Drone Downwash Formation Benchmark Runner")
    parser.add_argument(
        "--scenario",
        type=str,
        default="all",
        choices=["all"] + SCENARIO_NAMES,
        help="Scenario to run or 'all' to run all benchmark scenarios.",
    )
    parser.add_argument(
        "--controller",
        type=str,
        default="all",
        choices=["all", "naive", "soft_avoid", "feedforward", "ua_adc"],
        help="Controller to evaluate (default: 'all').",
    )
    parser.add_argument(
        "--num-drones",
        type=int,
        default=None,
        help="Swarm size N in [2, 10]. If omitted, uses scenario default.",
    )
    parser.add_argument("--gui", action="store_true", help="Enable PyBullet GUI rendering.")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory for logs and figures.")
    args = parser.parse_args()

    if args.num_drones is not None:
        if args.num_drones < 2 or args.num_drones > 10:
            parser.error(
                f"Invalid --num-drones {args.num_drones}. Swarm size must be between 2 and 10."
            )

    true_model = load_true_downwash_model("configs/downwash_true.yaml")
    scenarios = SCENARIO_NAMES if args.scenario == "all" else [args.scenario]

    all_summary_rows: List[Dict[str, Any]] = []

    for sc in scenarios:
        results = run_scenario_benchmark(
            scenario_name=sc,
            true_downwash_model=true_model,
            num_drones=args.num_drones,
            output_dir=args.output_dir,
            selected_controller=args.controller,
            gui=args.gui,
        )

        for ctrl_name, trial in results.items():
            row = {
                "scenario": sc,
                "controller": ctrl_name,
                "num_drones": trial["ref_positions"].shape[1],
                **trial["summary"],
            }
            # Remove array fields for CSV storage
            row.pop("per_drone_rmse", None)
            row.pop("settling_times", None)
            all_summary_rows.append(row)

    # Save consolidated summary CSVs
    os.makedirs(args.output_dir, exist_ok=True)
    df_summary = pd.DataFrame(all_summary_rows)
    csv_path = os.path.join(args.output_dir, "multi_drone_comparison.csv")
    csv_all4_path = os.path.join(args.output_dir, "multi_drone_comparison_all4.csv")
    df_summary.to_csv(csv_path, index=False)
    df_summary.to_csv(csv_all4_path, index=False)
    print(f"\n[SUCCESS] Multi-drone benchmark complete! Summary saved to {csv_path} and {csv_all4_path}")


if __name__ == "__main__":
    main()
