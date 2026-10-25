"""Unified CLI runner for the Downwash Swarm Formation Control Project.

Executes all project stages, runs the pytest suite, generates all figures and datasets,
and validates the complete experimental pipeline end-to-end.
"""
import sys
import os
import argparse
import subprocess
from datetime import datetime


def run_command(cmd_list: list, description: str) -> bool:
    """Run a subprocess command and stream its output."""
    print("\n" + "=" * 75)
    print(f"  RUNNING: {description}")
    print(f"  COMMAND: {' '.join(cmd_list)}")
    print("=" * 75)

    start = datetime.now()
    result = subprocess.run(cmd_list)
    duration = (datetime.now() - start).total_seconds()

    if result.returncode == 0:
        print(f"[SUCCESS] {description} completed in {duration:.2f}s")
        return True
    else:
        print(f"[ERROR] {description} failed with return code {result.returncode}")
        return False


def main() -> None:
    """CLI Entrypoint for running all experiments and tests."""
    parser = argparse.ArgumentParser(description="Downwash Swarm Project Unified Master Runner")
    parser.add_argument("--skip-tests", action="store_true", help="Skip running the pytest test suite.")
    parser.add_argument("--skip-grid", action="store_true", help="Skip the 48-condition characterisation grid sweep.")
    parser.add_argument("--skip-benchmarks", action="store_true", help="Skip the multi-drone benchmark scenarios.")
    parser.add_argument("--skip-robustness", action="store_true", help="Skip the model error robustness sweep.")
    parser.add_argument("--skip-scaling", action="store_true", help="Skip the swarm-size scaling sweep and analysis.")
    parser.add_argument(
        "--controller",
        type=str,
        default="all",
        choices=["all", "naive", "soft_avoid", "feedforward", "ua_adc"],
        help="Controller to run (default: 'all').",
    )
    args = parser.parse_args()

    python_exe = sys.executable

    print("\n" + "#" * 75)
    print("  DOWNWASH-AWARE FORMATION CONTROL FOR MULTI-DRONE SWARMS")
    print("  Comprehensive University Research Evaluation Pipeline")
    print("#" * 75)

    all_passed = True

    # 1. Run full test suite
    if not args.skip_tests:
        ok = run_command(
            [python_exe, "-m", "pytest", "-v"],
            "Complete Pytest Test Suite",
        )
        all_passed = all_passed and ok

    # 2. Run Single Drone Baseline (Phase 1)
    ok = run_command(
        [python_exe, "-m", "src.experiments.single_drone_baseline"],
        "Phase 1: Single Drone Orbit Baseline Tracking",
    )
    all_passed = all_passed and ok

    # 3. Run Downwash Characterisation Grid Sweep (Phase 2)
    if not args.skip_grid:
        ok = run_command(
            [python_exe, "-m", "src.experiments.downwash_characterisation"],
            "Phase 2: 48-Condition Aerodynamic Downwash Grid Sweep & Heatmap Generation",
        )
        all_passed = all_passed and ok

    # 4. Run Multi-Drone Benchmark Scenarios (Phase 5)
    if not args.skip_benchmarks:
        ok = run_command(
            [python_exe, "-m", "src.experiments.multi_drone_comparison", "--scenario", "all", "--controller", args.controller],
            f"Phase 5: Multi-Drone Scenario Benchmarks ({args.controller} across Scenarios)",
        )
        all_passed = all_passed and ok

    # 5. Run Robustness & Sensitivity Sweep (Phase 6)
    if not args.skip_robustness:
        ok = run_command(
            [python_exe, "-m", "src.experiments.robustness_sweep", "--controller", args.controller],
            f"Phase 6: Robustness & Model Error Analysis ({args.controller})",
        )
        all_passed = all_passed and ok

    # 6. Run Swarm-Size Scaling Sweep & Analysis (Extension)
    if not args.skip_scaling:
        ok = run_command(
            [python_exe, "-m", "src.experiments.scaling_sweep", "--seeds", "20", "--controller", args.controller],
            f"Swarm-Size Scaling Sweep (N in [2, 3, 5, 7, 10] across 20 Seeds, {args.controller})",
        )
        all_passed = all_passed and ok
        ok = run_command(
            [python_exe, "-m", "src.analysis.scaling_plots"],
            "Swarm-Size Scaling Analysis & Figure Generation (docs/scaling_summary.md)",
        )
        all_passed = all_passed and ok

    # 7. Generate UA-ADC Signature Plot
    ok = run_command(
        [python_exe, "tools/ua_adc_signature.py"],
        "UA-ADC Signature Visualisation Plot",
    )
    all_passed = all_passed and ok

    print("\n" + "=" * 75)
    if all_passed:
        print("  ALL PROJECT EXPERIMENTAL PIPELINES COMPLETED SUCCESSFULLY!")
        print("  Datasets saved to: results/")
        print("  Publication figures saved to: results/figures/")
    else:
        print("  SOME PIPELINES FAILED OR EXITED WITH ERRORS. CHECK LOGS ABOVE.")
    print("=" * 75)


if __name__ == "__main__":
    main()
