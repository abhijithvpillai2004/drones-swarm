"""Interactive 3D PyBullet flight demo.

Runs one scenario with one controller in a live PyBullet window, prints a
telemetry line 4 times per second, and saves the full trace to a CSV file so
the numbers shown on screen can be checked afterwards (the CSV is written from
the same arrays the terminal reads).

Examples:
    python demo_gui.py --controller ua_adc --scenario hover_column_3d --num-drones 3
    python demo_gui.py --controller ua_adc --scenario vertical_swap --num-drones 2
    python demo_gui.py --controller soft_avoid --scenario orbit_stack --num-drones 8 --duration 16
    python demo_gui.py --controller ua_adc --scenario hover_column_3d --headless   # no window

Window controls: rotate = left-drag, zoom = right-drag or wheel, pan = Ctrl + left-drag.
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.experiments.catalog import CONTROLLER_NAMES, SCENARIO_NAMES, make_controller, make_scenario
from src.paths import RESULTS_DIR
from src.sim.episode import (
    GROUND_CONTACT_Z,
    NEAR_COLLISION_DIST,
    EpisodeConfig,
    load_true_model,
    run_episode,
)

# Demo flights are longer than the experiment runs so they are easy to watch.
DEMO_DURATIONS = {
    "hover_column_3d": 12.0,
    "vertical_swap": 10.0,
    "crossing_pair": 10.0,
    "orbit_stack": 16.0,
    "line": 10.0,
    "v_formation": 10.0,
    "grid": 10.0,
    "staggered": 10.0,
}
HUD_EVERY = 12            # control steps between terminal lines (48 Hz / 12 = 4 Hz)
CLOSE_PROXIMITY = 0.25    # [m] status shows "Close" below this


def _status(min_d: float, min_z: float) -> str:
    if min_z < GROUND_CONTACT_Z:
        return "GROUND CONTACT"
    if min_d < NEAR_COLLISION_DIST:
        return "NEAR-COLLISION"
    if min_d < CLOSE_PROXIMITY:
        return "Close"
    return "Nominal"


class _Hud:
    """Terminal HUD plus optional 3D labels above each drone."""

    def __init__(self, n: int, show_labels: bool) -> None:
        self.n = n
        self.show_labels = show_labels
        self.label_ids: List[int] = [-1] * n
        self.printed_rows: List[Dict[str, Any]] = []

    def header(self, is_ua: bool) -> None:
        alt = " ".join(f"{'z' + str(d):>6}" for d in range(self.n))
        extra = "  UA-ADC: c=confidence a=blend(0 avoid..1 ff) r=conflict risk" if is_ua else ""
        line = f"{'t[s]':>6} | {alt} | {'min d':>6} | {'status':<14}{extra}"
        print(line)
        print("-" * len(line))

    def __call__(self, step: int, t: float, info: Dict[str, Any]) -> None:
        pos = info["positions"]
        min_d = info["min_dist"]
        min_z = float(pos[:, 2].min())
        if step % HUD_EVERY == 0:
            alt = " ".join(f"{z:6.2f}" for z in pos[:, 2])
            line = f"{t:6.2f} | {alt} | {min_d:6.3f} | {_status(min_d, min_z):<14}"
            if info["is_ua"]:
                line += "  " + " ".join(
                    f"[{d}] c={info['confidence'][d]:.2f} a={info['blend'][d]:.2f} r={info['conflict_risk'][d]:.2f}"
                    for d in range(self.n)
                )
            print(line)
            self.printed_rows.append({"step": step, "t": t, "min_d": min_d})
            if self.show_labels:
                self._labels(info)

    def _labels(self, info: Dict[str, Any]) -> None:
        import pybullet as p
        for d in range(self.n):
            txt = f"D{d}"
            if info["is_ua"]:
                txt += f" c={info['confidence'][d]:.2f} a={info['blend'][d]:.2f}"
            try:
                self.label_ids[d] = p.addUserDebugText(
                    txt, (info["positions"][d] + np.array([0.0, 0.0, 0.08])).tolist(),
                    textColorRGB=[0.1, 0.1, 0.1], textSize=1.1,
                    replaceItemUniqueId=self.label_ids[d] if self.label_ids[d] >= 0 else -1,
                    physicsClientId=info["client"],
                )
            except Exception:
                self.show_labels = False  # labels are cosmetic; never stop the run for them


def save_trace_csv(res: Any, path: Path) -> None:
    """Write the full per-step, per-drone trace that the HUD reads from."""
    import pandas as pd
    T, N = res.pos.shape[:2]
    rows = []
    for k in range(T):
        for d in range(N):
            rows.append({
                "step": k, "t": res.t[k], "drone": d,
                "x": res.pos[k, d, 0], "y": res.pos[k, d, 1], "z": res.pos[k, d, 2],
                "ref_x": res.ref[k, d, 0], "ref_y": res.ref[k, d, 1], "ref_z": res.ref[k, d, 2],
                "err_m": float(np.linalg.norm(res.pos[k, d] - res.ref[k, d])),
                "true_fz": res.true_force[k, d, 2],
                "confidence": res.confidence[k, d], "blend": res.blend[k, d],
                "blend_raw": res.blend_raw[k, d], "conflict_risk": res.conflict_risk[k, d],
                "theta": res.theta[k, d],
            })
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _hold_until_enter(controller: Any, scenario: Any, duration: float):
    """Return an on_finish callback that keeps the swarm hovering until Enter."""
    def _hold(aviary: Any, obs: np.ndarray) -> None:
        print("\n[INFO] Flight finished. Drones keep hovering; rotate/zoom freely.")
        print("[INFO] Press Enter here (or close the window) to stop.")
        stop = threading.Event()
        threading.Thread(target=lambda: (sys.stdin.readline(), stop.set()), daemon=True).start()
        final_pos, _ = scenario.evaluate(duration)
        dt = 1.0 / aviary.CTRL_FREQ
        while not stop.is_set():
            try:
                rpms, _, _ = controller.compute_actions(dt, obs, final_pos, np.zeros_like(final_pos))
                obs, _, _, _, _ = aviary.step(rpms)
                time.sleep(dt)
            except Exception:  # window closed by the user
                break
    return _hold


def run_gui_demo(
    scenario_key: str,
    controller_key: str,
    num_drones: Optional[int] = None,
    duration_sec: Optional[float] = None,
    loop: bool = False,
    playback_speed: float = 0.6,
    headless: bool = False,
    seed: Optional[int] = None,
    log: bool = True,
    hold: bool = True,
) -> Optional[Path]:
    """Run the demo. Returns the path of the last saved CSV (or None)."""
    duration = duration_sec if duration_sec is not None else DEMO_DURATIONS[scenario_key]
    scenario = make_scenario(scenario_key, num_drones, duration)
    n = scenario.num_drones
    controller = make_controller(controller_key, n)
    true_model = load_true_model()

    print("\n" + "=" * 75)
    print(f"  Scenario {scenario_key} | Controller {controller_key} | N = {n} | {duration:.1f} s"
          f"{' | headless' if headless else f' | speed {playback_speed:.2f}x'}")
    print("  Drone 0 is the top drone: nothing is above it, so its confidence stays 0 (correct).")
    print("=" * 75)

    csv_path: Optional[Path] = None
    run = 1
    try:
        while True:
            hud = _Hud(n, show_labels=not headless)
            hud.header(controller_key == "ua_adc")
            last = (not loop)
            res = run_episode(
                scenario, controller, true_model,
                EpisodeConfig(seed=seed, gui=not headless,
                              realtime_speed=None if headless else playback_speed),
                on_step=hud,
                on_finish=_hold_until_enter(controller, scenario, duration) if (last and hold and not headless) else None,
            )
            m = res.metrics
            print("-" * 75)
            print(f"  RMSE {100 * m['rmse_mean_m']:.2f} cm | min separation {m['min_sep_m']:.3f} m"
                  f" | lowest z {m['min_z_m']:.3f} m | near-collision steps {m['near_collision_steps']}"
                  f" | ground contact {'YES' if m['ground_contact'] else 'no'}")
            if controller_key == "ua_adc":
                rise = m.get("conf_rise_time_max_s")
                print(f"  mean confidence {m['confidence_mean']:.2f} | mean blend {m['blend_mean']:.2f}"
                      f" | slowest in-wake confidence > 0.5 at "
                      f"{'never' if rise is None and m['in_wake_drones'] else ('n/a' if rise is None else f'{rise:.2f} s')}")
            if log:
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                csv_path = RESULTS_DIR / "demo_logs" / f"{stamp}_{scenario_key}_{controller_key}_N{n}.csv"
                save_trace_csv(res, csv_path)
                print(f"  trace saved: {csv_path}")
            if not loop:
                break
            run += 1
            print(f"\n[INFO] Replay #{run} (Ctrl+C to stop)")
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
    return csv_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Live 3D PyBullet swarm demo")
    ap.add_argument("--scenario", default="vertical_swap", choices=SCENARIO_NAMES)
    ap.add_argument("--controller", default="soft_avoid", choices=CONTROLLER_NAMES)
    ap.add_argument("--num-drones", type=int, default=None, help="swarm size N in [2, 10]")
    ap.add_argument("--duration", type=float, default=None, help="flight duration [s]")
    ap.add_argument("--speed", type=float, default=0.6, help="playback speed (1.0 = real time)")
    ap.add_argument("--loop", action="store_true", help="replay until Ctrl+C")
    ap.add_argument("--headless", action="store_true", help="no window; run as fast as possible")
    ap.add_argument("--seed", type=int, default=None,
                    help="add seeded 3 mm start jitter (default: deterministic run)")
    ap.add_argument("--no-log", action="store_true", help="do not save the CSV trace")
    ap.add_argument("--no-hold", action="store_true", help="close the window when the flight ends")
    a = ap.parse_args()
    if a.num_drones is not None and not 2 <= a.num_drones <= 10:
        ap.error(f"--num-drones must be between 2 and 10 (got {a.num_drones})")
    run_gui_demo(a.scenario, a.controller, a.num_drones, a.duration, a.loop, a.speed,
                 a.headless, a.seed, not a.no_log, not a.no_hold)


if __name__ == "__main__":
    main()
