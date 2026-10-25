"""Numbered menu of the review demos (same set as Demo_Commands.txt).

    python demo_menu.py          # pick a number, the 3D window opens
    python demo_menu.py 3        # run demo 3 directly

Each entry says what to expect, so the presenter can read it aloud.
"""
from __future__ import annotations

import sys
from typing import List, NamedTuple, Optional

from demo_gui import run_gui_demo


class Demo(NamedTuple):
    title: str
    scenario: str
    controller: str
    n: int
    duration: Optional[float]
    expect: str


DEMOS: List[Demo] = [
    Demo("Stacked hover - Naive", "hover_column_3d", "naive", 3, None,
         "Middle and bottom drones sag below their set heights."),
    Demo("Stacked hover - Soft-Avoid", "hover_column_3d", "soft_avoid", 3, None,
         "Lower drones move sideways out of the wake column."),
    Demo("Stacked hover - Feedforward", "hover_column_3d", "feedforward", 3, None,
         "Drones hold height with almost no motion."),
    Demo("Stacked hover - UA-ADC", "hover_column_3d", "ua_adc", 3, None,
         "Confidence c on drones 1 and 2 rises toward 1 within about a second; blend a follows; no sag."),
    Demo("Vertical swap - Soft-Avoid", "vertical_swap", "soft_avoid", 2, None,
         "Drones steer apart and pass with clearance."),
    Demo("Vertical swap - Feedforward", "vertical_swap", "feedforward", 2, None,
         "Drones follow the shared column and come very close (no avoidance)."),
    Demo("Vertical swap - UA-ADC (with conflict override)", "vertical_swap", "ua_adc", 2, None,
         "Conflict risk r goes to 1 on approach, blend a drops to 0, drones pass like Soft-Avoid."),
    Demo("Crossing pair - UA-ADC", "crossing_pair", "ua_adc", 2, None,
         "Drones cross at different heights with tracking close to feedforward."),
    Demo("Orbit stack N=5 - UA-ADC", "orbit_stack", "ua_adc", 5, 16.0,
         "All drones hold altitude while orbiting."),
    Demo("Orbit stack N=8 - UA-ADC (edge of the limit)", "orbit_stack", "ua_adc", 8, 16.0,
         "Drones stay up by dodging sideways, but some pairs come closer than 0.12 m."),
    Demo("Orbit stack N=8 - Feedforward (density limit)", "orbit_stack", "feedforward", 8, 16.0,
         "No dodging: the bottom drones cannot out-thrust the stacked wake and reach the floor."),
    Demo("Orbit stack N=10 - UA-ADC (density limit)", "orbit_stack", "ua_adc", 10, 16.0,
         "Expected failure: too many drones packed in the wake; bottom drones fall, as real drones would."),
    Demo("V-formation N=5 - Naive", "v_formation", "naive", 5, None,
         "Coplanar: no stacking, so every controller holds the shape."),
    Demo("Hover column N=7 - UA-ADC", "hover_column_3d", "ua_adc", 7, None,
         "Larger stack; UA-ADC stays steady."),
]


def _menu() -> Optional[int]:
    print("\nDRONE SWARM DEMOS")
    print("=" * 60)
    for i, d in enumerate(DEMOS, 1):
        print(f" {i:2d}. {d.title}")
    print("  q. quit")
    choice = input("\nPick a demo number: ").strip().lower()
    if choice in ("q", "quit", ""):
        return None
    return int(choice) if choice.isdigit() and 1 <= int(choice) <= len(DEMOS) else -1


def run(idx: int) -> None:
    d = DEMOS[idx - 1]
    print(f"\n>>> {d.title}\n    Expect: {d.expect}")
    print(f"    Command: python demo_gui.py --controller {d.controller} --scenario {d.scenario} "
          f"--num-drones {d.n}" + (f" --duration {d.duration:g}" if d.duration else ""))
    run_gui_demo(d.scenario, d.controller, d.n, d.duration)


def main() -> None:
    if len(sys.argv) > 1:
        run(int(sys.argv[1]))
        return
    while True:
        c = _menu()
        if c is None:
            break
        if c < 0:
            print("Not a valid number.")
            continue
        run(c)


if __name__ == "__main__":
    main()
