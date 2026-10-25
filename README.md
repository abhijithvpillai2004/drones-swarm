# Uncertainty-Aware Adaptive Control for Drone Swarms

Simulation study of rotor downwash in multi-drone formations (PyBullet, Crazyflie 2.x, N = 2 to 10).
Four controllers are compared: **A Naive**, **B Soft-Avoid**, **C Feedforward**, and **D UA-ADC**,
which learns the wake online, tracks its confidence, and blends avoidance with compensation.

> *"We quantify aerodynamic interaction between multirotors in formation and compare avoidance,
> compensation, and uncertainty-aware adaptive blending strategies for downwash, including their
> robustness to model error."*

Simulation only, no hardware. The wake model and the individual strategies come from prior work
(Preiss et al. 2017, Hönig et al. 2018, Shi et al. 2020). The contribution is the confidence-driven blend,
a conflict override for head-on approaches, and a controlled test under model mismatch.

---

## 1. Setup (Windows)

pybullet has **no prebuilt Windows wheels on PyPI**, so pip has to compile it. Three options:

**A. This folder already has a working `venv/`** (built on the original PC, Python 3.12). A venv only
works on the machine it was made on; on another PC use B or C.

**B. Python 3.12 + the bundled wheel (no compiler needed)**
```bash
python -m venv venv
venv\Scripts\python -m pip install wheels\pybullet-3.2.7-cp312-cp312-win_amd64.whl
venv\Scripts\python -m pip install -r requirements.txt
venv\Scripts\python -m pip install -e external/gym-pybullet-drones --no-deps
```

**C. Conda (any Python, no compiler needed)**
```bash
conda env create -f environment.yml
conda activate downwash_swarm
pip install -e external/gym-pybullet-drones --no-deps
```

`setup_env.ps1` automates B/C (`powershell -ExecutionPolicy Bypass -File setup_env.ps1`).
`external/gym-pybullet-drones` is included in this copy; on a fresh clone, get it with
`git clone https://github.com/utiasDSL/gym-pybullet-drones.git external/gym-pybullet-drones`.

Check the install:
```bash
venv\Scripts\python -m pytest -q        # 131 tests
```

---

## 2. Running it

**Live 3D demo** (rotate: left-drag, zoom: wheel, pan: Ctrl + left-drag):
```bash
python demo_menu.py                     # numbered list of the review demos with what to expect
python demo_gui.py --controller ua_adc --scenario hover_column_3d --num-drones 3
python demo_gui.py --controller ua_adc --scenario vertical_swap --num-drones 2
python demo_gui.py --controller ua_adc --scenario orbit_stack --num-drones 8 --duration 16
```
Options: `--headless` (no window), `--speed 1.0` (real time), `--seed 3` (3 mm start jitter), `--loop`, `--no-log`.
For UA-ADC the terminal shows per drone `c` (confidence), `a` (blend: 0 = avoid, 1 = feedforward) and
`r` (conflict risk). Every run saves its full trace to `results/demo_logs/*.csv`, the same values the
terminal prints. Drone 0 is the top drone; nothing is above it, so its confidence stays 0, which is correct.

Scenarios: `hover_column_3d`, `vertical_swap`, `crossing_pair`, `orbit_stack`, `line`, `v_formation`, `grid`, `staggered`.
Controllers: `naive`, `soft_avoid`, `feedforward`, `ua_adc`.

**Studies** (parallel, about 1 s per simulation; outputs in `results/studies/`):
```bash
python tools/studies.py all             # everything below, then docs/results_v2.md
python tools/studies.py regression      # 8 scenarios x N = 2..10 x 4 controllers, strict pass criteria
python tools/studies.py vswap           # vertical swap failure rate, 20 seeds
python tools/studies.py gateb           # model mismatch, 20 seeds, paired Wilcoxon
python tools/studies.py wake            # orbit-stack collapse vs wake strength
python tools/studies.py stress          # 1.3x force + noise + 2-step delay, 20 seeds
python tools/studies.py report          # rebuild the report from saved CSVs only
```
The original pipeline still works: `python run_all.py`, `python -m src.experiments.scaling_sweep --seeds 20`.

---

## 3. Results

Every number below is copied from **[docs/results_v2.md](docs/results_v2.md)**, which is generated from
the CSVs in `results/studies/`. Pass/fail rules are in [configs/pass_criteria.yaml](configs/pass_criteria.yaml);
they were written before the strict regression was run (decision D-012).

**Strict regression** (8 scenarios × N = 2..10, 72 runs per controller; fail on ground contact,
separation ≤ 0.12 m, slow confidence, or UA-ADC worse than Soft-Avoid):

| Controller | Passed |
|---|---|
| A Naive | 60 / 72 |
| B Soft-Avoid | 69 / 72 |
| C Feedforward | 60 / 72 |
| D UA-ADC | 69 / 72 (60 / 72 with the conflict override disabled) |

All remaining failures are `orbit_stack` at N = 8 to 10 (every controller). Naive and Feedforward also
fail every `vertical_swap`, because the reference path itself passes through the other drone.

**Vertical swap, N = 2, 20 seeds** (failure = ground contact or separation ≤ 0.12 m):

| Variant | Failures | Mean min. separation | RMSE |
|---|---|---|---|
| Soft-Avoid | 0 / 20 | 0.209 m | 11.53 cm |
| UA-ADC | 0 / 20 | 0.212 m | 6.90 cm |
| UA-ADC, override disabled | 20 / 20 | 0.027 m | 103.90 cm |
| Naive / Feedforward | 20 / 20 | 0.037 / 0.029 m | |

**Model mismatch (Gate B), hover column N = 3, 20 seeds**, RMSE in cm (mean ± std); assumed wake = scale × true:

| Scale | Feedforward | UA-ADC | UA better in | Learned gain θ (ideal) |
|---|---|---|---|---|
| 1.0 | 0.99 ± 0.05 | 1.10 ± 0.05 | 2 / 20 seeds | 1.00 (1.00) |
| 1.5 | 1.99 ± 0.03 | 1.40 ± 0.05 | 20 / 20 seeds | 0.66 (0.67) |
| 3.0 | 6.99 ± 0.03 | 1.62 ± 0.06 | 20 / 20 seeds | 0.33 (0.33) |

Feedforward is better when its model is right; UA-ADC is better when the model over-predicts the wake,
and its learned gain matches the true ratio. The 0.5 row and the Naive and Soft-Avoid columns are in
results_v2.

**Dense formations are a physical limit, not a bug.** Hover thrust is 0.265 N and the Crazyflie's maximum
thrust is about 2.25× that (0.60 N), which leaves about 0.33 N of margin. The summed wake on the lowest drone of
a tight stack (capped at 0.45 N in `downwash_true.yaml`) is larger than that margin, so a drone that stays
under the stack cannot hold its height. This is the same reason real drones cannot fly packed tightly on top
of each other. In the orbit stack (drones about 0.14 m apart vertically):

| N | Naive / Feedforward | Soft-Avoid / UA-ADC |
|---|---|---|
| 7 | fly | fly |
| 8 | fall | stay up by dodging sideways, but come closer than 0.12 m |
| 9, 10 | fall | fall |

So avoidance buys one extra drone; beyond that nothing helps. Size alone is not the problem: 10 drones in a
`line`, `grid` or a `hover_column_3d` with 0.45 m spacing all fly, because the drones are not packed into each
other's wake. The wake study confirms the collapse depends on wake strength: at 0.75× strength N = 8 and 9
fly with every controller.

**Stress case** (1.3× force, observation noise, 2-step delay, 20 seeds): ground contact Naive 0, Feedforward 1,
Soft-Avoid 4, UA-ADC 4. Delay is still the hardest condition for the avoidance-based controllers.

---

## 4. What changed in this version

Details and reasons for every item are in [docs/decisions.md](docs/decisions.md) (D-007 to D-013).

- **UA-ADC conflict override (D-007).** A closest-point-of-approach check forces avoidance when two drones are
  about to meet, whatever the confidence. Vertical swap goes from 20/20 failures to 0/20.
- **Soft-Avoid chatter fixes (D-013).** The escape direction is held fixed inside the wake core, and the
  offset velocity is low-pass filtered. Under a 2-step delay this cut ground contacts from 4/8 to 0/8
  (8-seed check) and brought Soft-Avoid's nominal error spread from ±0.28 to ±0.10 cm.
- **Exact blend in UA-ADC.** Fixes a rounding drift that made UA-ADC differ from Soft-Avoid where they should match.
- **Seeds actually used (D-009).** `run_single_trial` ignored its seed, so "N-seed" runs repeated one
  simulation. All new studies go through `src/sim/episode.py`, which applies the seed explicitly.
- **Estimator uses the vertical part of thrust (D-008).**
- **Pass criteria + checker (D-012):** `configs/pass_criteria.yaml`, `src/evaluation/criteria.py`.
- **Demo:** rebuilt on the shared runner; CSV log of exactly what is printed; blend and risk shown; labels in the
  3D view; `--headless`; `demo_menu.py`.
- **Portability (D-011):** removed hard-coded `D:/Drones_Project/...` paths; configs resolve from the project root.
- **Removed (D-010):** `run_batch.py`, `src/experiments/runner.py`, `characterise_downwash.py`. They imported
  classes that never existed and could not run.
- Tests: 82 → 131.

---

## 5. Repository map

| Path | Contents |
|---|---|
| `src/controllers/` | Controllers A to D (`naive`, `soft_avoid`, `feedforward`, `ua_adc`) and the PID |
| `src/models/` | Wake model (`downwash.py`) and the online estimator (`disturbance_estimator.py`) |
| `src/sim/` | `DownwashAviary` (applies the TRUE wake) and `episode.py` (the shared run loop) |
| `src/experiments/` | `catalog.py` (scenario and controller names), `batch.py` (parallel runs), original sweeps |
| `src/evaluation/` | Pass-criteria checker |
| `configs/` | TRUE and ASSUMED wake, UA-ADC settings, pass criteria, scenarios |
| `tools/studies.py` | All studies for the open items |
| `results/studies/` | Study CSVs (`*_checked.csv` include per-criterion verdicts) |
| `docs/` | `results_v2.md` (generated), `decisions.md`, original reports |

**Anti-circularity rule:** only the simulator reads `configs/downwash_true.yaml`. Controllers B, C and D read
`configs/downwash_assumed.yaml` and observed states only.

---

## 6. Limitations

- Simulation only; no hardware flight.
- The wake is an empirical shape with linear superposition (capped). There is no turbulence, ground effect or wind.
- Wake strength and footprint are taken from the literature and are not calibrated. The dense-formation
  results depend on them (see the wake study).
- The strict regression uses one seed per cell; the vertical swap, mismatch and stress studies use 20.
- Under command delay, the avoidance-based controllers (B and D) still lose drones in some seeds.
- Older documents in `docs/` (final_report, project_summary, scaling_summary) and the root-level CSVs come from the
  previous version and were not regenerated; use `docs/results_v2.md` for current numbers.
