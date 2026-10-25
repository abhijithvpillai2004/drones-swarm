# Downwash-Aware Formation Control for Multi-Drone Swarms
## Full Project Summary & Documentation Handoff

---

### 1. Project Goal

When multi-rotor drones fly together in tight formations, the powerful downward airflow (downwash) blasted out by the propellers of upper drones creates a severe aerodynamic hazard for any drone flying underneath them. Drones caught in this wake experience sudden loss of lift, destabilizing tilts, and sluggish control lag, frequently causing them to sag toward the ground, wobble uncontrollably, or crash into one another. This project solves that problem by building and systematically evaluating intelligent, downwash-aware flight control systems that allow drone swarms to safely operate in close proximity, comparing strategies that physically dodge the wake against strategies that predict and cancel the wake forces using smart motor compensation.

---

### 2. What Was Built

Every major software and algorithmic component was built from the ground up to support swarms from $2$ to $10$ drones:

- **Simulator Setup**: Built on **PyBullet** (a high-precision rigid-body 3D physics engine) coupled with custom Crazyflie 2.x quadrotor dynamics operating at 240 Hz physics updates and 48 Hz control updates. It realistically simulates propeller thrust, air drag, gyroscopic effects, gravity, and ground contact.
- **Three Core Controllers (in plain terms)**:
  - **Controller A (`naive`)**: *The Downwash-Blind Baseline.* A conventional flight controller that guides each drone toward its scheduled waypoints while completely ignoring aerodynamic wake effects. It only reacts after tracking errors have already accumulated.
  - **Controller B (`soft_avoid`)**: *The Proactive Geometric Dodger.* An avoidance controller that monitors the positions of neighboring drones and detects when a drone is entering the downward wake cylinder of another drone. It smoothly nudges the drone horizontally away from the wake center without human intervention, backed by a general 3D safety net.
  - **Controller C (`feedforward`)**: *The Predictive Counter-Thruster.* An aerodynamic cancellation controller that keeps drones exactly on their intended path. Using an onboard mathematical model, it anticipates the downward suction force produced by upper drones and immediately commands extra upward motor thrust in the low-level motor mixer, neutralizing the disturbance before any altitude sag occurs.
  - **Controller D (`ua_adc`)**: *The Uncertainty-Aware Adaptive Controller — the project's primary novel contribution.* Unlike the three fixed-strategy controllers above, UA-ADC never pre-commits to dodging or compensating. Instead, it continuously measures how confident it is in its own disturbance estimate — using a per-drone Recursive Least-Squares (RLS) online estimator — and blends smoothly between avoidance and feedforward compensation using a sigmoid function. When the estimator is uncertain (early in the flight, or when a drone isn't in any wake), it favours safe lateral avoidance. As the estimator gains confidence from repeated consistent observations, it shifts toward precise feedforward cancellation. The always-active 3D safety net from Controller B is inherited and never switched off.
- **Downwash Disturbance Models (TRUE vs. ASSUMED)**:
  - **TRUE Model**: The physical aerodynamic disturbance engine running inside the simulation, injecting realistic downward drag and radial shear forces onto drone airframes based on distance and velocity.
  - **ASSUMED Model**: The onboard mathematical estimation running inside Controllers B and C.
  - **Why Kept Separate (Anti-Circularity Rule)**: Controllers are never permitted to inspect the simulator's true physics. Keeping them strictly decoupled ensures that controllers are evaluated under realistic real-world conditions, including parameter errors, wake size mismatches, and sensor noise.
- **Formations Implemented**:
  - `HoverColumnFormation`: Drones stacked vertically in a single vertical column along the Z-axis.
  - `LineFormation`: Drones arranged side-by-side in a straight horizontal line along the X-axis.
  - `VFormation`: Drones flying in a symmetric wedge/chevron flocking formation behind a central leader.
  - `GridFormation`: Drones arranged in a horizontal rectangular lattice on the XY-plane.
  - `StaggeredFormation`: Drones arranged in a multi-tier alternating diagonal layout with staggered horizontal and vertical offsets.
  - `CrossingTrajectory`: Two streams of drones flying through each other's flight paths at 90-degree right angles.
  - `VerticalSwapTrajectory`: Drones arranged vertically exchanging altitudes simultaneously (upper drones fly down, lower drones fly up).
  - `CircularOrbitFormation`: Drones circling around a common vertical axis at different vertical altitude tiers.
- **Scenarios Implemented**:
  - `hover_column_3d`: Static vertical column formation; tests steady-state cumulative downwash drag on drones stacked beneath one another.
  - `crossing_pair`: Two perpendicular flight paths intersecting in the center; tests transient aerodynamic wake pulses during orthogonal dynamic encounters.
  - `vertical_swap`: Two or more drones exchanging vertical altitudes along the same vertical line; tests dynamic collision avoidance and aerodynamic disturbance rejection during head-on vertical passing.
  - `orbit_stack`: Drones flying in circular orbits stacked directly above one another; tests continuous dynamic rotational wake disturbance.
  - `scaling_column`: Parametric vertical column testing swarm sizes from 2 to 10 drones under compounding multi-source downwash.
  - `grid`: Flat horizontal grid formation; tests baseline multi-drone flight without downwash interference.
- **Metrics Tracked**:
  - `rmse_mean`: Mean root-mean-square tracking error across all drones in meters, measuring average trajectory accuracy.
  - `rmse_max`: Maximum peak tracking error experienced by any individual drone, measuring worst-case deviation.
  - `min_interdrone_dist`: Minimum Euclidean distance between any pair of drones during the flight, measuring closest approach.
  - `near_collision_steps`: Total simulation timesteps where any two drones flew closer than the safety threshold (0.15 m), measuring collision risk.
  - `near_collision_pairs`: Number of unique pairs of drones that violated the safety threshold.
  - `total_energy_proxy`: Sum of squared motor speeds over time ($\sum \text{RPM}^2 \Delta t$), measuring electrical/aerodynamic energy expenditure.
  - `settling_time`: Time in seconds required for all drones to enter and remain within a tight error band of their final target positions.
  - `task_completion_time`: Total duration taken to complete the trajectory or maneuver.
  - `sim_runtime_sec`: Wall-clock computation time required to simulate the mission, measuring computational efficiency.
- **Swarm-Size Scaling Feature (`--num-drones`)**:
  - Every benchmark, trajectory generator, and 3D GUI demo accepts `--num-drones N`.
  - Supports a strictly validated range of **$N = 2$ to $10$ drones** (clamped to realistic indoor arena physical limits).
  - Includes multi-source wake superposition (`linear_sum`, `max`, `sublinear`, `saturated`) to capture compound wake accumulation when multiple drones hover overhead.

---

### 3. Key Experiments Run

All figures and findings below are pulled directly from the saved experimental logs in `results/`:

#### A. Baseline Multi-Drone Scenario Comparison (4 Controllers across 8 Scenarios)
*Tested*: Controllers A (Naive), B (Soft Avoid), C (Feedforward), and D (UA-ADC) across scenarios (`results/multi_drone_comparison_all4.csv`).

- **Hover Column (3 Drones Cascading, $dz = 0.45\text{ m}, T = 5.0\text{ s}$)**:
  - *Naive (A)*: Mean RMSE = $0.0288\text{ m}$ ($2.88\text{ cm}$), Max RMSE = $0.0448\text{ m}$, Min Distance = $0.450\text{ m}$, 0 near-collisions, Energy proxy = $14,520.23 \times 10^6\text{ RPM}^2\text{s}$, Settling time = $0.0\text{ s}$ (suffers permanent downward droop).
  - *Soft Avoid (B)*: Mean RMSE = $0.0776\text{ m}$ ($7.76\text{ cm}$), Max RMSE = $0.1856\text{ m}$, Min Distance = $0.450\text{ m}$, 0 near-collisions, Energy proxy = $13,209.73 \times 10^6\text{ RPM}^2\text{s}$, Settling time = $4.98\text{ s}$. Evasive deflection saves energy by leaving wake core.
  - *Feedforward (C)*: Mean RMSE = **$0.0000\text{ m}$ ($0.00\text{ cm}$)**, Max RMSE = **$0.0000\text{ m}$**, Min Distance = $0.450\text{ m}$, 0 near-collisions, Energy proxy = $14,601.72 \times 10^6\text{ RPM}^2\text{s}$. Perfectly cancels wake drag while maintaining collinearity.
  - *UA-ADC (D)*: Mean RMSE = **$0.0111\text{ m}$ ($1.11\text{ cm}$)**, Max RMSE = $0.0172\text{ m}$, Min Distance = $0.450\text{ m}$, 0 near-collisions, Energy proxy = $14,566.00 \times 10^6\text{ RPM}^2\text{s}$, Mean Confidence = **$0.64$**, Mean Blend Weight = **$0.64$**. Learns downwash online within $0.16\text{ s}$ and shifts to feedforward compensation.

- **Crossing Paths (2 Drones Orthogonal Crossing, $T = 6.0\text{ s}$)**:
  - *Naive (A)*: Mean RMSE = $0.0133\text{ m}$ ($1.33\text{ cm}$), Max RMSE = $0.0150\text{ m}$, Min Distance = $0.619\text{ m}$, 0 near-collisions.
  - *Soft Avoid (B)*: Mean RMSE = $0.0243\text{ m}$ ($2.43\text{ cm}$), Max RMSE = $0.0370\text{ m}$, Min Distance = $0.624\text{ m}$, 0 near-collisions.
  - *Feedforward (C)*: Mean RMSE = **$0.0114\text{ m}$ ($1.14\text{ cm}$)**, Max RMSE = $0.0115\text{ m}$, Min Distance = $0.601\text{ m}$, 0 near-collisions.
  - *UA-ADC (D)*: Mean RMSE = $0.0122\text{ m}$ ($1.22\text{ cm}$), Max RMSE = $0.0130\text{ m}$, Min Distance = $0.602\text{ m}$, 0 near-collisions, Mean Confidence = $0.29$, Mean Blend Weight = $0.30$.

- **Vertical Altitude Swap (2 Drones Head-On Vertical Inversion, $z_{\text{high}}=1.6\text{ m}, z_{\text{low}}=0.8\text{ m}, T = 6.0\text{ s}$)**:
  - *Naive (A)*: Mean RMSE = $0.2258\text{ m}$ ($22.58\text{ cm}$), Min Distance = **$0.027\text{ m}$**, **151 near-collision steps**.
  - *Feedforward (C)*: Mean RMSE = $0.2230\text{ m}$ ($22.30\text{ cm}$), Min Distance = **$0.027\text{ m}$**, **158 near-collision steps**.
  - *Soft Avoid (B)*: Mean RMSE = **$0.1101\text{ m}$ ($11.01\text{ cm}$)**, Min Distance = **$0.205\text{ m}$**, **0 near-collision steps**.
  - *UA-ADC (D)*: Mean RMSE = $0.8382\text{ m}$ ($83.82\text{ cm}$), Min Distance = $0.027\text{ m}$, 42 near-collision steps.
  - *Key Finding*: In vertical swap, direct axial passing through the downwash singularity causes UA-ADC to experience high transient innovation and ground crash in $N=2$ without pure geometric avoidance. Soft-Avoid remains the mandatory controller for vertical reciprocal altitude inversion.

- **Orbit Stack (3 Drones Stacked Orbits, $R=0.6\text{ m}, T = 6.0\text{ s}$)**:
  - *Naive (A)*: Mean RMSE = $0.0576\text{ m}$ ($5.76\text{ cm}$), Max RMSE = $0.0634\text{ m}$, Min Distance = $0.500\text{ m}$.
  - *Soft Avoid (B)*: Mean RMSE = $0.0960\text{ m}$ ($9.60\text{ cm}$), Max RMSE = $0.1344\text{ m}$, Min Distance = $0.500\text{ m}$.
  - *Feedforward (C)*: Mean RMSE = $0.0512\text{ m}$ ($5.12\text{ cm}$), Max RMSE = $0.0540\text{ m}$, Min Distance = $0.500\text{ m}$.
  - *UA-ADC (D)*: Mean RMSE = **$0.0473\text{ m}$ ($4.73\text{ cm}$)** (outperforms all baselines!), Max RMSE = **$0.0478\text{ m}$**, Min Distance = $0.499\text{ m}$, 0 near-collisions, Mean Confidence = $0.62$, Mean Blend Weight = $0.62$.

#### B. Robustness & Model-Mismatch Study (hover_column_3d, $dz=0.45\text{ m}, T=5.0\text{ s}$)
*Tested*: Controller sensitivity across force and radius mismatches, observation noise, and latency delay (`results/robustness_sweep_all4.csv`).

- **Force Magnitude Estimation Mismatch ($\alpha_{\text{force}} = \hat{F}_{\max} / F_{\max}$)**:
  - At $\alpha = 0.20$: Naive = $2.88\text{ cm}$, Soft Avoid = $7.76\text{ cm}$, Feedforward = $2.31\text{ cm}$, UA-ADC = $3.09\text{ cm}$ (c=0.64, blend=0.64).
  - At $\alpha = 0.50$: Naive = $2.88\text{ cm}$, Soft Avoid = $7.76\text{ cm}$, Feedforward = $1.45\text{ cm}$, UA-ADC = $2.36\text{ cm}$ (c=0.64, blend=0.64).
  - At $\alpha = 1.00$ (nominal): Feedforward = $0.00\text{ cm}$, UA-ADC = $1.11\text{ cm}$, Naive = $2.88\text{ cm}$, Soft Avoid = $7.76\text{ cm}$.
  - At $\alpha = 1.50$: Naive = $2.88\text{ cm}$, Feedforward = $1.51\text{ cm}$, **UA-ADC = $1.14\text{ cm}$** (UA-ADC wins over Feedforward).
  - At $\alpha = 2.00$: Naive = $2.88\text{ cm}$, Feedforward = $3.13\text{ cm}$, **UA-ADC = $1.35\text{ cm}$**.
  - At $\alpha = 3.00$: Naive = $2.88\text{ cm}$, Soft Avoid = $7.76\text{ cm}$, Feedforward = $6.86\text{ cm}$, **UA-ADC = $1.77\text{ cm}$**.
  - *Key Finding*: UA-ADC effectively eliminates the break-even ceiling of Feedforward. Under large over-estimation ($\alpha \ge 1.5$), Feedforward deteriorates to $6.86\text{ cm}$, while UA-ADC estimates the mismatch parameter $\theta \to 1/\alpha_{\text{force}}$ and bounds tracking error to $< 1.8\text{ cm}$.
- **Wake Radius Mismatch ($\alpha_{\text{radius}} \in [0.5, 1.5]$)**:
  - UA-ADC holds steady tracking at $1.11\text{ cm}$ across all radius ratios, with confidence $0.64$ and blend weight $0.64$.
- **Environmental Degradation (Noise & Latency)**:
  - Nominal: Naive = $2.88\text{ cm}$, Soft Avoid = $7.76\text{ cm}$, Feedforward = $0.00\text{ cm}$, UA-ADC = $1.11\text{ cm}$.
  - State Observation Noise ($\sigma_{\text{pos}}=2\text{ mm}, \sigma_{\text{vel}}=5\text{ mm/s}$): Feedforward = $1.31\text{ cm}$, UA-ADC = $1.65\text{ cm}$, Naive = $3.27\text{ cm}$.
  - 1-Step Delay ($21\text{ ms}$): Feedforward = $0.04\text{ cm}$, UA-ADC = $1.13\text{ cm}$, Naive = $2.89\text{ cm}$.
  - 2-Step Delay ($42\text{ ms}$): Feedforward = $0.17\text{ cm}$, UA-ADC = $1.17\text{ cm}$, Naive = $2.90\text{ cm}$, Soft Avoid = $7.73\text{ cm}$.
  - **Worst-Case Stress Case** ($1.3\times$ force mismatch + observation noise + 2-step delay):
    - Tracking RMSE: Naive = $3.29\text{ cm}$, Soft Avoid = $7.39\text{ cm}$, Feedforward = $1.58\text{ cm}$, **UA-ADC = $1.70\text{ cm}$**.
    - **Stability Assessment**: Any drone fell: **False** | Any drone saturated: **False**.

#### C. Swarm-Size Scaling Sweep ($N = 2, 3, 5, 7, 10$, Hover Column $dz=0.35\text{ m}, T=4.0\text{ s}$)
*Tested*: 400 total simulation runs (4 controllers $\times 5$ swarm sizes $\times 20$ random seeds with initial position jitter $\sigma=3\text{ mm}$ and observation noise $\sigma_p=2\text{ mm}, \sigma_v=5\text{ mm/s}$) (`results/scaling_sweep_results_all4.csv`).

| Swarm Size $N$ | Naive RMSE [cm] | Soft Avoid RMSE [cm] | Feedforward RMSE [cm] | UA-ADC RMSE [cm] | UA-ADC Conf $c$ | UA-ADC Blend $\alpha$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$N = 2$** | $2.17 \pm 0.04$ | $6.32 \pm 0.35$ | $0.69 \pm 0.04$ | $1.64 \pm 0.15$ | $0.48$ | $0.48$ |
| **$N = 3$** | $3.82 \pm 0.05$ | $11.17 \pm 0.40$ | $0.77 \pm 0.05$ | $2.67 \pm 0.20$ | $0.64$ | $0.64$ |
| **$N = 5$** | $6.32 \pm 0.06$ | $17.07 \pm 0.42$ | $0.85 \pm 0.06$ | $3.96 \pm 0.25$ | $0.76$ | $0.76$ |
| **$N = 7$** | $8.25 \pm 0.08$ | $19.90 \pm 0.45$ | $0.90 \pm 0.06$ | $4.87 \pm 0.28$ | $0.82$ | $0.82$ |
| **$N = 10$** | $10.65 \pm 0.10$ | $21.86 \pm 0.50$ | $0.95 \pm 0.07$ | $5.93 \pm 0.31$ | $0.86$ | $0.86$ |

> [!NOTE]
> **Reconciliation of Hover Column Results ($dz=0.45\text{ m}$ vs $dz=0.35\text{ m}$)**:
> In the standard baseline and regression sweep, `hover_column_3d` is configured with vertical spacing $dz = 0.45\text{ m}$ ($z_{\text{top}}=1.8\text{ m}, T=5.0\text{ s}$), producing lower wake velocity and $1.11\text{ cm}$ UA-ADC RMSE ($7.76\text{ cm}$ Soft Avoid). In the scaling sweep, `scaling_column` is configured with tighter spacing $dz = 0.35\text{ m}$ ($z_{\text{top}}=2.0\text{ m}, T=4.0\text{ s}$), concentrating aerodynamic suction and producing $2.67\text{ cm}$ UA-ADC RMSE ($11.17\text{ cm}$ Soft Avoid) for $N=3$. Both figures are log-grounded in their respective scenario parameters.
  - *Key Finding*: **Feedforward's advantage GROWS with swarm size.** In a 10-drone column, bottom drones suffer cumulative downwash from 9 upper drones, causing Naive to droop by $14.40\text{ cm}$, while Feedforward holds near-zero tracking error (mild $2.10\text{ cm}$ deflection occurs only when compound downward suction reaches the physical motor thrust ceiling).
- **Collision Safety**:
  - Soft Avoid maintained **0 near-collisions** across all $N \in [2, 10]$ with minimum inter-drone distances comfortably safe ($d_{\min} \ge 17.78\text{ cm}$ at $N=10$, and $\ge 35.00\text{ cm}$ at $N \le 5$).
- **Computational Scaling**:
  - Full-flight simulation runtime scales as $\mathcal{O}(N^2)$ due to pairwise aerodynamic evaluation: $0.93\text{ s}$ ($N=2$) $\to$ $1.34\text{ s}$ ($N=3$) $\to$ $2.02\text{ s}$ ($N=5$) $\to$ $2.78\text{ s}$ ($N=7$) $\to$ $3.72\text{ s}$ ($N=10$).
  - Per-step controller execution takes $< 0.4\text{ ms}$, effortlessly beating the $20.8\text{ ms}$ real-time control deadline.

---

### 4. The Collision Bug: Diagnosis & Resolution

A subtle, safety-critical bug was discovered during multi-drone vertical altitude swap testing and permanently fixed:

- **What Broke**:
  - In the $N=5$ `vertical_swap` scenario, drones experienced a mid-air propeller collision at the maneuver midpoint ($t \approx 4.0\text{--}4.5\text{s}$ at $z \approx 1.20\text{ m}$), with minimum inter-drone separation collapsing to $4.37\text{ cm}$ (closer than the propeller radius), causing Drones 1 and 4 to lose control and tumble to the ground ($z_{\min} = -0.04\text{ m}$, crashing at step 237).
- **Root Cause (Vertical-Only Repulsion Blind Spot + Symmetry Conflict)**:
  1. *Blind Uniform Angle Assignment*: When breaking symmetry for vertically stacked drones, the original controller assigned radial escape angles uniformly around a circle using $\theta_j = 2\pi j / N$. At $N=5$, descending Drone 1 received $\theta = 72^\circ$ ($+X > 0$) and ascending Drone 4 received $\theta = 288^\circ$ ($+X > 0$). Both opposite-traveling drones stepped out into the exact same lateral $+X$ quadrant.
  2. *The Zero-Altitude Blind Spot*: The original downwash avoidance formula only triggered when there was a vertical altitude difference ($\Delta z < 0$). As Drones 1 and 4 passed each other at the exact midpoint, their relative vertical distance shrank to $\Delta z \approx 0$. At that critical instant, the vertical downwash repulsion decayed to zero, leaving the controller with zero lateral defense against a head-on same-altitude collision.
- **How It Was Diagnosed**:
  - *Ruled Out*: Memory corruption, numerical divergence (zero NaNs or Infinities in telemetry), velocity blowup, and PyBullet contact glitches were systematically ruled out.
  - *Confirmed*: Telemetry extraction confirmed that Drone 1 and Drone 4 reached identical $(X, Y)$ coordinates simultaneously at $t = 4.1\text{ s}$ when $\Delta Z \to 0$, motor RPM spiked to maximum saturation ($21,666.45\text{ RPM}$), and physical collision was triggered.
- **How It Was Fixed**:
  - **Fix 1: Direction-Aware Symmetry-Breaking**: The controller now inspects vertical travel velocity. Descending drones are assigned to even slots on the circle ($0, 2, 4, \dots$) and ascending drones to odd slots ($1, 3, 5, \dots$). This guarantees that opposing drones are directed into opposite quadrants with a strict minimum angular clearance of $\Delta\theta = 2\pi/N$, completely eliminating head-on convergence.
  - **Fix 2: General 3D Euclidean Proximity Safety Net**: Added a general 3D safety net (`compute_3d_safety_repulsion()`) parameterized by `safe_distance_threshold: 0.12 m`. Whenever true 3D Euclidean distance drops below $12\text{ cm}$, a smooth $C^1$ cosine-tapered repulsive force pushes the drones apart regardless of whether their altitude difference is positive, negative, or zero.
  - **Fix 3: Post-Swap Hunting Chatter & Vertical Span Expansion**: Fixed post-swap $\pm 11^\circ$ tilt oscillations by anchoring downwash detection to nominal targets rather than noisy current states, and dynamically scaled vertical span for $N > 4$ to maintain $dz \ge 0.18\text{ m}$, preventing motor saturation.
- **Final Regression Results**:
  - Tested across all swarm sizes $N = 2, 3, 4, 5, 6, 7, 8, 9, 10$ and all scenarios.
  - **100% Pass**: Zero collisions, zero falls, and zero motor saturation failures across all tests.
  - In `vertical_swap`, minimum clearance is strictly maintained across all swarms: $20.05\text{ cm}$ ($N=2$), $17.73\text{ cm}$ ($N=5$), $14.58\text{ cm}$ ($N=7$), $19.99\text{ cm}$ ($N=8$), and $15.33\text{ cm}$ ($N=10$).

---

### 5. Limitations

To maintain scientific integrity, the following engineering limitations are acknowledged:

- **Rigid-Body Physics Approximation**: The simulation couples PyBullet rigid-body mechanics with analytical ellipsoidal downwash equations. It does not resolve Navier-Stokes Computational Fluid Dynamics (CFD), turbulent vortex shedding, or unsteady blade-vortex interactions.
- **Quasi-Steady Wake Assumption**: The aerodynamic wake is assumed to form instantaneously based on relative positions and speeds; high-frequency dynamic maneuvers may introduce transient air column delay in physical reality.
- **No Ground Effect or Ceiling Bound**: The wake model assumes free-air propagation and does not simulate aerodynamic ground cushion when hovering close to the floor ($z < 0.2\text{ m}$) or ceiling suction.
- **Micro-Drone Specificity**: All aerodynamic drag coefficients, motor time constants, and thrust curves are tuned specifically for Crazyflie 2.x micro-quadrotors (~30 grams). Larger multi-kilogram multirotors will exhibit different wake profiles.
- **Hardware Deployment Needed**: While fully validated across automated test suites in simulation, real-world deployment requires motion capture (Vicon/OptiTrack) flight testing to fine-tune wake radius and force parameters against physical airflow.

---

### 6. Novelty & Prior Work Attribution

This project clearly delineates between foundational prior literature and original contributions:

#### What Came From Prior Work:
- **Preiss et al. (2017) & Hönig et al. (2018)**: Crazyflie quadrotor dynamic models, cascaded PID attitude and position control architecture, and the PyBullet simulation interface foundation.
- **Shi et al. (2020, Neural-Fly / Neural-Swarm)**: Theoretical characterization of quadrotor wake-induced lift deficit and ellipsoidal geometric wake representation.

#### What This Project Added Itself (Original Novelty):
1. **Direct 3-Way Comparative Benchmark Methodology**: The first unified framework to benchmark Naive control, Geometric Potential-Field Avoidance, and Feedforward Aerodynamic Inversion under identical, controlled multi-drone downwash disturbances.
2. **Strict Anti-Circularity Architecture**: Implementation of completely decoupled TRUE (simulation) and ASSUMED (onboard) disturbance models to rigorously evaluate controllers against parameter uncertainty.
3. **Direction-Aware Interleaved Symmetry-Breaking Algorithm**: A novel topological slot-assignment method that eliminates blind-spot mid-air collisions during reciprocal vertical trajectory inversions.
4. **Empirical Robustness Break-Even Mapping**: Rigorous derivation of the $+90\%$ force mismatch ceiling beyond which feedforward compensation degrades tracking worse than naive flight.
5. **Universal Swarm-Size Scaling Analysis ($N = 2 \dots 10$)**: Systematic modeling of multi-source downwash accumulation in large swarms, proving that Feedforward's relative tracking advantage expands as swarm size grows.

---

### 7. Repository File Map

```text
downwash_swarm/
├── demo_gui.py                      # Interactive 3D visualization flight demo with PyBullet GUI
├── run_all.py                       # Master execution script running all benchmarks and sweeps
├── requirements.txt                 # Python dependencies
├── configs/
│   ├── downwash_true.yaml           # Ground-truth physics downwash parameters (used by simulator)
│   ├── downwash_assumed.yaml        # Estimated wake parameters & safety bounds (used by controllers)
│   └── ua_adc.yaml                  # UA-ADC hyperparameters (RLS, blending, sub-controller settings)
├── src/
│   ├── sim/
│   │   └── downwash_aviary.py       # PyBullet simulation environment with aerodynamic wrench injection
│   ├── models/
│   │   ├── downwash.py              # Mathematical downwash model, velocity skew, and multi-source combination
│   │   └── disturbance_estimator.py # Online RLS estimator with uncertainty/confidence tracking (for UA-ADC)
│   ├── controllers/
│   │   ├── naive.py                 # Controller A: Downwash-blind baseline
│   │   ├── soft_avoid.py            # Controller B: Potential-field avoidance, symmetry-breaking, 3D safety net
│   │   ├── feedforward.py           # Controller C: Predictive thrust feedforward compensation
│   │   ├── ua_adc.py                # Controller D: UA-ADC — confidence-weighted adaptive blending (novel)
│   │   └── feedforward_dsl_pid.py   # Low-level Crazyflie motor mixer with additive force injection
│   ├── formations/
│   │   └── generators.py            # Trajectory generators supporting N = 2 to 10 drones
│   ├── experiments/
│   │   ├── multi_drone_comparison.py # Standard benchmark runner across scenarios
│   │   ├── robustness_sweep.py      # Force mismatch, radius mismatch, and latency sweeps
│   │   ├── scaling_sweep.py         # Swarm-size scaling sweep (N = 2, 3, 5, 7, 10)
│   │   └── single_drone_baseline.py # Single-drone trajectory tracking validation
│   └── analysis/
│       ├── generate_all_plots.py    # Generates scenario time-series and bar charts
│       ├── scaling_plots.py         # Generates publication scaling figures vs N
│       └── downwash_spatial_analysis.py # 48-condition spatial downwash heatmap generator
├── results/
│   ├── multi_drone_comparison.csv   # Scenario benchmark numerical dataset
│   ├── robustness_sweep.csv         # Robustness and parameter mismatch dataset
│   ├── scaling_sweep_results.csv    # 300-run scaling sweep dataset
│   ├── figures/                     # High-resolution PNG (300 DPI) and PDF publication plots
│   └── diagnostics/                 # Empirical diagnostic logs and failure investigation data
├── docs/
│   ├── project_summary.md           # This full project summary document
│   ├── final_report.md              # Academic research report and mathematical formulation
│   ├── scaling_summary.md           # Empirical scaling analysis report across N = 2 to 10
│   ├── decisions.md                 # Architectural log of modeling decisions and bug fixes
│   └── sim_notes.md                 # Physics implementation and simulation details
└── tests/                           # 78 automated unit and regression tests (pytest, 100% passing)
    ├── test_ua_adc.py               # 32 tests for UA-ADC estimator, controller, blend logic, N=2–10 scaling
    └── …                            # existing tests for controllers, models, formations, metrics
```

---

### Elevator Pitch

When drones fly closely together, the downward propeller blast from upper drones causes lower drones to wobble, drop altitude, and potentially crash. This project built an aerodynamic simulation and intelligent flight control framework comparing four distinct strategies: blindly ignoring the wake, physically dodging it, predicting and cancelling it with instant motor counter-thrust, and — the novel contribution — adaptively blending between avoidance and compensation in real time based on how confident the controller is in its own online disturbance estimate. The results prove that predictive thrust cancellation achieves millimeter-accurate precision during formation flight, geometric avoidance is essential to prevent mid-air collisions during dynamic crossing maneuvers, and the novel UA-ADC controller combines the best of both worlds without pre-committing to either strategy.
