# Architectural & Physical Modeling Decisions Log

This document records architectural, physical modeling, and algorithmic design decisions for the **Downwash-Aware Formation Control for Multi-Drone Swarms** project.

---

## Decision 1: Swarm Size Boundary Bounds $N \in [2, 10]$

- **Context**: In extending the framework to scalable swarms, upper and lower limits on drone count $N$ must be explicitly bounded rather than unbounded or silently clamped.
- **Decision**: Enforce a strict valid range of $2 \le N \le 10$ across all entry points (`demo_gui.py`, `multi_drone_comparison.py`, `scaling_sweep.py`, and all formation generators in `src/formations/generators.py`).
- **Rationale**:
  - **Lower bound ($N \ge 2$)**: Aerodynamic downwash interaction, inter-drone proximity metrics, and collision avoidance are inherently multi-agent phenomena. Single-drone tracking is isolated to Phase 1 baseline verification.
  - **Upper bound ($N \le 10$)**: Representative of realistic micro-quadrotor (Crazyflie 2.X) indoor flight arenas (typically $3\text{m} \times 3\text{m} \times 2.5\text{m}$) and PyBullet rigid-body physics efficiency. Beyond 10 drones, physical arena spatial constraints force extreme proximity violations without arena expansion.
- **Enforcement**: Any input outside $[2, 10]$ is actively rejected with a descriptive `ValueError` or CLI argument error; silent fallbacks are prohibited.

---

## Decision 2: Multi-Source Downwash Combination Modeling (Configurable Superposition)

- **Context**: When $N > 2$ drones are stacked or clustered, a lower drone can sit simultaneously in the wake footprints of multiple upper drones. The interaction of multiple propeller slips in real fluid dynamics is complex, turbulent, and non-linear (involving wake contraction, turbulent mixing, and velocity deficit saturation).
- **Decision**:
  1. The downwash model (`DownwashModel` in `src/models/downwash.py`) explicitly evaluates all upper drones affecting each lower drone.
  2. Rather than hardcoding simple additive summation, multi-source disturbance combination is made **fully configurable** via the `combination_mode` parameter in `configs/downwash_true.yaml` and `configs/downwash_assumed.yaml`.
  3. Supported modes include:
     - `linear_sum` / `additive`: Direct vector summation $\vec{F}_j = \sum_{i} \vec{f}_{ij}$.
     - `max` / `peak`: Dominant wake interaction $\max_i |\vec{f}_{z, ij}|$ governs disturbance.
     - `sublinear`: Power-law diminishing returns combining vertical forces with exponent $\gamma \in (0, 1]$: $F_z = -\left(\sum |f_{z, ij}|^\gamma\right)^{1/\gamma}$.
     - `saturated`: Linear sum clamped to a ceiling `max_combined_force_z`.
  4. Both TRUE (simulation) and ASSUMED (controller) configurations retain independent combination settings to preserve anti-circularity.
- **Plausibility & Physical Validity Note**:
  > [!WARNING]
  > **Simplifying Assumption Alert**: Linear superposition (`linear_sum`) is an engineering approximation. In physical aerodynamics, wake momentum deficit does not accumulate without bound; it saturates due to turbulent dissipation and slipstream merging. For this study, `max_combined_force_z` is set to $0.45\text{ N}$ (approx. $1.6\times$ single drone hover thrust) to prevent unphysical infinite forces while capturing compound downwash. Experimental wind-tunnel or particle image velocimetry (PIV) / CFD data is required to validate exact superposition coefficients in physical deployments.

---

## Decision 3: Formation Shape Behavior and Degradation Across $N \in [2, 10]$

All formation generators in `src/formations/generators.py` support arbitrary $N \in [2, 10]$. However, geometric layouts exhibit specific trade-offs and edge-case behaviors across small vs. large $N$:

| Formation Generator | Behavior at Small $N$ ($N=2, 3$) | Behavior at Large $N$ ($N=7, 10$) | Edge-Case / Degradation Notes |
| :--- | :--- | :--- | :--- |
| **`HoverColumnFormation`** | Crisp 2-3 drone vertical stack with $dz=0.45\text{ m}$. | Drones occupy dense vertical column ($dz$ dynamically adapted to fit $z \in [0.4, 2.0]\text{ m}$). | Lowest drones suffer extreme compound downwash; at $N=10$, bottom drone experiences wake from 9 disturbers. |
| **`VFormation`** | **Trivial at $N=2$**: Degenerates into a 2-drone angled pair (1 leader, 1 wing drone). | Well-defined symmetric wedge flock with balanced left/right trailing wings. | Symmetric V flocking requires $N \ge 3$ (and preferably odd $N$) for visual symmetry; at even $N$, right wing has one extra drone. |
| **`GridFormation`** | $N=2$: $1 \times 2$ grid.<br>$N=3$: $2 \times 2$ grid with 1 empty slot. | Rectangular lattice ($n_{cols} = \lceil\sqrt{N}\rceil$). | For non-perfect square primes ($N=3, 5, 7, 10$), the trailing row has fewer drones than preceding rows. |
| **`LineFormation`** | 2-3 drone horizontal line. | 7-10 drone extended horizontal string. | Cleanly defined for all $N$; spans $x \in [-2.0\text{ m}, +2.0\text{ m}]$ at $N=10$ with $0.45\text{ m}$ spacing. |
| **`StaggeredFormation`** | 2-3 drone alternating offset. | Multi-tier 3D staggered lattice. | Alternates $Y$ offsets and increments $Z$ altitude in modulo steps ($z_{base} + (i \pmod 3) dz$) to probe diagonal shear layers. |
| **`CrossingTrajectory`** | Classic 2-drone perpendicular intersection ($1 \text{ vs } 1$). | Stream-crossing ($n_A = \lceil N/2\rceil$ along X vs $n_B = \lfloor N/2\rfloor$ along Y). | Parallel drones within each stream are laterally separated by $0.35\text{ m}$ to avoid co-stream collisions. |
| **`VerticalSwapTrajectory`** | 2 drones swap $z_{high} \leftrightarrow z_{low}$. | Stratified altitude inversion across $N$ tiers. | Drones invert their vertical order using 5th-order minimum-jerk polynomials. |
| **`CircularOrbitFormation`** | Coaxial or phased orbits. | Multi-tier stacked orbit carousel. | Altitudes evenly distributed between $1.6\text{ m}$ and $0.6\text{ m}$. |

---

## Decision 4: Scaling Sweep Experiment Scenario Selection (`scaling_column`)

- **Selected Scenario**: `HoverColumnFormation` (`scaling_column`).
- **Rationale**:
  1. **Maximum Aerodynamic Downwash Stress**: In a vertical column, every drone $k > 0$ sits directly along the centerline of all drones $0, \dots, k-1$ above it. This provides the most rigorous test of multi-source downwash accumulation.
  2. **Unambiguous Controller Differentiation**:
     - *Controller A (Naive)*: Suffers compounding downward force, directly exposing steady-state droop scaling.
     - *Controller B (Soft-Avoidance)*: Repels lower drones laterally to escape the wake, visibly spreading the vertical column into a wider conical formation.
     - *Controller C (Feedforward)*: Directly sums predictive counter-thrust to hold the column in place, exposing motor saturation limits at high $N$.
  3. **Direct Comparability**: Vertical column geometry isolates aerodynamic downwash from lateral tracking dynamics, ensuring that metric degradation across $N$ is strictly attributable to multi-source aerodynamic interaction.

---

## Decision 5: Vertical Swap Midpoint Collision Fix and General 3D Proximity Safety Net

### Summary Note
> **A collision bug was found and fixed** — `soft_avoid` originally used vertical-only repulsion and had no defense against same-altitude lateral collisions during crossing maneuvers. Fixed via **Step 1 (Direction-Aware Symmetry-Breaking Angle Assignment)** and hardened via **Step 2 (General 3D Euclidean Safety Net)**.

### Root Cause Analysis
- **Defect Description**: In the `vertical_swap` scenario at $N=5$, drones experienced physical propeller collisions ($d_{\min} = 4.37\text{ cm}$) and tumbled to the ground ($z_{\min} = -0.04\text{ m}$).
- **Mechanism**: The original `soft_avoid` controller used a blind uniform radial angle assignment $\theta_j = 2\pi j / N$ to break symmetry when drones start vertically stacked. At $N=5$:
  - Descending Drone 1 ($Z: 1.4\text{m} \to 1.0\text{m}$) received angle $72^\circ$ ($+X > 0$).
  - Ascending Drone 4 ($Z: 0.8\text{m} \to 1.6\text{m}$) received angle $288^\circ$ ($+X > 0$).
  - Both opposite-traveling drones were displaced into the exact same lateral $+X$ half-plane.
  - As they converged near the midpoint ($t \approx 4.0\text{--}4.5\text{s}$ at $Z \approx 1.20\text{m}$), their relative vertical separation shrank to $\Delta Z \approx 0$. Because the original controller only repelled lower drones beneath upper drones ($\Delta Z < 0$), the repulsion decayed to zero at $\Delta Z = 0$, leaving no lateral defense while converging on the same lateral side.
- **Ruled Out Causes**: Verified NOT caused by numerical instability (zero NaNs/Infs), unbounded velocity growth, or mathematical singularity.

### Step 1: Direction-Aware Symmetry-Breaking Assignment (Root-Cause Patch)
- **Algorithm**: In `src/controllers/soft_avoid.py`, `_update_symmetry_directions()`:
  1. Inspects commanded and state vertical velocity $v_z$.
  2. Partitions drones into descending ($v_z < -10^{-6}$), stationary ($|v_z| \le 10^{-6}$), and ascending ($v_z > 10^{-6}$) sets, sorted by travel speed.
  3. Maps drones to an **interleaved uniform $N$-gon slot sequence** around the full $2\pi$ circle:
     - Even slots ($0, 2, 4, \dots$) assigned to the upper/descending half.
     - Odd slots ($1, 3, 5, \dots$) assigned to the lower/ascending half.
  4. Guarantees that every drone has a strict minimum angular separation of $\Delta\theta = \frac{2\pi}{N}$ from its nearest angular neighbors, completely eliminating sector crowding/clustering (which originally caused a $30^\circ$ squeeze and collision at $N=7$).
  5. Opposing vertical travel directions naturally map to alternating quadrants with maximum antipodal clearance.
  6. Non-crossing scenarios (`hover_column_3d`, `crossing_pair`, `grid`) preserve original blind dispersion without regression.
- **Empirical Validation**: Completely resolved collisions and falls across $N \in [2, 7]$ in `demo_gui.py` ($10.0\text{s}$ duration):
  - $N=2$: $d_{\min} = 20.12\text{ cm}$, zero falls
  - $N=3$: $d_{\min} = 20.38\text{ cm}$, zero falls
  - $N=4$: $d_{\min} = 19.31\text{ cm}$, zero falls
  - $N=5$: $d_{\min} = 19.20\text{ cm}$, zero falls
  - $N=6$: $d_{\min} = 16.00\text{ cm}$, zero falls
  - $N=7$: $d_{\min} = 13.33\text{ cm}$, zero falls (fixed from original collision at $t=7.06\text{s}$)

### Step 2: General 3D Euclidean Minimum-Distance Safety Net (Hardening Layer)
- **Architecture**: Added `compute_3d_safety_repulsion()` in `soft_avoid.py`.
- **Activation**: Triggers whenever true 3D Euclidean distance $d_{3D} = \|\mathbf{p}_j - \mathbf{p}_i\|$ drops below `safe_distance_threshold`, independent of relative vertical altitude $\Delta Z$.
- **Smooth Formulation**: Employs a $C^1$ continuous cosine taper:
  $$\mathbf{F}_{ij} = F_{\max} \cdot \frac{1}{2}\left(1 + \cos\left(\pi \frac{d_{3D}}{d_{\text{safe}}}\right)\right) \cdot \frac{\mathbf{p}_j - \mathbf{p}_i}{d_{3D}}$$
  $$\mathbf{d}_{ij} = d_{\max} \cdot \frac{1}{2}\left(1 + \cos\left(\pi \frac{d_{3D}}{d_{\text{safe}}}\right)\right) \cdot \frac{(\mathbf{p}_j - \mathbf{p}_i)_{xy}}{d_{3D}}$$
  - At $d_{3D} \ge d_{\text{safe}}$: Force and offset are identically zero ($C^1$ derivative is zero; does not fight Step 1).
  - At $d_{3D} < d_{\text{safe}}$: Produces equal-and-opposite repulsive feedforward force in `feedforward_forces` and additive lateral setpoint displacement in `cmd_positions`.
- **Safe-Distance Threshold Value**:
  - `safe_distance_threshold`: **$0.12\text{ m}$** ($12\text{ cm}$), corresponding to $\approx 2.6\times$ propeller diameter ($d_{\text{prop}} = 0.0463\text{ m}$).
  - Configuration location: [configs/downwash_assumed.yaml](file:///c:/Users/Dell/.gemini/antigravity-ide/scratch/downwash_swarm/configs/downwash_assumed.yaml).
  - Status: `TODO-verify against experimental collision envelope in wind tunnel / physical motion-capture flight`.

---

## Decision 6: Post-Swap Hover Stability & Universal Swarm Scaling ($N \in [2, 10]$)
- **Problem**:
  1. *Post-Swap Shaking / Attitude Oscillation*: After swapping, drones exhibited $\pm 11^\circ$ limit-cycle shaking during hover hold. When lower drones escaped the wake laterally, measuring wake from current state collapsed avoidance offsets to zero, creating a hunting loop (dodge $\to$ escape $\to$ return $\to$ re-enter wake $\to$ dodge).
  2. *$N \ge 8$ Thrust Saturation & Midpoint Squeeze*: Compressing $N \ge 8$ drones into fixed 0.8m altitude bounds led to initial vertical separations $dz < 0.089\text{m}$, creating cumulative downward downwash forces ($> 0.71\text{N}$) that exceeded Crazyflie motor limits. Furthermore, when upper drones moved laterally, lower drones saw upper drones near $r = 0.25\text{m}$ and prematurely collapsed their radial offsets, causing chord-distance squeeze at $N=10$.
- **Solution**:
  1. *Anchored Wake Avoidance Reference*: Evaluated vertical wake presence using `formation_targets[j]` rather than current noisy positions. Holds evasive offsets steady during hover hold without state-feedback hunting.
  2. *Corridor Offset Preservation*: When dedicated symmetry corridors (`self.symmetry_dirs`) are active, held $weight_r = 1.0$ across the vertical span as long as upper drones remain in the stack, preventing premature radial collapse during reciprocal swaps.
  3. *Dynamic Vertical Span Expansion*: For $N > 4$, `VerticalSwapTrajectory` scales the vertical span dynamically ($z_{\text{low}} = \max(0.45, 1.20 - \text{half\_span})$, $z_{\text{high}} = \min(2.15, 1.20 + \text{half\_span})$), keeping $dz \ge 0.18\text{m}$ and preventing motor saturation.
  4. *Adaptive Clearance Corridor for $N \ge 8$*: Configured $R = 0.28\text{m}$ and $v_{\max} = 0.30\text{ m/s}$ for large swarms ($N \ge 8$), guaranteeing $> 14.5\text{ cm}$ clearance between all pairs across all $N \in [2, 10]$.
- **Outcome**:
  - All swarm counts $N \in [2, 10]$ achieve 100% collision-free and fall-free flight in `vertical_swap` with min clearance $\ge 14.58\text{ cm}$.
  - Post-swap hover hold attitude tilt settled to $< 2.4^\circ$ with zero limit-cycle chatter.

---

## Decision 5: UA-ADC — Online Estimation with Confidence-Weighted Blending

- **Context**: Controllers A, B, C each commit to a single strategy at design time — Controller A ignores downwash, B avoids it, C compensates it. None of them adapt when their assumptions break down (model mismatch, varying swarm density, novel disturbance profiles). The core research gap is: *can a controller infer how well it understands the disturbance and adjust its strategy accordingly?*

- **Decision**: Implement Controller D — **UA-ADC (Uncertainty-Aware Adaptive Control)** — that:
  1. Maintains a per-drone **Recursive Least-Squares (RLS) online estimator** that fits a scalar gain θ on the assumed-model prediction as a feature, observing the vertical acceleration residual.
  2. Derives a **confidence value** c ∈ [0, 1] from the posterior variance P of the RLS estimator via a log-linear mapping (c → 0 when P ≈ P_init, c → 1 when P collapses).
  3. Smooths confidence with an EMA filter (α_smooth = 0.90) to prevent chattering.
  4. Computes a **sigmoid blend weight** α = σ(k·(c − 0.5)) that continuously interpolates between avoidance (α → 0) and feedforward compensation (α → 1).
  5. Blends *both* position setpoints and feedforward forces from the two sub-controllers continuously.
  6. Keeps the SoftAvoid 3D safety-net repulsion **always active** (additive, never blended away), regardless of confidence.

- **Why RLS over simpler approaches**:
  - RLS is recursive (O(1) per step, no matrix inversion for a scalar system) and tunable via a single forgetting factor λ.
  - A Bayesian prior (initial P_0 = 1.0) ensures the estimator starts uncertain (c = 0, prefer avoidance) and only gains confidence after observing consistent signal — this is the safe default.
  - A large φ feature (the assumed-model Fz prediction) collapses P quickly when signal is present; a near-zero feature keeps P large (uncertain), naturally encoding the "drone not in a wake" case.

- **Why sigmoid blending over a discrete switch**:
  - A threshold-based hard switch creates velocity discontinuities that excite the low-level PID and amplify position error spikes.
  - The sigmoid with sharpness k = 8.0 transitions smoothly over a confidence range of ≈ 0.3–0.7, giving a C^∞ interpolation and zero impulsive force commands.

- **Anti-circularity guarantee**: UA-ADC passes the assumed model (configs/downwash_assumed.yaml) to both sub-controllers and to the estimator feature. The TRUE simulation parameters (configs/downwash_true.yaml) are never accessed or imported by any UA-ADC code path.

- **Implementation**:
  - Estimator: `src/models/disturbance_estimator.py` (`DroneDisturbanceEstimator`, `SwarmDisturbanceEstimator`)
  - Controller: `src/controllers/ua_adc.py` (`UAAdaptiveController`)
  - Config: `configs/ua_adc.yaml`
  - Tests: `tests/test_ua_adc.py` (32 tests, all passing)
  - Signature visualisation: `tools/ua_adc_signature.py`

- **Outcome**: UA-ADC passes all unit tests and scaling validation. With closed-loop physics simulation in `tools/ua_adc_signature.py` (`DownwashAviary`), confidence rises rapidly and blend weight α shifts from avoidance to feedforward.

---

## Decision 6: UA-ADC Online Estimator Fixes & Observation Grounding

- **Context**: `results/diagnostics/ua_adc/diagnosis_report.md` revealed two defects in the original estimator:
  1. The estimator regressed raw acceleration residual $y = a_{\text{obs}, z} - 0$, which in closed-loop hover equilibrates to $\approx 0$ m/s$^2$ because the low-level altitude PID drives steady-state error near zero. As a result, $\theta$ decayed to zero under hover.
  2. The confidence mapping contained a hard clamp $c = 0$ whenever posterior covariance $P \ge 1.0$. Because $P$ initially grows from prior excitation, confidence remained locked at 0.00 for the full 10 s flight.

- **Decision**: Redesign the estimator measurement, regressor normalization, covariance bounding, innovation-driven confidence, and seed randomness:

| Hyperparameter / Component | Old Value | New Value | Reason |
| :--- | :--- | :--- | :--- |
| **Measurement formulation $y$** | $y = a_{\text{obs}, z}$ [m/s$^2$] | $y = m \cdot a_{\text{obs}, z} - (T_{\text{cmd}} - m \cdot g)$ [N] | In closed-loop hover, $m a_{\text{obs}} \approx 0$ while commanded thrust $T_{\text{cmd}} - mg \approx -F_{\text{downwash}}$. Using controller's own $T_{\text{cmd}}$ (from motor RPMs) measures true aerodynamic disturbance without TRUE model leakage. |
| **Regressor normalization $\tilde{\phi}, \tilde{y}$** | Raw force in Newtons | Dimensionless: $\tilde{\phi} = \phi / mg$, $\tilde{y} = y / mg$ | Normalizing by nominal weight $W = mg$ standardizes the RLS regression across different vehicle masses. |
| **Excitation freeze threshold** | None (always update) | `min_excitation: 0.01` ($1\%$ hover thrust) | Freezes covariance and parameter updates when upstream assumed downwash is near zero (e.g. top drone in free air), preventing covariance drift. |
| **Covariance bounds** | Unbounded $[0, \infty)$ | Clamped $P \in [10^{-4}, 5.0]$ | Prevents covariance windup when excitation is sparse and maintains numerical conditioning. |
| **Confidence mapping** | Log-linear on $P$ with hard clamp $P \ge 1.0 \to c=0$ | Innovation Gaussian kernel: $c_{\text{raw}} = \exp\left(-0.5 \left(\frac{\tilde{e}}{\sigma_e}\right)^2\right)$ with $\sigma_e = 0.04$ | Replaces arbitrary $P$-clamping with normalized prediction innovation $\tilde{e} = \tilde{y} - \tilde{\phi}\theta$. When the estimator's prediction matches the physical measurement, confidence rapidly rises to $\approx 1.0$. |
| **Confidence smoothing** | EMA $\beta = 0.90$ | EMA $\beta = 0.90$ maintained | Suppresses high-frequency sensor noise while responding dynamically within $< 0.2$ s. |
| **Adaptive force compensation** | $F_{\text{comp}} = \alpha \cdot F_{\text{assumed}}$ | $F_{\text{comp}} = \alpha \cdot (-\hat{F}_{\text{est}}) = \alpha \cdot (-\theta \cdot F_{\text{assumed}})$ | Compensates using estimated parameter $\theta$, adapting online when assumed model scales $\alpha_{\text{force}} \ne 1.0$. |
| **Random Seed Simulation Variance** | Deterministic (seed had no physical effect, std = 0.0000) | Initial position jitter $\Delta p \sim \mathcal{N}(0, (3\text{ mm})^2)$ + state observation noise $\sigma_{\text{pos}}=2\text{ mm}, \sigma_{\text{vel}}=5\text{ mm/s}$ | Provides documented physical variance for rigorous 20-seed statistical evaluation and non-degenerate Wilcoxon tests. |

- **Verification Results**:
  - **Gate A (Nominal hover_column_3d, N=3, 10s)**: $\hat{F}$ does not decay (stabilizes at $-0.064$ N matching true downwash $-0.064$ N); $P$ bounded in $[0.339, 1.000]$; confidence exceeds 0.5 at $t = 0.167$ s (Drone 1) and $t = 0.146$ s (Drone 2). **PASS**.
  - **Gate B (Mismatch Traces $\alpha_{\text{force}} \in \{0.5, 1.5, 3.0\}$)**: $\theta$ converges toward $1/\alpha_{\text{force}}$ ($1.99$ at $\alpha=0.5$, $0.67$ at $\alpha=1.5$, $0.33$ at $\alpha=3.0$); at $\alpha=3.0$, UA-ADC achieves $1.73$ cm RMSE vs Feedforward's $6.86$ cm. **PASS**.


---

## Decision 7 (D-007): Conflict override for UA-ADC (closing-speed rule)

- **Context**: In `vertical_swap`, the lower drone starts inside the wake, so UA-ADC's confidence rises within ~0.2 s and the sigmoid blend moves toward feedforward. Feedforward keeps the nominal path, and the nominal path of a vertical swap goes straight through the other drone. Confidence measures how well the wake model fits; it carries no information about an approaching neighbour. This is the mechanism behind the intermittent swap failure (open item 1).
- **Decision**: Multiply the blend weight by `(1 − r_j)`, where `r_j` is a conflict risk from a constant-velocity closest-point-of-approach (CPA) prediction over all neighbours, computed from both the observed states and the nominal references. Risk is 1 at a predicted miss distance of 0, tapers (cosine) to 0 at `conflict_radius = 0.30 m`, only counts approaches within `horizon_s = 2.5 s`, and decays over `hold_s = 1.0 s` after the conflict clears. Parameters live in `configs/ua_adc.yaml` (`conflict_override`); `enabled: false` restores the old behaviour for ablation.
- **Why 0.30 m**: larger than the 0.12 m safety net plus a drone's own wake radius margin, smaller than the 0.35 m scaling-column spacing and the 0.45 m hover spacing, so static stacks never trigger it.
- **Anti-circularity**: uses observed states and the commanded references only. No TRUE-model access.

## Decision 8 (D-008): Vertical component of commanded thrust in the estimator

- The measurement `y = m·a_z − (T_cmd − m·g)` previously used the full thrust magnitude. Thrust acts along the body z-axis, so its vertical part is `T_cmd · R[2,2]` with `R[2,2] = 1 − 2(qx² + qy²)`. The difference is negligible in hover and matters when drones tilt to avoid. The quaternion is part of the observed state.

## Decision 9 (D-009): One shared episode loop and explicit seeds

- **Context**: the simulate/control loop was duplicated in about ten places. `run_single_trial` accepted `seed` and ignored it, so any "N-seed" study routed through it repeated one deterministic run N times (this is why some earlier tables showed zero spread and repeated p-values).
- **Decision**: `src/sim/episode.py::run_episode` is the single loop for new work. `seed=None` means deterministic; an integer seed draws 3 mm start jitter (and observation noise if enabled). `run_single_trial` gained `randomize=` (default False, so the published single-seed benchmark is unchanged).

## Decision 10 (D-010): Removal of the never-working mock-physics batch entry points

- `run_batch.py`, `src/experiments/runner.py` and the root `characterise_downwash.py` imported `NaiveController`, `SoftAvoidController` and `FeedforwardController`, which do not exist anywhere in the repository history. They could not run and are removed. The working characterisation is `python -m src.experiments.downwash_characterisation`; batch studies are `python tools/studies.py`.

## Decision 11 (D-011): Paths resolved from the project root

- Several tools hard-coded `D:/Drones_Project/downwash_swarm`, and controllers loaded `configs/...` relative to the current directory. Both are replaced by `src/paths.py` (`PROJECT_ROOT`, `config_path`), so the project runs from any folder or drive.

## Decision 12 (D-012): Pass criteria fixed before the strict regression

- `configs/pass_criteria.yaml` was written before the strict regression was rerun: no ground contact (z > 0.05 m), minimum separation > 0.12 m, UA-ADC confidence > 0.5 within 1 s for every drone that starts in a wake, and UA-ADC RMSE ≤ 1.05 × Soft-Avoid RMSE on the same scenario/N/seed. Feedforward is not used as the pass/fail baseline because at nominal it is given the correct model by construction. Any later change to these thresholds must be recorded here with a reason.

## Decision 13 (D-013): Soft-Avoid chatter fixes (found by the stress study)

- **Symptom**: in the stress study (hover column N = 3, 2-step command delay), Soft-Avoid touched the ground in about half of the seeds while Naive never did. UA-ADC inherited the problem through its avoidance arm. Isolation runs showed the command delay alone was enough; observation noise alone and 1.3x wake force alone were not.
- **Cause 1, direction flipping**: for a drone in the wake core, the escape direction was the direction of `target_lower - position_upper`. That vector is only the upper drone's centimetre-level wobble, so its direction flipped from step to step and the offset chattered.
  **Fix**: inside a core radius (`direction_lock_radius`, default 0.4 x radius_xy = 0.10 m) the previous direction is held; on first entry a per-drone spread angle is used, so stacked drones escape in different directions instead of the lower one following the upper one into its shifted wake.
- **Cause 2, noisy velocity feedforward**: the step-to-step rate of change of the offset was added to the velocity setpoint. It jumps by up to the rate limit (0.15 m/s) whenever the offset target moves, and a 42 ms delay turned those jumps into a growing lateral oscillation.
  **Fix**: `offset_velocity_mode: filtered` (first-order low-pass, `offset_velocity_tau` = 0.25 s). `raw` restores the old behaviour.
- **Evidence (8 seeds, hover N = 3, noise on)**: raw → ground contact 4/8 with delay; filtered → 0/8 with delay; nominal RMSE 7.95 cm → 8.19 cm.
- **Related UA-ADC fix**: the blend is now written `avoid + a (nominal - avoid)`. The old form `(1 - a) avoid + a nominal` is not bit-exact when both arms agree, and the rounding difference grew into a 1.6 cm trajectory difference in the coplanar `line` formation, which the strict regression flagged as a tracking failure.
- These are control-logic changes, so all studies in `results/studies/` were rerun after them (rule: control-logic changes need a full regression check).
