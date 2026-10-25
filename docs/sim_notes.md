# Simulation & Physics Notes: gym-pybullet-drones Inspection

This document summarizes the architectural and physics inspection of the `gym-pybullet-drones` package (v2.2.0) conducted for the **Downwash-Aware Formation Control** project.

---

## 1. Physics Modes (`Physics` Enum)
Defined in `gym_pybullet_drones.utils.enums.Physics`:

| Physics Mode | Identifier | Description |
| :--- | :--- | :--- |
| `Physics.PYB` | `"pyb"` | Base PyBullet rigid body dynamics with motor thrust and torque applied to individual links. |
| `Physics.DYN` | `"dyn"` | Explicit analytical dynamics integration (custom Runge-Kutta ODE bypassing PyBullet contact/solver). |
| `Physics.PYB_GND` | `"pyb_gnd"` | PyBullet physics augmented with ground effect force based on Cheeseman-Bennett / Zhou et al. |
| `Physics.PYB_DRAG` | `"pyb_drag"` | PyBullet physics augmented with quadratic body drag ($F_{drag} \propto -v^2$). |
| `Physics.PYB_DW` | `"pyb_dw"` | PyBullet physics with built-in analytical downwash model. |
| `Physics.PYB_GND_DRAG_DW` | `"pyb_gnd_drag_dw"` | Combines ground effect, aerodynamic drag, and built-in downwash. |

---

## 2. Built-in Downwash Model: Analysis & Equations
Located in `gym_pybullet_drones.envs.BaseAviary._downwash(nth_drone)` (lines 785–812):

### Governing Equations
For a lower drone $j$ and an upper drone $i$:
$$\Delta z = z_i - z_j$$
$$\Delta_{xy} = \sqrt{(x_i - x_j)^2 + (y_i - y_j)^2}$$

If $\Delta z > 0$ (drone $i$ is above drone $j$) and $\Delta_{xy} < 10\text{ m}$:
$$\alpha = DW\_COEFF\_1 \cdot \left(\frac{r_{prop}}{4 \cdot \Delta z}\right)^2$$
$$\beta = DW\_COEFF\_2 \cdot \Delta z + DW\_COEFF\_3$$
$$\mathbf{F}_{dw} = \begin{bmatrix} 0 \\ 0 \\ -\alpha \exp\left(-\frac{1}{2}\left(\frac{\Delta_{xy}}{\beta}\right)^2\right) \end{bmatrix}$$

The force is injected into PyBullet via:
```python
p.applyExternalForce(
    objectUniqueId=self.DRONE_IDS[nth_drone],
    linkIndex=4,
    forceObj=downwash,
    posObj=[0, 0, 0],
    flags=p.LINK_FRAME,
    physicsClientId=self.CLIENT
)
```

### Parameter Values (from `cf2x.urdf`)
- $DW\_COEFF\_1 = 2267.18$
- $DW\_COEFF\_2 = 0.16$
- $DW\_COEFF\_3 = -0.11$
- $r_{prop} = 0.0231348\text{ m}$
- Crazyflie 2.x mass: $m = 0.027\text{ kg}$
- Hover thrust per drone: $\approx m \cdot g \approx 0.265\text{ N}$

### Identified Limitations of Built-in Model
1. **Singularity at small $\Delta z$**: As $\Delta z \to 0$, $\alpha \propto 1 / \Delta z^2 \to \infty$, leading to explosive, unphysical force spikes if drones approach the same altitude.
2. **Zero-Crossing Singularity in $\beta$**: When $\Delta z = -DW\_COEFF\_3 / DW\_COEFF\_2 = 0.11 / 0.16 = 0.6875\text{ m}$, the wake radius parameter $\beta$ becomes zero, causing catastrophic division by zero in $\Delta_{xy}/\beta$.
3. **No Horizontal Skew**: The wake is assumed strictly vertical and symmetric regardless of the upper drone's forward velocity.
4. **Conclusion for our Project**: We will utilize the **Hönig et al. (2018)** and **Neural-Swarm (Shi et al., 2020)** ellipsoidal downwash models as our standalone **TRUE** and **ASSUMED** models, ensuring smooth, bounded wake geometry and velocity-skewed wake dynamics without mathematical singularities.

---

## 3. Controlling $N$ Drones
In `gym-pybullet-drones`, multi-drone control operates via `CtrlAviary`:
- **Initialization**:
  ```python
  env = CtrlAviary(
      drone_model=DroneModel.CF2X,
      num_drones=N,
      initial_xyzs=initial_positions,  # shape (N, 3)
      initial_rpys=initial_orientations,# shape (N, 3)
      physics=Physics.PYB,             # Base PyBullet physics (clean disturbance injection)
      pyb_freq=240,
      ctrl_freq=48,
      gui=False
  )
  ```
- **Controller Allocation**:
  An individual `DSLPIDControl(drone_model=DroneModel.CF2X)` instance is maintained for each drone $j \in \{0, \dots, N-1\}$.
- **Control Stepping**:
  Each control cycle, `computeControlFromState` calculates the required 4 motor RPMs from the 20-element drone state and reference setpoints:
  ```python
  action = np.zeros((N, 4))
  for j in range(N):
      action[j, :], _, _ = ctrl[j].computeControlFromState(
          control_timestep=env.CTRL_TIMESTEP,
          state=obs[j],
          target_pos=target_pos[j],
          target_rpy=target_rpy[j],
          target_vel=target_vel[j]
      )
  obs, reward, terminated, truncated, info = env.step(action)
  ```

---

## 4. State & Observation Representation
Each drone's kinematic state vector (length 20) returned by `_getDroneStateVector(j)`:
- `pos` (`state[0:3]`): Cartesian position $[x, y, z]$ in world frame (meters).
- `quat` (`state[3:7]`): Orientation quaternion $[q_x, q_y, q_z, q_w]$.
- `rpy` (`state[7:10]`): Euler angles $[\phi, \theta, \psi]$ (roll, pitch, yaw) in radians.
- `vel` (`state[10:13]`): Linear velocity $[v_x, v_y, v_z]$ in world frame (m/s).
- `ang_vel` (`state[13:16]`): Angular velocity $[\omega_x, \omega_y, \omega_z]$ in body frame (rad/s).
- `last_clipped_action` (`state[16:20]`): Last executed motor RPMs.

With `obs_type=ObservationType.KIN`, `obs` is a NumPy array of shape `(N, 20)`.

---

## 5. Frequencies & Timing
- **Simulation Frequency (`pyb_freq`)**: Default $240\text{ Hz}$ ($\Delta t_{sim} \approx 4.167\text{ ms}$).
- **Control Frequency (`ctrl_freq`)**: Default $48\text{ Hz}$ ($\Delta t_{ctrl} \approx 20.833\text{ ms}$).
- **Aggregate Ratio (`PYB_STEPS_PER_CTRL`)**:
  $$\text{Steps per control} = \frac{f_{sim}}{f_{ctrl}} = \frac{240}{48} = 5$$
  During each `env.step(action)`, PyBullet performs 5 internal physics steps with the motor inputs held constant.

---

## 6. How to Apply External Disturbance Forces
PyBullet provides `p.applyExternalForce`:
```python
p.applyExternalForce(
    objectUniqueId=env.DRONE_IDS[drone_idx],
    linkIndex=-1,                     # Base link / Center of mass
    forceObj=[fx, fy, fz],            # 3D disturbance force vector
    posObj=[0.0, 0.0, 0.0],           # Point of application relative to link COM
    flags=p.WORLD_FRAME,              # Apply directly in Cartesian world coordinates
    physicsClientId=env.CLIENT
)
```
For our TRUE downwash module, we can inject disturbance forces directly during or before physics steps, completely decoupled from the drone controllers.

---

## 7. Multi-Source Overlapping Downwash Handling & Combination Modes

When scaling swarms to $N \in [2, 10]$ drones, lower drones frequently operate within the overlapping wake influence zones of two or more upper quadrotors simultaneously (e.g. in multi-drone vertical columns or dense formations).

### Multi-Disturber Superposition Framework
For each lower drone $j$, `DownwashModel.compute_swarm_forces()` iterates over all upper drones $i$ where $z_i > z_j$. The individual pair disturbance vectors $\mathbf{f}_{ij}$ are collected and combined using the configurable `combination_mode`:

1. **`linear_sum` / `additive` (Default Superposition)**:
   $$\mathbf{F}_j = \sum_{i: z_i > z_j} \mathbf{f}_{ij}$$
2. **`max` / `peak` (Dominant Wake)**:
   $$\mathbf{F}_j = \arg\max_{\mathbf{f}_{ij}} |f_{z, ij}|$$
3. **`sublinear` (Diminishing Returns Power-Law)**:
   $$F_{z, j} = -\left( \sum_{i: z_i > z_j} |f_{z, ij}|^{\gamma} \right)^{1/\gamma}, \quad \gamma \in (0, 1]$$
4. **`saturated` (Capped Multi-Rotor Disturbance)**:
   Direct summation clamped by a safety ceiling: $|F_{z, j}| \le \text{max\_combined\_force\_z}$ (default $0.45\text{ N}$).

Both `configs/downwash_true.yaml` (simulator ground truth) and `configs/downwash_assumed.yaml` (controller model) configure these combination modes independently, maintaining strict anti-circularity while reflecting realistic aerodynamic multi-rotor interactions.

