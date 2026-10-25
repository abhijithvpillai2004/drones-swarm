"""Verification script: Run multi-drone headless simulation with CtrlAviary and DSLPIDControl."""
import sys
import numpy as np
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

def test_multi_drone_headless():
    num_drones = 3
    init_xyzs = np.array([
        [0.0, 0.0, 0.5],
        [0.5, 0.0, 0.7],
        [-0.5, 0.0, 0.9]
    ])
    
    print(f"[INFO] Initializing CtrlAviary with {num_drones} drones headless...")
    env = CtrlAviary(
        drone_model=DroneModel.CF2X,
        num_drones=num_drones,
        initial_xyzs=init_xyzs,
        physics=Physics.PYB,
        pyb_freq=240,
        ctrl_freq=48,
        gui=False,
        obstacles=False
    )
    
    ctrl = [DSLPIDControl(drone_model=DroneModel.CF2X) for _ in range(num_drones)]
    action = np.zeros((num_drones, 4))
    
    target_pos = np.array([
        [0.0, 0.0, 1.0],
        [0.5, 0.0, 1.0],
        [-0.5, 0.0, 1.0]
    ])
    
    steps = int(2.0 * env.CTRL_FREQ)  # 2 seconds at 48 Hz = 96 steps
    print(f"[INFO] Stepping simulation for {steps} control steps ({steps * env.PYB_STEPS_PER_CTRL} physics steps)...")
    
    for step in range(steps):
        obs, reward, terminated, truncated, info = env.step(action)
        for j in range(num_drones):
            action[j, :], _, _ = ctrl[j].computeControlFromState(
                control_timestep=env.CTRL_TIMESTEP,
                state=obs[j],
                target_pos=target_pos[j]
            )
            
    final_positions = np.array([obs[j][0:3] for j in range(num_drones)])
    print("[SUCCESS] Headless simulation completed successfully!")
    for j in range(num_drones):
        pos_err = np.linalg.norm(final_positions[j] - target_pos[j])
        print(f"  Drone {j}: Target={target_pos[j]}, Final Pos={np.round(final_positions[j], 3)}, Err={pos_err:.4f}m")
        assert pos_err < 0.1, f"Drone {j} tracking error too large: {pos_err}m"
        
    env.close()

if __name__ == "__main__":
    test_multi_drone_headless()
    print("All checks passed.")
