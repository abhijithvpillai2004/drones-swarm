"""Environment wrapper that injects TRUE aerodynamic downwash disturbance forces.

Subclasses CtrlAviary to apply external disturbance forces computed by the TRUE
DownwashModel to lower drones at each physics simulation sub-step.
"""
from typing import Optional, Dict, Any
import numpy as np
import pybullet as p

from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

from src.models.downwash import DownwashModel


class DownwashAviary(CtrlAviary):
    """Multi-drone aviary with decoupled aerodynamic downwash disturbance injection."""

    def __init__(
        self,
        drone_model: DroneModel = DroneModel.CF2X,
        num_drones: int = 2,
        initial_xyzs: Optional[np.ndarray] = None,
        initial_rpys: Optional[np.ndarray] = None,
        physics: Physics = Physics.PYB,
        pyb_freq: int = 240,
        ctrl_freq: int = 48,
        gui: bool = False,
        record: bool = False,
        obstacles: bool = False,
        user_debug_gui: bool = False,
        downwash_model: Optional[DownwashModel] = None,
    ) -> None:
        """Initialize DownwashAviary.

        Args:
            downwash_model: TRUE disturbance model instance. If None, downwash is disabled.
        """
        super().__init__(
            drone_model=drone_model,
            num_drones=num_drones,
            initial_xyzs=initial_xyzs,
            initial_rpys=initial_rpys,
            physics=physics,
            pyb_freq=pyb_freq,
            ctrl_freq=ctrl_freq,
            gui=gui,
            record=record,
            obstacles=obstacles,
            user_debug_gui=user_debug_gui,
        )
        self.downwash_model: Optional[DownwashModel] = downwash_model
        # Buffer to record the last applied disturbance force on each drone [fx, fy, fz]
        self.last_applied_disturbances: np.ndarray = np.zeros((self.NUM_DRONES, 3))

    def step(self, action: np.ndarray):
        """Step simulation, applying motor commands and TRUE downwash disturbances."""
        # Pre-process / clip action as in BaseAviary
        action = np.reshape(action, (self.NUM_DRONES, 4))
        clipped_action = np.zeros((self.NUM_DRONES, 4))
        for i in range(self.NUM_DRONES):
            clipped_action[i, :] = np.clip(action[i, :], 0, self.MAX_RPM)

        # Internal simulation sub-stepping
        for _ in range(self.PYB_STEPS_PER_CTRL):
            # Update kinematics between aggregate steps
            self._updateAndStoreKinematicInformation()

            # 1. Base motor thrust & torque physics
            for i in range(self.NUM_DRONES):
                self._physics(clipped_action[i, :], i)

            # 2. Inject TRUE downwash disturbance forces
            if self.downwash_model is not None and self.NUM_DRONES > 1:
                disturb_forces = self.downwash_model.compute_swarm_forces(
                    positions=self.pos,
                    velocities=self.vel,
                )
                self.last_applied_disturbances = disturb_forces.copy()

                for j in range(self.NUM_DRONES):
                    f_disturb = disturb_forces[j]
                    if np.any(f_disturb != 0.0):
                        # NOTE: with WORLD_FRAME, posObj is a WORLD-frame point.
                        # Apply at the drone's own base/CoM position so the
                        # disturbance is a pure force (no spurious torque).
                        p.applyExternalForce(
                            objectUniqueId=self.DRONE_IDS[j],
                            linkIndex=-1,  # base link
                            forceObj=f_disturb.tolist(),
                            posObj=self.pos[j].tolist(),
                            flags=p.WORLD_FRAME,
                            physicsClientId=self.CLIENT,
                        )
            else:
                self.last_applied_disturbances = np.zeros((self.NUM_DRONES, 3))

            # Step PyBullet rigid body dynamics
            p.stepSimulation(physicsClientId=self.CLIENT)
            self.last_clipped_action = clipped_action

        # Post-step update and observations
        self._updateAndStoreKinematicInformation()
        obs = self._computeObs()
        reward = self._computeReward()
        terminated = self._computeTerminated()
        truncated = self._computeTruncated()
        info = self._computeInfo()
        self.step_counter = self.step_counter + (1 * self.PYB_STEPS_PER_CTRL)

        return obs, reward, terminated, truncated, info
