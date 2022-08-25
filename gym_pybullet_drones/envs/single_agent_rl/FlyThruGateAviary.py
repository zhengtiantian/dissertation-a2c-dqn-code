import numpy as np
import pybullet as p
import pkg_resources
import torch.autograd

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType, \
    BaseSingleAgentAviary
from gym_pybullet_drones.utils.utils import handle_best_SINR_weight, action2dir, get_approx_dir, get_next_pos
class FlyThruGateAviary(BaseSingleAgentAviary):
    """Single agent RL problem: fly through a gate."""

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics = Physics.PYB,
                 freq: int = 240,
                 aggregate_phy_steps: int = 1,
                 gui=False,
                 record=False,
                 obs: ObservationType = ObservationType.KIN,
                 act: ActionType = ActionType.VEL,
                 target_xyz=None,
                 p_distance=None,
                 buildings=None,
                 building_ids=None,
                 base_stations=None,
                 use_dqn_like_reward=True,
                 **kwargs):
        """Initialization of a single agent RL environment.

        Using the generic single agent RL superclass.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        freq : int, optional
            The frequency (Hz) at which the physics engine steps.
        aggregate_phy_steps : int, optional
            The number of physics steps within one call to `BaseAviary.step()`.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation in folder `files/videos/`.
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision)
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, or waypoint with PID control)

        """

        super().__init__(drone_model=drone_model,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         freq=freq,
                         aggregate_phy_steps=aggregate_phy_steps,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act,
                         target_xyz=target_xyz,
                         p_distance=p_distance,
                         buildings=buildings,
                         building_ids=building_ids,
                         base_stations=base_stations,
                         **kwargs)
        self.ideal_vel = 1
        self.MAX_ERROR_WITH_TARGET = 8
        self.USE_DQN_LIKE_REWARD = use_dqn_like_reward
        self.detect_span = 2
    ################################################################################

    def _addObstacles(self):
        """Add obstacles to the environment.

        Extends the superclass method and add the gate build of cubes and an architrave.

        """
        buildings, buildings_ids, baseStations = [], [], []

        id = p.loadURDF(pkg_resources.resource_filename('gym_pybullet_drones', 'objects/myCube2.urdf'),
                        [5, 5, 5.5],
                        p.getQuaternionFromEuler([0, 0, 0]),
                        physicsClientId=self.CLIENT
                        )
        print("Obstacle id:" + str(id))
        buildings.append([[4, 4, 0], [6, 6, 11]])
        buildings_ids.append(id)

        id = p.loadURDF(pkg_resources.resource_filename('gym_pybullet_drones', 'objects/myCube2.urdf'),
                        [10, 10, 5.5],
                        p.getQuaternionFromEuler([0, 0, 0]),
                        physicsClientId=self.CLIENT
                        )
        print("Obstacle id:" + str(id))
        buildings.append([[9, 9, 0], [11, 11, 11]])
        buildings_ids.append(id)

        id = p.loadURDF(pkg_resources.resource_filename('gym_pybullet_drones', 'objects/myCube.urdf'),
                        [15, 35, 4],
                        p.getQuaternionFromEuler([0, 0, 0]),
                        physicsClientId=self.CLIENT
                        )
        print("Obstacle id:" + str(id))
        buildings.append([[14, 34, 0], [16, 36, 10]])
        buildings_ids.append(id)

        id = p.loadURDF(pkg_resources.resource_filename('gym_pybullet_drones', 'objects/baseStation.urdf'),
                        [1, 45, 1],
                        p.getQuaternionFromEuler([0, 0, 0]),
                        physicsClientId=self.CLIENT
                        )
        baseStations.append([1, 45, 2])
        buildings_ids.append(id)

        id = p.loadURDF(pkg_resources.resource_filename('gym_pybullet_drones', 'objects/baseStation.urdf'),
                        [45, 1, 1],
                        p.getQuaternionFromEuler([0, 0, 0]),
                        physicsClientId=self.CLIENT
                        )
        baseStations.append([45, 1, 2])
        buildings_ids.append(id)
        # #
        id = p.loadURDF(pkg_resources.resource_filename('gym_pybullet_drones', 'objects/baseStation.urdf'),
                        [48, 36, 1],
                        p.getQuaternionFromEuler([0, 0, 0]),
                        physicsClientId=self.CLIENT
                        )
        baseStations.append([48, 36, 2])
        buildings_ids.append(id)

        return buildings, buildings_ids, baseStations

    ################################################################################

    def _getGradOfRewardbyUAVxyz(self, UAVxyzs) -> np.ndarray:
        last_uav_pos = UAVxyzs

        distance = self._calculateDistance(UAVxyzs, self.TARGET_XYZ)

        last_TARGET_dir = self.TARGET_XYZ - last_uav_pos
        last_TARGET_dir = last_TARGET_dir / np.linalg.norm(last_TARGET_dir)

        grad = self._getGradOfBestSINRbyUAVxyz(last_uav_pos)
        norm = np.linalg.norm(grad, 2)
        if norm > np.finfo(np.float32).eps:
            last_best_SINR_dir = grad / norm
        else:
            last_best_SINR_dir = np.zeros_like(grad)

        is_dir_opposite = np.dot(last_best_SINR_dir, last_TARGET_dir) < 0

        best_SINR_weight = min(0.85, 1.2 * distance / self.P_DISTANCE)
        # best_SINR_weight = handle_best_SINR_weight(best_SINR_weight, is_dir_opposite, max_best_SINR_weight=0.4)
        # best_SINR_weight = handle_best_SINR_weight(best_SINR_weight, is_dir_opposite, max_best_SINR_weight=0.8)
        last_final_dir = best_SINR_weight * last_best_SINR_dir + (1 - best_SINR_weight) * last_TARGET_dir
        return last_final_dir

    def _computeDQNLikeReward(self):
        cur_uav_pos = self.pos[0]
        last_uav_pos = self.last_pos[0]

        distance = self._calculateDistance(last_uav_pos, self.TARGET_XYZ)

        last_final_dir = self._getGradOfRewardbyUAVxyz(last_uav_pos)

        last_dir = action2dir(self.new_last_action)
        # print(f"last_dir = {last_dir}")
        _, rewards = get_approx_dir([last_dir], last_final_dir)
        reward = rewards[0]

        reward = -1 if reward < 0 else reward

        last_next_uav_poses = get_next_pos([last_dir], last_uav_pos, self.detect_span, v=1)
        # collisionInfos = self._detectCollisionByMotion(last_uav_pos, last_next_uav_poses)
        collisionInfos = self._new_detectCollisionByMotion(last_uav_pos, last_next_uav_poses)
        collisionInfo = collisionInfos[0]

        if self._detectCollision(1) or collisionInfo:
            # reward = -1
            reward = -10

        if distance <= self.MAX_ERROR_WITH_TARGET:
            reward = 10000

        return reward
    def _computeReward(self):
        """Computes the current reward value.

        Returns
        -------
        float
            The reward.

        """
        # cur_dis =
        # print("c:" + str(self.pos[0]))
        # print("c:"+str(cur_dis))
        # print("p:" + str(self.P_DISTANCE))
        # reward = 0

        # if cur_dis < self.P_DISTANCE:
        #     reward = 5
        #     self.P_DISTANCE = cur_dis
        # else:
        #     reward = -1
        if self.USE_DQN_LIKE_REWARD:
            return self._computeDQNLikeReward()

        txy = 0
        tz = 0.025
        reward = 0
        cur_uav_pos = self.pos[0]
        last_uav_pos = self.last_pos[0]
        # print(cur_uav_pos)
        distance = self._calculateDistance(cur_uav_pos, self.TARGET_XYZ)
        last_distance = self._calculateDistance(last_uav_pos, self.TARGET_XYZ)

        # reward += - distance / self.P_DISTANCE / self.SIM_FREQ * 0.1
        bestSINR = self._getBestSINRbyUAVxyz(cur_uav_pos)
        last_bestSINR = self._getBestSINRbyUAVxyz(last_uav_pos)
        # print(bestSINR)

        last_dir = action2dir(self.new_last_action)
        # print(f"last_dir = {last_dir}")

        last_next_uav_poses = get_next_pos([last_dir], last_uav_pos, self.detect_span, v=1)
        # collisionInfos = self._detectCollisionByMotion(last_uav_pos, last_next_uav_poses)
        collisionInfos = self._new_detectCollisionByMotion(last_uav_pos, last_next_uav_poses)
        collisionInfo = collisionInfos[0]

        if collisionInfo:
            reward += -1 / self.SIM_FREQ
        elif self._detectCollision(1):
            reward += -3 / self.SIM_FREQ
            # print("collision reward:" + str(reward) + str(cur_uav_pos))
            # return reward

        if cur_uav_pos[2] < tz:
            # print("out of bundary reward:" + str(reward) + str(cur_uav_pos))
            reward += -2 / self.SIM_FREQ
            # print("out of bundary reward:" + str(reward) + str(cur_uav_pos))
            # return reward
        # elif cur_uav_pos[0] > self.TARGET_XYZ[0] or cur_uav_pos[1] > self.TARGET_XYZ[1] or cur_uav_pos[2] > \
        #         self.TARGET_XYZ[2]:
        #     print("reward:" + str(reward)+ str(cur_uav_pos))
        #     reward = -1
        #     return reward

        # cur_x = self.pos[0][0]
        # cur_y = self.pos[0][1]
        # cur_z = self.pos[0][2]
        #
        # reward += cur_x / self.TARGET_XYZ[0]
        # reward += cur_y / self.TARGET_XYZ[1]
        # reward += cur_z / self.TARGET_XYZ[2]

        if distance <= self.MAX_ERROR_WITH_TARGET:
            reward += 1000
        # elif distance > self.P_DISTANCE * 0.3 \
        #         and self.step_counter + 1 >= self.EPISODE_LEN_SEC * self.SIM_FREQ:
        #     pass
            # reward -= 20
        # elif distance < self.P_DISTANCE * 0.2:
        #     reward += 4.5
        # elif distance < self.P_DISTANCE * 0.3:
        #     reward += 4
        # elif distance < self.P_DISTANCE * 0.4:
        #     reward += 3.5
        # elif distance < self.P_DISTANCE * 0.5:
        #     reward += 3
        # elif distance < self.P_DISTANCE * 0.6:
        #     reward += 2.5
        # elif distance < self.P_DISTANCE * 0.7:
        #     reward += 2
        # elif distance < self.P_DISTANCE * 0.8:
        #     reward += 1.5
        # elif distance < self.P_DISTANCE * 0.95:
        #     reward += 1

        # reward += (bestSINR / 100) ** 3 / self.SIM_FREQ
        if bestSINR >= last_bestSINR:
            reward += (bestSINR - last_bestSINR)
        elif bestSINR >= 24.5:
            reward += 0
        elif distance >= self.P_DISTANCE * 0.3:
            reward += (bestSINR - last_bestSINR) * 10

        # if distance <= last_distance:
        if last_distance <= distance:
            reward += (last_distance - distance) * 2
        else:
            reward += (last_distance - distance)
        # print(p.calculateMassMatrix(2,[0,0,0]))
        # if p.performCollisionDetection() is not None:
        #     print(p.performCollisionDetection())
        if self.step_counter % (10 * self.SIM_FREQ) == 0:
            print(f"r: {reward:.6f}", end=", \t")
        return reward
        # state = self._getDroneStateVector(0)
        # norm_ep_time = (self.step_counter/self.SIM_FREQ) / self.EPISODE_LEN_SEC
        # return -10 * np.linalg.norm(np.array([0, -2*norm_ep_time, 0.75])-state[0:3])**2

    ################################################################################

    def _computeDone(self):
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """
        txy = 0
        tz = 0.025

        cur_pos = self.pos[0]
        cur_dis = self._calculateDistance(self.pos[0], self.TARGET_XYZ)
        # cur_time = self.step_counter / self.SIM_FREQ
        cur_time = self.step_counter * self.TIMESTEP
        cur_vel = np.linalg.norm(self.vel[0])
        length = cur_vel * cur_time
        if self.step_counter / self.SIM_FREQ > self.EPISODE_LEN_SEC or cur_dis < self.MAX_ERROR_WITH_TARGET \
                or cur_pos[2] < 5:  # or self._detectCollision(1):
            # self.P_DISTANCE = self._calculateDistance(self.INIT_XYZS[0], self.TARGET_XYZ)
            print("====== end")
            print('====== final_pos: %s, final_dis: %.2f, final_vel: %.2f, time: %.2f, path length: %.2f' %
                  (str(cur_pos.round(2)), cur_dis, cur_vel, cur_time, length))

        # print("p:" + str(self.P_DISTANCE))
        cur_uav_pos = self.pos[0]
        cur_dis = self._calculateDistance(self.pos[0], self.TARGET_XYZ)
        if self.step_counter % (10 * self.SIM_FREQ) == 0:
            print(f"a: {self.total_reward}, \tv: {self.vel[0]}, \tp: {self.pos[0]}, "
                  f"\td: {cur_dis}, \ts: {self._getBestSINRbyUAVxyz(cur_uav_pos)} "
                  + ("\t[collision] " if self._detectCollision(1) else "")
                  + ("\t[boundary] " if cur_uav_pos[0] < txy or cur_uav_pos[1] < txy or cur_uav_pos[2] < tz else ""))

        if self.step_counter + 1 >= self.EPISODE_LEN_SEC * self.SIM_FREQ:
            # self.P_DISTANCE = self._calculateDistance(self.INIT_XYZS[0], self.TARGET_XYZ)
            print("end")
            return True
        elif cur_dis <= self.MAX_ERROR_WITH_TARGET:
            print("end ----------------------------------------------------------------------------------- arrived")
            return True

        else:
            return False

    ################################################################################

    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        return {"answer": 42}  #### Calculated by the Deep Thought supercomputer in 7.5M years

    ################################################################################

    def _clipAndNormalizeState(self,
                               state
                               ):
        """Normalizes a drone's state to the [-1,1] range.

        Parameters
        ----------
        state : ndarray
            (20,)-shaped array of floats containing the non-normalized state of a single drone.

        Returns
        -------
        ndarray
            (20,)-shaped array of floats containing the normalized state of a single drone.

        """
        MAX_LIN_VEL_XY = 3
        MAX_LIN_VEL_Z = 1

        # MAX_XY = MAX_LIN_VEL_XY * self.EPISODE_LEN_SEC
        MAX_XY = self.TARGET_XYZ[:-1].max()
        MAX_Z = MAX_LIN_VEL_Z * self.EPISODE_LEN_SEC

        MAX_DIS = self.P_DISTANCE
        MAX_SINR = 30

        MAX_PITCH_ROLL = np.pi  # Full range

        clipped_pos_xy = np.clip(state[0:2], -MAX_XY, MAX_XY)
        clipped_pos_z = np.clip(state[2], 0, MAX_Z)
        clipped_rp = np.clip(state[7:9], -MAX_PITCH_ROLL, MAX_PITCH_ROLL)
        clipped_vel_xy = np.clip(state[10:12], -MAX_LIN_VEL_XY, MAX_LIN_VEL_XY)
        clipped_vel_z = np.clip(state[12], -MAX_LIN_VEL_Z, MAX_LIN_VEL_Z)

        # add clipped distance
        clipped_dis = np.clip(state[20], -MAX_DIS, MAX_DIS)
        clipped_sinr = np.clip(state[21], -MAX_SINR, MAX_SINR)

        if self.GUI:
            self._clipAndNormalizeStateWarning(state,
                                               clipped_pos_xy,
                                               clipped_pos_z,
                                               clipped_rp,
                                               clipped_vel_xy,
                                               clipped_vel_z
                                               )

        normalized_pos_xy = clipped_pos_xy / MAX_XY
        normalized_pos_z = clipped_pos_z / MAX_Z
        normalized_rp = clipped_rp / MAX_PITCH_ROLL
        normalized_y = state[9] / np.pi  # No reason to clip
        normalized_vel_xy = clipped_vel_xy / MAX_LIN_VEL_XY
        normalized_vel_z = clipped_vel_z / MAX_LIN_VEL_XY
        normalized_ang_vel = state[13:16] / np.linalg.norm(state[13:16]) if np.linalg.norm(
            state[13:16]) != 0 else state[13:16]

        # add normalized distance
        normalized_dis = clipped_dis / MAX_DIS
        normalized_sinr = clipped_sinr / MAX_SINR

        norm_and_clipped = np.hstack([normalized_pos_xy,
                                      normalized_pos_z,
                                      state[3:7],
                                      normalized_rp,
                                      normalized_y,
                                      normalized_vel_xy,
                                      normalized_vel_z,
                                      normalized_ang_vel,
                                      state[16:20],
                                      normalized_dis,
                                      normalized_sinr
                                      ]).reshape(22, )

        return norm_and_clipped

    ################################################################################

    def _clipAndNormalizeStateWarning(self,
                                      state,
                                      clipped_pos_xy,
                                      clipped_pos_z,
                                      clipped_rp,
                                      clipped_vel_xy,
                                      clipped_vel_z,
                                      ):
        """Debugging printouts associated to `_clipAndNormalizeState`.

        Print a warning if values in a state vector is out of the clipping range.
        
        """
        if not (clipped_pos_xy == np.array(state[0:2])).all():
            print("[WARNING] it", self.step_counter,
                  "in FlyThruGateAviary._clipAndNormalizeState(), clipped xy position [{:.2f} {:.2f}]".format(state[0],
                                                                                                              state[1]))
        if not (clipped_pos_z == np.array(state[2])).all():
            print("[WARNING] it", self.step_counter,
                  "in FlyThruGateAviary._clipAndNormalizeState(), clipped z position [{:.2f}]".format(state[2]))
        if not (clipped_rp == np.array(state[7:9])).all():
            print("[WARNING] it", self.step_counter,
                  "in FlyThruGateAviary._clipAndNormalizeState(), clipped roll/pitch [{:.2f} {:.2f}]".format(state[7],
                                                                                                             state[8]))
        if not (clipped_vel_xy == np.array(state[10:12])).all():
            print("[WARNING] it", self.step_counter,
                  "in FlyThruGateAviary._clipAndNormalizeState(), clipped xy velocity [{:.2f} {:.2f}]".format(state[10],
                                                                                                              state[
                                                                                                                  11]))
        if not (clipped_vel_z == np.array(state[12])).all():
            print("[WARNING] it", self.step_counter,
                  "in FlyThruGateAviary._clipAndNormalizeState(), clipped z velocity [{:.2f}]".format(state[12]))

    def _detectCollision(self, UAV_ID):
        for i in self.BUILDING_IDS:
            if p.getContactPoints(UAV_ID, i):
                return True

        return False

    def _new_detectCollisionByMotion(self, UAVxyz, nextUAVxyzs):
        collisionInfos = []
        for nextUAVxyz in nextUAVxyzs:
            collisionInfo = self._new_blockedByBuildings(UAVxyz, nextUAVxyz)
            collisionInfos.append(collisionInfo)

        collisionInfos = np.array(collisionInfos)
        return collisionInfos