import numpy as np
import pybullet as p
import pkg_resources

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary_new_dqn import ActionType, ObservationType, \
    BaseSingleAgentAviary_new_dqn
from gym_pybullet_drones.utils.utils import dir2action, get_next_pos, get_max_index, handle_best_SINR_weight, \
    action2dir, get_index, get_approx_dir
from tools.BoxIntersectLine import Point, LineSegment, Box


class FlyThruGateAviary_new_dqn(BaseSingleAgentAviary_new_dqn):
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
                 episode_len_sec: int = 320,
                 ideal_vel=2
                 ):
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
                         episode_len_sec=episode_len_sec
                         )

        # self.action = np.array([0, 0, 0, 1])
        self.dir_list = np.array([
            # 左前0、前1、右前2
            [1, 1, 0],
            [1, 0, 0],
            [1, -1, 0],
            # 左3、（停）、右4
            [0, 1, 0],
            # [0, 0, 0],
            [0, -1, 0],
            # 左后5、后6、右后7
            [-1, 1, 0],
            [-1, 0, 0],
            [-1, -1, 0],
        ])

        self.approx_dir_list = np.array([
            # 左前0、前1、右前2
            [1.00, 1.00, 0.01],
            [1.00, 0.01, 0.01],
            [1.00, -1.00, 0.01],
            # 左3、（停）、右4
            [0.01, 1.00, 0.01],
            # [0.01, 0.01, 0.01],
            [0.01, -1.00, 0.01],
            # 左后5、后6、右后7
            [-1.00, 1.00, 0.01],
            [-1.00, 0.01, 0.01],
            [-1.00, -1.00, 0.01],
        ])
        # self.action_list = np.array([dir2action(item) for item in self.dir_list])

        init_dis = self._calculateDistance(self.INIT_XYZS[0], self.TARGET_XYZ)
        # self.ideal_vel = round(init_dis / self.EPISODE_LEN_SEC * 1.5, 2)
        # self.ideal_vel = 1
        self.ideal_vel = ideal_vel
        self.action_list = np.array([dir2action(item, self.ideal_vel) for item in self.dir_list])

        # self.MAX_ERROR_WITH_TARGET = 0.5
        # self.MAX_ERROR_WITH_TARGET = 1
        self.MAX_ERROR_WITH_TARGET = 8
        self.init_distance = self._calculateDistance(self.INIT_XYZS[0], self.TARGET_XYZ)
        self.bestSINR_span = 1
        self.detect_span = 5

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

        txy = 0
        tz = 0

        cur_uav_pos = self.pos[0]
        last_uav_pos = self.last_pos[0]
        # print(cur_uav_pos)
        distance = self._calculateDistance(cur_uav_pos, self.TARGET_XYZ)
        # print(cur_uav_pos)

        last_TARGET_dir = self.TARGET_XYZ - last_uav_pos
        last_TARGET_dir = last_TARGET_dir / np.linalg.norm(last_TARGET_dir)
        # last_uav_SINR_poses = get_next_pos(self.dir_list, last_uav_pos, self.bestSINR_span)
        last_uav_SINR_poses = get_next_pos(self.approx_dir_list, last_uav_pos, self.bestSINR_span, v=self.ideal_vel)
        last_bestSINRs = self._getBestSINRbyUAVxyz(last_uav_SINR_poses)
        last_bestSINRs_index = get_max_index(last_bestSINRs)
        last_best_SINR_dir = self.dir_list[last_bestSINRs_index]
        last_best_SINR_dir = last_best_SINR_dir / np.linalg.norm(last_best_SINR_dir)

        is_dir_opposite = np.dot(last_best_SINR_dir, last_TARGET_dir) < 0

        best_SINR_weight = distance / self.init_distance
        # best_SINR_weight = handle_best_SINR_weight(best_SINR_weight, is_dir_opposite, max_best_SINR_weight=0.4)
        best_SINR_weight = handle_best_SINR_weight(best_SINR_weight, is_dir_opposite, max_best_SINR_weight=0.8)
        last_final_dir = best_SINR_weight * last_best_SINR_dir + (1 - best_SINR_weight) * last_TARGET_dir

        self.last_dir = action2dir(self.new_last_action)
        self.last_dir = self.last_dir.astype(np.int)
        index = get_index(self.dir_list, self.last_dir)
        _, rewards = get_approx_dir(self.dir_list, last_final_dir)
        reward = rewards[index]

        max_index = get_max_index(rewards)
        # reward = 1 if max_index == index else reward / 10
        # reward = 1 if max_index == index else reward

        # reward = -1 if reward < 0 else reward
        # reward = -5 if reward < 0 else reward
        reward = -1 if reward < 0 else reward



        # TARGET_dir = self.TARGET_XYZ - cur_uav_pos
        # BestSINR_dir = None
        # self.action = dir2action(TARGET_dir)
        # _, rewards_ = get_approx_dir(self.dir_list, TARGET_dir)
        #
        # # 不能预先知道下一时刻是否碰到障碍物！！！
        # last_next_uav_poses = get_next_pos(self.dir_list, last_uav_pos, self.TIMESTEP)
        last_next_uav_poses = get_next_pos(self.approx_dir_list, last_uav_pos, self.detect_span, v=self.ideal_vel)
        # collisionInfos = self._detectCollisionByMotion(last_uav_pos, last_next_uav_poses)
        collisionInfos = self._new_detectCollisionByMotion(last_uav_pos, last_next_uav_poses)
        collisionInfo = collisionInfos[index]




        # reward = 1 - distance / self.P_DISTANCE

        # bestSINR = self._getBestSINRbyUAVxyz(cur_uav_pos)
        # print(bestSINR)

        if self._detectCollision(1) or collisionInfo or cur_uav_pos[2] < 5:
            # reward = -1
            reward = -10
        # elif cur_uav_pos[0] < txy or cur_uav_pos[1] < txy or cur_uav_pos[2] < tz:
        #     reward = -1
        # elif cur_uav_pos[0] > self.TARGET_XYZ[0] or cur_uav_pos[1] > self.TARGET_XYZ[1] or cur_uav_pos[2] > \
        #         self.TARGET_XYZ[2]:
        #     reward = -1

        # elif distance < self.P_DISTANCE * 0.1:
        #     reward += 5
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

        # print(p.calculateMassMatrix(2,[0,0,0]))
        # if p.performCollisionDetection() is not None:
        #     print(p.performCollisionDetection())

        if reward > 0:
            positive = True
            # reward = 1 if index == max_index else -1
        else:
            positive = False

        if self.step_counter / self.SIM_FREQ > self.EPISODE_LEN_SEC:
            if distance > self.MAX_ERROR_WITH_TARGET:
                reward = -1

        # if distance < 20:
        #     reward *= 100

        # reward *= 10

        if self.step_counter % self.SIM_FREQ == 0:
            print("positive reward: %s, max reward: %s, reward: %.2f, action: %d, distance: %.2f, collisionInfo: %s, best_SINR_weight: %.1f, last_bestSINRs: %s, rewards: %s, cur_uav_pos: %s, collisionInfos: %s" \
                  % (str(positive), str(index == max_index), reward, index, distance, str(collisionInfo), best_SINR_weight, str(last_bestSINRs.round(2)), str(rewards.round(2)), str(cur_uav_pos.round(2)), str(collisionInfos)))
        return reward

    ################################################################################

    def _computeDone(self):
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """
        txy = 0
        tz = 0.015
        # print("p:" + str(self.P_DISTANCE))
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
            return True
        # elif cur_dis < self.P_DISTANCE * 0.05:
        #     print("end")
        #     return True
        # elif cur_pos[0] < txy or cur_pos[1] < txy or cur_pos[2] < tz:
        #
        #     print("end")
        #     return True

        # elif p.getContactPoints(1, 2) or p.getContactPoints(1, 3):
        #     print("end")
        #     return True
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
        # MAX_XY = 10
        MAX_XY = self.TARGET_XYZ[:-1].max()

        MAX_Z = MAX_LIN_VEL_Z * self.EPISODE_LEN_SEC

        MAX_DIS = self.P_DISTANCE

        MAX_PITCH_ROLL = np.pi  # Full range

        clipped_pos_xy = np.clip(state[0:2], -MAX_XY, MAX_XY)
        clipped_pos_z = np.clip(state[2], 0, MAX_Z)
        clipped_rp = np.clip(state[7:9], -MAX_PITCH_ROLL, MAX_PITCH_ROLL)
        clipped_vel_xy = np.clip(state[10:12], -MAX_LIN_VEL_XY, MAX_LIN_VEL_XY)
        clipped_vel_z = np.clip(state[12], -MAX_LIN_VEL_Z, MAX_LIN_VEL_Z)

        # add clipped distance
        clipped_dis = np.clip(state[20], -MAX_DIS, MAX_DIS)

        if self.GUI:
            if self.step_counter / self.SIM_FREQ > self.EPISODE_LEN_SEC:
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

        norm_and_clipped = np.hstack([normalized_pos_xy,
                                      normalized_pos_z,
                                      state[3:7],
                                      normalized_rp,
                                      normalized_y,
                                      normalized_vel_xy,
                                      normalized_vel_z,
                                      normalized_ang_vel,
                                      state[16:20],
                                      normalized_dis
                                      ]).reshape(21, )

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

    def _detectCollisionByMotion(self, UAVxyz, nextUAVxyzs):
        collisionInfos = []
        for nextUAVxyz in nextUAVxyzs:
            collisionInfo = self._blockedByBuildings(UAVxyz, nextUAVxyz)
            collisionInfos.append(collisionInfo)

        collisionInfos = np.array(collisionInfos)
        return collisionInfos

    def _new_detectCollisionByMotion(self, UAVxyz, nextUAVxyzs):
        collisionInfos = []
        for nextUAVxyz in nextUAVxyzs:
            collisionInfo = self._new_blockedByBuildings(UAVxyz, nextUAVxyz)
            collisionInfos.append(collisionInfo)

        collisionInfos = np.array(collisionInfos)
        return collisionInfos
