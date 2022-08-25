import datetime
import time

from gym.utils.env_checker import check_env
from stable_baselines3 import DQN
from stable_baselines3.dqn import MlpPolicy

from gym_pybullet_drones.envs.single_agent_rl.FlyThruGateAviary_new_dqn import FlyThruGateAviary_new_dqn

import numpy as np
import argparse

if __name__ == "__main__":
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(
        description='Single agent reinforcement learning example script using TakeoffAviary')
    parser.add_argument('--initial_x', default=0, type=int,)
    parser.add_argument('--total_timesteps', default=100000, type=int, )
    parser.add_argument('--ideal_vel', default=2, type=int, )
    parser.add_argument('--gamma', default=0.9, type=float, )
    args = parser.parse_args()

    initial_xyzs = np.array([args.initial_x, 0, 10]).reshape(1, 3)
    # initial_xyzs = np.array([20, 0, 0]).reshape(1, 3)
    ds = "test2B13SNR_VEL_2D"
    load_path = "models_dqn/model2022811_" + ds + ".ckpt"
    env = FlyThruGateAviary_new_dqn(gui=True, record=False, freq=240, initial_xyzs=initial_xyzs, ideal_vel=args.ideal_vel)

    try:
        # model = DQN.load(load_path)
        model = DQN(MlpPolicy, env, verbose=1, learning_starts=5000, exploration_fraction=0.2, gamma=args.gamma)
    except:
        #### Check the environment's spaces ########################
        print("[INFO] Action space:", env.action_space)
        print("[INFO] Observation space:", env.observation_space)

        model = DQN(MlpPolicy, env, verbose=1, learning_starts=5000, exploration_fraction=0.2, gamma=args.gamma)

    check_env(env, warn=True, skip_render_check=True)

    print(model.policy)

    model.set_env(env)
    # model.learn(total_timesteps=6000000)  # Typically not enough
    # model.learn(total_timesteps=600000)  # Typically not enough
    # model.learn(total_timesteps=100000)  # Typically not enough
    # model.learn(total_timesteps=200000)  # Typically not enough
    model.learn(total_timesteps=args.total_timesteps)

    model.save("models_dqn/model" + str(
        datetime.datetime.now().year) + str(datetime.datetime.now().month) + str(
        datetime.datetime.now().day) + '_' + ds + "(" + str(args.initial_x) + ")" + ".ckpt")
