"""Script demonstrating the use of `gym_pybullet_drones`' Gym interface.

Class TakeoffAviary is used as a learning env for the A2C and PPO algorithms.

Example
-------
In a terminal, run as:

    $ python learn.py

Notes
-----
The boolean argument --rllib switches between `stable-baselines3` and `ray[rllib]`.
This is a minimal working example integrating `gym-pybullet-drones` with 
reinforcement learning libraries `stable-baselines3` and `ray[rllib]`.
It is not meant as a good/effective learning example.

"""
import time
import argparse
import numpy as np
from stable_baselines3 import A2C, DQN, DDPG, TD3, PPO, SAC
from stable_baselines3.common.env_checker import check_env

from gym_pybullet_drones.envs.single_agent_rl.FlyThruGateAviary_new_dqn import FlyThruGateAviary_new_dqn
from gym_pybullet_drones.utils.Logger import Logger

if __name__ == "__main__":

    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(
        description='Single agent reinforcement learning example script using TakeoffAviary')
    parser.add_argument('--initial_x', default=0, type=int, )
    parser.add_argument('--total_timesteps', default=100000, type=int, )
    args = parser.parse_args()

    initial_xyzs = np.array([args.initial_x, 0, 10]).reshape(1, 3)
    env = FlyThruGateAviary_new_dqn(gui=False, record=False, freq=60, initial_xyzs=initial_xyzs)

    # env = FlyThruGateAviary_new_dqn(gui=True,record=False, freq=240)
    print("[INFO] Action space:", env.action_space)
    print("[INFO] Observation space:", env.observation_space)
    check_env(env, warn=True,  skip_render_check=True )



    #### Show (and record a video of) the models's performance ##
    ds = "test2B13SNR_VEL_2D"
    # load_path = "models_dqn/model2022811_" + ds + ".ckpt"
    # load_path = "models_dqn/model2022818_" + ds + ".ckpt"
    # load_path = "models_dqn/model2022819_" + ds + ".ckpt"

    load_path = "models_dqn/model2022819_" + ds + "(" + str(args.initial_x) + ")" + ".ckpt"

    model = DQN.load(load_path,gamma=0.9)

    print(model.policy)

    logger = Logger(logging_freq_hz=int(env.SIM_FREQ / env.AGGR_PHY_STEPS),
                    num_drones=1
                    )
    obs = env.reset()
    start = time.time()



    for _ in range(1):
        for i in range(env.EPISODE_LEN_SEC * env.SIM_FREQ):
            action, _states = model.predict(obs, deterministic=True)

            # print("action:" + str(action))
            obs, reward, done, info = env.step(action)
            logger.new_log(drone=0,
                       timestamp=i / env.SIM_FREQ,
                       # state=np.hstack([obs[0:3], np.zeros(4), obs[3:15], np.resize(action, (4))]),
                       state=np.hstack([obs[0:2]]),
                       control=np.zeros(12)
                       )
            if i % env.SIM_FREQ == 0:
                env.render()
                print(done)
            # sync(i, start, env.TIMESTEP)
            if done:
                obs = env.reset()
                break

    # logger.visualize(env)
    logger.new_visualize(env)
    env.close()
    logger.plot()
