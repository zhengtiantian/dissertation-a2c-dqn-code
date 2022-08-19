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
import gym
import numpy as np
from stable_baselines3 import A2C, DQN, DDPG, TD3, PPO, SAC
from stable_baselines3.a2c import MlpPolicy
from stable_baselines3.common.env_checker import check_env

from gym_pybullet_drones.envs.single_agent_rl import BaseSingleAgentAviary, FlyThruGateAviary
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.single_agent_rl.TakeoffAviary import TakeoffAviary
from gym_pybullet_drones.utils.utils import sync, str2bool

if __name__ == "__main__":

    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(
        description='Single agent reinforcement learning example script using TakeoffAviary')
    parser.add_argument('--rllib', default=False, type=str2bool,
                        help='Whether to use RLlib PPO in place of stable-baselines A2C (default: False)', metavar='')
    ARGS = parser.parse_args()

    #### Check the environment's spaces ########################
    env = FlyThruGateAviary(gui=True,record=False)
    print("[INFO] Action space:", env.action_space)
    print("[INFO] Observation space:", env.observation_space)
    check_env(env, warn=True,  skip_render_check=True )

    #### Train the models_A2C #######################################

    model = A2C(MlpPolicy,
                env,
                verbose=1
                )
    model.learn(total_timesteps=3000000)  # Typically not enough


    #### Show (and record a video of) the models_A2C's performance ##

    logger = Logger(logging_freq_hz=int(env.SIM_FREQ / env.AGGR_PHY_STEPS),
                    num_drones=1
                    )
    obs = env.reset()
    start = time.time()
    for i in range(3 * env.SIM_FREQ):
        if not ARGS.rllib:
            action, _states = model.predict(obs,
                                            deterministic=True
                                            )


        print("action:" + str(action))
        obs, reward, done, info = env.step(action)
        logger.log(drone=0,
                   timestamp=i / env.SIM_FREQ,
                   state=np.hstack([obs[0:3], np.zeros(4), obs[3:15], np.resize(action, (4))]),
                   control=np.zeros(12)
                   )
        if i % env.SIM_FREQ == 0:
            env.render()
            print(done)
        sync(i, start, env.TIMESTEP)
        if done:
            obs = env.reset()
    env.close()
    logger.plot()
