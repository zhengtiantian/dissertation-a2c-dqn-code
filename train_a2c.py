import pickle
import datetime
import time
import os

from gym.utils.env_checker import check_env
from stable_baselines3 import A2C, PPO, SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.a2c import MlpPolicy
# from stable_baselines3.ppo import MlpPolicy
# import fly_a2c
from gym_pybullet_drones.envs.single_agent_rl import FlyThruGateAviary

model_dict = {
    'A2C': A2C
}

task_dict = {
    'AT=0_RW=0': {
        'initial_xyzs': [[0.0, 0.0, 10.0]],
        'use_dqn_like_reward': True,
        'batch_size': 1
    },
    'AT=7_RW=0': {
        'initial_xyzs': [[7.0, 0.0, 10.0]],
        'use_dqn_like_reward': True,
        'batch_size': 1
    },
    'AT=20_RW=0': {
        'initial_xyzs': [[20.0, 0.0, 10.0]],
        'use_dqn_like_reward': True,
        'batch_size': 1
    },
    'Integated_RW=0': {
        'initial_xyzs': None,
        'use_dqn_like_reward': True,
        'batch_size': 3
    }
}

if __name__ == "__main__":
    MODEL_NAME = "A2C"
    TASK_NAME = "AT=0_RW=0"
    # TASK_NAME = "AT=7_RW=0"
    # TASK_NAME = "AT=20_RW=0"

    MODEL = model_dict[MODEL_NAME]
    TASK = task_dict[TASK_NAME].copy()
    MODEL_PATH = f"models_{MODEL_NAME}_{TASK_NAME}"
    CUR_EPISODE = 0
    
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)
    for file in os.listdir(MODEL_PATH):
        if file.startswith("model") and file.split('_')[1].isdigit() and file.endswith('.ckpt'):
            CUR_EPISODE = max(CUR_EPISODE, int(file.split('_')[1]))

    TOT_EPISODE = 1600
    SAVE_EPISODE = 3
    BATCH_SIZE = TASK.pop('batch_size')
    SIM_FREQ = 30
    EPISODE_LEN_SEC = 160

    env = FlyThruGateAviary(gui=False, record=False,
                            freq=SIM_FREQ, episode_len_sec=EPISODE_LEN_SEC, **TASK)

    ds = "test4B7Bs13S3R_VEL_2D"
    # load_path = os.path.join(MODEL_PATH, f"model_{CUR_EPISODE}_{ds}.ckpt")
    load_path = os.path.join(MODEL_PATH, f"model_{CUR_EPISODE}_{ds}.ckpt")
    try:
        if MODEL is A2C:
            model = A2C.load(load_path, verbose=0)
        elif MODEL is SAC:
            model = SAC.load(load_path, verbose=1, gradient_steps=-1)
        else:
            # with open(load_path, 'rb') as f:
            #     model = pickle.load(f)
            model = MODEL.load(load_path,
                               gamma=0.90,
                               # gamma=0.999,
                               ent_coef=0.0001,
                               verbose=1,
                               n_steps=env.SIM_FREQ * env.EPISODE_LEN_SEC * BATCH_SIZE)
    except:
        #### Check the environment's spaces ########################
        print("[INFO] Action space:", env.action_space)
        print("[INFO] Observation space:", env.observation_space)
        if MODEL is A2C:
            model = A2C('MlpPolicy', env, verbose=1)
        elif MODEL is SAC:
            model = SAC('MlpPolicy', env, verbose=1, gradient_steps=-1)
        else:
            model = MODEL('MlpPolicy', env,
                          gamma=0.9,
                          # gamma=1,
                          ent_coef=0.0001,
                          verbose=1,
                          n_steps=env.SIM_FREQ * env.EPISODE_LEN_SEC * BATCH_SIZE,
                          policy_kwargs={
                              'net_arch': [128, 128, dict(pi=[64], vf=[64])]
                          })
        CUR_EPISODE = 0

    check_env(env, warn=True, skip_render_check=True)

    print(model.policy)
    model.set_env(env)

    for _ in range(TOT_EPISODE):
        print(f"current episode = {CUR_EPISODE + SAVE_EPISODE * (_ + 1)}")
        model.policy.train()
        model.learn(total_timesteps=env.SIM_FREQ * env.EPISODE_LEN_SEC * BATCH_SIZE * SAVE_EPISODE)  # Typically not enough
        model.policy.eval()
        model.save(os.path.join(MODEL_PATH, f"model_{CUR_EPISODE + SAVE_EPISODE * (_ + 1)}_{ds}.ckpt"))
        # fly_a2c.main()



