import pickle
import time
import os
from stable_baselines3 import A2C, PPO, SAC

from gym_pybullet_drones.envs.single_agent_rl import FlyThruGateAviary
from gym.utils.env_checker import check_env
from gym_pybullet_drones.utils.Logger import Logger
import numpy as np
from gym_pybullet_drones.utils.utils import sync

model_dict = {
    'A2C': A2C,

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


def main(model=None, env=None):
    MODEL_NAME = "A2C"
    # TASK_NAME = "AT=0_RW=0"
    TASK_NAME = "AT=0_RW=0"
    # TASK_NAME = "AT=20_RW=0"
    # TASK_NAME = "Integated_RW=0"

    MODEL = model_dict[MODEL_NAME]
    TASK = task_dict[TASK_NAME].copy()
    MODEL_PATH = f"models_{MODEL_NAME}_{TASK_NAME}"
    CUR_EPISODE = 0
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)
    for file in os.listdir(MODEL_PATH):
        if file.startswith("model") and file.endswith(".ckpt"):
            CUR_EPISODE = max(CUR_EPISODE, int(file.split('_')[1]))
    # CUR_EPISODE = 3
    TOT_EPISODE = 1600
    SAVE_EPISODE = 10
    BATCH_SIZE = TASK.pop('batch_size')
    SIM_FREQ = 30
    EPISODE_LEN_SEC = 160

    ds = "test4B7Bs13S3R_VEL_2D"
    load_path = os.path.join(MODEL_PATH, f"model_6_{ds}.ckpt")
    # load_path = os.path.join(MODEL_PATH, f"model_{CUR_EPISODE}_{ds}.pkl")

    if env is None:
        env = FlyThruGateAviary(gui=False, record=False,
                                freq=SIM_FREQ, episode_len_sec=EPISODE_LEN_SEC, **TASK)

    if model is None:
        # with open(load_path, 'rb') as f:
        #     model = pickle.load(f)
        model = MODEL.load(load_path,
                         gamma=0.9,
                         # gamma=0.999,
                         ent_coef=0.0001,
                         verbose=1,
                         n_steps=env.SIM_FREQ * env.EPISODE_LEN_SEC * BATCH_SIZE
                         )



    check_env(env, warn=True, skip_render_check=True)
    model.set_env(env)

    logger = Logger(logging_freq_hz=int(env.SIM_FREQ / env.AGGR_PHY_STEPS),
                    num_drones=1
                    )
    env.seed(0)
    model.set_random_seed(0)
    print(model.policy)
    obs = env.reset()
    # logger.visualize(env, map_size=(500, 500), save_as=os.path.join(MODEL_PATH, f"result_{CUR_EPISODE}_{ds}.png"))
    print("start test: \n\n")
    for _ in range(BATCH_SIZE):
        for i in range(env.EPISODE_LEN_SEC * env.SIM_FREQ):
            action, _states = model.predict(obs, deterministic=True)

            # print("action:" + str(action))
            obs, reward, done, info = env.step(action)
            logger.log(drone=0,
                       timestamp=i / env.SIM_FREQ,
                       state=np.hstack([obs[0:3], np.zeros(4), obs[3:15], np.resize(action, (4))]),
                       control=np.zeros(12)
                       )
            # if i % env.SIM_FREQ == 0:
            #     env.render()
            #     print(done)
            # sync(i, start, env.TIMESTEP)
            if done:
                obs = env.reset()
                break

    logger.new_visualize(env, map_size=(500, 500), save_as=os.path.join(MODEL_PATH, f"result_{CUR_EPISODE}_{ds}.png"))
    env.close()



if __name__ == "__main__":
    main()
