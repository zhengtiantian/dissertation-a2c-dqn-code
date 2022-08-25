import os
from datetime import time

import numpy as np
from gym.utils.env_checker import check_env
from matplotlib import pyplot as plt
from stable_baselines3 import DQN

from fly_a2c import model_dict
from gym_pybullet_drones.envs.single_agent_rl.FlyThruGateAviary_new_dqn import FlyThruGateAviary_new_dqn
from gym_pybullet_drones.utils.Logger import Logger

from gym_pybullet_drones.envs.single_agent_rl import FlyThruGateAviary

task_dict = {
    'AT=0_RW=0': {
        'initial_xyzs': [[0.0, 0.0, 10.0]],
        'use_dqn_like_reward': True
    },
    'AT=7_RW=0': {
        'initial_xyzs': [[7.0, 0.0, 10.0]],
        'use_dqn_like_reward': True
    },
    'AT=20_RW=0': {
        'initial_xyzs': [[20.0, 0.0, 10.0]],
        'use_dqn_like_reward': True
    },
    'Integated_RW=0': {
        'initial_xyzs': None,
        'use_dqn_like_reward': True
    }
}

fig = plt.figure(figsize=(11, 11))
ax = fig.add_subplot(111)

a2c = []
dqn = []

ax.set_xlim(0, 500)
ax.set_ylim(0, 500)
matrix = np.ones((500, 500))

MODEL_NAME = "A2C"
TASK_NAME = "AT=20_RW=0"
# TASK_NAME = "AT=7_RW=0"
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
# BATCH_SIZE = TASK.pop('batch_size')
SIM_FREQ = 60
EPISODE_LEN_SEC = 160

ds = "test4B7Bs13S3R_VEL_2D"
load_path = os.path.join(MODEL_PATH, f"model_18_{ds}.ckpt")
# load_path = os.path.join(MODEL_PATH, f"model_{CUR_EPISODE}_{ds}.pkl")


env = FlyThruGateAviary(gui=False, record=False,
                        freq=SIM_FREQ, episode_len_sec=EPISODE_LEN_SEC, **TASK)

for i in range(500):
    for j in range(500):
        matrix[j, i] = env._getBestSINRbyUAVxyz([i / 10, j / 10, 10])
colorbar = fig.colorbar(ax.imshow(matrix))
colorbar.set_label('SINR(dB)')

MAX_XY = env.TARGET_XYZ[:-1].max()

model = MODEL.load(load_path,
                   gamma=0.9,
                   # gamma=0.999,
                   ent_coef=0.0001,
                   verbose=1,
                   n_steps=env.SIM_FREQ * env.EPISODE_LEN_SEC * 1
                   )

check_env(env, warn=True, skip_render_check=True)
model.set_env(env)

logger = Logger(logging_freq_hz=int(env.SIM_FREQ / env.AGGR_PHY_STEPS),
                num_drones=1
                )
env.seed(0)
model.set_random_seed(0)

obs = env.reset()
# logger.visualize(env, map_size=(500, 500), save_as=os.path.join(MODEL_PATH, f"result_{CUR_EPISODE}_{ds}.png"))
print("start test: \n\n")
for _ in range(1):
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

_states1, _timesteps1 = logger.visualize(env)

for drone in range(env.NUM_DRONES):
    x = _states1[drone, 0] * MAX_XY * 10
    y = _states1[drone, 1] * MAX_XY * 10
    t = _timesteps1[drone]
    a2c.append(x)
    a2c.append(y)
    a2c.append(t)
    print('a2c shapes' + str(x.shape) + ':' + str(y.shape) + ':' + str(t.shape))

initial_xyzs = np.array([20, 0, 10]).reshape(1, 3)
env1 = FlyThruGateAviary_new_dqn(gui=False, record=False, freq=SIM_FREQ, episode_len_sec=EPISODE_LEN_SEC,
                                 initial_xyzs=initial_xyzs)

check_env(env1, warn=True, skip_render_check=True)

#### Show (and record a video of) the models's performance ##
ds = "test2B13SNR_VEL_2D"
# load_path = "models_dqn/model2022811_" + ds + ".ckpt"
# load_path = "models_dqn/model2022818_" + ds + ".ckpt"
# load_path = "models_dqn/model2022819_" + ds + ".ckpt"

load_path = "models_dqn/model2022819_" + ds + "(20)" + ".ckpt"

model = DQN.load(load_path, gamma=0.9)

print(model.policy)

logger = Logger(logging_freq_hz=int(env1.SIM_FREQ / env1.AGGR_PHY_STEPS),
                num_drones=1
                )
obs = env1.reset()

for _ in range(1):
    for i in range(env1.EPISODE_LEN_SEC * env1.SIM_FREQ):
        action, _states = model.predict(obs, deterministic=True)

        # print("action:" + str(action))
        obs, reward, done, info = env1.step(action)
        logger.new_log(drone=0,
                       timestamp=i / env1.SIM_FREQ,
                       # state=np.hstack([obs[0:3], np.zeros(4), obs[3:15], np.resize(action, (4))]),
                       state=np.hstack([obs[0:2]]),
                       control=np.zeros(12)
                       )
        if i % env1.SIM_FREQ == 0:
            env1.render()
            print(done)

        if done:
            obs = env1.reset()
            break

_states2, _timesteps2 = logger.visualize(env1)

for drone in range(env1.NUM_DRONES):
    x = _states2[drone, 0] * MAX_XY * 10
    y = _states2[drone, 1] * MAX_XY * 10
    t = _timesteps2[drone]
    dqn.append(x)
    dqn.append(y)
    dqn.append(t)
    print('dqn shapes' + str(x.shape) + ':' + str(y.shape) + ':' + str(t.shape))

ax.scatter(a2c[0], a2c[1], s=1, label='A2C', color='m')
ax.scatter(dqn[0], dqn[1], s=1, label='DQN', color='b')

circle = plt.Circle((env.TARGET_XYZ[0] * 10, env.TARGET_XYZ[1] * 10), env.MAX_ERROR_WITH_TARGET * 10, color='r',
                    fill=False, label='end position range')
plt.gcf().gca().add_artist(circle)

xs = [0, 70, 200]
ys = [1, 1, 1]
plt.plot(xs, ys, "o", color="black", label='start point')
plt.plot(-1, -1, "s", color="black", label='buildings')

plt.xlabel('x meters')
plt.ylabel('y meters')
plt.legend(loc=7)

plt.show()

# plot line graph
fig = plt.figure(figsize=(11, 11))
ax = fig.add_subplot(111)

a2cSINR = []
a2cTotalSINR = 0
a2ci = 0
a2cbestSINR = float("-inf")
a2cworstSINR = float("inf")
a2cdistance = 0
for i,v in enumerate(a2c[0]):
    SINR = env._getBestSINRbyUAVxyz([a2c[0][i], a2c[1][i], 10])
    a2cSINR.append(SINR)
    a2cTotalSINR = a2cTotalSINR + SINR
    if SINR > a2cbestSINR:
        a2cbestSINR = SINR

    if SINR < a2cworstSINR:
        a2cworstSINR = SINR

    a2ci = a2ci + 1
    if i > 0:
        a2cdistance = a2cdistance + env._calculateDistance2D([a2c[0][i], a2c[1][i], 10],
                                                           [a2c[0][i - 1], a2c[1][i - 1], 10])

print('averageSINR:' + str(a2cTotalSINR / a2ci) + ' bestSINR:' + str(a2cbestSINR) + ' worstSINR:' + str(
    a2cworstSINR) + 'distance:' + str(a2cdistance * 10))


dqnSINR = []
dqnTotalSINR = 0
dqni = 0
dqnbestSINR = float("-inf")
dqnworstSINR = float("inf")
dqndistance = 0
for i,v in enumerate(dqn[0]):
    SINR = env._getBestSINRbyUAVxyz([dqn[0][i], dqn[1][i], 10])
    dqnSINR.append(SINR)
    dqnTotalSINR = dqnTotalSINR + SINR
    if SINR > dqnbestSINR:
        dqnbestSINR = SINR

    if SINR < dqnworstSINR:
        dqnworstSINR = SINR

    dqni = dqni + 1
    if i > 0:
        dqndistance = dqndistance + env._calculateDistance2D([dqn[0][i], dqn[1][i], 10],
                                                           [dqn[0][i - 1], dqn[1][i - 1], 10])

print('averageSINR:' + str(dqnTotalSINR / a2ci) + ' bestSINR:' + str(dqnbestSINR) + ' worstSINR:' + str(
    dqnworstSINR) + 'distance:' + str(dqndistance * 10))

plt.plot(a2c[2],a2cSINR, color="r", label='a2c')
plt.plot(dqn[2],dqnSINR, color="b", label='dqn')
plt.xlabel('times')
plt.ylabel('SINR(dB)')


plt.legend(loc=7)

plt.show()

env.close()
env1.close()
