import math

import numpy as np
import seaborn as sns
import matplotlib.pylab as plt

from gym_pybullet_drones.envs.single_agent_rl import FlyThruGateAviary

env = FlyThruGateAviary(gui=False, record=False)

metrix = np.ones((500, 500))

cols, rows = metrix.shape

x = []
y = []
dx = []
dy = []

for i in range(rows):
    for j in range(cols):
        metrix[j, i] = env._getBestSINRbyUAVxyz([i/10, j/10, 10])
        if i % 20 == 10 and j % 20 == 10:
            # grad = env._getGradOfBestSINRbyUAVxyz([i/10, j/10, 10])
            grad = env._getGradOfRewardbyUAVxyz([i / 10, j / 10, 10])
            x.append(i)
            y.append(j)
            norm = (grad[0] ** 2 + grad[1] ** 2) ** 0.5
            dx.append(grad[0] / norm)
            dy.append(grad[1] / norm)
            print(i, j, dx[-1], dy[-1])

x = np.array(x)
y = np.array(y)
dx = np.array(dx)
dy = np.array(dy)

ax = sns.heatmap(metrix)
# fig = plt.figure(figsize=(10, 10))
# ax = fig.add_subplot(111)
# plt.colorbar(ax.imshow(metrix))
ax.quiver(x, y, dx, dy)
ax.set_xlim(0, 500)
ax.set_ylim(0, 500)
plt.show()
