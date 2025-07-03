from utils.env import Env
from utils.utils import *
import matplotlib.pyplot as plt

done=False
env = Env()
env.reset()
plt.ion()  # 开启交互模式
fig = plt.figure()

while True:
    plt.clf()
    done = env.step()
    env.plot()
    # print(type(env.robot.frontier_cluster_centers))  # nd.array
    # print(type(env.robot.global_frontier))  # set
    
    if done:
        env.reset()
        # continue
        # break

    plt.pause(0.05)

plt.ioff()  # 关闭交互模式
plt.show()
