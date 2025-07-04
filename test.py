import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from utils.env import Env_SAC, Env
from utils.agent import FrontierSACAgent, Agent
from utils.network import PolicyNet, Encoder

def test_greedy():
    agent = Agent()
    env = Env(agent)
    obs = env.reset()
    plt.ion()  # 开启交互模式
    plt.figure()
    while True:
        plt.clf()
        action = agent.get_action(obs)
        env.plot(action)
        obs, reward, done = env.step(action)
        if done:
            obs = env.reset()
        plt.pause(0.05)


def test_sac(model_path=None):
    # 初始化网络
    encoder = Encoder()
    policy_net = PolicyNet()
    if model_path is not None:
        encoder.load_state_dict(torch.load(model_path))
        policy_net.load_state_dict(torch.load(model_path))
    # 初始化agent
    agent = FrontierSACAgent()
    # 初始化环境
    env = Env_SAC(agent)
    # reset
    obs = env.reset()

    plt.ion()  # 开启交互模式
    plt.figure()

    while True:
        plt.clf()
        # get action
        frontier_tensor, traj_tensor, mask_tensor = get_obs_tensor(obs)
        traj_enc, frontier_embed = encoder(frontier_tensor, traj_tensor, mask_tensor)
        logits = policy_net(frontier_embed, traj_enc, mask_tensor)
        action_index = torch.argmax(logits.squeeze(0))
        action = agent.frontier_cluster_centers[action_index]  # action is the next point
        # print(f"action: {action}")
        # step
        env.plot(action)
        obs, reward, done = env.step(action)
        if done:
            obs = env.reset()
            continue
        
        plt.pause(0.05)


def get_obs_tensor(obs:tuple[np.ndarray, np.ndarray, np.ndarray])->tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    frontier_tensor = torch.tensor(obs[0], dtype=torch.float32).unsqueeze(0)
    traj_tensor = torch.tensor(obs[1], dtype=torch.float32).unsqueeze(0)
    mask_tensor = torch.tensor(obs[2], dtype=torch.bool).unsqueeze(0)
    return frontier_tensor, traj_tensor, mask_tensor


if __name__ == "__main__":
    # test_sac()
    test_greedy()