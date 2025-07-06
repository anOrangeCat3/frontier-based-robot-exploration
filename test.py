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
    policy_net = PolicyNet()
    # 加载模型
    if model_path is not None:
        checkpoint = torch.load(model_path,weights_only=False)
        policy_net.load_state_dict(checkpoint['policy_net'])
        print(f"Loaded policy from episode {checkpoint.get('episode', 'Unknown')}")
    
    policy_net.eval()
    # 初始化agent
    agent = FrontierSACAgent()
    # 初始化环境
    env = Env_SAC(agent,mode='test')
    # reset
    obs = env.reset()

    plt.ion()  # 开启交互模式
    plt.figure()

    while True:
        plt.clf()
        # get action
        frontier_tensor, traj_tensor, mask_tensor = get_obs_tensor(obs)
        logits = policy_net(frontier_tensor, traj_tensor, mask_tensor)
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
    # 0705_214750：traj inpit_dim =3
    # 0706_003355：traj inpit_dim =4
    # checkpoints/sac_training_20250706_003355/sac_model_episode_37500.pth
    test_sac(model_path='./checkpoints/sac_training_20250706_003355/sac_model_episode_42900.pth')
    # test_greedy()