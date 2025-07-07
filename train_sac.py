import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime

from utils.network import PolicyNet, QNet
from utils.agent import FrontierSACAgent
from utils.env import Env_SAC
from utils.parameter import *


class ReplayBuffer:
    """经验回放缓冲区"""
    def __init__(self):
        self.buffer = deque(maxlen=REPLAY_BUFFER_CAPACITY)
    
    def push(self, state, action, reward, next_state, done):
        """添加经验到缓冲区"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """从缓冲区随机采样"""
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)


def train_sac(resume_from_checkpoint=None):
    # 在训练开始时创建时间戳文件夹
    start_time = datetime.now()
    timestamp = start_time.strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = f"checkpoints/sac_training_{timestamp}"
    if resume_from_checkpoint:
        # 从checkpoint恢复
        print(f"从checkpoint恢复训练: {resume_from_checkpoint}")
    else:
        # 新训练的
        print(f"开始新训练")
    print(f"模型保存路径: {checkpoint_dir}")
    print(f"训练开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 创建检查点目录
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 保存训练配置信息
    config_info = {
        'start_time': start_time.strftime('%Y-%m-%d %H:%M:%S'),
        'timestamp': timestamp,
        'n_clusters': N_CLUSTERS,
        'learning_rate': LEARNING_RATE,
        'batch_size': BATCH_SIZE,
        'replay_buffer_capacity': REPLAY_BUFFER_CAPACITY,
        'train_episodes': TRAIN_EPISODES,
        'resume_from': resume_from_checkpoint,
        # 添加其他重要参数
    }
    
    # 保存配置文件
    with open(f"{checkpoint_dir}/training_config.json", 'w') as f:
        json.dump(config_info, f, indent=2)
    
    # 创建训练日志文件
    log_file = f"{checkpoint_dir}/training_log.txt"
    
    def log_message(message):
        timestamp = datetime.now().strftime('%Y%m%d%H%M')
        log_entry = f"[{timestamp}] {message}"
        print(log_entry)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry + '\n')
    
    # 初始化网络
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 初始化网络
    policy_net = PolicyNet().to(device)
    q_net_1 = QNet().to(device)
    q_net_2 = QNet().to(device)
    target_q_net_1 = QNet().to(device)
    target_q_net_2 = QNet().to(device)
    
    # 优化器
    policy_optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    q1_optimizer = optim.Adam(q_net_1.parameters(), lr=LEARNING_RATE)
    q2_optimizer = optim.Adam(q_net_2.parameters(), lr=LEARNING_RATE)
    
    # 自动熵调节
    target_entropy = 0.1 * np.log(N_CLUSTERS)
    log_alpha = torch.FloatTensor([-2]).to(device)
    log_alpha.requires_grad = True
    alpha_optimizer = optim.Adam([log_alpha], lr=LEARNING_RATE)
    alpha = log_alpha.exp()
    
    # 训练统计
    episode_rewards = []
    episode_lengths = []
    episode_dists = []
    losses = {'policy': [], 'q1': [], 'q2': [], 'alpha': []}
    total_steps = 0
    start_episode = 0
    
    # 加载checkpoint（如果指定）
    if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
        print(f"Loading checkpoint from: {resume_from_checkpoint}")
        checkpoint = torch.load(resume_from_checkpoint, map_location=device, weights_only=False)
        
        # 加载网络参数
        policy_net.load_state_dict(checkpoint['policy_net'])
        q_net_1.load_state_dict(checkpoint['q_net_1'])
        q_net_2.load_state_dict(checkpoint['q_net_2'])
        
        # 重新初始化目标网络
        target_q_net_1.load_state_dict(q_net_1.state_dict())
        target_q_net_2.load_state_dict(q_net_2.state_dict())
        
        # 加载训练统计
        start_episode = checkpoint.get('episode', 0)
        episode_rewards = checkpoint.get('episode_rewards', [])
        episode_lengths = checkpoint.get('episode_lengths', [])
        episode_dists = checkpoint.get('episode_dists', [])
        losses = checkpoint.get('losses', {'policy': [], 'q1': [], 'q2': [], 'alpha': []})
        
        # 估算总步数（这是一个近似值）
        total_steps = len(losses['policy']) * UPDATE_INTERVAL if losses['policy'] else 0
        
        log_message(f"Checkpoint loaded successfully!")
        log_message(f"Resuming from episode: {start_episode}")
        log_message(f"Previous episodes trained: {len(episode_rewards)}")
        log_message(f"Previous training steps: {total_steps}")
        
        if episode_rewards:
            log_message(f"Previous average reward: {np.mean(episode_rewards[-10:]):.2f}")
            
        # 打印checkpoint信息
        if 'training_start_time' in checkpoint:
            print(f"Original training started: {checkpoint['training_start_time']}")
        if 'elapsed_time_formatted' in checkpoint:
            print(f"Previous training time: {checkpoint['elapsed_time_formatted']}")
    else:
        # 新训练或checkpoint不存在
        if resume_from_checkpoint:
            print(f"Warning: Checkpoint file not found: {resume_from_checkpoint}")
            print("Starting new training instead...")
        
        # 初始化目标网络
        target_q_net_1.load_state_dict(q_net_1.state_dict())
        target_q_net_2.load_state_dict(q_net_2.state_dict())
        log_message("Starting new training")
    
    # 经验回放
    replay_buffer = ReplayBuffer()
    
    # 环境和智能体
    agent = FrontierSACAgent()
    env = Env_SAC(agent=agent,mode='train')

    log_message(f"Training will continue from episode {start_episode} to {TRAIN_EPISODES}")

    for episode in range(start_episode, TRAIN_EPISODES):
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        episode_dist = 0
        
        while not done:
            frontier_tensor, traj_tensor, mask_tensor = get_obs_tensor(obs)
            frontier_tensor = frontier_tensor.to(device)
            traj_tensor = traj_tensor.to(device)
            mask_tensor = mask_tensor.to(device)
            with torch.no_grad():
                logits = policy_net(frontier_tensor, traj_tensor, mask_tensor)
        
            # 创建概率分布
            probs = F.softmax(logits, dim=-1)
            dist = Categorical(probs)
            action_index = dist.sample()
            next_waypoint = agent.frontier_cluster_centers[action_index]

            next_obs, reward, done = env.step(next_waypoint)
            
            # 只有在非终止状态时才存储经验
            if not done:
                replay_buffer.push(obs, action_index, reward, next_obs, done)
            
            obs = next_obs
            episode_reward += reward
            episode_length += 1
            episode_dist += agent.travel_dist[-1]
            total_steps += 1
            
            # 训练网络
            if len(replay_buffer) > WARMUP_STEPS and total_steps % UPDATE_INTERVAL == 0:
                # 采样批次数据
                # print(len(replay_buffer))
                batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = replay_buffer.sample(BATCH_SIZE)
                
                # 转换为张量
                batch_frontier_tensors = []
                batch_traj_tensors = []
                batch_mask_tensors = []
                batch_next_frontier_tensors = []
                batch_next_traj_tensors = []
                batch_next_mask_tensors = []
                
                for i in range(BATCH_SIZE):
                    # 当前状态
                    frontier_tensor, traj_tensor, mask_tensor = get_obs_tensor(batch_states[i])
                    batch_frontier_tensors.append(frontier_tensor)
                    batch_traj_tensors.append(traj_tensor)
                    batch_mask_tensors.append(mask_tensor)
                    
                    # 下一状态处理
                    if not batch_dones[i]:
                        # 非终止状态：正常处理next_state
                        next_frontier_tensor, next_traj_tensor, next_mask_tensor = get_obs_tensor(batch_next_states[i])
                    else:
                        # 终止状态：创建dummy next_state（反正会被done mask掉）
                        next_frontier_tensor = torch.zeros_like(frontier_tensor)
                        next_traj_tensor = torch.zeros_like(traj_tensor)
                        next_mask_tensor = torch.zeros_like(mask_tensor, dtype=torch.bool)
                    
                    batch_next_frontier_tensors.append(next_frontier_tensor)
                    batch_next_traj_tensors.append(next_traj_tensor)
                    batch_next_mask_tensors.append(next_mask_tensor)
                
                # 拼接批次
                batch_frontier_tensors = torch.cat(batch_frontier_tensors, dim=0).to(device)
                batch_traj_tensors = torch.cat(batch_traj_tensors, dim=0).to(device)
                batch_mask_tensors = torch.cat(batch_mask_tensors, dim=0).to(device)
                batch_next_frontier_tensors = torch.cat(batch_next_frontier_tensors, dim=0).to(device)
                batch_next_traj_tensors = torch.cat(batch_next_traj_tensors, dim=0).to(device)
                batch_next_mask_tensors = torch.cat(batch_next_mask_tensors, dim=0).to(device)
                
                batch_actions = torch.tensor(batch_actions, dtype=torch.long).to(device)
                batch_rewards = torch.tensor(batch_rewards, dtype=torch.float32).to(device)
                batch_dones = torch.tensor(batch_dones, dtype=torch.bool).to(device)
                
                # 计算当前Q值 - 使用批次数据
                q1_values = q_net_1(batch_frontier_tensors, batch_traj_tensors, batch_mask_tensors)
                q2_values = q_net_2(batch_frontier_tensors, batch_traj_tensors, batch_mask_tensors)

                current_q1 = q1_values.gather(1, batch_actions.unsqueeze(1)).squeeze(1)
                current_q2 = q2_values.gather(1, batch_actions.unsqueeze(1)).squeeze(1)

                # 计算目标Q值 - 使用批次数据
                with torch.no_grad():
                    next_logits = policy_net(batch_next_frontier_tensors, batch_next_traj_tensors, batch_next_mask_tensors)
                    next_probs = F.softmax(next_logits, dim=-1)
                    next_log_probs = F.log_softmax(next_logits, dim=-1)
                    
                    next_q1 = target_q_net_1(batch_next_frontier_tensors, batch_next_traj_tensors, batch_next_mask_tensors)
                    next_q2 = target_q_net_2(batch_next_frontier_tensors, batch_next_traj_tensors, batch_next_mask_tensors)
                    next_q = torch.min(next_q1, next_q2)
                    
                    next_v = (next_probs * (next_q - alpha * next_log_probs)).sum(dim=-1)
                    target_q = batch_rewards + GAMMA * next_v * (~batch_dones)

                # Q网络损失
                q1_loss = F.mse_loss(current_q1, target_q)
                q2_loss = F.mse_loss(current_q2, target_q)

                # 更新Q网络
                q1_optimizer.zero_grad()
                q1_loss.backward()
                q1_optimizer.step()

                q2_optimizer.zero_grad()
                q2_loss.backward()
                q2_optimizer.step()

                # 策略网络损失 - 重新编码以避免计算图冲突
                current_logits = policy_net(batch_frontier_tensors, batch_traj_tensors, batch_mask_tensors)
                current_probs = F.softmax(current_logits, dim=-1)
                current_log_probs = F.log_softmax(current_logits, dim=-1)

                with torch.no_grad():
                    current_q1_detached = q_net_1(batch_frontier_tensors, batch_traj_tensors, batch_mask_tensors)
                    current_q2_detached = q_net_2(batch_frontier_tensors, batch_traj_tensors, batch_mask_tensors)
                    current_q_detached = torch.min(current_q1_detached, current_q2_detached)

                policy_loss = (current_probs * (alpha * current_log_probs - current_q_detached)).sum(dim=-1).mean()

                # 更新策略网络
                policy_optimizer.zero_grad()
                policy_loss.backward()
                policy_optimizer.step()

                # 自动熵调节
                alpha_loss = -(log_alpha * (current_log_probs + target_entropy).detach()).mean()
                alpha_optimizer.zero_grad()
                alpha_loss.backward()
                alpha_optimizer.step()
                alpha = log_alpha.exp()

                # 软更新目标网络
                update_target_networks(q_net_1, target_q_net_1, TAU)
                update_target_networks(q_net_2, target_q_net_2, TAU)

                # 记录损失
                losses['policy'].append(policy_loss.item())
                losses['q1'].append(q1_loss.item())
                losses['q2'].append(q2_loss.item())
                losses['alpha'].append(alpha_loss.item())

        # 记录episode统计
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_dists.append(episode_dist)
        # 打印进度
        if episode % 20 == 0 and len(replay_buffer) > WARMUP_STEPS:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_length = np.mean(episode_lengths[-10:])
            avg_dist = np.mean(episode_dists[-10:])
            print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Avg Length: {avg_length:.1f}, Avg Dist: {avg_dist:.1f}")
            
            # 打印最近的损失统计
            if len(losses['policy']) > 0:
                recent_policy_loss = np.mean(losses['policy'][-10:]) if len(losses['policy']) >= 10 else np.mean(losses['policy'])
                recent_q1_loss = np.mean(losses['q1'][-10:]) if len(losses['q1']) >= 10 else np.mean(losses['q1'])
                recent_q2_loss = np.mean(losses['q2'][-10:]) if len(losses['q2']) >= 10 else np.mean(losses['q2'])
                recent_alpha_loss = np.mean(losses['alpha'][-10:]) if len(losses['alpha']) >= 10 else np.mean(losses['alpha'])
                
                print(f"Recent Policy Loss: {recent_policy_loss:.4f}, Q1 Loss: {recent_q1_loss:.4f}, "
                      f"Q2 Loss: {recent_q2_loss:.4f}, Alpha Loss: {recent_alpha_loss:.4f}, Alpha: {alpha.item():.3f}")

        # 保存模型
        if episode % 300 == 0 and len(replay_buffer) > WARMUP_STEPS:
            current_time = datetime.now()
            elapsed_time = current_time - start_time
            
            # 保存模型
            checkpoint_data = {
                'policy_net': policy_net.state_dict(),
                'q_net_1': q_net_1.state_dict(),
                'q_net_2': q_net_2.state_dict(),
                'episode': episode,
                'episode_rewards': episode_rewards,
                'episode_lengths': episode_lengths,
                'episode_dists': episode_dists,
                'losses': losses,
                # 添加时间信息
                'training_start_time': start_time.strftime('%Y-%m-%d %H:%M:%S'),
                'checkpoint_time': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                'elapsed_time_seconds': elapsed_time.total_seconds(),
                'elapsed_time_formatted': str(elapsed_time).split('.')[0]
            }
            
            model_path = f"{checkpoint_dir}/sac_model_episode_{episode}.pth"
            torch.save(checkpoint_data, model_path)
            
            print(f"Episode {episode}: 模型已保存到 {model_path}")
            print(f"已训练时间: {str(elapsed_time).split('.')[0]}")
    
    # 训练结束时保存最终模型
    end_time = datetime.now()
    total_time = end_time - start_time
    
    final_checkpoint = {
        'policy_net': policy_net.state_dict(),
        'q_net_1': q_net_1.state_dict(),
        'q_net_2': q_net_2.state_dict(),
        'episode': TRAIN_EPISODES,
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'episode_dists': episode_dists,
        'losses': losses,
        'training_start_time': start_time.strftime('%Y-%m-%d %H:%M:%S'),
        'training_end_time': end_time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_training_time': str(total_time).split('.')[0],
        'total_training_seconds': total_time.total_seconds()
    }
    
    torch.save(final_checkpoint, f"{checkpoint_dir}/sac_model_final.pth")
    
    print(f"训练完成！")
    print(f"总训练时间: {str(total_time).split('.')[0]}")
    print(f"最终模型保存到: {checkpoint_dir}/sac_model_final.pth")
    
    return checkpoint_dir

def get_obs_tensor(obs: tuple[np.ndarray, np.ndarray, np.ndarray], device='cuda') -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    frontier_tensor = torch.tensor(obs[0], dtype=torch.float32, device=device).unsqueeze(0)
    traj_tensor = torch.tensor(obs[1], dtype=torch.float32, device=device).unsqueeze(0)
    mask_tensor = torch.tensor(obs[2], dtype=torch.bool, device=device).unsqueeze(0)
    return frontier_tensor, traj_tensor, mask_tensor

def update_target_networks(q_net, target_q_net, tau):
    """软更新目标网络"""
    for target_param, param in zip(target_q_net.parameters(), q_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


if __name__ == "__main__":
    # 指定要恢复的checkpoint路径
    # checkpoint_path = "checkpoints/sac_training_20250705_010619/sac_model_episode_6900.pth"
    # # 从checkpoint恢复训练
    # train_sac(resume_from_checkpoint=checkpoint_path)
    
    # 如果要开始新训练，使用：
    train_sac()