import numpy as np
import os
from skimage import io
from skimage.measure import block_reduce
import matplotlib.pyplot as plt

from utils.parameter import *
from utils.agent import Agent, FrontierSACAgent
from utils.utils import get_position_in_map_from_coords, A_star


class Env:
    def __init__(self,agent:Agent):
        '''
        Attributes:
            ground_truth: ndarray
            ground_truth_size: ndarray
            robot_position_in_map: ndarray
        '''
        self.pixel_size = PIXEL_SIZE  # meter

        self.robot=agent  # 可替换为其他agent

        self.ground_truth=None  # pixel
        self.ground_truth_size=None  # pixel
        self.robot_position_in_map=None  # pixel
        
        self.explored_rate=0
        self.explored_rate_change=0
        self.episode_index = 1 
        self.step_count=0

        # map info 用于坐标系转换
        self.belief_origin_x=None
        self.belief_origin_y=None

    def reset(self):
        self.ground_truth, self.robot_position_in_map = self.import_ground_truth(self.episode_index)
        self.ground_truth_size = np.shape(self.ground_truth)  # pixel

        self.belief_origin_x = -np.round(self.robot_position_in_map[0] * self.pixel_size, 1)  # meter
        self.belief_origin_y = -np.round(self.robot_position_in_map[1] * self.pixel_size, 1)  # meter

        self.explored_rate = 0
        self.explored_rate_change = 0
        self.step_count = 0  

        self.robot.reset(self.ground_truth_size,
                         self.belief_origin_x,
                         self.belief_origin_y)
        self.update_robot_info()
        self.update_env_info()

        self.episode_index += 1
        obs = self.robot.get_obs()
        return obs

    def step(self, action:np.ndarray)->tuple[np.ndarray, float, bool]:
        self.robot.move(action)
        # 移动到下一个目标点, robot_position_in_map和robot.position同时更新
        self.update_position()
        # 更新机器人信息
        self.update_robot_info()
        # 更新环境信息
        self.update_env_info()
        # 计算奖励
        # self.calculate_reward()
        if self.explored_rate >= EXPLORATION_RATE_THRESHOLD:
            done = True
        else:
            done = False
        # 步数+1
        self.step_count += 1
        next_obs = self.robot.get_obs()
        reward = 0  # greedy agent没有reward
        return next_obs, reward, done

    def update_env_info(self):
        # update robot_position_in_map
        self.robot_position_in_map = get_position_in_map_from_coords(self.robot.position, 
                                                                     self.robot.belief_map_info)
        # update explored rate
        self.update_explored_rate()
        self.step_count += 1

    def update_explored_rate(self):
        if self.step_count == 0:
            self.explored_rate = np.sum(self.robot.belief_map_info.map == FREE) / np.sum(self.ground_truth == FREE)
            old_explored_rate = self.explored_rate
        else:
            old_explored_rate = self.explored_rate
            self.explored_rate = np.sum(self.robot.belief_map_info.map == FREE) / np.sum(self.ground_truth == FREE)
        self.explored_rate_change = self.explored_rate - old_explored_rate
        
    def update_position(self):
        # update robot position
        self.robot.update_robot_position()
        # update robot_position_in_map
        self.robot_position_in_map = get_position_in_map_from_coords(self.robot.position, 
                                                                     self.robot.belief_map_info)

    def update_robot_info(self):
        # update robot belief map
        self.robot.update_belief_map(self.robot_position_in_map,
                                     self.ground_truth)
        # update global frontier and frontier cluster centers
        self.robot.update_frontier()

    def import_ground_truth(self, episode_index:int)->tuple[np.ndarray,np.ndarray]:
        map_dir = f'maps/train/'
        map_list = os.listdir(map_dir)
        map_index = episode_index % np.size(map_list)
        ground_truth = (io.imread(map_dir + '/' + map_list[map_index], 1) * 255).astype(int)
        ground_truth = block_reduce(ground_truth, 2, np.min)  # 图像降采样

        robot_position_in_map = np.nonzero(ground_truth == 208)
        robot_position_in_map = np.array([np.array(robot_position_in_map)[1, 10], np.array(robot_position_in_map)[0, 10]])

        ground_truth = (ground_truth > 150) | ((ground_truth <= 80) & (ground_truth >= 50))
        ground_truth = ground_truth * 254 + 1

        return ground_truth, robot_position_in_map
    
    def plot(self, action:np.ndarray=None):
        plt.imshow(self.robot.belief_map_info.map, cmap='gray')
        plt.axis('off')
        # robot position    
        plt.plot(self.robot_position_in_map[0],self.robot_position_in_map[1],'mo',markersize=5,zorder=10)
        
        if len(self.robot.frontier_cluster_centers) > 0:
            frontier_points = get_position_in_map_from_coords(self.robot.frontier_cluster_centers,
                                                              self.robot.belief_map_info)
            plt.scatter(frontier_points[:,0], frontier_points[:,1], c='darkred', s=10, marker='o', alpha=1, zorder=2)
        
        # trajectory
        if len(self.robot.trajectory) > 0:
            # 先画所有历史路径
            for i, path in enumerate(self.robot.trajectory):  # 除了最后一条
                if path is not None and len(path) > 1:
                    # 修复：将 list[np.ndarray] 转换为 np.ndarray
                    path_array = np.array(path)
                    path = get_position_in_map_from_coords(path_array, self.robot.belief_map_info)
                    plt.plot(path[:,0], path[:,1], 'b-', linewidth=2, alpha=1, zorder=1)
        # 画当前路径
        current_path = A_star(self.robot.position, action, self.robot.belief_map_info)
        if current_path is not None and len(current_path) > 1:
                # 修复：将 list[np.ndarray] 转换为 np.ndarray
                current_path_array = np.array(current_path)
                path = get_position_in_map_from_coords(current_path_array, self.robot.belief_map_info)
                plt.plot(path[:,0], path[:,1], 'r--', linewidth=1.5, alpha=1, zorder=1)
        # next waypoint
        next_waypoint = get_position_in_map_from_coords(action,self.robot.belief_map_info)
        plt.scatter(next_waypoint[0], next_waypoint[1], c='red', s=13, marker='o', alpha=1, zorder=3)
        plt.title(f"Step {self.step_count}  Explored ratio: {self.explored_rate*100:.4g}%  Travel distance: {self.robot.dist:.4g}m")
        plt.show()


class Env_SAC(Env):
    def __init__(self, agent: FrontierSACAgent):
        super().__init__(agent)

    def step(self, action:np.ndarray)->tuple[np.ndarray, float, bool]:
        self.robot.move(action)
        # 移动到下一个目标点, robot_position_in_map和robot.position同时更新
        self.update_position()
        # 更新机器人信息
        self.update_robot_info()
        # 得到next_obs
        next_obs = self.robot.get_obs()
        # 更新环境信息
        self.update_env_info()
        terminate, truncated = self.check_finish()
        # TODO: 计算奖励
        reward=self.calculate_reward(terminate, truncated)
        done=terminate or truncated
        self.step_count += 1
        return next_obs, reward, done
    
    def check_finish(self)->tuple[bool,bool]:
        if len(self.robot.waypoint) >= MAX_EPISODE_STEP:
            truncated = True
        else:
            truncated = False
        if self.explored_rate >= EXPLORATION_RATE_THRESHOLD:
            terminate = True
        else:
            terminate = False
        return terminate, truncated
    
    def calculate_reward(self, terminate:bool, truncated:bool)->float:
        '''
        1. exploration rate change 需要参考, 但是并不是主要的, 因为目标点就是前沿, 不管如何行动都exploration_rate_change>0
        2. 考虑每一步的exploration rate change和每一步的distance cost, 如果比值较大，说明这一步探索效率较高，则奖励更高
        3. 考虑当前总长度(np.sum(self.robot.travel_dist))和总探索率(explored_rate)，如果比值较大，说明长期来看(多步)探索效率较高，则奖励更高
        4. 完成探索(terminate=True)，固定奖励
        5. 未完成探索(truncated=True)，固定惩罚
        '''
        # === 1. exploration rate change 作为参考 ===
        base_exploration_reward = self.explored_rate_change * BASE_EXPLORATION_REWARD_WEIGHT
        # === 2. 单步探索效率奖励 (核心奖励) ===
        single_step_exploration_reward = self.explored_rate_change/self.robot.travel_dist[-1] * SINGLE_STEP_EXPLORATION_REWARD_WEIGHT
        # === 3. 长期探索效率奖励 ===
        long_term_exploration_reward = self.explored_rate/self.robot.dist * LONG_TERM_EXPLORATION_REWARD_WEIGHT
        # === 4. 完成探索奖励 ===
        finish_exploration_reward = FINISH_EXPLORATION_REWARD if terminate else 0
        # === 5. 未完成探索惩罚 ===
        not_finish_exploration_penalty = NOT_FINISH_EXPLORATION_PENALTY if truncated else 0
        # === 6. 总奖励 ===
        total_reward = base_exploration_reward + single_step_exploration_reward + long_term_exploration_reward + finish_exploration_reward + not_finish_exploration_penalty

        print(f'exploration_rate_change: {self.explored_rate_change}, explored_rate: {self.explored_rate}\n'
              f'base_exploration_reward: {base_exploration_reward}\n'
              f'single_step_exploration_reward: {single_step_exploration_reward}\n'
              f'long_term_exploration_reward: {long_term_exploration_reward}\n'
              f'finish_exploration_reward: {finish_exploration_reward}\n'
              f'not_finish_exploration_penalty: {not_finish_exploration_penalty}\n'
              f'total_reward: {total_reward}\n'
              f'--------------------------------')

        return total_reward
    