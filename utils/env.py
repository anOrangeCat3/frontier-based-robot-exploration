import numpy as np
import os
from skimage import io
from skimage.measure import block_reduce
import matplotlib.pyplot as plt

from utils.parameter import *
from utils.agent import Agent, FrontierSACAgent
from utils.utils import get_position_in_map_from_coords


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
        self.episode_index = 0 
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
        self.robot.get_action()

        self.episode_index += 1

        return obs

    def step(self, action):
        # TODO：移动到下一个目标点, robot_position_in_map和robot.position同时更新
        self.update_position()
        # TODO：更新机器人信息
        self.update_robot_info()
        # TODO：更新环境信息
        self.update_env_info()
        # TODO：计算奖励
        # self.calculate_reward()
        if self.explored_rate >= EXPLORATION_RATE_THRESHOLD:
            done = True
        else:
            # TODO：得到下一个目标点
            self.robot.get_action()
            done = False

        return next_obs, reward, done

    def update_env_info(self):
        # update robot_position_in_map
        self.robot_position_in_map = get_position_in_map_from_coords(self.robot.position, 
                                                                     self.robot.belief_map_info)
        # update explored rate
        self.update_explored_rate()
        self.step_count += 1

    def update_explored_rate(self):
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
    
    def plot(self):
        plt.imshow(self.robot.belief_map_info.map, cmap='gray')
        plt.axis('off')
        # robot position
        plt.plot(self.robot_position_in_map[0],self.robot_position_in_map[1],'mo',markersize=5,zorder=10)
        # frontier cluster centers
        if len(self.robot.frontier_cluster_centers) > 0:
            frontier_points = get_position_in_map_from_coords(self.robot.frontier_cluster_centers,
                                                              self.robot.belief_map_info)
            plt.scatter(frontier_points[:,0], frontier_points[:,1], c='darkred', s=10, marker='o', alpha=1, zorder=2)
        # next waypoint
        if len(self.robot.waypoint) > 0:
            next_waypoint = get_position_in_map_from_coords(self.robot.waypoint[-1],
                                                              self.robot.belief_map_info)
            plt.scatter(next_waypoint[0], next_waypoint[1], c='red', s=13, marker='o', alpha=1, zorder=3)
        if len(self.robot.trajectory) > 0:
            # 先画所有历史路径（较淡）
            for i, path in enumerate(self.robot.trajectory[:-1]):  # 除了最后一条
                if path is not None and len(path) > 1:
                    path = get_position_in_map_from_coords(path, self.robot.belief_map_info)
                    plt.plot(path[:,0], path[:,1], 'b-', linewidth=2, alpha=1, zorder=1)
            
            # 再画当前路径（突出显示）
            if len(self.robot.trajectory) > 0:
                current_path = self.robot.trajectory[-1]
                if current_path is not None and len(current_path) > 1:
                    path = get_position_in_map_from_coords(current_path, self.robot.belief_map_info)
                    plt.plot(path[:,0], path[:,1], 'r--', linewidth=1.5, alpha=1, zorder=1)
        plt.title(f"Step {self.step_count}  Explored ratio: {self.explored_rate*100:.4g}%  Travel distance: {self.robot.travel_dist:.4g}m")
        plt.show()

    
    