import numpy as np
import torch
from utils.parameter import *
from utils.utils import *
from utils.sensor import sensor_work

class Agent:
    def __init__(self):
        '''
        Attributes:
            global_frontier: set
            frontier_cluster_centers: np.ndarray
            trajectory: list[np.ndarray]
            waypoint: list[np.ndarray]
            travel_dist: float
        '''
        self.scan_range = SENSOR_RANGE/PIXEL_SIZE

        self.position = np.array([])
        self.belief_map_info = None
        
        self.global_frontier = set()
        self.frontier_cluster_centers = np.array([])
        self.trajectory = []
        self.waypoint = []
        self.travel_dist = []
        self.dist = 0

    def reset(self,
              ground_truth_size:np.ndarray,
              belief_origin_x:np.ndarray,
              belief_origin_y:np.ndarray):
        belief_map = np.ones(ground_truth_size) * 127
        self.belief_map_info = MapInfo(belief_map, belief_origin_x, belief_origin_y)
        self.position = np.array([0.0, 0.0])
        self.travel_dist = []
        self.dist = 0
        self.trajectory = []
        self.waypoint = []
        self.waypoint.append(np.array([0.0, 0.0]))

    def get_obs(self):
        '''对于greedy agent, obs为frontier_cluster_centers'''
        return self.frontier_cluster_centers

    def get_action(self,obs):
        '''对于greedy agent, action为选出最近路程的frontier_cluster_center'''
        best_waypoint = None
        min_distance = float('inf')
        for center in obs:
            path = A_star(self.position, center, self.belief_map_info)
            if path:  # 选出最近路程的frontier_cluster_center
                path_length = get_path_length(path)
                if path_length < min_distance:
                    min_distance = path_length
                    # best_path = np.array(path)
                    best_waypoint = center
        return best_waypoint

    def move(self, target_point:np.ndarray):
        '''
        将目标点添加到waypoint中
        A*得到路径
        计算路径长度添加到travel_dist中
        移动到目标点
        '''
        self.waypoint.append(target_point)
        path = A_star(self.position, target_point, self.belief_map_info)
        self.trajectory.append(path)
        dist = get_path_length(path)
        self.travel_dist.append(dist)
        self.dist += dist

    def update_robot_position(self):
        '''更新机器人位置, 将机器人位置更新为最新的waypoint(waypoint[-1])'''
        self.position = self.waypoint[-1]

    def update_belief_map(self,
                          robot_position_in_map:np.ndarray,
                          ground_truth:np.ndarray):
        self.belief_map_info.map = sensor_work(robot_position_in_map, 
                                      self.scan_range, 
                                      self.belief_map_info.map, 
                                      ground_truth)

    def update_frontier(self):
        '''更新frontier'''
        self.global_frontier = get_frontier_in_map(self.belief_map_info)
        frontier_cluster_centers = cluster_frontiers(self.global_frontier, n_clusters=N_CLUSTERS)

        valid_centers = []
        for center in frontier_cluster_centers:
            if not is_position_accessible(center, self.belief_map_info):
                nearest_accessible = find_nearest_accessible_position_spiral(center, self.belief_map_info)
                if nearest_accessible is not None:
                    valid_centers.append(nearest_accessible)
            else:
                valid_centers.append(center)
        self.frontier_cluster_centers = np.array(valid_centers)


class FrontierSACAgent(Agent):
    def __init__(self):
        super().__init__()

    def get_obs(self):
        '''
        obs = [归一化后的frontier_cluster_centers, 归一化后的waypoint, mask]
        '''
        norm_frontiers = self.normalize_frontiers(self.frontier_cluster_centers)
        norm_waypoint = self.normalize_waypoint(self.waypoint)
        mask = self.get_mask(self.waypoint)
        obs = [norm_frontiers, norm_waypoint, mask]
        return obs

    def normalize_frontiers(self, 
                            frontier_cluster_centers:np.ndarray)->np.ndarray:
        '''
        以当前机器人位置为原点
        '''
            # 处理空的前沿点数组
        if frontier_cluster_centers.size == 0:
            return frontier_cluster_centers  # 直接返回空数组
        norm_frontiers = frontier_cluster_centers - self.position
        return norm_frontiers
    
    def normalize_waypoint(self, 
                           waypoint:list)->np.ndarray:
        '''
        以当前机器人位置为原点
        然后填充到MAX_EPISODE_STEP
        '''
        norm_waypoint = np.zeros((MAX_EPISODE_STEP, 2))
        for i, way in enumerate(waypoint):
            norm_waypoint[i] = way - self.position
        return norm_waypoint
    
    def get_mask(self, 
                 waypoint:list)->np.ndarray:
        '''
        长度为MAX_EPISODE_STEP的bool
        其中前len(waypoint)为True, 其余为False
        '''
        mask = np.zeros(MAX_EPISODE_STEP, dtype=bool)
        mask[:len(waypoint)] = True
        return mask
    