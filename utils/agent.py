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
        self.travel_dist = 0

    def reset(self,
              ground_truth_size:np.ndarray,
              belief_origin_x:np.ndarray,
              belief_origin_y:np.ndarray):
        belief_map = np.ones(ground_truth_size) * 127
        self.belief_map_info = MapInfo(belief_map, belief_origin_x, belief_origin_y)
        self.position = np.array([0.0, 0.0])
        self.travel_dist = 0
        self.trajectory = []
        self.waypoint = []
        self.waypoint.append(np.array([0.0, 0.0]))

    def get_obs(self):
        '''对于greedy agent, obs为frontier_cluster_centers'''
        return self.frontier_cluster_centers

    def get_action(self):
        '''对于greedy agent, action为选出最近路程的frontier_cluster_center'''
        self.get_obs()
        best_waypoint = None
        min_distance = float('inf')
        for center in self.frontier_cluster_centers:
            path = A_star(self.position, center, self.belief_map_info)
            if path:  # 选出最近路程的frontier_cluster_center
                path_length = get_path_length(path)
                if path_length < min_distance:
                    min_distance = path_length
                    best_waypoint = center
                    best_path = np.array(path)
        self.waypoint.append(best_waypoint)
        self.trajectory.append(best_path)
        self.travel_dist += min_distance 

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
    def __init__(self, policy_net):
        super().__init__()
        self.policy_net = policy_net
    
    def get_obs(self):
        '''
        obs = [归一化后的frontier_cluster_centers, 归一化后的waypoint, mask]
        '''
        pass


    def get_action(self, obs):
        pass
