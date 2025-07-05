import numpy as np
from sklearn.cluster import KMeans
import heapq

from utils.parameter import *

class MapInfo:
    def __init__(self, map, map_origin_x, map_origin_y):
        self.map = map
        self.map_origin_x = map_origin_x
        self.map_origin_y = map_origin_y
        self.pixel_size = PIXEL_SIZE

    def update_map_info(self, map, map_origin_x, map_origin_y):
        self.map = map
        self.map_origin_x = map_origin_x
        self.map_origin_y = map_origin_y


def get_position_in_map_from_coords(coords:np.ndarray, map_info:MapInfo, check_negative=True):
    '''
    机器人坐标系 -> 地图坐标系
    '''
    single_position = False
    if coords.flatten().shape[0] == 2:
        single_position = True

    coords = coords.reshape(-1, 2)
    coords_x = coords[:, 0]
    coords_y = coords[:, 1]
    position_in_map_x = ((coords_x - map_info.map_origin_x) / map_info.pixel_size)
    position_in_map_y = ((coords_y - map_info.map_origin_y) / map_info.pixel_size)

    position_in_map = np.around(np.stack((position_in_map_x, position_in_map_y), axis=-1)).astype(int)

    if check_negative:
        assert sum(position_in_map.flatten() >= 0) == position_in_map.flatten().shape[0], print(position_in_map, coords, map_info.map_origin_x, map_info.map_origin_y)
    if single_position:
        return position_in_map[0]
    else:
        return position_in_map


def get_coords_from_position_in_map(position_in_map:np.ndarray, map_info:MapInfo):
    '''
    地图坐标系 -> 机器人坐标系
    '''
    position_in_map = position_in_map.reshape(-1, 2)
    position_in_map_x = position_in_map[:, 0]
    position_in_map_y = position_in_map[:, 1]
    coords_x = position_in_map_x * map_info.pixel_size + map_info.map_origin_x
    coords_y = position_in_map_y * map_info.pixel_size + map_info.map_origin_y
    coords = np.stack((coords_x, coords_y), axis=-1)
    coords = np.around(coords, 1)
    if coords.shape[0] == 1:
        return coords[0]
    else:
        return coords


def get_frontier_in_map(map_info:MapInfo)->set[tuple[float,float]]:
    '''
    提取前沿点
        Args: map_info 地图信息
        Returns: frontier_coords 前沿点集合, 机器人坐标系
    '''
    x_len = map_info.map.shape[1]
    y_len = map_info.map.shape[0]
    unknown = (map_info.map == UNKNOWN) * 1
    unknown = np.pad(unknown, ((1, 1), (1, 1)), 'constant', constant_values=0)
    unknown_neighbor = unknown[2:][:, 1:x_len + 1] + unknown[:y_len][:, 1:x_len + 1] + unknown[1:y_len + 1][:, 2:] \
                       + unknown[1:y_len + 1][:, :x_len] + unknown[:y_len][:, 2:] + unknown[2:][:, :x_len] + \
                       unknown[2:][:, 2:] + unknown[:y_len][:, :x_len]
    free_cell_indices = np.where(map_info.map.ravel(order='F') == FREE)[0]
    frontier_cell_1 = np.where(1 < unknown_neighbor.ravel(order='F'))[0]
    frontier_cell_2 = np.where(unknown_neighbor.ravel(order='F') < 8)[0]
    frontier_cell_indices = np.intersect1d(frontier_cell_1, frontier_cell_2)
    frontier_cell_indices = np.intersect1d(free_cell_indices, frontier_cell_indices)

    x = np.linspace(0, x_len - 1, x_len)
    y = np.linspace(0, y_len - 1, y_len)
    t1, t2 = np.meshgrid(x, y)
    cells = np.vstack([t1.T.ravel(), t2.T.ravel()]).T
    frontier_cell = cells[frontier_cell_indices]

    frontier_coords = get_coords_from_position_in_map(frontier_cell, map_info).reshape(-1, 2)
    if frontier_cell.shape[0] > 0 and FRONTIER_CELL_SIZE != PIXEL_SIZE:
        frontier_coords = frontier_coords.reshape(-1 ,2)
        frontier_coords = frontier_down_sample(frontier_coords)
    else:
        frontier_coords = set(map(tuple, frontier_coords))

    return frontier_coords


def frontier_down_sample(data, voxel_size=FRONTIER_CELL_SIZE):
    '''
    下采样
    '''
    voxel_indices = np.array(data / voxel_size, dtype=int).reshape(-1, 2)

    voxel_dict = {}
    for i, point in enumerate(data):
        voxel_index = tuple(voxel_indices[i])

        if voxel_index not in voxel_dict:
            voxel_dict[voxel_index] = point
        else:
            current_point = voxel_dict[voxel_index]
            if np.linalg.norm(point - np.array(voxel_index) * voxel_size) < np.linalg.norm(
                    current_point - np.array(voxel_index) * voxel_size):
                voxel_dict[voxel_index] = point
    downsampled_data = set(map(tuple, voxel_dict.values()))
    return downsampled_data

    
def cluster_frontiers(frontier_coords: set[tuple[np.ndarray, np.ndarray]], 
                      n_clusters=5) -> np.ndarray:
    """
    将前沿点聚类并返回聚类中心
    Args:
        frontier_coords: set of tuples, 前沿点坐标集合
        n_clusters: int, 聚类数量
    Returns:
        cluster_centers: np.ndarray, 聚类中心点坐标
    """
    # 转换数据格式
    if len(frontier_coords) == 0:
        # print("警告: 没有前沿点进行聚类")
        return np.array([]).reshape(0, 2)  # 返回正确形状的空数组
    
    frontier_list = list(frontier_coords)
    original_count = len(frontier_list)
    
    # print(f"原始前沿点数量: {original_count}")
    # 如果前沿点数量少于聚类数，复制前沿点
    if len(frontier_list) < n_clusters:
        # print(f"前沿点数量({len(frontier_list)}) < 聚类数({n_clusters})，开始复制前沿点...")
        needed_points = n_clusters - len(frontier_list)
        # 复制现有前沿点，添加小的随机偏移避免完全重复
        np.random.seed(42)  # 确保可重复性
        for i in range(needed_points):
            # 循环选择要复制的点
            base_point = frontier_list[i % len(frontier_list)]
            # 添加小的随机偏移
            noise_x = np.random.uniform(-0.1, 0.1)  # ±0.1米的随机偏移
            noise_y = np.random.uniform(-0.1, 0.1)
            new_point = (base_point[0] + noise_x, base_point[1] + noise_y)
            frontier_list.append(new_point)
        # print(f"复制后前沿点数量: {len(frontier_list)}")
    # 转换为NumPy数组
    frontier_array = np.array(frontier_list)
    # 如果前沿点数量等于聚类数，直接返回所有点
    if len(frontier_array) == n_clusters:
        # print(f"前沿点数量等于聚类数，直接返回所有点")
        return frontier_array
    
    # 进行K-means聚类
    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(frontier_array)
        
        cluster_centers = kmeans.cluster_centers_
        # print(f"聚类完成: {len(frontier_array)} 个前沿点 → {len(cluster_centers)} 个聚类中心")
        
        return cluster_centers
        
    except Exception as e:
        print(f"聚类失败: {e}")
        # 如果聚类失败，返回前n_clusters个点
        return frontier_array[:n_clusters]

def A_star(start:np.ndarray, 
           goal:np.ndarray, 
           map_info:MapInfo
           )->list[np.ndarray]:
    '''
    实现A*算法
    Args:
        start: np.ndarray, 起点坐标 (机器人坐标系)
        goal: np.ndarray, 终点坐标 (机器人坐标系)
        map_info: MapInfo, 地图信息
    Returns:
        path: list, 路径点列表 [np.array([x1,y1]), np.array([x2,y2]), ...] 如果无路径返回空列表
    '''
    # 确保输入是正确的numpy数组格式
    start = np.array(start).flatten()
    goal = np.array(goal).flatten()
    
    # 转换为网格坐标
    start_coords = start.reshape(1, -1)
    goal_coords = goal.reshape(1, -1)
    
    try:
        start_position_in_map_raw = get_position_in_map_from_coords(start_coords, map_info)
        goal_position_in_map_raw = get_position_in_map_from_coords(goal_coords, map_info)
        
        # 处理可能的标量返回值
        if isinstance(start_position_in_map_raw, np.ndarray):
            if start_position_in_map_raw.ndim == 0:
                # 如果是0维数组（标量），需要特殊处理
                start_position_in_map = np.array([start_position_in_map_raw.item(), 0])  # 这种情况通常不会发生
            elif start_position_in_map_raw.ndim == 1:
                start_position_in_map = start_position_in_map_raw
            else:
                start_position_in_map = start_position_in_map_raw[0]
        else:
            start_position_in_map = np.array(start_position_in_map_raw)
            
        if isinstance(goal_position_in_map_raw, np.ndarray):
            if goal_position_in_map_raw.ndim == 0:
                goal_position_in_map = np.array([goal_position_in_map_raw.item(), 0])
            elif goal_position_in_map_raw.ndim == 1:
                goal_position_in_map = goal_position_in_map_raw
            else:
                goal_position_in_map = goal_position_in_map_raw[0]
        else:
            goal_position_in_map = np.array(goal_position_in_map_raw)
            
        # 确保 start_cell 和 goal_cell 是1D数组且包含2个元素
        start_position_in_map = np.array(start_position_in_map).flatten()
        goal_position_in_map = np.array(goal_position_in_map).flatten()
        
        if len(start_position_in_map) != 2 or len(goal_position_in_map) != 2:
            print(f"坐标转换错误: start_position_in_map={start_position_in_map}, goal_position_in_map={goal_position_in_map}")
            return []
            
    except Exception as e:
        print(f"坐标转换错误: {e}")
        print(f"start: {start}, goal: {goal}")
        return []
    
    # 检查坐标是否在地图范围内
    map_height, map_width = map_info.map.shape
    if (not (0 <= start_position_in_map[0] < map_width and 0 <= start_position_in_map[1] < map_height) or
        not (0 <= goal_position_in_map[0] < map_width and 0 <= goal_position_in_map[1] < map_height)):
        print(f"坐标超出地图范围: start_position_in_map={start_position_in_map}, goal_position_in_map={goal_position_in_map}, map_size=({map_width}, {map_height})")
        return []
    
    # 检查起点和终点是否可通行
    if (map_info.map[start_position_in_map[1], start_position_in_map[0]] != FREE or 
        map_info.map[goal_position_in_map[1], goal_position_in_map[0]] != FREE):
        print(f"起点或终点不可通行: start_value={map_info.map[start_position_in_map[1], start_position_in_map[0]]}, goal_value={map_info.map[goal_position_in_map[1], goal_position_in_map[0]]}")
        return []
    
    def heuristic(position_in_map1, position_in_map2):
        """启发式函数 - 欧几里得距离"""
        return np.sqrt((position_in_map1[0] - position_in_map2[0])**2 + (position_in_map1[1] - position_in_map2[1])**2)
    
    def get_neighbors(position_in_map):
        """获取邻居节点"""
        x, y = position_in_map
        neighbors = []
        # 8方向移动
        directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            
            # 检查边界
            if 0 <= nx < map_width and 0 <= ny < map_height:
                # 检查是否可通行
                if map_info.map[ny, nx] == FREE:
                    # 计算移动成本 (对角线移动成本更高)
                    cost = np.sqrt(dx**2 + dy**2)
                    neighbors.append(((nx, ny), cost))
        
        return neighbors
    
    def reconstruct_path(came_from, current):
        """重构路径"""
        path_cells = []
        while current in came_from:
            path_cells.append(current)
            current = came_from[current]
        path_cells.append(current)  # 添加起点
        
        # 反转路径（从起点到终点）
        path_cells.reverse()
        
        # 转换网格坐标回机器人坐标系
        path = []
        for position_in_map in path_cells:
            position_in_map_coords = np.array(position_in_map).reshape(1, -1)
            real_coords = get_coords_from_position_in_map(position_in_map_coords, map_info)
            
            # 处理get_coords_from_cell_position的返回值
            if isinstance(real_coords, np.ndarray):
                if real_coords.ndim == 1:
                    path.append(real_coords)
                else:
                    path.append(real_coords.flatten())
            else:
                path.append(np.array(real_coords))
        
        return path
    
    # A*算法主体
    start_tuple = tuple(start_position_in_map.astype(int))
    goal_tuple = tuple(goal_position_in_map.astype(int))
    
    # 如果起点就是终点
    if start_tuple == goal_tuple:
        return [start, goal]
    
    # 优先队列: (f_score, current_cell)
    open_set = [(heuristic(start_position_in_map, goal_position_in_map), start_tuple)]
    
    # 记录已访问的节点
    closed_set = set()
    
    # 记录从起点到各节点的最短距离
    g_score = {start_tuple: 0.0}
    
    # 记录路径: 当前节点 -> 前一个节点
    came_from = {}
    
    while open_set:
        # 取出f值最小的节点
        current_f, current_position_in_map = heapq.heappop(open_set)
        
        # 如果已经访问过这个节点，跳过
        if current_position_in_map in closed_set:
            continue
            
        # 标记为已访问
        closed_set.add(current_position_in_map)
        
        # 如果到达目标
        if current_position_in_map == goal_tuple:
            # 重构并返回路径
            return reconstruct_path(came_from, current_position_in_map)
        
        # 探索邻居节点
        for neighbor_position_in_map, move_cost in get_neighbors(current_position_in_map):
            neighbor_tuple = tuple(neighbor_position_in_map)
            
            # 如果已经访问过，跳过
            if neighbor_tuple in closed_set:
                continue
            
            # 计算通过当前节点到邻居的距离
            tentative_g = g_score[current_position_in_map] + move_cost
            
            # 如果找到更短的路径，或者是第一次访问这个邻居
            if neighbor_tuple not in g_score or tentative_g < g_score[neighbor_tuple]:
                came_from[neighbor_tuple] = current_position_in_map
                g_score[neighbor_tuple] = tentative_g
                f_score = tentative_g + heuristic(neighbor_position_in_map, goal_position_in_map)
                heapq.heappush(open_set, (f_score, neighbor_tuple))
    # 无法找到路径
    return []


def get_path_length(path:list[np.ndarray])->float:
    """
    计算路径长度
    Args:
        path: list, 路径点列表 [np.array([x1,y1]), np.array([x2,y2]), ...]
    Returns:
        length: float, 路径总长度 (米)
    """
    if len(path) < 2:
        return 0.0
    
    total_length = 0.0
    for i in range(1, len(path)):
        segment_length = np.linalg.norm(
            np.array(path[i]) - np.array(path[i-1])
        )
        total_length += segment_length
    
    return total_length

def is_position_accessible(position, map_info:MapInfo):
        """
        检查位置是否可通行
        Args:
            position: np.array 或 tuple, 位置坐标 (机器人坐标系)
        Returns:
            bool: True如果可通行，False如果不可通行
        """
        try:
            # 确保输入是numpy数组
            if isinstance(position, (list, tuple)):
                position = np.array(position)
            
            # 转换为网格坐标
            position_in_map_raw = get_position_in_map_from_coords(
                position.reshape(1, -1), map_info
            )
            
            # 处理可能的标量返回值
            if isinstance(position_in_map_raw, np.ndarray):
                if position_in_map_raw.ndim == 0:
                    # 0维数组，无法索引
                    # print(f"位置 {position} 转换结果是标量: {cell_position_raw}")
                    return False
                elif position_in_map_raw.ndim == 1:
                    position_in_map = position_in_map_raw
                else:
                    position_in_map = position_in_map_raw[0]
            else:
                position_in_map = np.array(position_in_map_raw)
            
            # 确保是1维数组且有2个元素
            position_in_map = np.array(position_in_map).flatten()
            if len(position_in_map) != 2:
                # print(f"位置 {position} 转换结果格式错误: {position_in_map}")
                return False
            
            # 检查是否在地图范围内
            map_height, map_width = map_info.map.shape
            if not (0 <= position_in_map[0] < map_width and 0 <= position_in_map[1] < map_height):
                return False
            
            # 检查是否是自由空间
            position_value = map_info.map[position_in_map[1], position_in_map[0]]
            return position_value == FREE
            
        except Exception as e:
            # print(f"位置 {position} 不在可通行区域: {e}")
            return False
        
def find_nearest_accessible_position_spiral(center_coords, map_info:MapInfo, max_search_radius=3.0):
        """
        使用螺旋搜索找到距离指定中心点最近的可通行位置
        Args:
            center_coords: np.array 聚类中心坐标
            max_search_radius: float, 最大搜索半径 (米)
        Returns:
            np.array: 最近的可通行位置坐标，如果没有则返回None
        """
        # 确保输入是numpy数组
        if isinstance(center_coords, (list, tuple)):
            center_coords = np.array(center_coords)
        
        # 转换中心点为网格坐标
        try:
            center_position_in_map_raw = get_position_in_map_from_coords(
                center_coords.reshape(1, -1), map_info
            )
            
            # 处理可能的标量返回值
            if isinstance(center_position_in_map_raw, np.ndarray):
                if center_position_in_map_raw.ndim == 0:
                    print(f"中心点 {center_coords} 转换结果是标量")
                    return None
                elif center_position_in_map_raw.ndim == 1:
                    center_position_in_map = center_position_in_map_raw
                else:
                    center_position_in_map = center_position_in_map_raw[0]
            else:
                center_position_in_map = np.array(center_position_in_map_raw)
            
            # 确保是正确格式
            center_position_in_map = np.array(center_position_in_map).flatten()
            if len(center_position_in_map) != 2:
                print(f"中心点 {center_coords} 转换格式错误: {center_position_in_map}")
                return None
            
        except Exception as e:
            print(f"中心点坐标转换失败: {e}")
            return None
        
        # 检查是否在地图范围内
        map_height, map_width = map_info.map.shape
        if not (0 <= center_position_in_map[0] < map_width and 0 <= center_position_in_map[1] < map_height):
            print(f"中心点超出地图范围: {center_position_in_map}, 地图大小: {map_width}x{map_height}")
            return None
        
        # 最大搜索半径对应的网格数
        max_radius_cells = int(max_search_radius / map_info.pixel_size) + 1
        
        # 螺旋搜索：从中心开始，逐渐扩大搜索半径
        for radius in range(max_radius_cells + 1):
            # 在当前半径的所有位置中搜索
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    # 只搜索当前半径边界上的点（避免重复搜索内部点）
                    if abs(dx) != radius and abs(dy) != radius and radius > 0:
                        continue
                    
                    candidate_position_in_map = np.array([center_position_in_map[0] + dx, center_position_in_map[1] + dy])
                    
                    # 检查边界
                    if not (0 <= candidate_position_in_map[0] < map_width and 0 <= candidate_position_in_map[1] < map_height):
                        continue
                    
                    # 检查是否可通行
                    if map_info.map[candidate_position_in_map[1], candidate_position_in_map[0]] == FREE:
                        # 转换回实际坐标
                        candidate_coords = get_coords_from_position_in_map(
                            candidate_position_in_map.reshape(1, -1), map_info
                        )
                        
                        # 处理返回值
                        if isinstance(candidate_coords, np.ndarray):
                            if candidate_coords.ndim == 1:
                                result_coords = candidate_coords
                            else:
                                result_coords = candidate_coords.flatten()
                        else:
                            result_coords = np.array(candidate_coords)
                        
                        # 检查实际距离是否在搜索半径内
                        distance = np.linalg.norm(center_coords - result_coords)
                        if distance <= max_search_radius:
                            return result_coords
        return None

    