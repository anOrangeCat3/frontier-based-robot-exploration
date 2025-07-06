# map representation
FREE = 255  # value of free cells in the map
OCCUPIED = 1  # value of obstacle cells in the map
UNKNOWN = 127  # value of unknown cells in the map

# map and planning resolution
PIXEL_SIZE = 0.4  # meter, your map resolution
NODE_RESOLUTION = 4.0  # meter, your node resolution
FRONTIER_CELL_SIZE = 2 * PIXEL_SIZE  # do you want to downsample the frontiers

# sensor and utility range
SENSOR_RANGE = 16  # meter

# 探索参数
MAX_EPISODE_STEP = 32
EXPLORATION_RATE_THRESHOLD = 0.99  # 探索率阈值

# agent parameters
N_CLUSTERS = 16

# network parameters
EMBEDDING_DIM = 64
NUM_HEADS = 8
NUM_LAYERS = 4

# reward parameters
BASE_EXPLORATION_REWARD_WEIGHT = 3
SINGLE_STEP_EXPLORATION_REWARD_WEIGHT = 180
LONG_TERM_EXPLORATION_REWARD_WEIGHT = 300
FINISH_EXPLORATION_REWARD = 20
NOT_FINISH_EXPLORATION_PENALTY = -50

# training parameters
LEARNING_RATE = 1e-5
REPLAY_BUFFER_CAPACITY = 10000
BATCH_SIZE = 256
TRAIN_EPISODES = 50000
WARMUP_STEPS = 2000
UPDATE_INTERVAL = 4
GAMMA = 0.99 # 探索任务，所有步骤的奖励权重都是1
TAU = 0.005 # 软更新目标网络参数
