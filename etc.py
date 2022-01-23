import torch
import os
"""在地图上随机放置这些物体"""
num_cluster_center = 6
num_agents = 30
num_obstacles = 10
num_neighbor = 4

radius = 36  # 地图大小
target = (0, 0)  # 目标位置

num_epochs = 50  # 一共训练多少回合
max_step = 201  # 每回合一共做多少次决策
MAX_BUFFER = 100000  # buffer size

eta = 0.5
beta = 0.2
alpha = 0.8

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

# 路径定义
app_abspath = os.path.dirname(__file__)
OUTPUT_PATH = "%s\\output" % (app_abspath)  # 多了一层目录
ALGO_PATH = "%s\\mfddpg" % app_abspath  # 模型路径
MODELS_PATH = "%s\\Models\\" % ALGO_PATH  # 模型保存路径

# 状态空间维度: x,y
state_space_dim = 2
# 动作空间维度：x方向上的速度vx,y方向上的速度vy
action_space_dim = 2

distance_to_obs = radius /100  # 如果智能体到障碍物的距离小于这个值，则判定其碰到障碍
distance_to_target = radius / 100  # 如果智能体到目标地点的距离小于这个值，则判定到达目标地点
view_agent = radius / 10  # 智能体的视野

obstacle_penalty = -50  # 障碍物惩罚
dead_penalty = -1000  # 死亡惩罚

epsilon = 3
