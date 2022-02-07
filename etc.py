import torch
import os
"""在地图上随机放置这些物体"""
num_cluster_center = 6
num_agents = 50
num_obstacles = 20

radius = 4  # 地图大小
target = (0, 0)  # 目标位置

num_epochs = 50  # 一共训练多少回合
max_step = radius * 3 + 1  # 每回合一共做多少次决策

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

# 路径定义
app_abspath = os.path.dirname(__file__)
OUTPUT_PATH = "%s/output" % (app_abspath)  # 多了一层目录
ALGO_PATH = "%s/mfddpg" % app_abspath  # 模型路径
MODELS_PATH = "%s/models/" % ALGO_PATH  # 模型保存路径

# 状态空间维度: x,y
state_space_dim = 2
# 动作空间维度：x方向上的速度v_x,y方向上的速度v_y
action_space_dim = 2

distance_to_obs = 0.005  # 如果智能体到障碍物的距离小于这个值，则判定其碰到障碍
distance_to_target = 0.5  # 如果智能体到目标地点的距离小于这个值，则判定到达目标地点
view_agent = 0.1  # 智能体的视野

obstacle_penalty = -radius  # 障碍物惩罚
dead_penalty = -radius * 3  # 死亡惩罚