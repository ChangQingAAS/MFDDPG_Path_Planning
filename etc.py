import torch
import os
"""在地图上随机放置这些物体"""
num_cluster_center = 6
num_agents = 100
num_obstacles = 300
num_neighbor = 4

radius = 360  # 地图大小
target = (360, 360)  # 目标位置

num_epochs = 101  # 一共训练多少回合
max_step = 2001  # 每回合一共做多少次决策
MAX_BUFFER = 10000  # buffer size

eta = 0.5
beta = 0.2
alpha = 0.8

# USE_CUDA = torch.cuda.is_available()
USE_CUDA = False
device = torch.device("cuda" if USE_CUDA else "cpu")

# 路径定义
app_abspath = os.path.dirname(__file__)
TMP_PATH = "%s/tmp" % (app_abspath)
OUTPUT_PATH = "%s/output" % (app_abspath)  # 多了一层目录
MODELS_PATH = "%s/Models/" % OUTPUT_PATH  # 模型输出路径

# 状态空间维度: x,y
state_space_dim = 2
# 动作空间维度：x方向上的速度vx,y方向上的速度vy
action_space_dim = 2
# 设定速度的最大值
action_max = 1
