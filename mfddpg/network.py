import torch
import sys
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .config import *
sys.path.append("..")
from etc import  ALGO_PATH


# 扇入变量初始化，可用于初始化权重参数
def fanin_init(size, fanin=None):
    # fain = fanin or size[0]
    v = 0.06  # 这是一个超参
    return torch.Tensor(size).uniform_(-v, v)


class Critic(nn.Module):
    '''价值网络'''
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.fcs1 = nn.Linear(state_dim, 256)
        self.fcs1.weight.data = fanin_init(self.fcs1.weight.data.size())

        self.fcs2 = nn.Linear(256, 128)
        self.fcs2.weight.data = fanin_init(self.fcs2.weight.data.size())

        self.fca1 = nn.Linear(action_dim, 128)
        self.fca1.weight.data = fanin_init(self.fca1.weight.data.size())

        self.fca2 = nn.Linear(action_dim, 128)
        self.fca2.weight.data = fanin_init(self.fca2.weight.data.size())

        self.fc2 = nn.Linear(384, 128)
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())

        self.fc3 = nn.Linear(128, 1)
        self.fc3.weight.data = fanin_init(self.fc3.weight.data.size())

        self.init_write()

    def init_write(self):
        with open("%s/output/value_in_critic.csv" % ALGO_PATH, "w+") as f:
            f.write("value\n")
        with open("%s/output/loss_critic.csv" % ALGO_PATH, "w+") as f:
            f.write("times,loss\n")

    # 正向传播
    def forward(self, state, action, mean_action):
        s1 = F.relu(self.fcs1(state))
        s2 = F.relu(self.fcs2(s1))
        a1 = F.relu(self.fca1(action))
        a2 = F.relu(self.fca2(mean_action))
        x = torch.cat((s2, a1, a2), dim=1)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # print("value in critic is", x.shape)

        return x


class Actor(nn.Module):
    '''策略网络，四层网络'''
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # 全连接层
        self.fc1 = nn.Linear(state_dim, 256)
        # pylog.info(self.fc1.weight.data.size())
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())

        # 全连接层
        self.fc2 = nn.Linear(256, 128)
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())

        # 全连接层
        self.fc3 = nn.Linear(128, 64)
        self.fc3.weight.data = fanin_init(self.fc3.weight.data.size())

        # 全连接层
        self.fc4 = nn.Linear(64, action_dim)
        self.fc4.weight.data = fanin_init(self.fc4.weight.data.size())
        self.init_write()

    def init_write(self):
        with open("%s/output/action_in_actor.csv" % ALGO_PATH, "w+") as f:
            f.write("action[0], action[1]\n")
        with open("%s/output/loss_actor.csv" % ALGO_PATH, "w+") as f:
            f.write("times,loss\n")

    def forward(self, state):
        '''正向传播'''
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        action = torch.tanh(self.fc4(x))
        # print("action in actor is ", action.shape)
        # with open("%s/output/action_inactor.csv" % ALGO_PATH, "w+") as f:
        #     f.write("times,loss\n")
        return action