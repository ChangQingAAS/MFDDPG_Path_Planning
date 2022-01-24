from __future__ import division

import sys
import random
import datetime
import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

from . import update
from . import network
from .config import *
from . import OUnoise

sys.path.append("..")
from etc import ALGO_PATH
from . import buffer


class Trainer:
    '''训练器'''

    def __init__(self, state_dim, action_dim, dev, epoch=0, model_save_path=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.ram = buffer.MemoryBuffer(MAX_BUFFER)
        self.noise = OUnoise.OrnsteinUhlenbeckActionNoise(self.action_dim)
        self.device = dev

        # 策略网络（在线和目标）及其优化器
        self.actor = network.Actor(self.state_dim, self.action_dim).to(self.device)
        self.target_actor = network.Actor(self.state_dim, self.action_dim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), LEARNING_RATE)

        # 价值网络（在线和目标）及其优化器
        self.critic = network.Critic(self.state_dim, self.action_dim).to(self.device)
        self.target_critic = network.Critic(self.state_dim, self.action_dim).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), LEARNING_RATE)

        # 如果当前轮数大于0，则表示以前训练过，则载入训练过的模型
        if epoch > 0:
            print("load models")
            self.load_models(epoch, model_save_path)

        # 用在线网络更新目标网络
        update.hard_update(self.target_actor, self.actor)
        update.hard_update(self.target_critic, self.critic)

    def get_exploitation_action(self, state):
        '''利用模型产生动作'''
        state = Variable(torch.from_numpy(state).to(self.device))
        action = self.target_actor.forward(state).detach()
        return action.data.cpu().numpy()

    def get_exploration_action(self, state):
        '''随机探索产生动作'''
        state = Variable(torch.from_numpy(state).to(self.device))
        action = self.actor.forward(state).detach()
        new_action = action.data.cpu().numpy() + self.noise.sample()
        return new_action

    # 用深度强化学习算法得到智能体动作
    def get_action_from_drl_algo(self, x, y):
        state = [x, y]
        state = np.float32(state)  # 强制转化为浮点类型

        # epsilon的动作随机生成，其他的动作由模型生成
        number = random.randint(0, 1)
        if number < EPSILON:
            action = self.get_exploration_action(state)  # 通过模型来生成动作，利用
        else:
            action = self.get_exploitation_action(state)  # 随机生成动作
        with open("%s/output/action.csv" % ALGO_PATH, "a+") as f:
            # f.write("action from mfddpg is [%f,%f]" % (action[0], action[1]))
            f.write("%f,%f\n" % (action[0], action[1]))
        # print("action from mfddpg is ", action)
        return action

    # 根据动作执行结果，训练一次智能体
    def train_(self, train_step, state_last, action_last, reward_now, state_new, cur_step, mean_action):
        # 添加最新经验，并优化训练一把，再做决策
        self.ram.add(state_last, action_last, reward_now, state_new, mean_action)
        self.optimize(cur_step)
        # print("step in mfddpg is ", cur_step)
        train_step += 1

    def optimize(self, step):
        '''优化'''
        s1, a1, r1, s2, a_ = self.ram.sample(BATCH_SIZE)
        s1 = Variable(torch.from_numpy(s1).to(self.device))
        a1 = Variable(torch.from_numpy(a1).to(self.device))
        r1 = Variable(torch.from_numpy(r1).to(self.device))
        s2 = Variable(torch.from_numpy(s2).to(self.device))
        a_ = Variable(torch.from_numpy(a_).to(self.device))

        a2 = self.target_actor.forward(s2).detach()  # 根据S2得出的下一个动作a2
        # print("a2 is ", a2.size())
        next_val = torch.squeeze(self.target_critic.forward(s2, a2, a_).detach())  # 根据s2和a2得出的目标值
        # print("next_val is ", next_val.size())
        y_expectd = r1 + GAMMA * next_val  # 目标回报值
        # print("y_expected is ", y_expectd.size())

        y_predicted = torch.squeeze(self.critic.forward(s1, a1, a_))  # 主价值网络得到的回报值
        # print("y_predicted is ", y_predicted.size())
        loss_critic = F.smooth_l1_loss(y_predicted, y_expectd)  # 价值损失（回报值的损失）
        self.critic_optimizer.zero_grad()  # 优化器参数置为零
        with open("%s/output/loss_critic.csv" % ALGO_PATH, "a+") as f:
            f.write("%s,%s\n" % (step, loss_critic.item()))  # 将损失值保存到文件中
        loss_critic.backward()  # 反向传播
        self.critic_optimizer.step()  # 修改价值网络参数

        pred_a1 = self.actor.forward(s1)
        loss_actor = -1 * torch.sum(self.critic.forward(s1, pred_a1, a_))  # 决策损失（回报值的损失）
        self.actor_optimizer.zero_grad()
        with open("%s/output/loss_actor.csv" % ALGO_PATH, "a+") as f:
            f.write("%s,%s\n" % (step, loss_actor.item()))  # 将损失值保存到文件中
        loss_actor.backward()  # 反向传播
        self.actor_optimizer.step()  # 修改策略网络参数

        update.soft_update(self.target_actor, self.actor, TAU)
        update.soft_update(self.target_critic, self.critic, TAU)

    def get_models_path(self):
        return "./Models/"

    def save_model(self, episode_count, model_save_path=None):
        '''保存模型'''
        # torch.save(self.target_actor.state_dict(), './Models/' + str(episode_count) + '_actor.pt')
        if model_save_path == None:
            model_save_path = self.get_models_path()
        torch.save(self.target_actor.state_dict(), model_save_path + str(episode_count) + '_actor.pt')
        torch.save(self.target_critic.state_dict(), model_save_path + str(episode_count) + '_critic.pt')
        print("%s：%s Models saved successfully" % (datetime.datetime.now(), episode_count))
        # print("%s：轮数:%s 决策步数:%s  Reward:%.2f" %  (datetime.now(), episode_count, step, reward_now))

    def load_models(self, episode, model_save_path=None):
        '''载入以前训练过的模型, 包括策略网络和价值网络'''
        if model_save_path == None:
            model_save_path = self.get_models_path()
        # self.critic.load_state_dict(torch.load(self.get_models_path() + str(episode) + '_critic.pt'))
        self.critic.load_state_dict(torch.load(model_save_path + str(episode) + '_critic.pt'))
        # self.actor.load_state_dict(torch.load(self.get_models_path() + str(episode) + '_actor.pt'))
        self.actor.load_state_dict(torch.load(model_save_path + str(episode) + '_actor.pt'))
        update.hard_update(self.target_actor, self.actor)
        update.hard_update(self.target_critic, self.critic)
        print("Models loaded successfully")
