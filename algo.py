from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
import copy
import random

from kmeans_plus_plus import get_all
from RRT import get_action_by_RTT
from utils import *
from draw import *
from etc import *
from pic import write_loss
from file_utils import file

# 调用算法：MFDDPG
from mfddpg import buffer
from mfddpg import train


# 执行该动作是否会碰到障碍物
def is_hit_obstacle(agent, action):
    next_x = agent.x + action[0]
    next_y = agent.y + action[1]
    for item in CenterAgents[agent.group_id].obstacle_set:
        if int(next_x) - distance_to_obs <= item.x <= int(
                next_x) + distance_to_obs and int(
                    next_y) - distance_to_obs <= item.y <= int(
                        next_y) + distance_to_obs:
            return True

    return False


def get_obstacles_within_observation(agent, CenterAgent):
    for obstacle in obstacles:
        # 局部感知： 智能体只能看到部分
        if obstacle.x <= int(agent.x) + view_agent and obstacle.y <= int(
                agent.y) + view_agent:
            # print("info: \n        agent_%s find obstacle_%s " % (agent.id, obstacle.id))
            CenterAgent.obstacle_set.add(obstacle)


def get_action_from_all_clusters(agent, CenterAgents, action_from_algo):
    action_from_others_cluster = [0, 0]

    for item in CenterAgents:
        action_from_others_cluster[0] += item.average_action[0] / len(
            CenterAgents)
        action_from_others_cluster[1] += item.average_action[1] / len(
            CenterAgents)
    # 现加全部的，再把自己的减去
    action_from_others_cluster[0] -= CenterAgents[
        agent.group_id].average_action[0] / len(CenterAgents)
    action_from_others_cluster[1] -= CenterAgents[
        agent.group_id].average_action[1] / len(CenterAgents)

    action_from_its_cluster = CenterAgents[agent.group_id].average_action

    action[0] = beta * action_from_algo[0] + alpha * action_from_its_cluster[
        0] + (1 - beta - alpha) * action_from_others_cluster[0]
    action[1] = beta * action_from_algo[1] + alpha * action_from_its_cluster[
        1] + (1 - beta - alpha) * action_from_others_cluster[1]

    return action


def get_reward(centerAgent):
    reward = 0
    obstacle_reward = 0
    live_reward = 0
    # 接近目标的奖励
    distance_reward = -sqrt(
        pow((centerAgent.x - target[0]), 2) +
        pow((centerAgent.y - target[1]), 2))

    # 障碍物的惩罚
    for item in centerAgent.obstacle_set:
        if int(centerAgent.x) - 1 <= item.x <= int(centerAgent.x) + 1 and int(
                centerAgent.y) - 1 <= item.y <= int(centerAgent.y) + 1:
            obstacle_reward += obstacle_penalty

    if centerAgent.moving == False:
        live_reward = dead_penalty

    reward = distance_reward + obstacle_reward + live_reward
    return reward


# 用深度强化学习算法得到智能体动作
def get_action_from_drl_algo(trainer, x, y):
    state = [x, y]
    state = np.float32(state)  # 强制转化为浮点类型

    # 1/epilson的动作随机生成，其他的动作由模型生成
    temp = 1 / epilson
    number = random.randint(1, temp)
    if number % temp == 0:
        action = trainer.get_exploration_action(state)  # 通过模型来生成动作，利用
    else:
        action = trainer.get_exploitation_action(state)  # 随机生成动作。
    with open("./output/action.txt", "a+") as f:
        f.write("action from mfddpg is [%f,%f]" % (action[0], action[1]))
        f.write("\n")
    # print("action from mfddpg is ", action)
    return action


# 根据动作执行结果，训练一次智能体
def train_(trainer, ram, train_step, state_last, action_last, reward_now,
           state_new, cur_step, mean_action):
    # 添加最新经验，并优化训练一把，再做决策
    ram.add(state_last, action_last, reward_now, state_new, mean_action)
    trainer.optimize(cur_step)
    train_step += 1


# def get_mean_action_from_neigh(agents, id):
#     mean_action = [0, 0]
#     count = 1
#     if id < num_agents:
#         mean_action[0] = agents[id].action[0]
#         mean_action[1] = agents[id].action[1]
#     else:
#         return mean_action

#     for i in range(num_neighbor):
#         id += 1
#         if id < num_agents:
#             mean_action[0] = agents[id].action[0]
#             mean_action[1] = agents[id].action[1]
#             count += 1
#         else:
#             break

#     mean_action[0] /= count
#     mean_action[1] /= count
#     return mean_action

# 创建输出文件类，并进行文件初始化
out_file = file()
start_epoch, start_step, current_step = out_file.initialize()
print("start epoch: %s" % start_epoch)
print("start step: %s" % current_step)

# 初始化重放缓冲区
ram = buffer.MemoryBuffer(MAX_BUFFER)
# 初始化训练model
trainer = train.Trainer(
    state_space_dim,  # 状态空间维度： x,y
    # 这个参量在这里只有定义没有使用，是为了传给mfddpg的trainer加OU噪声，防止过拟合
    action_space_dim,
    # 在mfddpg里的actor，由动作空间大小算出来的OU噪声乘以action_max(在实际应用中，是为了对action进行缩放)
    action_max,
    ram,
    device,
    # None, 把loss写入文件的函数
    write_loss,
    start_epoch,
    MODELS_PATH)

# 当前决策步数
train_step = 0

# 初始化实体
agents, CenterAgents, obstacles = get_all()
show(agents, CenterAgents, obstacles, "./picture/init.png")

# 保存初始化的环境,以便于进行对环境reset
first_agents = []
for item in agents:
    first_agents.append(copy.copy(item))
first_center_agents = []
for item in CenterAgents:
    first_center_agents.append(copy.copy(item))

# 判断在运行算法之前，死掉了多少个智能体
before_dead = 0
for agent in agents:
    # 碰到障碍物
    for obstacle in obstacles:
        if int(agent.x) - distance_to_obs <= obstacle.x <= int(
                agent.x) + distance_to_obs and int(
                    agent.y) - distance_to_obs <= obstacle.y <= int(
                        agent.y) + distance_to_obs:
            print("before algo:\n")
            print("     agent %s stop because hit an obstacle" % agent.id)
            before_dead += 1
            agent.moving = False
# 把agent死亡数目放入info中
with open("./output/result.txt", "w") as f:
    f.write("before_algo, there are %s agents have hit obstacles" %
            before_dead)
    f.write("\n")
    f.write("\n")

# 画图
fig, ax = plt.subplots()

# algorithm
for epoch in range(num_epochs):

    # init: copy from first
    agents = []
    for item in first_agents:
        agents.append(copy.copy(item))
    CenterAgents = []
    for item in first_center_agents:
        CenterAgents.append(copy.copy(item))

    num_reach_target = 0
    num_hit_obs = 0
    num_out_of_range = 0

    for step in range(start_epoch, max_step):

        # 每一百步数，画图
        if step % 100 == 0:
            title = "epoch: " + str(epoch) + " step: " + str(step)
            draw_gif(ax, agents, CenterAgents, obstacles, title)

        for agent in agents:
            if agent.moving:
                # action_from_algo = get_action_from_drl_algo(
                #     trainer, agent.x, agent.y)
                # if not is_hit_obstacle(agent, action_from_algo):
                #     action = action_from_algo
                #     agent.take_action(action)
                # else:
                random_number = random.uniform(0, 1)
                if (random_number <= eta):
                    action = get_action_by_RTT(
                        agent, target,
                        CenterAgents[agent.group_id].obstacle_set)
                    # with open('./output/result.txt', 'a+') as f:
                    #     f.write( "agent_id_%s get action [%s, %s] from RRT\n" %(agent.id, action[0], action[1]))
                    agent.take_action(action)
                else:
                    action = get_action_from_all_clusters(
                        agent, CenterAgents, action_from_algo)
                    agent.take_action(action)

                # 增量求cluster平均动作
                CenterAgents[agent.group_id].average_action_num += 1
                CenterAgents[agent.group_id].average_action[0] += (
                    action[0] - CenterAgents[agent.group_id].average_action[0]
                ) / CenterAgents[agent.group_id].average_action_num
                CenterAgents[agent.group_id].average_action[1] += (
                    action[1] - CenterAgents[agent.group_id].average_action[1]
                ) / CenterAgents[agent.group_id].average_action_num

                # 观察障碍物，存入center_agent
                get_obstacles_within_observation(agent,
                                                 CenterAgents[agent.group_id])

                # 到达目标
                if agent.x in range(target[0] - distance_to_target, target[0] +
                                    distance_to_target) and agent.y in range(
                                        target[1] - distance_to_target,
                                        target[1] + distance_to_target):
                    # print("agent %s stop because Reach target" %agent.id)
                    num_reach_target += 1
                    agent.moving = False

                # 超出界限
                if abs(int(agent.x)) > radius or abs(int(agent.y)) > radius:
                    # print("agent %s stop because out of boundary"%agent.id)
                    num_out_of_range += 1
                    agent.moving = False

                # 碰到障碍物
                for obstacle in obstacles:
                    if agent.x <= obstacle.x + distance_to_obs and agent.y <= obstacle.y + distance_to_obs:
                        # print("agent %s stop because hit an obstacle" % agent.id)
                        num_hit_obs += 1
                        agent.moving = False

        for centerAgent in CenterAgents:
            if centerAgent.moving:
                action_from_algo = get_action_from_drl_algo(
                    trainer, centerAgent.x, centerAgent.y)
                if not is_hit_obstacle(centerAgent, action_from_algo):
                    action = action_from_algo
                    state_last = [centerAgent.x, centerAgent.y]
                    action_last = action
                    centerAgent.take_action(action)
                    reward_now = get_reward(centerAgent)
                    state_new = [centerAgent.x, centerAgent.y]
                    mean_action = centerAgent.average_action
                    train_(trainer, ram, train_step,
                           np.float32(state_last), action_last, reward_now,
                           np.float32(state_new), step, mean_action)
                    out_file.write_step(step)
                else:
                    random_number = random.uniform(0, 1)
                    if (random_number < eta):
                        action = get_action_by_RTT(centerAgent, target,
                                                   centerAgent.obstacle_set)
                        # with open('./output/result.txt', 'a+') as f:
                        #     f.write( "centerAgent_id_%s get action [%s, %s] from RRT\n" % (centerAgent.id, action[0], action[1]))
                        centerAgent.take_action(action)
                    else:
                        action = CenterAgents[
                            centerAgent.group_id].average_action
                        centerAgent.take_action(action)

                # 观察障碍物，存入center_agent
                get_obstacles_within_observation(centerAgent,
                                                 CenterAgents[agent.group_id])
                # 到达目标
                if target[0] - distance_to_target <= int(
                        centerAgent.x
                ) <= target[0] + distance_to_target and target[
                        1] - distance_to_target <= int(
                            centerAgent.y) <= target[1] + distance_to_target:
                    # print("agent %s stop because Reach target" %agent.id)
                    num_reach_target += 1
                    centerAgent.moving = False

                # 超出界限
                if abs(int(centerAgent.x)) > radius or abs(int(
                        centerAgent.y)) > radius:
                    # print("agent %s stop because out of boundary"%agent.id)
                    num_out_of_range += 1
                    centerAgent.moving = False

                # 碰到障碍物：
                for obstacle in obstacles:
                    if centerAgent.x <= obstacle.x + distance_to_obs and centerAgent.y <= obstacle.y + distance_to_obs:
                        # print("agent %s stop because hit an obstacle" % agent.id)
                        num_hit_obs += 1
                        centerAgent.moving = False

    # 保存一下已经训练的轮数
    trainer.save_model(epoch, MODELS_PATH)
    out_file.write_epoch(epoch)

    # TODO: 非集中式通信
    # 中心Agent交换各自发现的障碍物
    all_found_obstacles = set()
    obs_set = get_all_obstacles(obstacles)
    # 得到所有障碍物
    for item in CenterAgents:
        all_found_obstacles |= item.obstacle_set

    # 更新每个中心agent的障碍物集合和平均动作
    for item in CenterAgents:
        item.obstacle_set = all_found_obstacles
        item.average_action = [0, 0]
        item.average_action_num = 0

    num_is_moving = num_agents + num_cluster_center - num_reach_target - num_hit_obs - num_out_of_range

    # if epoch % 20 == 0:
    # pic_name = "./pic/epoch_" + str(epoch) + ".png"
    # show(agents, CenterAgents, obstacles, pic_name)

    with open('./output/result.txt', 'a+') as f:
        f.write("Epoch: %s\n" % epoch)
        f.write("     num_reach_target: %s\n" % num_reach_target)
        f.write("     num_hit_obs: %s\n" % num_hit_obs)
        f.write("     num_out_of_range: %s\n" % num_out_of_range)
        f.write("     num_is_moving: %s\n\n\n" % num_is_moving)
