import math
import random
from math import sqrt

from obstacle import Obstacle
from agent import Agent
from etc import *


def get_reward(agent, leader):
    reward = 0
    obstacle_reward = 0
    live_reward = 0
    # 接近目标的奖励,这里使用的是欧式距离
    distance_reward = -2* sqrt(pow((agent.x - target[0]), 2) + pow((agent.y - target[1]), 2))

    # 障碍物的惩罚
    for item in leader.obstacle_set:
        if agent.x - distance_to_obs <= item.x <= agent.x + distance_to_obs and agent.y - distance_to_obs <= item.y <= agent.y + distance_to_obs:
            obstacle_reward += obstacle_penalty
            break

    if agent.moving == False:
        live_reward = dead_penalty

    reward = distance_reward + obstacle_reward + live_reward
    return reward


def is_closer(agent, action, target):
    old_dis = pow((agent.x - target[0]), 2) + pow((agent.y - target[1]), 2)
    new_dis = pow((agent.x + action[0] - target[0]), 2) + pow((agent.y + action[1] - target[1]), 2)
    if new_dis < old_dis:
        return True
    return False


# 获取agent观察范围内的障碍物，并通信给中心agent
def get_obstacles_within_observation(agent, CenterAgent, obstacles):
    for obstacle in obstacles:
        # 局部感知： 智能体只能看到部分
        if agent.x - view_agent <= obstacle.x <= agent.x + view_agent or agent.y - view_agent <= obstacle.y <= agent.y + view_agent:
            # print("info: \n        agent_%s find obstacle_%s " % (agent.id, obstacle.id))
            CenterAgent.obstacle_set.add(obstacle)


# 执行该动作是否会碰到障碍物
def is_hit_obstacle(agent, action, CenterAgents):
    next_x = agent.x + action[0]
    next_y = agent.y + action[1]
    for item in CenterAgents[agent.group_id].obstacle_set:
        if next_x - distance_to_obs <= item.x <= next_x + distance_to_obs or next_y - distance_to_obs <= item.y <= next_y + distance_to_obs:
            return True

    return False


# 打印某个类的全部属性
def print_attr(x):
    if hasattr(x, '__dict__'):
        print(vars(x))  # add
        return vars(x)
    else:
        ret = {slot: getattr(x, slot) for slot in x.__slots__}
        for cls in type(x).mro():
            spr = super(cls, x)
            if not hasattr(spr, '__slots__'):
                break
            for slot in spr.__slots__:
                ret[slot] = getattr(x, slot)
        print(ret)  # add
        return ret


# 创建所有Agents
def generate_Agents(num_agents, radius):
    agents = [Agent() for _ in range(num_agents)]
    id = 1
    for agent in agents:
        # r = random.random() * radius
        # angle = random.random() * 2.0 * math.pi  # 这里有一个超参可以改变点的聚集状况
        # agent.x = r * math.cos(angle)
        # agent.y = r * math.sin(angle)
        agent.x = random.uniform(-1, 1) * radius
        agent.y = random.uniform(-1, 1) * radius
        agent.id = id
        id += 1
    return agents


# 创建所有Obstacles
def generate_Obstacles(num_agents, radius):
    # obstacles = [Obstacle() for _ in range(num_agents * 4)]
    # id = 1
    # for obstacle in obstacles:
    #     obstacle.x = random.randint(-radius, radius)
    #     obstacle.y = random.randint(-radius, radius)
    #     obstacle.id = id
    #     id += 1
    # return obstacles
    obstacles = []
    id = 0
    for _ in range(num_agents):
        items = []
        flag = True

        # 第1个
        item_1 = Obstacle()
        item_1.x = random.randint(-radius + 2, radius - 2)
        item_1.y = random.randint(-radius + 2, radius - 2)
        item_1.id = id + 1
        items.append(item_1)

        # # 第2个
        # item_2 = Obstacle()
        # item_2.x = item_1.x + 1
        # item_2.y = item_1.y
        # item_2.id = id + 2
        # items.append(item_2)

        # # 第3个
        # item_3 = Obstacle()
        # item_3.x = item_1.x
        # item_3.y = item_1.y - 1
        # item_3.id = id + 3
        # items.append(item_3)
        # id += 1

        # # 第4个
        # item_4 = Obstacle()
        # item_4.x = item_1.x + 1
        # item_4.y = item_1.y - 1
        # item_4.id = id + 4
        # items.append(item_4)

        # # 第5个
        # item_5 = Obstacle()
        # item_5.x = item_1.x + 1
        # item_5.y = item_1.y + 1
        # item_5.id = id + 5
        # items.append(item_5)

        # # 第6个
        # item_6 = Obstacle()
        # item_6.x = item_1.x
        # item_6.y = item_1.y + 1
        # item_6.id = id + 6
        # items.append(item_6)

        # # 第7个
        # item_7 = Obstacle()
        # item_7.x = item_1.x - 1
        # item_7.y = item_1.y
        # item_7.id = id + 7
        # items.append(item_7)

        # # 第8个
        # item_8 = Obstacle()
        # item_8.x = item_1.x - 1
        # item_8.y = item_1.y + 1
        # item_8.id = id + 8
        # items.append(item_8)

        # # 第9个
        # item_9 = Obstacle()
        # item_9.x = item_1.x - 1
        # item_9.y = item_1.y - 1
        # item_9.id = id + 9
        # items.append(item_9)

        for item in items:
            if item.x == target[0] and item.y == target[1]:
                flag = False

        if flag:
            obstacles += items
            id += 9

    return obstacles


# 得到两个点的距离
def get_distance_between_two_points(pointA, pointB):
    distance = sqrt(math.pow((pointA.x - pointB.x), 2) + math.pow((pointA.y - pointB.y), 2))
    return distance


# 返回所有的障碍物的坐标
def get_all_obstacles(Obstacles):
    obs_set = set()
    for item in Obstacles:
        obs_set.add((item.x, item.y))
    return obs_set


#
# def is_changed(agents, prev_agents):
#     for item in agents:
#         for prev_item in prev_agents:
#             if prev_item.id == item.id:
#                 flag = (prev_item.x == item.x) and (prev_item.y == item.y)
#                 print(prev_item.id, ": ", flag)

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

# # 从所有智能体群获取动作
# # action = beta * 来自drl的动作 + alpha * 来自该群的动作 + (1-beta - alpha) * 来自其他群的动作
# def get_action_from_all_clusters(agent, CenterAgents, action_from_algo):
#     action = [0, 0]
#     action_from_its_cluster = CenterAgents[agent.group_id].average_action
#     action_from_others_cluster = [0, 0]

#     # 先加全部的，再把自己的减去
#     for item in CenterAgents:
#         action_from_others_cluster[0] += item.average_action[0]
#         action_from_others_cluster[1] += item.average_action[1]

#     action_from_others_cluster[0] -= action_from_its_cluster[0]
#     action_from_others_cluster[1] -= action_from_its_cluster[1]

#     # 取平均
#     action_from_others_cluster = [i / (len(CenterAgents) - 1) for i in action_from_others_cluster]

#     action[0] = beta * action_from_algo[0] + alpha * action_from_its_cluster[0] + (
#         1 - beta - alpha) * action_from_others_cluster[0]
#     action[1] = beta * action_from_algo[1] + alpha * action_from_its_cluster[1] + (
#         1 - beta - alpha) * action_from_others_cluster[1]

#     return action