import math
import random
import copy
from numpy import average
import pylab
from utils import *
from draw import *
from etc import *
from obstacle import Obstacle
from agent import Agent
from center_agent import CenterAgent

FLOAT_MAX = 1e100

DEBUG_FLAG = False


# 找到该Agent属于的Group
def get_nearest_center(agent, cluster_center_group):
    min_index = agent.group_id
    min_distance = FLOAT_MAX
    for index, center in enumerate(cluster_center_group):
        distance = get_distance_between_two_points(agent, center)
        if (distance < min_distance):
            min_distance = distance
            min_index = index
    return (min_index, min_distance)


# 找到距离相对较大的中心,并完成第一次分类
def kMeansPlusPlus(agents, cluster_center_group):
    # 随机选一个点作为第一个中心点
    cluster_center_group[0] = copy.copy(random.choice(agents))
    distance_group = [0.0 for _ in range(len(agents))]
    sum = 0.0
    # 遍历后续中心点和all_agents以确定各个中心点的位置
    for index in range(1, len(cluster_center_group)):
        for i, agent in enumerate(agents):
            # 智能体i到(前index个中心点中)最近中心点的距离
            distance_group[i] = get_nearest_center(
                agent, cluster_center_group[:index])[1]
            sum += distance_group[i]
        sum *= random.random()
        for i, distance in enumerate(distance_group):
            sum -= distance
            if sum < 0:
                cluster_center_group[index] = copy.copy(agents[i])
                cluster_center_group[index].group_id = index
                break

    # 确定好分散的中心点后，对AGents进行分类
    for agent in agents:
        agent.group_id = get_nearest_center(agent, cluster_center_group)[0]

    for item in cluster_center_group:
        item.id = (item.group_id)

    if DEBUG_FLAG:
        print("in Kmeans++")
        for item in cluster_center_group:
            print(print_attr(item))
    return


def update_cluster_center_to_CenterAgents(cluster_center_group, agents):
    CenterAgents = [CenterAgent() for _ in range(len(cluster_center_group))]
    i = 0
    for cluster_center in cluster_center_group:
        CenterAgents[i].id = cluster_center.id
        CenterAgents[i].x = cluster_center.x
        CenterAgents[i].y = cluster_center.y
        CenterAgents[i].group_id = cluster_center.group_id
        CenterAgents[i].action = cluster_center.action
        CenterAgents[i].agents_list = []
        CenterAgents[i].average_action = [0, 0]
        i += 1

    for agent in agents:
        id = agent.group_id
        # for i in range(len(CenterAgents)):
        #     if CenterAgents[i].group_id == id:
        #         CenterAgents[i].agents_list.append(agent)
        CenterAgents[id].agents_list.append(agent)

    return CenterAgents


def kMeans(agents, cluster_center_number):
    # cluster_center_group = [Agent() for _ in range(cluster_center_number)]
    cluster_center_group = [None for _ in range(cluster_center_number)]
    kMeansPlusPlus(agents, cluster_center_group)
    # 此时已经基于一组离散中心点，完成分类了，即第一次正式分类，但是此时中心点并不是在群体中心，只是满足离散的条件

    # 下面开始进入不断优化地分类
    cluster_center_trace = [[cluster_center]
                            for cluster_center in cluster_center_group]
    tolerable_error, current_error = 5.0, FLOAT_MAX
    count = 0
    while current_error >= tolerable_error:
        count += 1
        count_center_number = [0 for _ in range(cluster_center_number)]
        current_center_group = [Agent() for _ in range(cluster_center_number)]
        for agent in agents:
            current_center_group[agent.group_id].x += agent.x
            current_center_group[agent.group_id].y += agent.y
            count_center_number[agent.group_id] += 1
        for index, center in enumerate(current_center_group):
            center.x /= count_center_number[index]
            center.y /= count_center_number[index]
        current_error = 0.0
        for index, singleTrace in enumerate(cluster_center_trace):
            singleTrace.append(current_center_group[index])
            current_error += get_distance_between_two_points(
                singleTrace[-1], singleTrace[-2])
            cluster_center_group[index].x = current_center_group[index].x
            cluster_center_group[index].y = current_center_group[index].y

        for agent in agents:
            agent.group_id = get_nearest_center(agent, cluster_center_group)[0]

    CenterAgents = update_cluster_center_to_CenterAgents(
        cluster_center_group, agents)

    return CenterAgents, cluster_center_trace


def main():
    num_cluster_center = 5
    num_agents = 100
    num_obstacles = 10
    radius = 360

    agents = generate_Agents(num_agents, radius)
    obstacles = generate_Obstacles(num_obstacles, radius)
    CenterAgents, cluster_center_trace = kMeans(agents, num_cluster_center)
    show(agents, CenterAgents, obstacles, "./show_result.png")


def get_all():
    agents = generate_Agents(num_agents, radius)
    obstacles = generate_Obstacles(num_obstacles, radius)
    CenterAgents, cluster_center_trace = kMeans(agents, num_cluster_center)
    return agents, CenterAgents, obstacles


if __name__ == '__main__':
    main()