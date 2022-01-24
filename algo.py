import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pylab
import numpy as np
import copy
import random

from kmeans_plus_plus import get_all
from RRT import get_action_by_RTT
from utils import *
from draw import *
from etc import *
from file_utils import file
from mfddpg import train

# 创建输出文件类，并进行文件初始化
out_file = file()
start_epoch = out_file.start_epoch
print("start epoch: %s " % (start_epoch))

# 初始化训练model
trainer = train.Trainer(
    state_space_dim,  # 状态空间维度：x,y 
    action_space_dim,
    device,
    start_epoch,
    MODELS_PATH)

# 当前决策步数
train_step = 0

# 初始化实体
agents, CenterAgents, obstacles = get_all()
# save_pic(agents, CenterAgents, obstacles, target, "./vis/init.png")

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
        if agent.x - distance_to_obs <= obstacle.x <= agent.x + distance_to_obs and agent.y - distance_to_obs <= obstacle.y <= agent.y + distance_to_obs:
            before_dead += 1
            agent.moving = False
            # print("agent:%s, [%s, %s]; obs: [%s,%s]" % (agent.id, agent.x, agent.y, obstacle.x, obstacle.y))
            break
with open("./output/result.txt", "w") as f:
    f.write("before_algo, there are %s agents have hit obstacles\n\n" % before_dead)

# 画图
# fig, ax = plt.subplots()

# algorithm
for epoch in range(start_epoch, num_epochs):
    # 画图
    fig = plt.figure()
    ims = []

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

    for step in range(max_step):
        # print("step in algo main is ", step)

        # 每一百步数，画图
        if step % radius == 0:
            title = "epoch: " + str(epoch) + " step: " + str(step)
            im = get_im(agents, CenterAgents, obstacles, target, title)
            ims.append(im)

        for agent in agents:
            if agent.moving:
                action_from_RRT = get_action_by_RTT(agent, target, CenterAgents[agent.group_id].obstacle_set)
                action = action_from_RRT
                agent.take_action(action_from_RRT)
                # action_from_algo = trainer.get_action_from_drl_algo(agent.x, agent.y)
                # if not (is_hit_obstacle(agent, action_from_algo, CenterAgents)
                #         or abs(agent.x + action_from_algo[0]) > radius or abs(agent.y + action_from_algo[1]) > radius):
                #     action = action_from_algo
                #     # with open('./output/result.txt', 'a+') as f:
                #     #     f.write("agent_id_%s get action [%s, %s] from ALGO\n" % (agent.id, action[0], action[1]))
                #     agent.take_action(action_from_algo)
                # else:
                #     # random_number = random.uniform(0, 1)
                #     # if random_number <= eta:
                #     action_from_RRT = get_action_by_RTT(agent, target, CenterAgents[agent.group_id].obstacle_set)
                #     action = action_from_RRT
                #     # with open('./output/result.txt', 'a+') as f:
                #     #     f.write("agent_id_%s get action [%s, %s] from RRT\n" % (agent.id, action[0], action[1]))
                #     agent.take_action(action_from_RRT)
                #     # else:
                #     #     action = get_action_from_all_clusters(agent, CenterAgents, action_from_algo)
                #     #     agent.take_action(action)

                # 增量方式求cluster平均动作
                leader = CenterAgents[agent.group_id]  #当前agent的leader，即该群的中心Agent
                leader.average_action_num += 1
                leader.average_action[0] += (action[0] - leader.average_action[0]) / leader.average_action_num
                leader.average_action[1] += (action[1] - leader.average_action[1]) / leader.average_action_num

                # 观察障碍物，存入center_agent
                get_obstacles_within_observation(agent, leader, obstacles)

                # 到达目标
                if target[0] - distance_to_target <= agent.x <= target[0] + distance_to_target and target[
                        1] - distance_to_target <= agent.y <= target[1] + distance_to_target:
                    # print("agent %s stop because Reach target" % agent.id)
                    num_reach_target += 1
                    agent.moving = False

                # 超出界限
                elif abs(agent.x) > radius or abs(agent.y) > radius:
                    print("agent %s stop because out of boundary" % agent.id)
                    num_out_of_range += 1
                    agent.moving = False

                # 碰到障碍物
                else:
                    for obstacle in obstacles:
                        if agent.x - distance_to_obs <= obstacle.x <= agent.x + distance_to_obs and agent.y - distance_to_obs <= obstacle.y <= agent.y + distance_to_obs:
                            print("agent:%s, [%s, %s]; obs: [%s,%s]" %
                                  (agent.id, agent.x, agent.y, obstacle.x, obstacle.y))
                            num_hit_obs += 1
                            agent.moving = False
                            break

        for centerAgent in CenterAgents:
            if centerAgent.moving:
                action_from_RRT = get_action_by_RTT(centerAgent, target, centerAgent.obstacle_set)
                action = action_from_RRT
                centerAgent.take_action(action_from_RRT)
                # action_from_algo = trainer.get_action_from_drl_algo(centerAgent.x, centerAgent.y)
                # if not (is_hit_obstacle(centerAgent, action_from_algo, CenterAgents)
                #         or abs(centerAgent.x + action_from_algo[0]) > radius
                #         or abs(centerAgent.y + action_from_algo[1]) > radius):
                #     action = action_from_algo
                #     # with open('./output/result.txt', 'a+') as f:
                #     #     f.write("centerAgent_id_%s get action [%s, %s] from algo\n" %
                #     #             (centerAgent.id, action[0], action[1]))
                #     state_last = [centerAgent.x, centerAgent.y]
                #     action_last = action
                #     centerAgent.take_action(action)
                #     reward_now = get_reward(centerAgent)
                #     state_new = [centerAgent.x, centerAgent.y]
                #     mean_action = centerAgent.average_action
                #     trainer.train_(train_step, np.float32(state_last), action_last, reward_now, np.float32(state_new),
                #                    step, mean_action)
                # else:
                #     # random_number = random.uniform(0, 1)
                #     # if random_number < eta:
                #     action_from_RRT = get_action_by_RTT(centerAgent, target, centerAgent.obstacle_set)
                #     action = action_from_RRT
                #     # with open('./output/result.txt', 'a+') as f:
                #     #     f.write("centerAgent_id_%s get action [%s, %s] from RRT\n" %
                #     #             (centerAgent.id, action[0], action[1]))
                #     centerAgent.take_action(action_from_RRT)
                #     # else:
                #     #     action = CenterAgents[centerAgent.group_id].average_action
                #     #     centerAgent.take_action(action)

                # 观察障碍物，存入center_agent
                get_obstacles_within_observation(centerAgent, CenterAgents[agent.group_id], obstacles)

                # 到达目标
                if centerAgent.x - distance_to_target <= target[
                        0] <= centerAgent.x + distance_to_target and centerAgent.y - distance_to_target <= target[
                            1] <= centerAgent.y + distance_to_target:
                    # print("centeragent %s stop because Reach target" % centerAgent.id)
                    num_reach_target += 1
                    centerAgent.moving = False

                # 超出界限
                elif abs(centerAgent.x) > radius or abs(centerAgent.y) > radius:
                    print("centerAgent %s stop because out of boundary" % centerAgent.id)
                    num_out_of_range += 1
                    centerAgent.moving = False

                # 碰到障碍物：
                else:
                    for obstacle in obstacles:
                        if centerAgent.x - distance_to_obs <= obstacle.x <= centerAgent.x + distance_to_obs and centerAgent.y - distance_to_obs <= obstacle.y <= centerAgent.y + distance_to_obs:
                            print("agent %s stop because hit an obstacle" % agent.id)
                            num_hit_obs += 1
                            centerAgent.moving = False
                            break

    ani = animation.ArtistAnimation(fig, ims, interval=500, repeat_delay=1000)
    gif_name = "./vis/epoch_" + str(epoch) + ".gif"
    ani.save(gif_name, writer='pillow')

    # 保存一下已经训练的轮数
    trainer.save_model(epoch, MODELS_PATH)
    out_file.write_epoch(epoch)
    # print("epoch is ", epoch)

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

    # if epoch % 5 == 0:
    #     pic_name = "./vis/epoch_" + str(epoch) + ".png"
    #     save_pic(agents, CenterAgents, obstacles, target, pic_name)

    with open('./output/result.txt', 'a+') as f:
        f.write("Epoch: %s\n" % epoch)
        f.write("     num_reach_target: %s\n" % num_reach_target)
        f.write("     num_hit_obs: %s\n" % num_hit_obs)
        f.write("     num_out_of_range: %s\n" % num_out_of_range)
        f.write("     num_is_moving: %s\n\n\n" % num_is_moving)
