import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import copy

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
save_pic(agents, CenterAgents, obstacles, target, "./vis/init.png")

# 保存初始化的环境,以便于进行对环境reset
first_agents = []
for item in agents:
    first_agents.append(copy.copy(item))
first_center_agents = []
for item in CenterAgents:
    first_center_agents.append(copy.copy(item))

with open("./output/result.csv", "w", encoding="utf-8") as f:
    f.write("epoch,num_reach_target,num_hit_obs,num_out_of_range,num_moving\n")

# 执行算法
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

    # 初始化参数
    num_reach_target = 0
    num_hit_obs = 0
    num_out_of_range = 0

    for step in range(max_step):
        # 每XX步数，画图
        if (step * 4) % radius == 0:
            im = get_im(agents, CenterAgents, obstacles, target)
            ims.append(im)

        for agent in agents:
            if agent.moving:
                #当前agent的leader，即该群的中心Agent
                leader = CenterAgents[agent.group_id]
                # 增量方式求cluster平均状态
                leader.average_action_num += 1
                leader.average_old_state[0] += (agent.x - leader.average_old_state[0]) / leader.average_action_num
                leader.average_old_state[1] += (agent.y - leader.average_old_state[1]) / leader.average_action_num

                action_from_algo = trainer.get_action_from_drl_algo(agent.x, agent.y)
                if (not is_hit_obstacle(agent, action_from_algo, CenterAgents)) and is_closer(
                        agent, action_from_algo, target):
                    action = action_from_algo
                    agent.take_action(action_from_algo)
                else:
                    action_from_RRT = get_action_by_RTT(agent, target, CenterAgents[agent.group_id].obstacle_set)
                    action = action_from_RRT
                    agent.take_action(action)
                # with open('./output/result.txt', 'a+') as f:
                #     f.write("agent_id_%s get action [%s, %s] from RRT\n" % (agent.id, action[0], action[1]))

                # 增量方式求cluster平均动作
                leader.average_action[0] += (action[0] - leader.average_action[0]) / leader.average_action_num
                leader.average_action[1] += (action[1] - leader.average_action[1]) / leader.average_action_num

                leader.average_state[0] += (agent.x - leader.average_state[0]) / leader.average_action_num
                leader.average_state[1] += (agent.y - leader.average_state[1]) / leader.average_action_num

                leader.average_reward += (get_reward(agent, leader) -
                                          leader.average_reward) / leader.average_action_num

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
                        if agent.x - distance_to_obs <= obstacle.x <= agent.x + distance_to_obs or agent.y - distance_to_obs <= obstacle.y <= agent.y + distance_to_obs:
                            # print("agent:%s, [%s, %s]; obs: [%s,%s]" % (agent.id, agent.x, agent.y, obstacle.x, obstacle.y))
                            print("agent_%s stop because hit an obstacle" % agent.id)
                            num_hit_obs += 1
                            agent.moving = False
                            break

        for centerAgent in CenterAgents:
            if centerAgent.moving:
                action_from_algo = trainer.get_action_from_drl_algo(centerAgent.x, centerAgent.y)
                if (not is_hit_obstacle(centerAgent, action_from_algo, CenterAgents)) and is_closer(
                        centerAgent, action_from_algo, target):
                    state_last = centerAgent.average_old_state
                    centerAgent.take_action(action_from_algo)
                    reward = centerAgent.average_reward
                    state_new = centerAgent.average_state
                    mean_action = centerAgent.average_action
                    trainer.train_(train_step, np.int16(state_last), mean_action, reward, np.int16(state_new), step)
                else:
                    action_from_RRT = get_action_by_RTT(centerAgent, target, centerAgent.obstacle_set)
                    action = action_from_RRT
                    centerAgent.take_action(action)
                #     # with open('./output/result.txt', 'a+') as f:
                #     #     f.write("centerAgent_id_%s get action [%s, %s] from RRT\n" %  (centerAgent.id, action[0], action[1]))

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
                        if centerAgent.x - distance_to_obs <= obstacle.x <= centerAgent.x + distance_to_obs or centerAgent.y - distance_to_obs <= obstacle.y <= centerAgent.y + distance_to_obs:
                            print("centeragent_%s stop because hit an obstacle" % centerAgent.id)
                            num_hit_obs += 1
                            centerAgent.moving = False
                            break

    ani = animation.ArtistAnimation(fig, ims, interval=500, repeat_delay=1000)
    gif_name = "./vis/gif/epoch_" + str(epoch) + ".gif"
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

    num_moving = num_agents + num_cluster_center - num_reach_target - num_hit_obs - num_out_of_range

    # if epoch % 5 == 0:
    #     pic_name = "./vis/epoch_" + str(epoch) + ".png"
    #     save_pic(agents, CenterAgents, obstacles, target, pic_name)

    with open('./output/result.csv', 'a+', encoding='utf-8') as f:
        f.write("%s,%s,%s,%s,%s\n" % (epoch, num_reach_target, num_hit_obs, num_out_of_range, num_moving))
