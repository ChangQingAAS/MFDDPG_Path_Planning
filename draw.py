import pylab
from obstacle import Obstacle
from agent import Agent
from center_agent import CenterAgent
import matplotlib.pyplot as plt


def show_cluster_analysis_results(agents, cluster_center_trace, name):
    color_list = ['.r', '.g', '.b', '.c', '.m', '.y', '.k']

    pylab.figure(figsize=(9, 9), dpi=80)
    for agent in agents:
        color = ''
        if agent.group_id >= len(color_list):
            color = color_list[-1]
        else:
            color = color_list[agent.group_id]
        pylab.plot(agent.x, agent.y, color)
    for single_trace in cluster_center_trace:
        pylab.plot([center.x for center in single_trace], [center.y for center in single_trace], 'k')
    pylab.savefig("./vis/init.png")
    pylab.show()


def save_pic(agents, CenterAgents, obstacles, target, name):
    color_list = ['.g', '.b', '.c', '.m', '.y', '.r']

    pylab.figure(figsize=(9, 9), dpi=100)

    # 黑色为障碍物
    for obstacle in obstacles:
        pylab.plot(
            obstacle.x,
            obstacle.y,
            color='#000000',
            marker="1",
        )

    # 彩色的为agent
    for agent in agents:
        color = ''
        if agent.group_id >= len(color_list):
            color = color_list[-1]
        else:
            color = color_list[agent.group_id]
        pylab.plot(agent.x, agent.y, color)

    #  簇中心
    for center_agent in CenterAgents:
        center_color_list = ['g', 'b', 'c', 'm', 'y', 'r']
        color = center_color_list[center_agent.group_id]
        pylab.plot(
            center_agent.x,
            center_agent.y,
            color=color,
            marker="D",
        )

    # 目标点
    pylab.plot(
        target[0],
        target[1],
        color="#FFD700",
        marker="p",
    )
    pylab.savefig(name)


def draw_gif(ax, agents, CenterAgents, obstacles, target, title):
    ax.cla()  # 清除键
    ax.set_title(title, fontsize=12, color='r')
    # 画点
    color_list = ['.g', '.b', '.c', '.m', '.y', '.r']
    # 黑色为障碍物
    for obstacle in obstacles:
        ax.plot(
            obstacle.x,
            obstacle.y,
            color='#000000',
            marker="1",
        )

    # 彩色的为agent
    for agent in agents:
        color = ''
        if agent.group_id >= len(color_list):
            color = color_list[-1]
        else:
            color = color_list[agent.group_id]
        ax.plot(agent.x, agent.y, color)

    #  簇中心
    for center_agent in CenterAgents:
        center_color_list = ['g', 'b', 'c', 'm', 'y', 'r']
        color = center_color_list[center_agent.group_id]
        pylab.plot(
            center_agent.x,
            center_agent.y,
            color=color,
            marker="D",
        )

    # 目标点
    pylab.plot(
        target[0],
        target[1],
        color="#FFD700",
        marker="p",
    )
    plt.pause(0.1)


def get_im(agents, CenterAgents, obstacles, target):
    # 目标点
    im = pylab.plot(target[0], target[1], color="#FFD700", marker='s', markersize=20)
    #  画点
    color_list = ['.g', '.b', '.c', '.m', '.y', '.r']
    # 黑色为障碍物
    for obstacle in obstacles:
        im.extend(pylab.plot(obstacle.x, obstacle.y, 'sk', ms=15))

    # 彩色的为agent
    for agent in agents:
        color = ''
        if agent.group_id >= len(color_list):
            color = color_list[-1]
        else:
            color = color_list[agent.group_id]
        im.extend(pylab.plot(agent.x, agent.y, color))

    #  簇中心
    for center_agent in CenterAgents:
        center_color_list = ['g', 'b', 'c', 'm', 'y', 'r']
        color = center_color_list[center_agent.group_id]
        im.extend(pylab.plot(
            center_agent.x,
            center_agent.y,
            color=color,
            marker="D",
        ))

    return im