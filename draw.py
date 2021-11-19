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
        pylab.plot([center.x for center in single_trace],
                   [center.y for center in single_trace], 'k')
    pylab.show()


def show(agents, CenterAgents, obstacles, name):
    color_list = ['.g', '.b', '.c', '.m', '.y','.r']

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
        center_color_list = ['g', 'b', 'c', 'm', 'y','r']
        color = center_color_list[center_agent.group_id]
        pylab.plot(
            center_agent.x,
            center_agent.y,
            color=color,
            marker="D",
        )

    pylab.savefig(name)
    # pylab.show()


def draw_gif(ax, agents, CenterAgents, obstacles, title):
    ax.cla()  # 清除键
    ax.set_title(title, fontsize=12, color='r')
    # 画点
    color_list = ['.g', '.b', '.c', '.m', '.y','.r']
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
        center_color_list = ['g', 'b', 'c', 'm', 'y','r']
        color = center_color_list[center_agent.group_id]
        pylab.plot(
            center_agent.x,
            center_agent.y,
            color=color,
            marker="D",
        )

    # ax.legend()
    plt.pause(0.1)