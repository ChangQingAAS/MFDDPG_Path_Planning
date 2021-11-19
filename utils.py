import math
import random
from obstacle import Obstacle
from agent import Agent
from center_agent import CenterAgent


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
        agent.x = int(random.uniform(-1, 1) * radius)
        agent.y = int(random.uniform(-1, 1) * radius)
        agent.id = id
        id += 1
    return agents


# 创建所有Obstacles
def generate_Obstacles(num_agents, radius):
    obstacles = [Obstacle() for _ in range(num_agents)]
    id = 1
    for obstacle in obstacles:
        obstacle.x = int(random.uniform(-1, 1) * radius)
        obstacle.y = int(random.uniform(-1, 1) * radius)
        obstacle.id = id
        id += 1
    return obstacles


def get_distance_between_two_points(pointA, pointB):
    distance = math.pow((pointA.x - pointB.x), 2) + math.pow(
        (pointA.y - pointB.y), 2)
    return distance


def get_all_obstacles(Obstacles):
    obs_set = set()
    for item in Obstacles:
        obs_set.add((item.x, item.y))
    return obs_set


def is_changed(agents, prev_agents):
    for item in agents:
        for prev_item in prev_agents:
            if prev_item.id == item.id:
                flag = (prev_item.x == item.x) and (prev_item.y == item.y)
                print(prev_item.id, ": ", flag)
