from agent import Agent
import random
from etc import *
import math


class CenterAgent(Agent):
    __slots__ = [
        "id", "x", "y", "group_id", "action", "obstacle_set", "average_action",
        "agents_list", "average_action_num", "moving"
    ]

    def __init__(self,
                 id=0,
                 x=0,
                 y=0,
                 group_id=0,
                 action=(0, 0),
                 obstacle_set=set(),
                 agents_list=[],
                 average_action=[0, 0]):
        super(CenterAgent, self).__init__()
        self.obstacle_set = obstacle_set
        self.agents_list = agents_list
        self.average_action = average_action
        self.average_action_num = 0

    def take_action(self, action):
        self.x += action[0]
        self.y += action[1]

    def get_random_action(self):
        action_x = random.uniform(-1, 1)
        action_y = random.uniform(-1, 1)
        self.action = [action_x, action_y]
        return self.action
