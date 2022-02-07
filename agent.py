from etc import *
import math
import random


class Agent:
    __slots__ = ["id", "x", "y", "group_id", "action", "moving"]

    def __init__(self, id=0, x=0, y=0, group_id=0, action=[0, 0]):
        self.id = id
        self.x = x
        self.y = y
        self.group_id = group_id
        self.action = action  # x,y两个方向前进的数值
        self.moving = True  # Agent是否在移动

    def take_action(self, action):
        self.x += action[0]
        self.y += action[1]

    def get_random_action(self):
        action_x = random.uniform(-1, 1)
        action_y = random.uniform(-1, 1)
        self.action = [action_x, action_y]
        return self.action
