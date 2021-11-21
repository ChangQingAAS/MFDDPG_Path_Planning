import numpy as np
import random
from collections import deque
from .config import *


class MemoryBuffer:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)
        self.maxSize = size
        self.len = 0

    def sample(self, count):
        batch = []
        count = min(count, self.len)
        batch = random.sample(self.buffer, count)

        state_arr = np.float32([arr[0] for arr in batch])
        action_arr = np.float32([arr[1] for arr in batch])
        reward_arr = np.float32([arr[2] for arr in batch])
        state_arr_ = np.float32([arr[3] for arr in batch])
        mean_action_arr = np.float32([arr[4] for arr in batch])

        return state_arr, action_arr, reward_arr, state_arr_, mean_action_arr

    def add(self, state, action, reward, state_, mean_action):
        trainsition = (state, action, reward, state_, mean_action)
        self.len += 1
        if self.len > self.maxSize:
            self.len = self.maxSize
        self.buffer.append(trainsition)
