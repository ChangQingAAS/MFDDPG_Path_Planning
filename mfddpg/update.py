import numpy as np
from .config import *


def soft_update(target, source, tau):
    '''软更新，在主网络参数基础上，做较小的改变，更新到目标网络'''
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    '''硬更新，在主网络参数基础上，做较小的改变，更新到目标网络'''
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)