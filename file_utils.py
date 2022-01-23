from __future__ import division
from __future__ import print_function

import os
import sys
sys.path.append("..")


class file():
    def __init__(self):
        super(file, self).__init__()

        # 绝对路径定义
        self.ABS_PATH = os.path.dirname(__file__)
        # print("绝对路径为：", self.ABS_PATH)

        # 输出路径定义
        self.OUTPUT_PATH = "%s/output/" % self.ABS_PATH
        # print("输出路径", self.OUTPUT_PATH)

        # 写入models的路径
        self.MODELS_PATH = "%s/Models/" % self.OUTPUT_PATH

        # 指定写入回合数的文件路径
        self.epoch_file_path = "%s/epoch.txt" % self.OUTPUT_PATH
        # print("写入回合数的路径：", self.epoch_file_path)

        # 获取已经训练的轮数,作为本次开始回合
        self.start_epoch = self.read_number(self.epoch_file_path)
        # print("start epoch: %s" % self.start_epoch)

        # # 指定写入step的文件路径
        # self.step_file_path = "%s/step.txt" % self.OUTPUT_PATH

        # # 获取需开始步数，即上次停止的位置
        # self.start_step = self.read_number(self.step_file_path)

        # # 当前step
        # self.cur_step = self.start_step
        # # print("start step: %s" % self.cur_step)

    def read_number(self, path):
        """
        description: 读取epoch和step用的
        """
        if not os.path.exists(path):
            return int(0)
        with open(path, "r") as f:
            try:
                ret = f.read()
                return int(ret)
            except:
                return 0

    def write_number(self, path, num):
        with open(path, "w") as f:
            f.write("%s" % num)

    def write_epoch(self, epoch):
        self.write_number(self.epoch_file_path, epoch)

    def write_step(self, step):
        self.write_number(self.step_file_path, step)

    #  保存最后的奖赏值
    def write_final_reward(self, reward, epochs):
        with open("%s/final_reward.csv" % self.OUTPUT_PATH, "a+") as f:
            f.write("%s,%s\n" % (epochs, reward))
