from __future__ import division
from __future__ import print_function

import os
import sys 
sys.path.append("..") 

import etc


class file():
    def __init__(self):
        super(file, self).__init__()

        # 绝对路径定义
        self.ABS_PATH = etc.app_abspath
        print(self.ABS_PATH)
        # 输出路径定义
        self.OUTPUT_PATH = "%s/output/" % self.ABS_PATH
        print(self.OUTPUT_PATH)
        # 指定写入回合数的文件路径
        self.epoch_file_path = "%s/epoch.txt" % self.OUTPUT_PATH
        print(self.epoch_file_path)
        # 本次开始回合
        self.start_epoch = 0
        # 指定写入step的文件路径
        self.step_file_path = "%s/step.txt" % self.OUTPUT_PATH
        # 本次开始step
        self.start_step = 0
        # 当前step
        self.cur_step = self.start_step

    def read_a_number(self, path):
        """
        description: 读取epoch和step用的
        """
        if not os.path.exists(path):
            return int(0)
        f = open(path, "r")
        ret = f.read()
        f.close()
        if not ret:
            return int(0)
        return int(ret)

    def initialize(self):
        # 获取已经训练的轮数
        self.start_epoch = self.read_a_number(self.epoch_file_path)
        print("start epoch: %s" % self.start_epoch)
        # 获取需开始步数，即上次停止的位置
        self.start_step = self.read_a_number(self.step_file_path)
        self.cur_step = self.start_step
        print("start step: %s" % self.cur_step)

        return self.start_epoch, self.start_step, self.cur_step

    def write_step(self, cur_step):
        f = open(self.step_file_path, "w")
        cur_step = str(cur_step)
        f.write(cur_step)
        f.close()

    def write_epoch(self, cur_epoch):
        f = open(self.epoch_file_path, "w")
        cur_epoch = str(cur_epoch)
        f.write(cur_epoch)
        f.close()

    def write_final_reward(reward, epochs):
        """
        保存最后的奖赏值
        :param reward: 奖赏值
        :param epochs: 回合
        :return:
        """
        file_path = "%s/final_reward.txt" % etc.OUTPUT_PATH
        if not os.path.exists(file_path):
            f = open(file_path, "w")
        else:
            f = open(file_path, "a")
        f.write("%s,%s\n" % (epochs, reward))
        f.close()


#################### 这部分暂时没用到 ####################

def create_path(path):
    """
    创建文件路径
    :param path: 文件路径
    :return:
    """
    if not os.path.isdir(path):
        os.makedirs(path)

def create_dir(path):
    """
    创建文件夹路径
    :param path:
    :return:
    """
    if not os.path.isdir(path):
        os.makedirs(path)

def write_file(con, name="default", path='./tmp_file/'):
    """
    写临时文件
    :param con: 写入文件内容
    :param name: 文件名
    :param path: 文件路径
    :return:
    """
    f = create_file(path, name)
    f.write(con)
    f.close()

def get_file_full_name(path, name):
    """
    获取文件全名
    :param path: 文件路径
    :param name: 文件名
    :return:
    """
    create_path(path)
    if path[-1] == "/":
        full_name = path + name
    else:
        full_name = path + "/" + name
    return full_name

def open_file(path, name, open_type='a'):
    """
    打开文件
    :param path: 文件路径
    :param name: 文件名
    :param open_type: 打开文件方式
    :return:
    """
    file_name = get_file_full_name(path, name)
    return open_file_with_full_name(file_name, open_type)

def create_file(path, name, open_type='w'):
    """
    创建文件
    :param path: 文件路径
    :param name: 文件名
    :param open_type: 创建文件的模式
    :return:
    """
    file_name = get_file_full_name(path, name)
    return open_file_with_full_name(file_name, open_type)

def check_is_have_file(path, name):
    """
    验证是否有文件
    :param path: 文件路径
    :param name: 文件名
    :return:
    """
    file_name = get_file_full_name(path, name)
    return os.path.exists(file_name)

def open_file_with_full_name(full_path, open_type):
    """
    使用绝对路径打开文件
    :param full_path: 文件的绝对路径
    :param open_type: 打开方式
    :return:
    """
    try:
        file_object = open(full_path, open_type)
        return file_object
    except Exception as e:
        if e.args[0] == 2:
            open(full_path, 'w')
        else:
            pass

def delete_dir(src):
    """
    删除文件或文件夹
    :param src: 文件或文件夹
    :return:
    """
    if os.path.isfile(src):
        try:
            os.remove(src)
        except Exception as e:
            pass
            return False
    elif os.path.isdir(src):
        for item in os.listdir(src):
            itemsrc = os.path.join(src, item)
            delete_dir(itemsrc)
        try:
            os.rmdir(src)
        except Exception as e:
            pass
            return False
    return True

def read_reward_file():
    # 读取奖励文档
    epochs_list = []
    reward_list = []
    f = open("%s/final_reward.txt" % etc.OUTPUT_PATH)
    con = f.read()
    f.close()
    con_lt = con.split("\n")
    for i in range(len(con_lt) - 1):
        lt = con_lt[i].split(',')
        epochs_list.append(int(lt[0]))
        reward_list.append(float(lt[1]))
    return epochs_list, reward_list