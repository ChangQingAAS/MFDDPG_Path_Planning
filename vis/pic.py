import matplotlib.pyplot as plt
import numpy as np
import etc
import os


def read_reward_file(self):
    # 读取奖励文档
    epochs_list = []
    reward_list = []
    with open("%s/final_reward.csv" % self.OUTPUT_PATH) as f:
        con = f.read()
    con_lt = con.split("\n")
    for i in range(len(con_lt) - 1):
        lt = con_lt[i].split(',')
        epochs_list.append(int(lt[0]))
        reward_list.append(float(lt[1]))
    return epochs_list, reward_list


def read_loss_file(file_name=""):
    # 读取损失文档
    epochs_list = []
    reward_list = []
    f = open(file_name)
    con = f.read()
    f.close()
    con_lt = con.split("\n")
    for i in range(len(con_lt) - 1):
        lt = con_lt[i].split(',')
        epochs_list.append(int(lt[0]))
        reward_list.append(float(lt[1]))
    return epochs_list, reward_list


def show_reward_pic():
    # 画出奖赏值的变化图
    epoch_list, reward_list = read_reward_file()
    reward_sum = 0.0
    for i in range(len(reward_list)):
        reward_sum += reward_list[i]

    reward_mean = reward_sum / len(reward_list)
    e = np.asarray(epoch_list)
    r = np.asarray(reward_list)
    plt.figure()
    plt.plot(e, r)
    plt.xlabel('Epochs')
    plt.ylabel('Reward')
    plt.show()
    plt.close()


def show_loss_pic(loss_name="loss_critic"):
    # 画出损失值的变化图
    file_name = "%s/%s.txt" % (etc.OUTPUT_PATH, loss_name)
    step_list, loss_list = read_loss_file(file_name)
    e = np.asarray(step_list)
    r = np.asarray(loss_list)
    plt.figure()
    plt.plot(e, r)
    plt.xlabel('steps')
    plt.ylabel(loss_name)
    # 使用plt.show(),然后使用plt.close()并不会关闭图，
    # plt.show()
    plt.draw()
    plt.pause(20)
    plt.close()


def show_pic():
    # 画图
    if etc.SHOW_FIGURE:
        # show_reward_pic()
        show_loss_pic("loss_actor")
        # show_loss_pic("loss_critic")
