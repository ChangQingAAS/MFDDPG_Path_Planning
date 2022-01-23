import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import csv
import random
import sys
import os


def get_data(path):
    x = []
    y = [] 
    with open(path , "r+", encoding="utf-8") as csvfile:
        epoch_number = []
        loss = []
        reader = csv.DictReader(csvfile)
        for row in reader:
            average_reward.append(float(row['average reward']))
            if x[algo] == []:
                epoch_number.append(int(row['epoch_number']))
    y[algo].append(average_reward)
    if x[algo] == []:
        x[algo].append(epoch_number)

    return x, y


def draw_all(algo_list, color, x, y, env, path):
    plt.cla()
    for i, algo in enumerate(reversed(algo_list)):
        sns.tsplot(time=x[algo], data=y[algo], condition=algo, linewidth=0.5, color=color[i])
        plt.ylabel('average reward', fontsize=10)
        plt.xlabel('num episodes', fontsize=10)
        # plt.yticks((0, 500, 100))
        # plt.xticks((0, 2000, 20))
        plt.title(env, fontsize=10)
        plt.legend(loc='lower right', fontsize=5)
        plt.savefig(path + "/vis/ALL-%s.png" % (env))


if __name__ == "__main__":
    path = sys.path[0].rsplit("\\", 1)[0]
    name = 'loss_critic.csv'
    path = path + '\output' + name
    x, y = get_data(path)
    draw(x, y, name)
