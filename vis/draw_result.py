import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pylab
import csv
import random
import sys
import os


def get_data(path):
    epoch = []
    num_reach_target = []
    num_hit_obs = []
    num_out_of_range = []
    num_moving = []
    with open(path, "r+", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            epoch.append(int(row['epoch']))
            num_reach_target.append(int(row['num_reach_target']))
            num_hit_obs.append(int(row['num_hit_obs']))
            num_out_of_range.append(int(row['num_out_of_range']))
            num_moving.append(int(row['num_moving']))
    return epoch, num_reach_target, num_hit_obs, num_out_of_range, num_moving


def draw(epoch, num_reach_target, num_hit_obs, num_out_of_range, num_moving, color_list, pic_name):
    plt.plot(epoch, num_reach_target, color=color_list[7])
    plt.plot(epoch, num_hit_obs, color=color_list[4])
    # plt.plot(epoch, num_out_of_range, color=color_list[0])
    plt.plot(epoch, num_moving, color=color_list[3])

    plt.legend(['num_reach_target', 'num_hit_obs', 'num_moving'], loc='lower right', fontsize=10)  # 图例
    plt.ylabel('number', fontsize=10)
    plt.xlabel('episodes', fontsize=10)
    plt.title('result', fontsize=10)
    plt.savefig(pic_name)


if __name__ == "__main__":
    path = '../output/result.csv'
    pic_name = '../output/result.png'
    color_list = sns.hls_palette(8, l=.3, s=.8)
    epoch, num_reach_target, num_hit_obs, num_out_of_range, num_moving = get_data(path)
    draw(epoch, num_reach_target, num_hit_obs, num_out_of_range, num_moving, color_list, pic_name)
