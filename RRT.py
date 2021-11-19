import matplotlib.pyplot as plt
import random
import math
import copy
from etc import *
from utils import *

show_animation = True


class Node(object):
    """
    RRT Node
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None


class RRT(object):
    def __init__(self, start, goal, obstacle_list, rand_area):
        """Setting Parameter

        Args:
            start [x,y]: 开始节点
            goal [x,y]: 目标节点
            obstacle_list [x,y,size]:  障碍物位置及大小
            rand_area [min,max]:  random sampling Area # 这里假设x的范围和y的范围属于(rand_area[0],rand_area[1])
        """
        self.start = Node(start[0], start[1])  # 归一化方便处理
        self.end = Node(goal[0], goal[1])
        self.min_rand = rand_area[0]
        self.max_rand = rand_area[1]
        # ------matter param------------
        # 这个写死的参数不太好定
        self.expandPis = 1 / 33 * math.sqrt(
            pow(goal[0] - start[0], 2) + pow(goal[1] - start[1], 2))
        # 选择终点的概率是0.05,这个参数matter
        self.goalSampleRate = 0.10
        self.EPSILON = 5
        self.NUMNODES = 10000
        # ----------------------------
        self.obstacleList = obstacle_list
        self.nodeList = [self.start]
        self.cuerrentState = "buildTree"
        self.count = 0

    def random_node(self):
        """
        产生随机节点
        """
        while True:
            node_x = random.uniform(self.min_rand, self.max_rand)
            node_y = random.uniform(self.min_rand, self.max_rand)
            node = Node(node_x, node_y)
            if self.is_collision(node) == False:
                return node

    def get_nearest_list_index(node_list, rnd):
        d_list = [(node.x - rnd[0])**2 + (node.y - rnd[1])**2
                  for node in node_list]
        min_index = d_list.index(min(d_list))
        return min_index

    def is_collision(self, new_node):
        for (ox, oy, size) in self.obstacleList:
            dx = ox - new_node.x
            dy = oy - new_node.y
            d = math.sqrt(dx * dx + dy * dy)
            if d <= size:
                return True
        return False

    def is_bound(self, new_node):
        if new_node.x < self.min_rand or new_node.x > self.max_rand or new_node.y < self.min_rand or new_node.y > self.max_rand:
            return True
        return False

    def distance(self, p1, p2):
        return math.sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) *
                         (p1.y - p2.y))

    def step_from_to(self, p, rnd, index):
        if self.distance(p, rnd) < self.expandPis:
            return rnd
        else:
            theta = math.atan2(rnd.y - p.y, rnd.x - p.x)
            new_node = copy.deepcopy(p)
            new_node.x += self.expandPis * math.cos(theta)
            new_node.y += self.expandPis * math.sin(theta)
            new_node.parent = index

            return new_node

    def planning(self):
        """
        Path planing
        """
        while True:
            if self.cuerrentState == "buildTree":
                self.count += 1
                if self.count < self.NUMNODES:
                    foundNext = False
                    while foundNext == False:
                        # Random Sampling
                        if random.random() >= self.goalSampleRate:
                            rnd = self.random_node()
                        else:
                            rnd = Node(self.end.x, self.end.y)

                        parentNode = self.nodeList[0]
                        for index, item in enumerate(self.nodeList):
                            # print(index)
                            if self.distance(item, rnd) <= self.distance(
                                    parentNode, self.end):
                                new_node = self.step_from_to(item, rnd, index)
                                if self.is_collision(new_node) == False:
                                    parentNode = item
                                    foundNext = True

                                if self.is_bound(new_node) == False:
                                    parentNode = item
                                    parentIndex = index
                                    foundNext = True

                    new_node = self.step_from_to(parentNode, rnd, parentIndex)
                    self.nodeList.append(new_node)
                    # self.draw_graph()

                    # check goal
                    dx = new_node.x - self.end.x
                    dy = new_node.y - self.end.y
                    d = math.sqrt(dx * dx + dy * dy)
                    if d <= self.expandPis:
                        # TODO： 感觉这里应该换一个参数表示
                        # print("Goal!")
                        self.currentState = 'goalFound'
                        return

                    # if len(self.nodeList) % 1000 == 0:
                    #     print("点_list的大小是： ", len(self.nodeList))

                else:
                    print("Ran out of nodes... :(")
                    return

    def find_path(self):
        # print("点_list的大小是： ", len(self.nodeList))

        path = [[self.end.x, self.end.y]]
        last_index = len(self.nodeList) - 1
        while self.nodeList[last_index].parent is not None:
            node = self.nodeList[last_index]
            path.append([node.x, node.y])
            last_index = node.parent
        path.append([self.start.x, self.start.y])
        return path

    def draw_graph(self, rnd=None):
        plt.clf()  # 清除上次画的图
        if rnd is not None:
            plt.plot(rnd[0], rnd[1], "^g")
        for node in self.nodeList:
            if node.parent is not None:
                plt.plot([node.x, self.nodeList[node.parent].x],
                         [node.y, self.nodeList[node.parent].y], "-g")

        for (ox, oy, size) in self.obstacleList:
            # plt.plot(ox, oy, "sk", ms=1 * size)  # 障碍物大小* 10
            plt.plot(ox, oy, color='#000000', marker="1")

        plt.plot(self.start.x, self.start.y, "^r")
        plt.plot(self.end.x, self.end.y, "^b")
        plt.axis([self.min_rand, self.max_rand, self.min_rand, self.max_rand])
        # plt.grid(True)
        plt.pause(0.001)

    def draw_static(self, path):
        """
        画出静态图像
        """
        plt.clf()  # 清除上次画的图

        for node in self.nodeList:
            if node.parent is not None:
                plt.plot([node.x, self.nodeList[node.parent].x],
                         [node.y, self.nodeList[node.parent].y], "-g")

        for (ox, oy, size) in self.obstacleList:
            plt.plot(ox, oy, color='#000000', marker="1")

        plt.plot(self.start.x, self.start.y, "^r")
        plt.plot(self.end.x, self.end.y, "^b")
        plt.axis([self.min_rand, self.max_rand, self.min_rand, self.max_rand])

        plt.plot([data[0] for data in path], [data[1] for data in path], '-r')
        # plt.grid(True)
        plt.show()


def main():
    obstacles = generate_Obstacles(num_obstacles, radius)
    obstacle_list = []
    for item in obstacles:
        obstacle_list.append((item.x, item.y, 1))

    # Set Initial parameters
    rrt = RRT(start=[-348, -338],
              goal=[348, 338],
              rand_area=[-1 * radius, radius],
              obstacle_list=obstacle_list)
    rrt.planning()
    path = rrt.find_path()
    print("path里的点的数量是： ", len(path))
    print(path)
    print("you should move to ", path[-2])

    # Draw final path
    if show_animation:
        plt.close()
        rrt.draw_static(path)


def get_action_by_RTT(start_point, goal_point, obstacles):
    obstacle_list = []
    for item in obstacles:
        obstacle_list.append((item.x, item.y, 1))

    # Set Initial parameters
    rrt = RRT(start=[start_point.x, start_point.y],
              goal=[goal_point[0], goal_point[1]],
              rand_area=[-1 * radius, radius],
              obstacle_list=obstacle_list)
    rrt.planning()
    path = rrt.find_path()

    dx = path[-2][0] - start_point.x
    dy = path[-2][1] - start_point.y
    l = math.sqrt(dx * dx + dy * dy)
    action_x = dx / l
    action_y = dy / l

    return [action_x, action_y]


if __name__ == '__main__':
    main()
