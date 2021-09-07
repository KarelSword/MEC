import profile
import numpy as np
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.spatial import ConvexHull
import pandas as pd
import miniball
import math
import time


# 无向图中点的类
class Point:
    index = -1  # 点的编号
    location = []  # 点的坐标
    neighbor_info = []  # 点的相邻信息
    degree = 0  # 点的度数
    belongs_to = -1  # 用于判定点属于哪一个MBS
    processed = 0  # 判断点有没有被执行过

    def __init__(self, location, index):
        self.location = location
        self.neighbor_info = []
        self.degree = 0
        self.index = index
        self.belongs_to = -1
        self.processed = 0

    def __del__(self):
        self.location = []
        self.neighbor_info = []
        self.degree = 0
        self.index = -1
        self.belongs_to = -1
        self.processed = 0


# MBS类
class MBS:
    inner_points = []  # MBS内部的点的索引
    count = 0  # MBS内部点的数量
    center = []  # MBS的中心位置
    processed = 0  # MBS是否处理过
    index = -1  # MBS的编号

    def __init__(self):
        self.inner_points = []
        self.count = 0
        self.center = []
        self.processed = 0
        self.index = -1

    def __del__(self):
        self.inner_points = []
        self.count = 0
        self.center = []
        self.processed = 0
        self.index = -1

    def print(self):
        print("now the index of mbs is: ")
        print(str(self.index))
        print(self.inner_points)
        print("_______________________")


# 读取点的位置信息
def read_location(points, points_set):
    points_location = []
    for key in points:
        location = points_set[str(key)].location
        points_location.append(location)

    return points_location


# 初始化操作
def initialize(number, distributed):
    # 初始化点的坐标
    np.random.seed()
    if distributed == "uniform":
        points = np.random.uniform(-100, 100, size=(number, 2))
        points = np.round(points, 2)
        """
        df = pd.DataFrame(points)
        df.to_csv('C:/Users/74412/Desktop/article/PandasNumpy.csv')
        # points = np.round(points, 3)
        """
    if distributed == "normal":
        points = np.random.normal(-100, 100, size=(number, 2))
    # 初始化点类的集合
    points_set = {}

    return points_set, points


# 求出边界点
def solve_convexhull(uncovered_points, points_set):
    if len(uncovered_points) >= 3:  # 如果seq 数目大于3,则执行convexhull程序
        uncovered_points = np.array(uncovered_points)
        try:
            boundarySetInOrder = ConvexHull(uncovered_points).vertices
        except:
            points_location = np.array(read_location(uncovered_points, points_set))
            index = np.lexsort([points_location[:, 1], points_location[:, 0]])
            points_location = points_location[index]

    if len(uncovered_points) == 2:  # 如果seq 数目等于2，逆时针选取点
        points_location = np.array(read_location(uncovered_points, points_set))
        seq_1 = points_location[0]
        seq_2 = points_location[1]
        if seq_1[0] < seq_2[0]:
            bound_order = [seq_1, seq_2]
        if seq_1[0] > seq_2[0]:
            bound_order = [seq_2, seq_1]
        if seq_1[0] == seq_2[0]:
            if seq_1[1] < seq_2[1]:
                bound_order = [seq_1, seq_2]
            if seq_1[1] > seq_2[1]:
                bound_order = [seq_2, seq_1]

    if len(uncovered_points) == 1:  # 如果seq 数目等于1，直接选取
        bound_order = uncovered_points

    return bound_order



# 主程序
def main():
    RADIUS = 50
    __number = 80
    __uav_position_set = []  # 无人机位置的点集
    __uncovered_points = []  # 未覆盖的点


    __points_set, __points = initialize(__number, "uniform")

    for i in range(__number):
        __uncovered_points.append(i)

    while __uncovered_points:  # seq 点的队列
        bound_order = solve_convexhull(__uncovered_points, __points_set)



if __name__ == "__main__":
    main()