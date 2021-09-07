"""
尝试通过K-center 问题来求出GDC问题的最优解
2021年1月16日
作者：karel
"""

import math
import time
import copy
import numpy as np
import pandas as pd
from scipy.spatial import distance
import miniball
from scipy.spatial import ConvexHull

from My_code.K_center import Draw


# 点的类
class Point:
    index = -1  # 点的编号
    location = []  # 点的坐标

    def __init__(self, location, index):
        self.location = location
        self.index = index

    def __del__(self):
        self.location = []
        self.index = -1


# MBS类
class MBS:
    inner_points = []  # MBS内部的点的索引
    count = 0  # MBS内部点的数量
    center = []  # MBS的中心位置
    index = -1  # MBS的编号

    def __init__(self):
        self.inner_points = []
        self.count = 0
        self.center = []
        self.index = -1

    def __del__(self):
        self.inner_points = []
        self.count = 0
        self.center = []
        self.index = -1


# 求出两点之间的距离
def solve_distance(A, B):
    return np.sqrt(np.square(A[0] - B[0]) + np.square(A[1] - B[1]))


# 对字典中的值进行排序，并且返回字典
def dic_sorted(dic, flag):
    """
    True 为降序，False 为升序
    :param dic:
    :param flag:
    :return:
    """
    dic = sorted(dic.items(), key=lambda x: x[1], reverse=flag)
    temp = {}
    for item in range(len(dic)):
        temp[dic[item][0]] = dic[item][1]
    return temp


# 读取点的位置信息
def read_location(points, points_set):
    points_location = []
    for key in points:
        location = points_set[str(key)].location
        points_location.append(location)

    return points_location


'''
# 求出来最远点的基本坐标
def furthest_violator(points, base, points_set):
    """
    求出来最远点的基本坐标
    :param points: 
    :param base:    基点，个数为多个
    :param points_set:  点的几何
    :return:
    """
    points_location = read_location(points, points_set)
    base_location = read_location(base, points_set)
    dis = distance.cdist(points_location, base_location, 'euclidean')

    index = np.argmax(dis)
    index = int(index / len(base))
    index = points[index]

    return index
'''


# 对未覆盖的点到MBS的距离进行排序
def order_distance(uncovered_points, ball_center, points_set):
    max_distance = {}
    max_distance_index = {}
    uncovered_location = read_location(uncovered_points, points_set)

    # dis 未覆盖的点到各个球中心的距离
    dis = distance.cdist(uncovered_location, ball_center, 'euclidean')

    # value:每个未覆盖的点到所有球的距离
    for index, value in enumerate(dis):
        max_dis = np.min(value)  # 返回该点距离最近的球的距离
        var = uncovered_points[index]  # 返回该点的序号
        max_distance[var] = max_dis  # 组成字典

    max_distance = dic_sorted(max_distance, True)   # 取其中的最大值

    max_point_index = list(max_distance.keys())[0]  # furthest violator的索引

    # 返回furthest violator在dis中的索引
    index_in_dis = np.where(np.array(uncovered_points) == max_point_index)[0][0]
    to_ball_dis = dis[index_in_dis]                 # furthest violator的距离

    for index, value in enumerate(to_ball_dis):
        max_distance_index[index] = value

    # furthest violator 到各个球的距离，排序之后
    max_distance_index = dic_sorted(max_distance_index, False)

    return max_distance, max_distance_index


# 初始化的操作
def initialize(number):
    np.random.seed()
    points_set = {}
    # points = np.random.normal(-100, 100, size=(number, 2))
    # points = np.random.uniform(-100, 100, size=(number, 2))
    points = pd.read_csv('C:/Users/74412/Desktop/article/k_center/k_center_0409.csv', usecols=[1, 2]).values

    """
    points = np.round(points, 2)
    df = pd.DataFrame(points)
    df.to_csv('C:/Users/74412/Desktop/article/k_center/k_center_0409.csv')
    """

    for i in range(0, len(points)):
        a = Point(points[i], i)
        points_set[str(i)] = a

    return points_set, points


# 迭代函数前的准备
def start(start_point, points_set, mbs_set, optimal, radius, uncovered_points, ball_center, mbs_count):
    """
    迭代函数前的准备，将0点并入到迭代函数中
    :param start_point          开始的点
    :param points_set:          点集合
    :param mbs_set:             MBS的集合
    :param optimal:             全局最优值
    :param radius:              局部最优值
    :param uncovered_points:    未覆盖的点
    :param ball_center:         MBS的坐标
    :param mbs_count:           MBS目前的数目
    :return:                    执行后的结果
    """
    current_point = 0  # 选取点0来作为初始的点

    mbs = MBS()
    mbs.inner_points.append(current_point)
    mbs.count = 1
    mbs.index = mbs_count
    mbs.center = points_set[str(current_point)].location
    mbs_set[str(mbs_count)] = mbs

    uncovered_points.remove(current_point)
    mbs_count += 1

    ball_center.append(list(mbs.center))

    return mbs_set, optimal, ball_center, mbs_count


# 核心递归函数
def core(points_set, mbs_set, optimal, radius, uncovered_points, ball_center, mbs_count, opt_ball_center):
    """

    :param points_set:          点集合
    :param mbs_set:             MBS的集合
    :param optimal:             全局最优值
    :param radius:              局部最优值
    :param uncovered_points:    未覆盖的点
    :param ball_center:         MBS的目前的坐标
    :param mbs_count:           MBS目前的数目
    :param opt_ball_center:     MBS目前最优的坐标
    :return:                    执行后的结果
    """
    opt = optimal

    # ball_distance,各点到球的距离，ball_index，各点对应最近的球
    ball_distance, ball_index = order_distance(uncovered_points, ball_center, points_set)

    current_point = list(ball_distance.keys())[0]  # 目前选取的点
    ball_key = list(ball_index.keys())[0]  # 目前选择的球的编号

    """
    print("+++++++++++++++++++++++++++++++++++++++")
    print("current_point:")
    print(current_point)
    """

    # 求目前到求的距离
    # to_center_distance = solve_distance(points_set[str(current_point)].location, mbs_set[str(ball_key)].center)
    to_center_distance = list(ball_index.values())[0]

    """
    print("to_center_distance:")
    print(to_center_distance)
    print("radius:")
    print(radius)
    """

    # 如果到球的距离小于目前的局部最优半径
    if to_center_distance <= radius:
        if radius < opt:
            opt = radius
            opt_ball_center = copy.deepcopy(ball_center)
            mbs_set[str(ball_key)].inner_points.append(current_point)

            # print("!!!include!!!")
            """
            if len(uncovered_points) > 0:
                uncovered_points.remove(current_point)
            """

            return opt, mbs_count, opt_ball_center

    # 球的顺序，为下面并入球的顺序
    ball_order = list(ball_index.keys())

    # 如果mbs_count的数目不满足__k
    if mbs_count < __k:
        ball_order.insert(0, mbs_count)

        for j in range(len(ball_order)):
            if j == 0:
                mbs = MBS()
                mbs.inner_points.append(current_point)
                mbs.count = 1
                mbs.index = mbs_count
                mbs.center = points_set[str(current_point)].location
                mbs_set[str(mbs_count)] = mbs

                uncovered_points.remove(current_point)
                mbs_count += 1

                ball_center.append(list(mbs.center))

                """
                print("------------------------------")
                print("mbs_count < __k")
                print("ball_center")
                print(ball_center)
                print("uncovered_points")
                print(uncovered_points)
                print("mbs_count")
                print(mbs_count)
                """

                opt, mbs_count, opt_ball_center = \
                    core(points_set, copy.deepcopy(mbs_set), opt, radius,
                         uncovered_points, copy.deepcopy(ball_center), mbs_count, copy.deepcopy(opt_ball_center))


                ball_center.pop()
                mbs_count -= 1

                mbs_set[str(mbs_count)].inner_points.remove(current_point)
                uncovered_points.append(current_point)

            else:
                inner_points = []

                ball_key = ball_order[j]
                # 将furthest violator并入到最近的MBS中，求最小包围圆
                inner_points.append(current_point)
                inner_points.extend(mbs_set[str(ball_key)].inner_points)
                inner_points_location = read_location(inner_points, points_set)

                temp_miniball = miniball.Miniball(inner_points_location)
                radii = math.sqrt(temp_miniball.squared_radius())

                """
                print("************************************")
                print("------------------------------")
                print("mbs_count < __k")
                print("current_point")
                print(current_point)
                print("ball_center")
                print(ball_center)
                print("radii")
                print(radii)
                print("radius")
                print(radius)
                print("uncovered_points")
                print(uncovered_points)
                print("mbs_count")
                print(mbs_count)
                """

                if radii < opt and radius < opt:
                    mbs_set[str(ball_key)].inner_points.append(current_point)
                    mbs_set[str(ball_key)].center = temp_miniball.center()
                    ball_center[ball_key] = temp_miniball.center()

                    if radii >= radius:
                        radius = radii

                    uncovered_points.remove(current_point)

                    if len(uncovered_points) > 0:
                        opt, mbs_count, opt_ball_center = \
                            core(points_set, copy.deepcopy(mbs_set), opt, radius,
                                 uncovered_points, copy.deepcopy(ball_center), mbs_count, copy.deepcopy(opt_ball_center))
                    else:
                        opt = radius
                        opt_ball_center = copy.deepcopy(ball_center)

                    mbs_set[str(ball_key)].inner_points.remove(current_point)
                    uncovered_points.append(current_point)

    # 递归程序
    else:
        for j in range(__k):
            inner_points = []

            ball_key = ball_order[j]
            # 将furthest violator并入到最近的MBS中，求最小包围圆
            inner_points.append(current_point)
            inner_points.extend(mbs_set[str(ball_key)].inner_points)
            inner_points_location = read_location(inner_points, points_set)

            temp_miniball = miniball.Miniball(inner_points_location)
            radii = math.sqrt(temp_miniball.squared_radius())

            """
            print("------------------------------")
            print("mbs_count == __k")
            print("radii")
            print(radii)
            print("optimal")
            print(opt)
            print("current_point_inner")
            print(current_point)
            print("uncovered_points")
            print(uncovered_points)
            print("mbs_count")
            print(mbs_count)
            """

            if radii < opt and radius < opt:
                mbs_set[str(ball_key)].inner_points.append(current_point)
                mbs_set[str(ball_key)].center = temp_miniball.center()
                ball_center[ball_key] = temp_miniball.center()

                if radii >= radius:
                    radius = radii

                uncovered_points.remove(current_point)

                if len(uncovered_points) > 0:
                    opt, mbs_count, opt_ball_center = \
                        core(points_set, copy.deepcopy(mbs_set), opt, radius, uncovered_points,
                             copy.deepcopy(ball_center), mbs_count, copy.deepcopy(opt_ball_center))

                if len(uncovered_points) == 0:
                    opt = radius
                    opt_ball_center = copy.deepcopy(ball_center)

                mbs_set[str(ball_key)].inner_points.remove(current_point)

                uncovered_points.append(current_point)

    # print(ball_center)

    return opt, mbs_count, opt_ball_center


# 主函数
def main():
    start_time = time.time()

    __number = 400   # 点的数目
    __mbs_count = 0  # mbs的数目
    __mbs_set = {}  # mbs的集合
    __uncovered_points = []  # 未覆盖点的集合
    __covered_points = []  # 已覆盖点的集合
    __optimal = float("inf")    # 最优值
    __radius = 0    # 半径
    __ball_center = []  # 球的中心

    # 生成点，初始化操作
    __points_set, __points = initialize(__number)

    # 对未覆盖的点进行初始化操作
    for i in range(__number):
        __uncovered_points.append(i)

    """
    # 读取点的位置信息
    __points_location = read_location(__uncovered_points, __points_set)
    hull = ConvexHull(__points_location)
    __start_point = hull.vertices[0]
    """

    # 为递归函数做前期准备
    __mbs_set, __optimal, __ball_center, __mbs_count = \
        start(0, __points_set, __mbs_set, __optimal, __radius,
              __uncovered_points, __ball_center, __mbs_count)

    __opt_ball_center = __ball_center.copy()

    # 核心的递归函数
    __optimal, __mbs_count, __opt_ball_center = \
        core(__points_set, copy.deepcopy(__mbs_set), __optimal, __radius, __uncovered_points,
             copy.deepcopy(__ball_center), __mbs_count, copy.deepcopy(__opt_ball_center))

    print("__optimal")
    print(__optimal)

    end_time = time.time()
    secs = end_time - start_time
    print(" took", secs, "seconds")

    return __optimal, __points, __opt_ball_center


if __name__ == "__main__":
    __k = 8

    opt, points, opt_ball_center = main()
    """
    # print(opt)
    Draw.show_scatter(points, 0, "#00CC66")
    Draw.draw_mbs(opt_ball_center, opt)
    Draw.plt.axis("equal")
    Draw.plt.show()
    """
    """
    mec_count = 0
    start_time = time.time()

    sum_opt = 0
    count = 0
    for i in range(1):
        # print("-----------------------")
        for j in range(5, 10):
            __k = j

            temp_opt, points, opt_ball_center = main()

            if temp_opt <= 50:
                count += j
                sum_opt += temp_opt
                break

    ave_opt = sum_opt/1
    ave_count = count/1
    print(ave_opt)
    print(ave_count)

    end_time = time.time()
    secs = end_time - start_time
    ave_time = secs/1
    print(" took", ave_time, "seconds")
    """
