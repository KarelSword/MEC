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

import profile

from My_code.K_center import Draw


# MBS类
class MBS:
    inner_points = []  # MBS内部的点的索引
    center = []  # MBS的中心位置
    index = -1  # MBS的编号

    def __init__(self):
        self.inner_points = []
        self.center = []
        self.index = -1

    def __del__(self):
        self.inner_points = []
        self.center = []
        self.index = -1


# 保护栈的操作
def protect_scene(mbs_set):
    new_set = {}
    for index, value in mbs_set.items():

        mbs = MBS()
        mbs.inner_points = value.inner_points.copy()
        mbs.index = value.index
        mbs.center = value.center.copy()

        new_set[str(index)] = mbs

    return new_set


# 对未覆盖的点到MBS的距离进行排序
def order_distance(uncovered_points, ball_center):
    """

    :param uncovered_points:
    :param ball_center:
    :return:
    """
    uncovered_location = __points[uncovered_points]

    # dis 未覆盖的点到各个球中心的距离
    dis = distance.cdist(uncovered_location, ball_center, 'euclidean')

    # 每个点到最近求的距离
    dis_row_min = np.min(dis, axis=1)

    # furthest violator在dis的距离
    index_in_dis = np.argmax(dis_row_min)

    # furthest violator在uncovered_points中的索引
    max_point_index = uncovered_points[index_in_dis]

    # furthest violator的距离
    to_ball_dis = dis[index_in_dis]

    to_ball_key = np.argsort(to_ball_dis)

    return max_point_index, to_ball_key, to_ball_dis


# 初始化的操作
def initialize(number):
    # points = np.random.normal(-100, 100, size=(number, 2))
    # points = np.random.uniform(-100, 100, size=(number, 2))
    points = pd.read_csv('C:/Users/74412/Desktop/article/k_center/k_center_test.csv', usecols=[1, 2]).values
    # points = pd.read_csv('F:/Scientific_Literature/写论文/test_data/Egypt - 7,146.csv', usecols=[1, 2]).values

    """
    points = np.round(points, 2)
    df = pd.DataFrame(points)
    df.to_csv('C:/Users/74412/Desktop/article/k_center/k_center_0410.csv')
    """

    return points


# 迭代函数前的准备
def start(mbs_set, optimal, uncovered_points, ball_center, mbs_count):
    """
    迭代函数前的准备，将0点并入到迭代函数中
    :param mbs_set:             MBS的集合
    :param optimal:             全局最优值
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
    mbs.center = __points[0]
    mbs_set[str(mbs_count)] = mbs

    uncovered_points.remove(current_point)
    mbs_count += 1

    ball_center.append(list(mbs.center))

    return mbs_set, optimal, ball_center, mbs_count


# 核心递归函数
def core(mbs_set, optimal, radius, uncovered_points, ball_center, mbs_count, opt_ball_center):
    """

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

    # chosen_point, ball_index，各点对应最近的球
    chosen_point, ball_index, ball_distance = order_distance(uncovered_points, ball_center)

    # 求目前最近到求的距离
    to_center_distance = ball_distance[ball_index[0]]

    # 如果到球的距离小于目前的局部最优半径
    if to_center_distance <= radius:
        if radius < opt:
            opt = radius
            opt_ball_center = ball_center.copy()

        return opt, mbs_count, opt_ball_center

    # 球的顺序，为下面并入球的顺序
    ball_order = list(ball_index)

    # 如果mbs_count的数目不满足__k
    if mbs_count < __k:
        # 在0的位置，插入mbs_count
        ball_order.insert(0, mbs_count)

        for j in range(len(ball_order)):
            if j == 0:
                mbs = MBS()
                mbs.inner_points.append(chosen_point)
                mbs.index = mbs_count
                mbs.center = __points[chosen_point]
                mbs_set[str(mbs_count)] = mbs

                uncovered_points.remove(chosen_point)
                mbs_count += 1

                ball_center.append(list(mbs.center))

                # 用来递归的mbs
                recursion_mbs_set = mbs_set
                mbs_set = protect_scene(mbs_set)

                opt, mbs_count, opt_ball_center = \
                    core(recursion_mbs_set, opt, radius,
                         uncovered_points.copy(), copy.copy(ball_center), mbs_count, copy.copy(opt_ball_center))

                ball_center.pop()
                mbs_count -= 1

                mbs_set[str(mbs_count)].inner_points.remove(chosen_point)
                uncovered_points.append(chosen_point)

            else:
                inner_points = []

                ball_key = ball_order[j]
                # 将furthest violator并入到最近的MBS中，求最小包围圆
                inner_points.append(chosen_point)
                inner_points.extend(mbs_set[str(ball_key)].inner_points)
                inner_points_location = __points[inner_points]

                temp_miniball = miniball.Miniball(inner_points_location)
                radii = math.sqrt(temp_miniball.squared_radius())

                if radii < opt and radius < opt:
                    mbs_set[str(ball_key)].inner_points.append(chosen_point)
                    mbs_set[str(ball_key)].center = temp_miniball.center()
                    ball_center[ball_key] = temp_miniball.center()

                    if radii >= radius:
                        radius = radii

                    uncovered_points.remove(chosen_point)

                    if len(uncovered_points) > 0:
                        recursion_mbs_set = mbs_set
                        mbs_set = protect_scene(mbs_set)

                        opt, mbs_count, opt_ball_center = \
                            core(recursion_mbs_set, opt, radius,
                                 uncovered_points.copy(), copy.copy(ball_center), mbs_count, copy.copy(opt_ball_center))
                    else:
                        opt = radius
                        opt_ball_center = copy.copy(ball_center)

                    mbs_set[str(ball_key)].inner_points.remove(chosen_point)
                    uncovered_points.append(chosen_point)

    # 递归程序
    else:
        for j in range(__k):
            inner_points = []

            ball_key = ball_order[j]
            # 将furthest violator并入到最近的MBS中，求最小包围圆
            inner_points.append(chosen_point)
            inner_points.extend(mbs_set[str(ball_key)].inner_points)
            inner_points_location = __points[inner_points]

            temp_miniball = miniball.Miniball(inner_points_location)
            radii = math.sqrt(temp_miniball.squared_radius())

            if radii < opt and radius < opt:
                mbs_set[str(ball_key)].inner_points.append(chosen_point)
                mbs_set[str(ball_key)].center = temp_miniball.center()
                ball_center[ball_key] = temp_miniball.center()

                if radii >= radius:
                    radius = radii

                uncovered_points.remove(chosen_point)

                if len(uncovered_points) > 0:
                    recursion_mbs_set = mbs_set
                    mbs_set = protect_scene(mbs_set)

                    opt, mbs_count, opt_ball_center = \
                        core(recursion_mbs_set, opt, radius, uncovered_points.copy(),
                             copy.copy(ball_center), mbs_count, copy.copy(opt_ball_center))

                if len(uncovered_points) == 0:
                    opt = radius
                    opt_ball_center = copy.copy(ball_center)

                mbs_set[str(ball_key)].inner_points.remove(chosen_point)

                uncovered_points.append(chosen_point)

    return opt, mbs_count, opt_ball_center


# 主函数
def main():
    start_time = time.time()

    __mbs_count = 0  # mbs的数目
    __mbs_set = {}  # mbs的集合
    __uncovered_points = []  # 未覆盖点的集合
    __optimal = float("inf")    # 最优值
    __radius = 0    # 半径
    __ball_center = []  # 球的中心

    # 对未覆盖的点进行初始化操作
    for i in range(__number):
        __uncovered_points.append(i)

    # 为递归函数做前期准备
    __mbs_set, __optimal, __ball_center, __mbs_count = \
        start(__mbs_set, __optimal,
              __uncovered_points, __ball_center, __mbs_count)

    __opt_ball_center = __ball_center.copy()

    # 核心的递归函数
    __optimal, __mbs_count, __opt_ball_center = \
        core(copy.deepcopy(__mbs_set), __optimal, __radius, __uncovered_points,
             copy.deepcopy(__ball_center), __mbs_count, copy.deepcopy(__opt_ball_center))

    # print("__optimal")
    # print(__optimal)

    end_time = time.time()
    secs = end_time - start_time
    print(" took", secs, "seconds")

    return __optimal, __opt_ball_center


if __name__ == "__main__":

    result = []
    for i in range(6, 20):

        __k = 5
        __number = i  # 点的数目

        # 生成点，初始化操作
        __points = initialize(__number)
        __points = __points[0:i]

        opt, opt_ball_center = main()
        result.append(opt)

    df = pd.DataFrame(result)
    df.to_csv('C:/Users/74412/Desktop/article/k_center/camp_RDF.csv')

    # profile.run('main()')

    """
    Draw.show_scatter(__points, 0, "#00CC66")
    Draw.draw_mbs(opt_ball_center, opt)
    Draw.plt.axis("equal")
    Draw.plt.show()
    """

    """
    mec_count = 0
    __number = 100  # 点的数目

    start_time = time.time()

    sum_opt = 0
    count = 0

    loop_number = 11
    for i in range(loop_number):
        for j in range(15, 20):
            __k = j

            __points = initialize(__number)
            temp_opt, opt_ball_center = main()

            if temp_opt <= 50:
                count += j
                sum_opt += temp_opt
                break

    ave_opt = sum_opt/loop_number
    ave_count = count/loop_number
    print(ave_opt)
    print(ave_count)

    end_time = time.time()
    secs = end_time - start_time
    ave_time = secs/loop_number
    print(" took", ave_time, "seconds")
    """
