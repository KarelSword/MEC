"""
尝试通过K-center 问题来求出GDC问题的最优解
2021年3月20日
作者：karel

SDF算法目前克服的问题。
1.RDF实现的问题。
2.SDF如何实现保护现场的问题。
    2.1
    2.2
    2.3
3.node入stack的问题。
4.if to_ball_key >= mbs_count的问题。
5.mbs radii,mbs数目不足的问题。
6.protect_scenne问题
"""
import math
import copy
import time
import profile

import numpy as np
import pandas as pd

import miniball
from scipy.spatial import distance

from My_code.K_center import Draw
from My_code.K_center import Check_Where_Wrong


# 保护栈的操作
def protect_scene(mbs_set):
    new_set = {}
    for index, value in mbs_set.items():

        mbs_inner_points = value.copy()

        new_set[str(index)] = mbs_inner_points

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

    # furthest violator在dis的索引
    index_in_dis = np.argmax(dis_row_min)

    # furthest violator在uncovered_points中的索引
    max_point_index = uncovered_points[index_in_dis]

    # furthest violator的距离
    to_ball_dis = dis[index_in_dis]

    to_ball_key = np.argsort(to_ball_dis)

    return max_point_index, to_ball_key, to_ball_dis


# 初始化的操作
def initialize(number):
    np.random.seed()
    # points = np.random.uniform(-100, 100, size=(number, 2))
    # points = np.random.normal(0, 1, size=(number, 2))
    # points = pd.read_csv('C:/Users/74412/Desktop/article/k_center/SDF_Q/k_center_test_042591.csv', usecols=[1, 2]).values
    # points = pd.read_csv('F:/Scientific_Literature/写论文/test_data/USA - 13,509.csv', usecols=[1, 2]).values

    name = 'C:/Users/74412/Desktop/article/Article_Data/3D/Normal/' + str(number) + '_' + str(index) + '.csv'
    points = pd.read_csv(name, usecols=[1, 2, 3]).values

    """
    points = np.round(points, 2)s
    df = pd.DataFrame(points)
    df.to_csv('C:/Users/74412/Desktop/article/k_center/k_center_test_0425.csv')
    """

    return points


# 核心递归函数
def core(uncovered_points):
    """
    :param uncovered_points:    未覆盖的点
    :return:                    执行后的结果
    """

    # 变量的初始化
    optimal = float("inf")    # 最优值
    opt_ball_center = []
    radius = 0                # 半径
    mec_count = 0             # MEC执行的次数

    ball_center = []          # 目前最优球的中心
    stack = []
    mbs_set = {}

    # stack 中每个节点的信息
    node = {"current_point": 0,
            "to_ball_key": 0,
            "to_ball_distance": 0,
            "radius": 0,
            "mbs_set": {},
            "uncovered_points": uncovered_points,
            "mbs_count": 0,
            "ball_center": []
            }

    stack.append(node)

    # 当栈不为空的时候持续执行
    while stack:

        # stack 的节点出栈
        node = stack.pop()

        current_point: int = node["current_point"]
        to_ball_key: int = node["to_ball_key"]
        to_center_distance: float = node["to_ball_distance"]
        radius: int = node["radius"]
        mbs_set: dict = node["mbs_set"]
        uncovered_points: list = node["uncovered_points"]
        mbs_count: int = node["mbs_count"]
        ball_center: list = node["ball_center"]

        # 如果要并入的ball_ket大于mbs_count
        if to_ball_key >= mbs_count:
            # 成功则将该点从未覆盖中移除
            uncovered_points.remove(current_point)

            # 创建一个新的mbs,将目前的点并入
            mbs_inner_points = [current_point]
            mbs_set[str(mbs_count)] = mbs_inner_points

            ball_center.append(__points[current_point])

            mbs_count += 1
            # next_point:下一个点, ball_index:下一个点对应球在ball_distance中的索引排序后,
            # ball_distance:各点对应最近的球的距离
            next_point, ball_index, ball_distance = order_distance(uncovered_points, ball_center)

            ball_order = list(ball_index)
            ball_order.reverse()

            for i in range(len(ball_order)):
                # stack 中每个节点的信息
                node = {"current_point": next_point,
                        "to_ball_key": ball_order[i],
                        "to_ball_distance": ball_distance[ball_order[i]],
                        "radius": radius,
                        "mbs_set": protect_scene(mbs_set),
                        "uncovered_points": uncovered_points.copy(),
                        "mbs_count": mbs_count,
                        "ball_center": ball_center.copy()
                        }

                stack.append(node)

            if mbs_count < __k:
                node = {"current_point": next_point,
                        "to_ball_key": mbs_count,
                        "to_ball_distance": 0,
                        "radius": radius,
                        "mbs_set": protect_scene(mbs_set),
                        "uncovered_points": uncovered_points.copy(),
                        "mbs_count": mbs_count,
                        "ball_center": ball_center.copy()
                        }
                stack.append(node)

        else:
            # 如果到球的距离小于目前的局部最优半径
            if to_center_distance < radius:
                # 如果比目前最优值更有，则需要更新
                if radius < optimal:
                    optimal = radius
                    opt_ball_center = copy.copy(ball_center)

                    continue

            # 检查是否满足迭代条件
            inner_points = [current_point]

            # 将furthest violator并入到最近的MBS中，求最小包围圆
            inner_points.extend(mbs_set[str(to_ball_key)])
            inner_points_location = __points[inner_points]
            temp_miniball = miniball.Miniball(inner_points_location)

            # radii将目前选取的点并入到候选球后的【该球】最小包围圆的半径
            radii = math.sqrt(temp_miniball.squared_radius())
            mec_count += 1

            # 判断将选取的点并入后是否优于目前的最优
            if radii < optimal and radius < optimal:

                # 成功则将该点从未覆盖中移除
                uncovered_points.remove(current_point)

                # 将目前的点并入到mbs中
                mbs_set[str(to_ball_key)].append(current_point)

                ball_center[to_ball_key] = temp_miniball.center()

                if radii >= radius:
                    radius = radii

                if len(uncovered_points) == 0:
                    optimal = radius
                    opt_ball_center = copy.copy(ball_center)

                    continue

                else:
                    # next_point:下一个点, ball_index:下一个点对应球在ball_distance中的索引排序后,
                    # ball_distance:各点对应最近的球的距离
                    next_point, ball_index, ball_distance = order_distance(uncovered_points, ball_center)

                    ball_order = list(ball_index)
                    ball_order.reverse()

                    for i in range(len(ball_order)):
                        # stack 中每个节点的信息

                        node = {"current_point": next_point,
                                "to_ball_key": ball_order[i],
                                "to_ball_distance": ball_distance[ball_order[i]],
                                "radius": radius,
                                "mbs_set": protect_scene(mbs_set),
                                "uncovered_points": uncovered_points.copy(),
                                "mbs_count": mbs_count,
                                "ball_center": ball_center.copy()
                                }

                        stack.append(node)

                    if mbs_count < __k:

                        node = {"current_point": next_point,
                                "to_ball_key": mbs_count,
                                "to_ball_distance": 0,
                                "radius": radius,
                                "mbs_set": protect_scene(mbs_set),
                                "uncovered_points": uncovered_points.copy(),
                                "mbs_count": mbs_count,
                                "ball_center": ball_center.copy()
                                }
                        stack.append(node)

    return optimal, opt_ball_center, mec_count


# 主函数
def main():
    start_time = time.time()

    __uncovered_points = []  # 未覆盖点的集合

    # 对未覆盖的点进行初始化操作
    for i in range(0, POINT_COUNT):
        __uncovered_points.append(i)

    optimal, opt_ball_center, mec_count = core(__uncovered_points)

    end_time = time.time()
    secs = end_time - start_time
    print(" took", secs, "seconds")

    return optimal, opt_ball_center, mec_count


if __name__ == "__main__":

    """
    __k = 17
    POINT_COUNT = 150  # 点的数目

    __points = initialize(POINT_COUNT)

    __optimal, __opt_ball_center, mec_count = main()

    print("__optimal")
    print(__optimal)

    print("__opt_ball_center")
    print(__opt_ball_center)
    
    print("mec_count")
    print(mec_count)
    """


    # 在已经生成的数据上，跑代码
    start_time = time.time()

    RADIUS = 50
    ave_mbs_num = 0
    min_mbs_num = 10000
    number = 50    # 执行的图的个数
    sum_ex_count = 0
    POINT_COUNT = 100  # 点的数目

    for index in range(50, 80):
        mbs_num = 0
        __points = initialize(POINT_COUNT)

        for j in range(1, 100):
            __k = j

            temp_opt, opt_ball_center, mec_count = main()

            if temp_opt <= RADIUS:
                ave_mbs_num += j
                sum_ex_count += mec_count
                mbs_num = j
                print("-----------------")
                break

        if mbs_num <= min_mbs_num:
            min_mbs_num = mbs_num

    ave_mbs_num = ave_mbs_num / number
    print(" ave_mbs_num", ave_mbs_num)
    print(" min_mbs_num", min_mbs_num)

    end_time = time.time()
    secs = end_time - start_time
    secs_ave = secs / number

    print(" took", secs, "seconds")
    print(" sec_ave", secs_ave)
    print(" sum_ex_count", sum_ex_count / number)


    """
    Draw.show_scatter(__points, 0, "#00CC66")
    Draw.draw_mbs(__opt_ball_center, __optimal)
    Draw.plt.axis("equal")
    Draw.plt.show()
    """
