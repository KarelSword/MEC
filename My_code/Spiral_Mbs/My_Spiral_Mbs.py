"""
本程序是自己对吕老师算法的复现

"""
import copy
import time

import numpy as np
from scipy.spatial import ConvexHull
import pandas as pd
import matplotlib.pyplot as plt
import profile

from My_code.Spiral_Mbs import My_LocalCover_v2
from My_code.Spiral_Mbs.Old import Spiral_Draw


# 将list转换为tuple
def list_to_tuple(original):
    result = []
    for item in original:
        result.append(tuple(item))
    return result


# 求解凸包的程序
def solve_convexhull(seq):
    if len(seq) >= 3:  # 如果seq 数目大于3,则执行convexhull程序
        seq_matrix = np.array(seq)
        try:
            boundarySetInOrder = list_to_tuple(seq_matrix[ConvexHull(seq_matrix).vertices])
        except:
            print("---------------------")
            index = np.lexsort([seq_matrix[:, 1], seq_matrix[:, 0]])
            boundarySetInOrder = seq_matrix[index]
            boundarySetInOrder = list_to_tuple(boundarySetInOrder)
    if len(seq) == 2:  # 如果seq 数目等于2，逆时针选取点
        seq_1 = seq[0]
        seq_2 = seq[1]
        if seq_1[0] < seq_2[0]:
            boundarySetInOrder = [seq_1, seq_2]
        if seq_1[0] > seq_2[0]:
            boundarySetInOrder = [seq_2, seq_1]
        if seq_1[0] == seq_2[0]:
            if seq_1[1] < seq_2[1]:
                boundarySetInOrder = [seq_1, seq_2]
            if seq_1[1] > seq_2[1]:
                boundarySetInOrder = [seq_2, seq_1]
    if len(seq) == 1:  # 如果seq 数目等于1，直接选取
        boundarySetInOrder = seq

    return boundarySetInOrder


# 执行程序
def execute(seq, radius):
    m = 0  #
    uav_position_set = []  # 无人机位置的点集
    count = 0  # 计数，记录miniball执行的次数
    convex_set = []

    while seq:  # seq 点的队列
        # print("m=",m)
        # 求凸包并按逆时针顺序排序
        boundarySetInOrder = solve_convexhull(seq)

        innerSet = list(set(seq).difference(set(boundarySetInOrder)))
        """
        # 测试用代码
        plt.title("overview")
        plt.scatter([dot[0] for dot in innerSet], [dot[1] for dot in innerSet], color='black')
        plt.scatter([dot[0] for dot in boundarySetInOrder], [dot[1] for dot in boundarySetInOrder], color='red')
        # x = [dot[0] for dot in res]
        # x.append(res[0][0])
        # y = [dot[1] for dot in res]
        # y.append(res[0][1])
        # plt.plot(x, y)
        # plt.xlim(-150, 150)
        # plt.ylim(-150, 150)
        plt.show()
        plt.plot([dot[0] for dot in boundarySetInOrder], [dot[1] for dot in boundarySetInOrder])
        # plt.xlim(-150, 150)
        # plt.ylim(-150, 150)
        plt.show()
        """
        # 在程序刚运行或是上一次循环迭代求得的凸包最后所有的点都被覆盖，
        # 则在当次迭代新求得的凸包中重新随机选择一点，（这里将第一个点作为随机点）
        if m == 0:
            # print(len(boundarySetInOrder))
            # print("----------------")
            # uav_position = boundarySetInOrder[10]
            index = np.random.choice(len(boundarySetInOrder), 1)[0]
            # print(index)
            uav_position = boundarySetInOrder[index]

        # 求当前UAV的位置，以便在进入下一个迭代的时候按逆时针方向选择其第一个最近的未被覆盖的点
        # sec_set = list(set(seq).difference(set(uav_position)))
        current_uav_position = boundarySetInOrder.index(uav_position)
        convex_set.append(uav_position)
        boundarySetInOrderSize = len(boundarySetInOrder)
        nextUAVInd = (current_uav_position + 1) % boundarySetInOrderSize

        # 传入：基站的位置，基站半径，内部点，其他边界点
        p_sec_boundary = copy.deepcopy(boundarySetInOrder)
        p_sec_boundary.remove(uav_position)

        uav_position, prio_set, miniball_count = My_LocalCover_v2.LocalCover(uav_position, radius, [uav_position],
                                                                             p_sec_boundary)
        count += miniball_count
        prioBoCopy = copy.deepcopy(prio_set)

        # 第二次调用localCover,优先点为上一次调用的优先结果点集
        # 返回结果即为该次放置无人机的位置以及覆盖的点
        # 传入：基站的位置，基站半径，内部点，其他非边界点
        uav_position, prio_set, miniball_count = My_LocalCover_v2.LocalCover(uav_position, radius, prio_set.copy(),
                                                                             innerSet)
        # uav_position, prio_set, miniball_count = My_LocalCover.LocalCover(uav_position, radius, [uav_position],
        # sec_set)
        count += miniball_count
        # prioBoCopy = copy.deepcopy(prio_set)

        uav_position_set.append(uav_position)
        # 计算下一个迭代距离当前选择的UAV位置第一近的未被覆盖的点
        flag = True

        while nextUAVInd != current_uav_position:
            if boundarySetInOrder[nextUAVInd] not in prioBoCopy:
                uav_position = boundarySetInOrder[nextUAVInd]
                m = m + 1
                flag = False
                break
            else:
                nextUAVInd = (nextUAVInd + 1) % boundarySetInOrderSize

        if flag:
            m = 0

        # 去除已覆盖的点，进入下一个无人机位置的计算
        seq = list(set(seq).difference(set(prio_set)))

    # print("count")
    # print(count)

    return uav_position_set, count, convex_set


def main():
    start_time = time.time()
    RADIUS = 250

    np.random.seed()
    # data = np.random.uniform(-100, 100, size=(500, 2))
    # data = np.random.normal(0, 1, size=(150, 2))

    # data = np.random.poisson(100, size=(800, 2))

    name = 'C:/Users/karel/Desktop/大论文/Data/' + str(50) + '_' + str(index) + '.csv'
    data = pd.read_csv(name, usecols=[1, 2]).values
    # data = pd.read_csv('F:/Scientific_Literature/写论文/test_data/Uruguay - 734.csv', usecols=[1, 2]).values
    # data = np.round(data, 3)
    data = list_to_tuple(data)
    # data = [(math.modf(dot[0])[1], math.modf(dot[1])[1]) for dot in data]
    # data = list(set(data))

    #  使用K-means
    # UAV_positionSet = My_KMeans.binarysearch(data, RADIUS)
    # print(*data,sep='\n')
    UAV_positionSet, ex_count, convex_points_set = execute(data, RADIUS)
    # print(len(UAV_positionSet))

    end_time = time.time()
    secs = end_time - start_time
    # print(" took", secs, "seconds")


    x = [dot[0] for dot in data]
    y = [dot[1] for dot in data]

    # 打印散点图
    plt.scatter(x, y, color='black')
    # for a, b in zip(x, y):
    #      plt.text(a, b, (a, b), ha='center', va='bottom', fontsize=10)
    # 按顺寻连接无人机的位置
    for UAV in UAV_positionSet:
        # print(UAV)
        # print("[", UAV[0], ",", UAV[1], "]", sep="")
        theta = np.arange(0, 2 * np.pi, 0.0001)
        x = UAV[0] + RADIUS * np.cos(theta)
        y = UAV[1] + RADIUS * np.sin(theta)
        plt.plot(x, y)

    x = [dot[0] for dot in UAV_positionSet]
    y = [dot[1] for dot in UAV_positionSet]
    plt.scatter(x[0], y[0], color='red')

    # 关闭了Spiral的顺序连接
    # plt.plot(x, y, color="red")
    plt.axis("equal")
    plt.show()


    # return len(UAV_positionSet)
    return len(UAV_positionSet), ex_count, UAV_positionSet


if __name__ == "__main__":

    # 读取固定集合的内容
    start_time = time.time()

    ave_mbs_num = 0
    min_mbs_num = float("inf")
    max_mbs_num = 0
    number = 1
    sum_ex_count = 0

    for index in range(number):
        mbs_num = 0

        mbs_num, _ex_count, UAV_positionSet = main()
        ave_mbs_num += mbs_num
        sum_ex_count += _ex_count
        if mbs_num <= min_mbs_num:
            min_mbs_num = mbs_num
        if mbs_num >= max_mbs_num:
            max_mbs_num = mbs_num
        # print("------------------")

    ave_mbs_num = ave_mbs_num / number
    print(" ave_mbs_num", ave_mbs_num)
    print(" min_mbs_num", min_mbs_num)
    print(" max_mbs_num", max_mbs_num)

    end_time = time.time()
    secs = end_time - start_time
    secs_ave = secs / number

    print(" took", secs, "seconds")
    print(" sec_ave", secs_ave)
    print(" sum_ex_count", sum_ex_count/number)


    """
    start_time = time.time()

    N, count, UAV_positionSet = main()
    print(N)
    # print(count)

    end_time = time.time()
    secs = end_time - start_time
    # print(" took", secs, "seconds")
    """

    # 输出的操作
    # df = pd.DataFrame(UAV_positionSet)
    # df.to_csv('C:/Users/karel/Desktop/大论文/Data/Spiral_VBS.csv')


    """
    # 分析Spiral算法在随机不读取情况下的效果
    start_time = time.time()

    ave_mbs_num = 0
    min_mbs_num = float("inf")
    number = 100
    sum_ex_count = 0

    for i in range(number):
        mbs_num = 0

        mbs_num, _ex_count = main()
        ave_mbs_num += mbs_num
        sum_ex_count += _ex_count
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
    print(" sum_ex_count", sum_ex_count/number)
    """

    """
    start_time = time.time()

    ave_mbs_num = 0
    min_mbs_num = float("inf")
    number = 5
    sum_ex_count = 0

    count_74 = 0
    count_76 = 0
    count_78 = 0
    count_80 = 0
    count_82 = 0
    count_84 = 0
    count_86 = 0
    count_88 = 0

    for j in range(100):
        ave_mbs_num = 0
        for i in range(number):
            mbs_num = 0

            mbs_num, _ex_count = main()
            ave_mbs_num += mbs_num
            sum_ex_count += _ex_count
            if mbs_num <= min_mbs_num:
                min_mbs_num = mbs_num

        ave_mbs_num = ave_mbs_num / number
        # print(" ave_mbs_num", ave_mbs_num)
        # print(" min_mbs_num", min_mbs_num)

        end_time = time.time()
        secs = end_time - start_time
        secs_ave = secs/number

        if ave_mbs_num == 7.4:
            count_74 += 1

        if ave_mbs_num == 7.6:
            count_76 += 1

        if ave_mbs_num == 7.8:
            count_78 += 1

        if ave_mbs_num == 8.0:
            count_80 += 1

        if ave_mbs_num == 8.2:
            count_82 += 1

        if ave_mbs_num == 8.4:
            count_84 += 1

        if ave_mbs_num == 8.6:
            count_86 += 1

        if ave_mbs_num == 8.8:
            count_88 += 1

    print("----7.4-----8.6----")
    print("count_74")
    print(count_74)
    print("count_76")
    print(count_76)
    print("count_78")
    print(count_78)
    print("count_80")
    print(count_80)
    print("count_82")
    print(count_82)
    print("count_84")
    print(count_84)
    print("count_86")
    print(count_86)
    print("count_88")
    print(count_88)

    # print(" took", secs, "seconds")
    # print(" sec_ave", secs_ave)
    # print(" sum_ex_count", sum_ex_count/number)
    """
