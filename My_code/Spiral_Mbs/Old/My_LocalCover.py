import numpy as np
from scipy.spatial import distance
import math
import miniball


# 求出两点之间的距离
def solve_distance(A, B):
    return np.sqrt(np.square(A[0] - B[0]) + np.square(A[1] - B[1]))


# 对字典中的值进行排序，并且返回字典
def dic_sorted(dic):
    dic = sorted(dic.items(), key=lambda x: x[1], reverse=False)
    temp = {}
    for item in range(len(dic)):
        temp[dic[item][0]] = dic[item][1]
    return temp


# 排除范围2r以外的点
def exclude_points_2r(p_prio, p_sec, r):
    temp_P_sec = []

    if len(p_prio) >= 1 and len(p_sec) >= 2:  # 如果p_sec长度大于等于2
        # prio_sec_distance:内部点和外部点之间的距离
        prio_sec_distance = distance.cdist(p_sec, p_prio, 'euclidean')

        # print(prio_sec_distance)
        cai = np.where(prio_sec_distance <= 2*r)[0]
        # print(cai)
        # print(set(cai))
        for value in set(cai):
            if np.sum(cai == value) == len(p_prio):
                temp_P_sec.append(p_sec[value])
        """
        for index, value in enumerate(prio_sec_distance):
            if np.max(value) <= 2*r:
                temp_P_sec.append(p_sec[index])
        # print("temp_P_sec_2rd")
        # print(temp_P_sec)
        # print("------------------")
        """

    elif len(p_prio) >= 2 and len(p_sec) == 1:
        p_sec_repeat = [p_sec[0], p_sec[0]]
        prio_sec_distance = distance.cdist(p_sec_repeat, p_prio, 'euclidean')

        if np.max(prio_sec_distance[0]) <= 2 * r:
            temp_P_sec.append(p_sec[0])

    else:
        for prio in p_prio:
            for sec in p_sec:
                if solve_distance(prio, sec) <= 2 * r:
                    temp_P_sec.append(sec)

    return temp_P_sec


# 更新范围为r内的点
def update_prio(candidate_points, wk, r):

    _p_prio = []
    _p_sec = []
    _distance_dic = {}
    # 如果候选点的数目大于2
    if len(candidate_points) >= 2:
        wk = [tuple(wk)]
        wk_distance = distance.cdist(wk, candidate_points, 'euclidean')

        a = np.where(wk_distance[0] <= r)[0]
        b = np.where(wk_distance[0] > r)[0]
        for value in a:
            _p_prio.append(candidate_points[value])
        for value in b:
            _p_sec.append(candidate_points[value])
            _distance_dic[candidate_points[value]] = wk_distance[0][value]

    # 如果候选点的数目为1
    if len(candidate_points) == 1:
        wk_distance = solve_distance(wk, candidate_points[0])

        if wk_distance <= r:
            _p_prio.append(candidate_points[0])
        else:
            _p_sec.append(candidate_points[value])
            _distance_dic[candidate_points[0]] = wk_distance

    return _p_prio, _distance_dic, _p_sec


def sort_distance(candidate_points, wk):

    _distance_dic = {}

    if len(candidate_points) >= 2:
        wk = [tuple(wk)]
        wk_distance = distance.cdist(wk, candidate_points, 'euclidean')

        for value in range(len(candidate_points)):
            _distance_dic[candidate_points[value]] = wk_distance[0][value]

    if len(candidate_points) == 1:
        wk_distance = solve_distance(wk, candidate_points[0])

        _distance_dic[candidate_points[0]] = wk_distance

    return _distance_dic


# LocalCover程序
def LocalCover(wk, r, p_prio_raw, p_sec_raw):
    """
    wk：基点，r：半径，P_prio_raw目前在半径中的点，P_sec_raw候选点
    """
    # 对数据进行备份
    count = 0   # 计算miniball执行的次数
    p_prio = p_prio_raw.copy()  # 对范围r的点进行备份
    # p_prio.append(wk)           # 将wk加入
    p_sec = p_sec_raw.copy()    # 对候选的点进行拷贝

    """
    print("initial")
    print("wk")
    print(wk)
    print("p_prio")
    print(p_prio)
    print("p_sec")
    print(p_sec)
    """
    # 排除范围2r外的点
    temp_P_sec = exclude_points_2r(p_prio, p_sec, r)

    # 确定候选点
    candidate_points = []
    candidate_points.extend(p_prio_raw)
    candidate_points.extend(temp_P_sec)

    """
    for point in candidate_points:
        temp_distance = solve_distance(wk, point)
        if temp_distance <= r:
            p_prio.append(point)
        else:
            distance_dic[point] = temp_distance
    """
    p_prio, distance_dic, p_sec = update_prio(candidate_points, wk, r)
    distance_dic = sort_distance(p_sec, wk)
    # p_sec = list(distance_dic.keys())
    p_sec = list(dic_sorted(distance_dic).keys())
    # print(len(p_sec))

    while p_sec:
        candidate = p_sec.pop()
        p_prio.append(candidate)
        mb = miniball.Miniball(p_prio)
        count += 1
        if math.sqrt(mb.squared_radius()) > r:
            p_prio.remove(candidate)
            continue

    mb = miniball.Miniball(p_prio)
    u = mb.center()

    """
    print("final")
    print("u")
    print(u)
    print("p_prio")
    print(p_prio)
    print("p_sec")
    print(p_sec)
    """

    return u, p_prio, count





