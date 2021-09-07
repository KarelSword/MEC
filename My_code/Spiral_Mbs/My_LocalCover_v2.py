import numpy as np
from scipy.spatial import distance
import math
import miniball


# 将list转换为tuple
def list_to_tuple(original):
    result = []
    for item in original:
        result.append(tuple(item))
    return result


# 求出两点之间的距离
def solve_distance(A, B):
    return np.sqrt(np.square(A[0] - B[0]) + np.square(A[1] - B[1]))


# 对字典中的值进行排序，并且返回字典
def dic_sorted(dic):
    # 这里reverse=True是因为待会出栈的时候是尾部先出
    dic = sorted(dic.items(), key=lambda x: x[1], reverse=True)
    temp = {}
    for item in range(len(dic)):
        temp[dic[item][0]] = dic[item][1]
    return temp


# 排除范围2r以外的点
def exclude_points_2r(p_prio, p_sec, r):
    temp_P_sec = []
    # 如果p_sec长度大于等于2
    # prio_sec_distance:内部点和外部点之间的距离

    prio_sec_distance = distance.cdist(p_sec, p_prio, 'euclidean')
    adjacency_matrix = np.where(prio_sec_distance <= 2*r, 1, 0)

    for index, value in enumerate(adjacency_matrix):
        if np.min(value) != 0:
            temp_P_sec.append(p_sec[index])

    """
    for index, value in enumerate(prio_sec_distance):
        if np.max(value) <= 2*r:
            temp_P_sec.append(p_sec[index])
    # print("temp_P_sec_2rd")
    # print(temp_P_sec)
    # print("------------------")
    """

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
            # _distance_dic[candidate_points[value]] = wk_distance[0][value]

    # 如果候选点的数目为1
    if len(candidate_points) == 1:
        wk_distance = solve_distance(wk, candidate_points[0])

        if wk_distance <= r:
            _p_prio.append(candidate_points[0])
        else:
            _p_sec.append(candidate_points[value])
            # _distance_dic[candidate_points[0]] = wk_distance

    return _p_prio, _p_sec


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
    p_prio = p_prio_raw.copy()
    p_sec = p_sec_raw.copy()
    count = 0   # 计算miniball执行的次数

    while p_sec:

        p_sec = exclude_points_2r(p_prio, p_sec, r)
        # print("p_sec_exclude_points_2r:")
        # print(p_sec)
        candidate_points = []
        candidate_points.extend(p_prio)
        candidate_points.extend(p_sec)

        p_prio, p_sec = update_prio(candidate_points, wk, r)
        # print("p_sec_update_prio:")
        # print(p_sec)

        distance_dic = sort_distance(p_sec, wk)

        p_sec = list(dic_sorted(distance_dic).keys())

        if p_sec:
            candidate = p_sec.pop()

            p_prio.append(candidate)
            mb = miniball.Miniball(p_prio)
            count += 1
            radii = math.sqrt(mb.squared_radius())
            if radii > r:
                p_prio.remove(candidate)

            else:
                wk = mb.center()
        """
        if len(p_sec_raw) > 0:
            Spiral_Draw.show_scatter(np.array(p_sec_raw), 0, "#00CC66")
        if len(p_sec) > 0:
            Spiral_Draw.show_scatter(np.array(p_sec), 0, "#0000FF")
        if len(p_prio) > 0:
            Spiral_Draw.show_scatter(np.array(p_prio), 0, "#FF0000")
        if len(p_prio_raw) > 0:
            Spiral_Draw.show_scatter(np.array(p_prio_raw), 0, "#000000")
        if candidate:
            Spiral_Draw.show_scatter(np.array(np.array([candidate])), 0, "#005500")

        Spiral_Draw.draw_mbs_one(wk, r)

        Spiral_Draw.plt.axis("equal")
        Spiral_Draw.plt.show()
        """
    mb = miniball.Miniball(p_prio)
    u = mb.center()

    return u, p_prio, count


if __name__ == "__main__":
    data = np.random.uniform(-100, 100, size=(10, 2))
    data = np.round(data, 2)
    wk = data[0]
    data = np.delete(data, 0, 0)
    wk = (wk[0], wk[1])
    data = list_to_tuple(data)

    LocalCover(wk, 50, [wk], data)


