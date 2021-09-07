# 对专利的算法的撰写
# 2020年9月25日
"""
本代码是对三维模式的探索
经过实际测试，发现自己代码运行的速度还可以
从GDC3.0继承而来
作者：karel
"""
import profile
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform, cdist
import miniball
import math
import time
from My_code.Clique_Method import Draw
from matplotlib import pyplot as plt

RADIUS = 30
DIAMETER = 2.0 * RADIUS


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
        print("_______________________")
        print(self.inner_points)


def list_to_tuple(original):
    result = []
    for item in original:
        result.append(tuple(item))
    return result


# 求出两点之间的距离
def solve_distance(A, B):
    return np.sqrt(np.square(A[0] - B[0]) + np.square(A[1] - B[1]))


# 初始化操作
def initialize(number, distributed):
    # 初始化点的坐标
    np.random.seed()
    if distributed == "uniform":
        points = np.random.uniform(-100, 100, size=(number, 3))
        # points = np.round(points, 3)
    if distributed == "normal":
        points = np.random.normal(-100, 100, size=(number, 2))
        # points = np.random.standard_normal(size=(number, 2)) * 100
        # points = np.round(points, 3)
    if distributed == "read":
        points = pd.read_csv('F:/Scientific_Literature_1017/PandasNumpy.csv', usecols=[1, 2]).values
        """
        data = pd.read_csv("深圳市-欢乐谷.csv", header=0, usecols=[2, 3, 4]).values
        data = [(math.modf(dot[0])[1], math.modf(dot[1])[1]) for dot in data]
        data = list(set(data[0:400]))
        points = np.array(data)
        """

    # 初始化点类的集合
    points_set = {}

    # ————————————————————————————————————
    # 构建无向图，返回邻接矩阵,初始化邻接矩阵
    adjacency_matrix, points_distance, anti_adjacency_matrix = create_graph(points)
    # 确定点的度数,初始化点的度数列表
    degree = determine_degree(adjacency_matrix)
    # 确定点的邻接矩阵
    adjacency_list = determine_adjacency_list(adjacency_matrix)

    for i in range(0, len(points)):
        a = Point(points[i], i)
        a.neighbor_info = adjacency_list[i]
        a.degree = degree[str(i)]
        points_set[str(i)] = a

    return points_set, points, adjacency_matrix, degree, points_distance, anti_adjacency_matrix


# 创建无向图
def create_graph(points):
    """本代码段根据点集创建无向图，当任意两点之间的距离小于2r时，则认为他们是相邻的"""
    points_distance = squareform(pdist(points))
    points_distance = np.where(points_distance <= DIAMETER, 1, points_distance)
    points_distance = np.where(points_distance > DIAMETER, 0, points_distance)
    adjacency_matrix = points_distance
    np.fill_diagonal(adjacency_matrix, 0)

    anti_adjacency_matrix = np.where(adjacency_matrix == 1, -1, adjacency_matrix)
    tmp_matrix = np.ones(len(points))
    anti_adjacency_matrix = np.add(anti_adjacency_matrix, tmp_matrix)
    np.fill_diagonal(anti_adjacency_matrix, 0)

    return adjacency_matrix, points_distance, anti_adjacency_matrix


# 确定所有点的度数
def determine_degree(adjacency_matrix):
    degree = {}
    """本代码段根据点和点之间的相邻信息确定任一点的度数信息"""
    temp_degree = np.array(adjacency_matrix).sum(axis=1)
    for row_index, row_item in enumerate(adjacency_matrix):
        degree[str(row_index)] = temp_degree[row_index]

    return degree


# 创建无向图的邻接表
def determine_adjacency_list(adjacency_matrix):
    adjacency_list = []
    for index, item in enumerate(adjacency_matrix):
        a = np.where(item == 1)[0].tolist()
        # print(a)
        adjacency_list.append(a)

        # print(adjacency_list)
    return adjacency_list


# 对字典中的值进行排序，并且返回字典
def dic_sorted(dic, flag):
    dic = sorted(dic.items(), key=lambda x: x[1], reverse=flag)
    temp = {}
    for item in range(len(dic)):
        temp[dic[item][0]] = dic[item][1]
    return temp


# 初始化
def randomization_1_start(points_set, degree):
    """
    初始化随机构造若干个MBS
    一个点开始
    :return:
    """
    mbs_set = {}  # mbs_set:mbs的点集合
    mbs_index = 0

    degree_sorted = dic_sorted(degree.copy(), False)

    for key, value in degree_sorted.items():
        mbs = MBS()
        mbs.inner_points.append(int(key))
        mbs.count = 1
        mbs.index = mbs_index
        mbs_set[str(mbs_index)] = mbs

        points_set[str(key)].belongs_to = mbs_index
        mbs_index += 1

    return mbs_set


# 确定任一点加入团后，是否还能构成一个团
def check_if_clique(candidate, clique):
    """
    确定任一点加入团后，是否还能构成一个团

    :param candidate: 候选的点
    :param clique:    要并入的团
    :return:          可以返回True，否则Flase
    """
    # print("candidate point:", candidate.index)
    # print("set(clique.inner_points)", set(clique.inner_points))
    # print("set(candidate.neighbor_info)", set(candidate.neighbor_info))
    if set(clique.inner_points) <= set(candidate.children_info):
        # print("True")
        return True
    else:
        # print("False")
        return False


# 读取mbs中所有点的位置信息，并且转换为tuple类型
def read_mbs_location(mbs, points_set):
    mbs_innerpoints_tuple = []
    for item in mbs.inner_points:
        temp = tuple(points_set[str(item)].location)
        mbs_innerpoints_tuple.append(temp)

    return mbs_innerpoints_tuple


# 读取任意点集的位置信息
def read_location(points_index, points_set):
    points_location = []
    for item in points_index:
        temp = tuple(points_set[str(item)].location)
        points_location.append(temp)

    return points_location


# 删除点的操作
def delete_point(index, points_set):
    """
    删除节点的操作
    :param points_set:
    :param index:要删除的点的编号
    :return:
    """
    ne_points = points_set[str(index)].children_info
    for item in ne_points:
        points_set[str(item)].children_info.remove(index)
        points_set[str(item)].children_num -= 1

    del points_set[str(index)]

    return points_set


# 对临近点的度数进行排序
def sort_ne_degree(ne, points_set):
    degree_temp = {}
    for item in ne:
        degree_temp[str(item)] = points_set[str(item)].children_num

    degree_temp = dic_sorted(degree_temp, False)
    degree_temp = list(map(int, degree_temp))

    return degree_temp


# 对全局的点进行度数排序
def sort_all_degree(degree, points_set):
    for key, value in degree.copy().items():
        if points_set[str(key)].processed == 1:
            del degree[str(key)]

    degree_temp = dic_sorted(degree, False)

    return degree_temp


# 对度数和距离组合排序
def sort_degree_distance(ne, points_set, center):
    degree_temp = {}
    distance_dic = {}

    if len(ne) > 1:
        ne_points_location = read_location(ne, points_set)
        center = [tuple(center)]

        wk_distance = cdist(center, ne_points_location, 'euclidean')
        for value in range(len(wk_distance[0])):
            distance_dic[ne[value]] = wk_distance[0][value]
        distance_value = list(distance_dic.values())
        # p_sec = list(distance_dic.keys())

        for item in ne:
            degree_temp[str(item)] = points_set[str(item)].children_num
        degree_value = list(degree_temp.values())

        index = np.lexsort([distance_value, degree_value])
        ne = np.array(ne)
        return ne[index]

    if len(ne) <= 1:
        return ne


# 对相邻点的距离进行排序
def sort_distance(ne_points_index, center, points_set):
    distance_dic = {}
    ne_points_location = read_location(ne_points_index, points_set)
    center = [tuple(center)]

    if len(ne_points_location) >= 1:
        wk_distance = cdist(center, ne_points_location, 'euclidean')
        for value in range(len(wk_distance[0])):
            distance_dic[ne_points_index[value]] = wk_distance[0][value]
        distance_dic = dic_sorted(distance_dic, False)
        p_sec = list(distance_dic.keys())
        # print(p_sec)
        return p_sec

    if len(ne_points_location) == 0:
        # print(ne_points_index)
        return ne_points_index


# 排除执行过的点
def exclude_processed_points(ne_points_index, points_set):
    delete_set = []
    for index in ne_points_index:
        processed_flag = points_set[str(index)].processed
        if processed_flag == 1:
            delete_set.append(index)

    for index in delete_set:
        ne_points_index.remove(index)

    return ne_points_index


# 更新图的拓扑信息
def update_graph(adjacency_matrix, mbs, points_set):
    adjacency_matrix[..., mbs.inner_points] = 0
    adjacency_matrix[mbs.inner_points, ...] = 0

    degree = determine_degree(adjacency_matrix)
    adjacency_list = determine_adjacency_list(adjacency_matrix)

    for index in range(0, len(points_set)):
        points_set[str(index)].children_info = adjacency_list[index]
        points_set[str(index)].children_num = degree[str(index)]

    return adjacency_matrix, points_set, degree


# 核心执行模块
def core(points_set, degree, points, adjacency_matrix):
    """
    核心合并的模块

    :return:
    """
    # 初始化的操作
    mbs_set = randomization_1_start(points_set, degree)
    num_mbs = 0
    num_count = 0
    num_clique = 0
    num_sec = 0
    num_sec_success = 0
    delete_points = []
    processed_points = []
    degree = sort_all_degree(degree.copy(), points_set)

    # 遍历mbs_set,即遍历MBS集合

    while degree:
        degree = sort_all_degree(degree.copy(), points_set)
        if degree == {}:
            break
        num_mbs += 1
        degree_index = list(degree.keys())
        chosen_point = int(degree_index[0])
        element = points_set[str(chosen_point)].belongs_to

        # 尝试是否存在这个MBS
        try:
            mbs_set[str(element)]
        except KeyError:
            continue

        # 如果该MBS已经被处理过，则直接跳过该点
        if mbs_set[str(element)].processed == 1:
            continue

        # chosen_point = mbs_set[str(element)].inner_points[0]
        # chosen_point_ne 选择的点的相邻的点
        chosen_point_ne = points_set[str(chosen_point)].children_info
        chosen_point_ne = exclude_processed_points(chosen_point_ne, points_set)

        # 对该点相邻的点，依照度数顺序进行排序
        chosen_point_ne = sort_ne_degree(chosen_point_ne, points_set)

        # 读取MBS中点位置信息
        mbs_inner_points = read_mbs_location(mbs_set[str(element)], points_set)
        # 遍历选取的点的所有临近的点
        for candidate in chosen_point_ne:
            # candidate_mbs:候选点所属于的mbs
            candidate_mbs = mbs_set[str(points_set[str(candidate)].belongs_to)]

            # 如果满足以下条件：
            # 1.candidate_mbs的大小小于目前的大小
            # 2.候选点加入后可以成团
            # 3.候选点加入后可以形成最小包围圆
            num_count += 1
            # if candidate_mbs.processed == 0:
            if candidate_mbs.count <= mbs_set[str(element)].count:
                num_clique += 1
                if check_if_clique(points_set[str(candidate)], mbs_set[str(element)]):
                    # 先将候选点加入mbs集合中
                    mbs_inner_points.append(tuple(points_set[str(candidate)].location))
                    temp_miniball = miniball.Miniball(mbs_inner_points)

                    num_sec += 1  # 执行miniball的计数器

                    if (math.sqrt(temp_miniball.squared_radius())) <= RADIUS:

                        mbs_set[str(element)].inner_points.append(candidate)  # 将候选点加入到现在的mbs
                        mbs_set[str(element)].count += 1  # 更新现在mbs点的数目
                        mbs_set[str(element)].center = temp_miniball.center()

                        candidate_mbs.inner_points.remove(candidate)  # 将候选点从原来的mbs中删除
                        candidate_mbs.count -= 1

                        points_set[str(candidate)].belongs_to = mbs_set[str(element)].index
                        points_set[str(candidate)].processed = 1
                        processed_points.append(candidate)
                        # points_set = delete_point(candidate, points_set)

                        num_sec_success += 1

                        if candidate_mbs.count == 0:
                            del mbs_set[str(candidate_mbs.index)]
                    else:
                        mbs_inner_points.remove(tuple(points_set[str(candidate)].location))

        mbs_set[str(element)].processed = 1
        points_set[str(chosen_point)].processed = 1
        processed_points.append(chosen_point)

        # points_set = delete_point(chosen_point, points_set)
        # adjacency_matrix_copy = adjacency_matrix.copy()
        # adjacency_matrix, points_set, degree = update_graph(adjacency_matrix, mbs_set[str(element)], points_set)

        """
        for item in mbs_set[str(element)].inner_points:
            item_ne = points_set[str(item)].neighbor_info
            for value in item_ne:
                points_set[str(value)].neighbor_info.remove(item)
                points_set[str(value)].degree -= 1
        """
        """
        # 下面的程序是为了排除bug而运行的
        # 绘制出点和点之间的相邻信息
        Draw.draw_plot(points, adjacency_matrix_copy)

        #  阅读出mbs的位置信息，并转换为tuple类型
        mbs_points = read_mbs_location(mbs_set[str(element)], points_set)
        mbs_points = list_to_tuple(mbs_points)

        # 选取的点的相邻的点
        chosen_point_ne = points[chosen_point_ne]
        chosen_point_ne = list_to_tuple(chosen_point_ne)

        # 需要删除删除的点
        delete_points.extend(mbs_points)

        # 剩下还在的点
        other_points = list_to_tuple(points)

        for item in delete_points:
            try:
                other_points.remove(item)
            except:
                pass

        # 转换成ndarray类型，方便绘制
        mbs_points = np.array(mbs_points)
        chosen_point_ne = np.array(chosen_point_ne)
        other_points = np.array(other_points)
        start_point = points[chosen_point]
        start_point = [start_point]

        Draw.draw_txt(points)
        Draw.show_scatter_debug_2(other_points, "blue")
        Draw.show_scatter_debug_2(chosen_point_ne, "green")
        Draw.show_scatter_debug_2(mbs_points, "purple")
        Draw.show_scatter_debug_2(start_point, "black")

        mbs_set[str(element)].center = miniball.Miniball(mbs_points).center()
        Draw.draw_mbs_temp(mbs_set[str(element)])

        Draw.plt.xlim(-100, 100)
        Draw.plt.ylim(-100, 100)
        Draw.plt.axis("equal")
        Draw.plt.show()
        print("----------------------------")        
        """
    """
    print("num_mbs")
    print(num_mbs)
    print("num_count")
    print(num_count)
    print("num_clique")
    print(num_clique)
    print("num_sec")
    print(num_sec)
    print("num_sec_success")
    print(num_sec_success)
    """
    # print("---------one over------------")

    return mbs_set, num_sec


# 计算每个MBS的中心
def get_center(mbs_set, points_set):
    for key, value in mbs_set.items():
        inner_points = read_mbs_location(value, points_set)
        temp_miniball = miniball.Miniball(inner_points)
        value.center = temp_miniball.center()

    return mbs_set


# 显示点集的度数信息和相邻信息
def show_points(points_set):
    for key, value in points_set.items():
        print("the points" + key)
        print(value.children_info)
        print(value.belongs_to)


def show_mbs(mbs_set):
    for key, value in mbs_set.items():
        value.print()


# 主程序片段
def main():
    # start_time = time.time()
    # __points_set，点类的集合，__points点的列表集合，__adjacency_matrix邻接矩阵
    __points_set, __points, __adjacency_matrix, __degree, __points_distance, __anti_adjacency_matrix \
        = initialize(40, "uniform")  # uniform, normal
    __adjacency_matrix = np.array(__adjacency_matrix)
    # __mbs_set = randomization_1_start(__points_set, __degree)

    fig = plt.figure()
    ax1 = plt.axes(projection='3d')
    ax1.scatter3D(__points[:, 0], __points[:, 1], __points[:, 2])
    ax1.plot3D(__points[:, 0], __points[:, 1], __points[:, 2], 'r--',)
    plt.show()

    __mbs_set, count = core(__points_set.copy(), __degree.copy(), __points.copy(), __adjacency_matrix.copy())

    __mbs_set = get_center(__mbs_set, __points_set)


    # Draw.draw_plot(__points, __adjacency_matrix)
    """
    Draw.draw_mbs(__mbs_set)
    Draw.show_scatter(__points)
    Draw.plt.axis("equal")
    Draw.plt.show()
    """
    # end_time = time.time()
    # secs = end_time - start_time
    # print(" took", secs, "seconds")


    return len(__mbs_set), count


if __name__ == "__main__":
    main()
    # profile.run('main()')
    """
    start_time = time.time()

    ave_mbs_num = 0
    min_mbs_num = 10000
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
    print(" sum_ex_count", sum_ex_count / number)
    """
