# 对专利的算法的撰写
# 2020年9月25日
"""
本代码是对专利代码实现的撰写
经过实际测试，发现自己代码运行的速度还可以
作者：karel
"""
import profile
import numpy as np
from scipy.spatial.distance import pdist, squareform, cdist
import miniball
import math
import time
from My_code.Clique_Method import Draw

RADIUS = 50
DIAMETER = 2 * RADIUS
LET = 1.7 * RADIUS


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
        points = np.random.uniform(-100, 100, size=(number, 2))
        # points = np.round(points, 3)
    if distributed == "normal":
        points = np.random.normal(-100, 100, size=(number, 2))
        # points = np.random.standard_normal(size=(number, 2)) * 100
        # points = np.round(points, 3)
    '''
    data = pd.read_csv("深圳市-欢乐谷.csv", header=0, usecols=[2, 3, 4]).values
    data = [(math.modf(dot[0])[1], math.modf(dot[1])[1]) for dot in data]
    data = list(set(data[0:400]))
    points = np.array(data)
    '''
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
    points_distance = np.where(points_distance <= LET, 1, points_distance)
    points_distance = np.where(points_distance > LET, 0, points_distance)
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


# 合并范围r内的点
def include_r_point(ne_points_index, center, points_set, mbs_set):
    """
    :param mbs_set:
    :param points_set:
    :param ne_points_index: 备选的点的序号
    :param center:
    :return:
    """
    p_prio = []
    p_sec = []
    ne_points_location = read_location(ne_points_index, points_set)
    center = [tuple(center)]

    if len(ne_points_location) >= 2:
        center_distance = cdist(center, ne_points_location, 'euclidean')
        a = np.where(center_distance[0] <= RADIUS)[0]
        b = np.where(center_distance[0] > RADIUS)[0]

        for item in a:
            if points_set[str(item)].processed == 0:
                p_prio.append(ne_points_index[item])
            else:
                p_sec.append(ne_points_index[item])
        for item in b:
            p_sec.append(ne_points_index[item])

    if len(ne_points_location) == 1:
        center_distance = solve_distance(center[0], ne_points_location[0])

        if center_distance <= RADIUS and points_set[str(ne_points_index[0])].processed == 0:
            p_prio.append(ne_points_index[0])
        else:
            p_sec.append(ne_points_index[0])

    return p_prio, p_sec


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
    for index in ne_points_index:
        if points_set[str(index)].processed == 1:
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

    # 遍历mbs_set,即遍历MBS集合
    # for element in range(len(mbs_set)):
    while True:

        degree = sort_all_degree(degree.copy(), points_set)
        if degree == {}:
            break
        # print(degree)
        degree_index = list(degree.keys())
        chosen_point = int(degree_index[0])
        # print(chosen_point)
        element = points_set[str(chosen_point)].belongs_to

        num_mbs += 1

        # 尝试是否存在这个MBS
        try:
            mbs_set[str(element)]
        except KeyError:
            continue

        # 如果该MBS已经被处理过，则直接跳过该点
        if mbs_set[str(element)].processed == 1:
            continue

        # chosen_point_ne 选择的点的相邻的点
        chosen_point_ne = points_set[str(chosen_point)].children_info
        chosen_point_ne = exclude_processed_points(chosen_point_ne, points_set)
        center = points_set[str(chosen_point)].location

        chosen_point_ne = sort_degree_distance(chosen_point_ne, points_set, center)
        # 对该点相邻的点，依照度数顺序进行排序
        # chosen_point_ne = sort_ne_degree(chosen_point_ne, points_set)

        # 读取MBS中点位置信息
        mbs_inner_points = read_mbs_location(mbs_set[str(element)], points_set)
        # 遍历选取的点的所有临近的点
        for candidate in chosen_point_ne:
            # candidate_mbs:候选点所属于的mbs

            center = mbs_set[str(element)].center

            candidate_mbs = mbs_set[str(points_set[str(candidate)].belongs_to)]

            if center:
                if solve_distance(points_set[str(candidate)].location, center) <= RADIUS:

                    mbs_set[str(element)].inner_points.append(candidate)  # 将候选点加入到现在的mbs
                    mbs_set[str(element)].count += 1  # 更新现在mbs点的数目

                    candidate_mbs.inner_points.remove(candidate)  # 将候选点从原来的mbs中删除
                    candidate_mbs.count -= 1

                    points_set[str(candidate)].belongs_to = mbs_set[str(element)].index
                    points_set[str(candidate)].processed = 1

                    if candidate_mbs.count == 0:
                        del mbs_set[str(candidate_mbs.index)]

                    continue


            # 如果满足以下条件：
            # 1.candidate_mbs的大小小于目前的大小
            # 2.候选点加入后可以成团
            # 3.候选点加入后可以形成最小包围圆
            num_count += 1
            if candidate_mbs.processed == 0:
                # if candidate_mbs.count <= mbs_set[str(element)].count:
                num_clique += 1
                mbs_inner_points.append(tuple(points_set[str(candidate)].location))
                if check_if_clique(points_set[str(candidate)], mbs_set[str(element)]):
                    # 先将候选点加入mbs集合中
                    temp_miniball = miniball.Miniball(mbs_inner_points)

                    num_sec += 1  # 执行miniball的计数器

                    mbs_set[str(element)].inner_points.append(candidate)  # 将候选点加入到现在的mbs
                    mbs_set[str(element)].count += 1  # 更新现在mbs点的数目
                    mbs_set[str(element)].center = temp_miniball.center()

                    candidate_mbs.inner_points.remove(candidate)  # 将候选点从原来的mbs中删除
                    candidate_mbs.count -= 1

                    points_set[str(candidate)].belongs_to = mbs_set[str(element)].index
                    points_set[str(candidate)].processed = 1

                    # mbs_inner_points.append(tuple(__points_set[str(candidate)].location))
                    # print("RADIUS:")
                    # print(math.sqrt(temp_miniball.squared_radius()))

                    num_sec_success += 1

                    if candidate_mbs.count == 0:
                        del mbs_set[str(candidate_mbs.index)]
                else:
                    mbs_inner_points.remove(tuple(points_set[str(candidate)].location))

        mbs_set[str(element)].processed = 1
        points_set[str(chosen_point)].processed = 1

        # adjacency_matrix_copy = adjacency_matrix.copy()
        adjacency_matrix, points_set, degree = update_graph(adjacency_matrix, mbs_set[str(element)], points_set)

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
        # print(value.degree)
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
        = initialize(400, "uniform")  # uniform, normal
    __adjacency_matrix = np.array(__adjacency_matrix)
    # __mbs_set = randomization_1_start(__points_set, __degree)
    __mbs_set, count = core(__points_set.copy(), __degree.copy(), __points.copy(), __adjacency_matrix.copy())

    __mbs_set = get_center(__mbs_set, __points_set)

    """
    # Draw.draw_mbs(__mbs_set)

    # Draw.show_scatter(__points)

    # Draw.plt.axis("equal")

    # Draw.plt.show()
    # Draw.plt.clf()

    # Draw.draw_plot(__points, __adjacency_matrix)
    # Draw.show_scatter(__points)

    # Draw.plt.axis("equal")
    # Draw.plt.show()
    # Draw.plt.clf()

    Draw.draw_mbs(__mbs_set)
    Draw.show_scatter(__points)
    Draw.plt.axis("equal")
    Draw.plt.show()
    """

    return len(__mbs_set), count


if __name__ == "__main__":
    # main()
    # profile.run('main()')

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
