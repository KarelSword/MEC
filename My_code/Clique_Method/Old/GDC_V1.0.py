# 对专利的算法的撰写
# 2020年9月25日
"""
本代码是对专利代码实现的撰写
经过实际测试，发现自己代码运行的速度还可以
作者：karel
"""
import numpy as np
from scipy.spatial.distance import pdist, squareform
import miniball
import math
import time

from My_code.Clique_Method import Draw

RADIUS = 40.00
DIAMETER = 2*RADIUS


# 无向图中点的类
class Point:
    index = -1  # 点的编号
    location = []  # 点的坐标
    neighbor_info = []  # 点的相邻信息
    degree = 0  # 点的度数
    selected = 0  # 用于判定randomization()中点是否被处理过
    belongs_to = -1  # 用于判定点属于哪一个MBS

    def __init__(self, location, index):
        self.location = location
        self.neighbor_info = []
        self.degree = 0
        self.index = index

    def __del__(self):
        self.location = []
        self.neighbor_info = []
        self.degree = 0
        self.index = -1


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


'''
# 返回任意两点之间的距离
def distance(item, jtem):
    """本代码段返回任意两点之间的距离，输入为任意两点的坐标，返回为两点之间的距离"""
    return np.hypot(item[0] - jtem[0], item[1] - jtem[1])
'''


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
    adjacency_matrix = create_graph(points)
    # 确定点的度数,初始化点的度数列表
    degree = determine_degree(adjacency_matrix)
    # 确定点的邻接矩阵
    adjacency_list = determine_adjacency_list(points, adjacency_matrix)

    for i in range(0, len(points)):
        a = Point(points[i], i)
        a.neighbor_info = adjacency_list[i]
        a.degree = degree[str(i)]
        points_set[str(i)] = a

    return points_set, points, adjacency_matrix, degree


# 创建无向图
def create_graph(points):
    adjacency_matrix = [[0 for col in range(len(points))] for row in range(len(points))]
    """本代码段根据点集创建无向图，当任意两点之间的距离小于2r时，则认为他们是相邻的"""
    points_distance = squareform(pdist(points))
    for i, item in enumerate(points_distance):
        for j, jtem in enumerate(item):
            if jtem <= DIAMETER:
                adjacency_matrix[i][j] = 1
            else:
                adjacency_matrix[i][j] = 0
            if i == j:
                adjacency_matrix[i][j] = -1
    """
    for i, item in enumerate(__points):
        for j, jtem in enumerate(__points):
            if np.hypot(item[0] - jtem[0], item[1] - jtem[1]) < 2 * RADIUS:
                adjacency_matrix[i][j] = 1
            else:
                adjacency_matrix[i][j] = 0
            if i == j:
                adjacency_matrix[i][j] = -1
    """
    return adjacency_matrix


# 确定所有点的度数
def determine_degree(adjacency_matrix):
    degree = {}
    """本代码段根据点和点之间的相邻信息确定任一点的度数信息"""
    for row_index, row_item in enumerate(adjacency_matrix):
        degree[str(row_index)] = 0
        for column_index, column_item in enumerate(row_item):
            if column_item == 1:
                degree[str(row_index)] = degree[str(row_index)] + 1
    return degree


# 创建无向图的邻接表
def determine_adjacency_list(points, adjacency_matrix):
    adjacency_list = []
    for i, item in enumerate(points):
        a = []
        for ne_index in range(0, len(points)):
            if adjacency_matrix[i][ne_index] == 1:
                a.append(ne_index)
        adjacency_list.append(a)
    return adjacency_list


# 初始化随机构造若干个MBS
def randomization():
    """
    初始化随机构造若干个MBS

    :return:
    """
    mbs_set = {}  # mbs_set:mbs的点集合
    points_index = list((__points_set.keys()))  # points_index:目前还存在的点的集合
    points_index = np.array(points_index).astype(np.int).tolist()  # 将str转换成list
    np.random.shuffle(points_index)  # 对points_index的随机化操作

    mbs_index = 0

    # 遍历points_index中所有的点
    for item in points_index:
        # temp_list 储存目前选取的点(item)的相邻信息
        sign = 0

        temp_list = __points_set[str(item)].neighbor_info
        # 对temp_list随机化
        np.random.shuffle(temp_list)

        if __points_set[str(item)].selected == 1:
            continue

        __points_set[str(item)].selected = 1

        # 遍历temp_list，即为遍历目前选取的点(item)的相邻信息
        for value in temp_list:
            # 如果目前的点尚未被选取则
            if __points_set[str(value)].selected == 0:
                mbs = MBS()
                mbs.inner_points.append(item)
                mbs.inner_points.append(value)
                mbs.count = len(mbs.inner_points)
                mbs.index = mbs_index
                mbs_set[str(mbs_index)] = mbs

                __points_set[str(value)].belongs_to = mbs_index
                __points_set[str(item)].belongs_to = mbs_index
                __points_set[str(value)].selected = 1

                sign = 1
                mbs_index += 1
                break
        if sign == 0:
            mbs = MBS()
            mbs.inner_points.append(item)
            mbs.count = len(mbs.inner_points)
            mbs.index = mbs_index
            mbs_set[str(mbs_index)] = mbs

            __points_set[str(item)].belongs_to = mbs_index
            mbs_index += 1

    '''
    print("mbs_set")
    # count = 0

    for key, value in mbs_set.items():
        print(key)
        print(value.inner_points)
        # count += value.count
    # print("points len  of mbs_set")
    # print(count)
    print("len(__delete_set)")
    print(len(__delete_set))
    '''

    return mbs_set


def randomization_1_start():
    """
    初始化随机构造若干个MBS
    一个点开始
    :return:
    """
    mbs_set = {}  # mbs_set:mbs的点集合
    points_index = list((__points_set.keys()))  # points_index:目前还存在的点的集合
    points_index = np.array(points_index).astype(np.int).tolist()  # 将str转换成list
    np.random.shuffle(points_index)  # 对points_index的随机化操作

    mbs_index = 0


# 确定任一点加入团后，是否还能构成一个团
def check_if_clique(candidate, clique):
    """
    确定任一点加入团后，是否还能构成一个团

    :param candidate: 候选的点
    :param clique:    要并入的团
    :return:          可以返回True，否则Flase
    """
    if set(clique.inner_points) <= set(candidate.children_info):
        return True
    else:
        return False


# 读取mbs中所有点的位置信息，并且转换为tuple类型
def read_loacation(mbs):
    mbs_innerpoints_tuple = []
    for item in mbs.inner_points:
        temp = tuple(__points_set[str(item)].location)
        mbs_innerpoints_tuple.append(temp)

    return mbs_innerpoints_tuple


def core():
    """
    核心合并的模块

    :return:
    """
    # 初始化的操作
    mbs_set = randomization()
    num_mbs = 0
    num_count = 0
    num_clique = 0
    num_sec = 0
    num_sec_success = 0
    # 遍历mbs_set,即遍历MBS集合
    for element in range(len(mbs_set)):
        num_mbs += 1
        # 尝试是否存在这个MBS
        try:
            mbs_set[str(element)]
        except KeyError:
            continue
        # 如果改MBS已经被处理过，则直接跳过该点
        if mbs_set[str(element)].processed == 1:
            continue
        # 如果是一个点的MBS，则直接遍历
        if mbs_set[str(element)].count == 1:
            chosen_point = mbs_set[str(element)].inner_points[0]
            # chosen_point_ne 选择的点的相邻的点
            chosen_point_ne = __points_set[str(chosen_point)].neighbor_info
        # 如果是两个点的MBS，则遍历其中度数较少的点
        if mbs_set[str(element)].count == 2:
            point_1 = mbs_set[str(element)].inner_points[0]
            point_2 = mbs_set[str(element)].inner_points[1]
            if __degree[str(point_1)] < __degree[str(point_2)]:
                chosen_point = point_1
                # chosen_point_ne 选择的点的相邻的点
                chosen_point_ne = __points_set[str(chosen_point)].neighbor_info
                chosen_point_ne.remove(point_2)
            else:
                chosen_point = point_2
                # chosen_point_ne 选择的点的相邻的点
                chosen_point_ne = __points_set[str(chosen_point)].neighbor_info
                chosen_point_ne.remove(point_1)

        # 读取MBS中点位置信息
        mbs_inner_points = read_loacation(mbs_set[str(element)])
        # 遍历选取的点的所有临近的点
        for candidate in chosen_point_ne:
            # candidate_mbs:候选点所属于的mbs
            '''
            try:
                mbs_set[str(__points_set[str(candidate)].belongs_to)]
            except KeyError:
                print("error_key")
                print(candidate)
                continue
            '''
            candidate_mbs = mbs_set[str(__points_set[str(candidate)].belongs_to)]
            # 如果满足以下条件：
            # 1.candidate_mbs的大小小于目前的大小
            # 2.候选点加入后可以成团
            # 3.候选点加入后可以形成最小包围圆
            num_count += 1
            if candidate_mbs.count <= mbs_set[str(element)].count:
                # num_clique += 1
                if check_if_clique(__points_set[str(candidate)], mbs_set[str(element)]):
                    mbs_inner_points.append(tuple(__points_set[str(candidate)].location))
                    temp_miniball = miniball.Miniball(mbs_inner_points)
                    num_sec += 1
                    if (math.sqrt(temp_miniball.squared_radius())) <= RADIUS:
                        # print("now the mbs points are:")
                        # print(mbs_set[str(element)].inner_points)
                        # print("now the candidate point:")
                        # print(__points_set[str(candidate)].index)

                        mbs_set[str(element)].inner_points.append(candidate)  # 将候选点加入到现在的mbs
                        mbs_set[str(element)].count += 1  # 更新现在mbs点的数目
                        # mbs_set[str(element)].center = temp_miniball.center()

                        candidate_mbs.inner_points.remove(candidate)  # 将候选点从原来的mbs中删除
                        candidate_mbs.count -= 1

                        __points_set[str(candidate)].belongs_to = mbs_set[str(element)].index

                        # mbs_inner_points.append(tuple(__points_set[str(candidate)].location))
                        # print("RADIUS:")
                        # print(math.sqrt(temp_miniball.squared_radius()))
                        num_sec_success += 1

                        if candidate_mbs.count == 0:
                            del mbs_set[str(candidate_mbs.index)]

        mbs_set[str(element)].processed = 1
        # print("-------------------")
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

    return mbs_set


# 计算每个MBS的中心
def get_center(mbs_set):
    for key, value in mbs_set.items():
        inner_points = read_loacation(value)
        temp_miniball = miniball.Miniball(inner_points)
        value.center = temp_miniball.center()


# 显示点集的度数信息和相邻信息
def show_points():
    for key, value in __points_set.items():
        print("the points" + key)
        # print(value.degree)
        print(value.neighbor_info)
        print(value.belongs_to)


def show_mbs(mbs_set):
    for key, value in mbs_set.items():
        value.print()


if __name__ == "__main__":
    start_time = time.time()
    result = []
    # __points_set，点类的集合，__points点的列表集合，__adjacency_matrix邻接矩阵
    __points_set, __points, __adjacency_matrix, __degree = initialize(400, "uniform")  # uniform, normal
    __adjacency_matrix = np.array(__adjacency_matrix)
    # show_points()

    # __delete_set, __points_set = pre_deal_points(__points_set.copy())
    # print(__adjacency_matrix)
    # processed__adjacency_matrix = update_adjacency_matrix(__adjacency_matrix.copy(), __delete_set)
    # __degree = update_degree(__adjacency_matrix)

    __mbs_set = core()
    print(len(__mbs_set))

    get_center(__mbs_set)
    # show_mbs(__mbs_set)
    # draw_mbs(__mbs_set)

    # show_points()

    end_time = time.time()
    secs = end_time - start_time
    print(" took", secs, "seconds")

    """
    Draw.draw_plot(__points, __adjacency_matrix)
    Draw.show_scatter(__points)

    Draw.plt.axis("equal")
    Draw.plt.show()
    """
    print(__degree)






