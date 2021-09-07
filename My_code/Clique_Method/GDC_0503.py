"""
简化版本，用来出版
作者：karel

打算出版的版本
5月3日
在GDC 0411的基础上进一步进行改进的工作
"""
import profile
import numpy as np
from scipy.spatial.distance import pdist, squareform, cdist
import pandas as pd
import miniball
import math
import time
from My_code.Clique_Method import Draw

RADIUS = 50
DIAMETER = 2 * RADIUS


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
    if distributed == "uniform":
        points = np.random.uniform(-100, 100, size=(number, 2))
        points = np.round(points, 2)
        # df = pd.DataFrame(points)
        # df.to_csv('C:/Users/74412/Desktop/article/GDC_test_data/PandasNumpy.csv')
        # points = np.round(points, 3)
    if distributed == "normal":
        points = np.random.normal(100, 50, size=(number, 2))
        # points = np.random.standard_normal(size=(number, 2)) * 100
        # points = np.round(points, 3)
    if distributed == "poisson":
        points = np.random.poisson(lam=(100, 100), size=(number, 2))
    if distributed == "read":
        # global index
        name = 'C:/Users/74412/Desktop/article/Article_Data/Uniform/' + str(number) + '_' + str(index) + '.csv'
        points = pd.read_csv(name, usecols=[1, 2]).values
        # points = pd.read_csv('F:/Scientific_Literature/写论文/test_data/Luxembourg - 980.csv', usecols=[1, 2]).values
    # 初始化点类的集合
    points_set = {}

    # ————————————————————————————————————
    # 构建无向图，返回邻接矩阵,初始化邻接矩阵
    adjacency_matrix, points_distance = create_graph(points)
    # 确定点的度数,初始化点的度数列表
    degree = determine_degree(adjacency_matrix)
    # 确定点的邻接矩阵
    adjacency_list = determine_adjacency_list(adjacency_matrix)

    for i in range(0, len(points)):
        a = Point(points[i], i)

        a.neighbor_info = adjacency_list[i]
        a.degree = degree[str(i)]
        points_set[str(i)] = a

    return points_set, points, adjacency_matrix, degree, points_distance


# 创建无向图
def create_graph(points):
    """本代码段根据点集创建无向图，当任意两点之间的距离小于2r时，则认为他们是相邻的"""
    points_distance = squareform(pdist(points))
    points_distance = np.where(points_distance <= DIAMETER, 1, points_distance)
    points_distance = np.where(points_distance > DIAMETER, 0, points_distance)
    adjacency_matrix = points_distance
    np.fill_diagonal(adjacency_matrix, 0)

    return adjacency_matrix, points_distance


# 确定所有点的度数
def determine_degree(adjacency_matrix):
    """本代码段根据点和点之间的相邻信息确定任一点的度数信息"""
    degree = {}
    """本代码段根据点和点之间的相邻信息确定任一点的度数信息"""
    temp_degree = adjacency_matrix.sum(axis=1)
    for row_index, row_item in enumerate(adjacency_matrix):
        degree[str(row_index)] = temp_degree[row_index]

    return degree


# 创建无向图的邻接表
def determine_adjacency_list(adjacency_matrix):
    adjacency_list = []

    for item in adjacency_matrix:
        a = np.where(item == 1)[0]

        adjacency_list.append(a)

    return adjacency_list


# 更新邻接矩阵
def update_adjacency_list(adjacency_matrix, mbs, points_set):
    tmp = set()
    for item in mbs.inner_points:
        item_ne = points_set[str(item)].neighbor_info
        tmp.update(item_ne)

    tmp.difference_update(mbs.inner_points)

    for index in tmp:
        a = np.where(adjacency_matrix[index] == 1)[0]
        points_set[str(index)].neighbor_info = a

    return points_set


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
        mbs.center = points_set[str(key)].location
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
    # 判断clique内部的点是否是candidate内部点的子集
    if set(clique.inner_points).issubset(set(candidate.neighbor_info)):
        return True
    else:
        return False


# 排除形成不了团的点
def exclude_no_clique(ne_points, clique, points_set):
    p_sec = []
    for item in ne_points:
        if check_if_clique(points_set[str(item)], clique):
            p_sec.append(item)
    return p_sec


# 读取mbs中所有点的位置信息
def read_mbs_location(mbs, points):
    """
    读取mbs中所有点的位置信息
    :param mbs: 输入的mbs
    :param points:  points坐标信息
    :return:
    """
    inner_points_location = points[mbs.inner_points]

    return inner_points_location.tolist()


# 读取任意点集的位置信息
def read_location(points_index, points):
    """
    读取任意点集的位置信息
    :param points_index:
    :param points:
    :return:
    """
    points_location = points[points_index]

    return points_location


# 对临近点的度数进行排序
def sort_ne_degree(ne, points_set):
    degree_temp = {}
    for item in ne:
        degree_temp[str(item)] = points_set[str(item)].degree

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


# 删除节点后，更新整个图的信息
def update_graph(adjacency_matrix, mbs, points_set):
    adjacency_matrix[..., mbs.inner_points] = 0
    adjacency_matrix[mbs.inner_points, ...] = 0

    degree = determine_degree(adjacency_matrix)

    for index in range(0, len(points_set)):

        points_set[str(index)].degree = degree[str(index)]

    return adjacency_matrix, points_set, degree


# 合并范围r内的点
def include_r_point(ne_points_index, center, points):
    """
    :param points:
    :param ne_points_index: 备选的点的序号
    :param center:
    :return:
    """
    p_prio = []
    p_sec = []
    ne_points_location = read_location(ne_points_index, points)
    center = [tuple(center)]

    if len(ne_points_location) >= 2:
        center_distance = cdist(center, ne_points_location, 'euclidean')
        a = np.where(center_distance[0] <= RADIUS)[0]
        b = np.where(center_distance[0] > RADIUS)[0]

        for item in a:
            p_prio.append(ne_points_index[item])
        for item in b:
            p_sec.append(ne_points_index[item])

    if len(ne_points_location) == 1:
        center_distance = solve_distance(center[0], ne_points_location[0])

        if center_distance <= RADIUS:
            p_prio.append(ne_points_index[0])
        else:
            p_sec.append(ne_points_index[0])

    return p_prio, p_sec


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

    # 遍历mbs_set,即遍历MBS集合
    while True:
        degree = sort_all_degree(degree.copy(), points_set)

        if degree == {}:
            break
        degree_index = list(degree.keys())
        chosen_point = int(degree_index[0])

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

        chosen_point_ne = points_set[str(chosen_point)].neighbor_info

        chosen_point_ne = exclude_processed_points(chosen_point_ne, points_set)

        # 对该点相邻的点，依照度数顺序进行排序
        chosen_point_ne = sort_ne_degree(chosen_point_ne, points_set)

        chosen_point_ne.reverse()

        while chosen_point_ne:

            center = mbs_set[str(element)].center

            # 合并范围r内的所有点
            p_prio, chosen_point_ne = include_r_point(chosen_point_ne, center, points)
            for value in p_prio:
                mbs_set[str(element)].inner_points.append(value)  # 将候选点加入到现在的mbs
                mbs_set[str(element)].count += 1  # 更新现在mbs点的数目

                candidate_mbs = mbs_set[str(points_set[str(value)].belongs_to)]

                candidate_mbs.inner_points.remove(value)  # 将候选点从原来的mbs中删除
                candidate_mbs.count -= 1

                points_set[str(value)].belongs_to = mbs_set[str(element)].index
                points_set[str(value)].processed = 1
                # processed_points.append(value)

                if candidate_mbs.count == 0:
                    del mbs_set[str(candidate_mbs.index)]

            if chosen_point_ne:
                candidate = chosen_point_ne.pop()
            else:
                break

            mbs_inner_points = read_mbs_location(mbs_set[str(element)], points)

            # candidate_mbs:候选点所属于的mbs
            candidate_mbs = mbs_set[str(points_set[str(candidate)].belongs_to)]

            # 如果满足以下条件：
            # 1.candidate_mbs的大小小于目前的大小
            # 2.候选点加入后可以成团
            # 3.候选点加入后可以形成最小包围圆
            num_count += 1
            if candidate_mbs.processed == 0:
                # if candidate_mbs.count <= mbs_set[str(element)].count:
                num_clique += 1
                if check_if_clique(points_set[str(candidate)], mbs_set[str(element)]):
                    # 先将候选点加入mbs集合中
                    mbs_inner_points.append(points[candidate].tolist())
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
                        # processed_points.append(candidate)

                        # print("candidate true!!!")

                        num_sec_success += 1

                        if candidate_mbs.count == 0:
                            del mbs_set[str(candidate_mbs.index)]
                    else:
                        mbs_inner_points.remove(points[candidate].tolist())

            # print("-------------------------------------")
        mbs_set[str(element)].processed = 1
        points_set[str(chosen_point)].processed = 1

        adjacency_matrix, points_set, degree = update_graph(adjacency_matrix, mbs_set[str(element)], points_set)
        points_set = update_adjacency_list(adjacency_matrix, mbs_set[str(element)], points_set)

    return mbs_set, num_sec


# 显示点集的度数信息和相邻信息
def show_points(points_set):
    for key, value in points_set.items():
        print("the points" + key)
        # print(value.degree)
        print(value.neighbor_info)
        print(value.belongs_to)


# 输出mbs的信息
def show_mbs(mbs_set):
    for key, value in mbs_set.items():
        print(len(value.inner_points))
        # value.print()


# 主程序片段
def main():
    start_time = time.time()
    # __points_set，点类的集合，__points点的列表集合，__adjacency_matrix邻接矩阵
    __points_set, __points, __adjacency_matrix, __degree, __points_distance \
        = initialize(200, "read")  # uniform, normal, poisson, read
    __adjacency_matrix = np.array(__adjacency_matrix)
    # __mbs_set = randomization_1_start(__points_set, __degree)
    __mbs_set, count = core(__points_set.copy(), __degree.copy(), __points.copy(), __adjacency_matrix.copy())


    end_time = time.time()
    secs = end_time - start_time
    # print(" took", secs, "seconds")

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
    """
    Draw.draw_plot(__points, __adjacency_matrix)
    Draw.show_scatter(__points, 1, "#00CC66")
    Draw.plt.axis("equal")
    Draw.plt.show()
    """
    """
    Draw.show_scatter(__points, 0, "#00CC66")  # #33CCFF,#00CC66
    # Draw.draw_plot(__points, __adjacency_matrix)
    Draw.draw_mbs(__mbs_set)
    # Draw.draw_mbs_for_Show_off(__mbs_set, __points_set)
    Draw.plt.axis("equal")
    Draw.plt.show()
    """
    return len(__mbs_set), count


if __name__ == "__main__":

    # 在已经生成的数据上，跑代码
    start_time = time.time()

    ave_mbs_num = 0
    min_mbs_num = 10000
    number = 100
    sum_ex_count = 0

    for index in range(number):
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
    start_time = time.time()
    
    n, count = main()
    # profile.run('main()')
    print(n)
    print(count)
    end_time = time.time()
    secs = end_time - start_time
    # print(" took", secs, "seconds")
    """

    """
    start_time = time.time()

    ave_mbs_num = 0
    min_mbs_num = 10000
    number = 50
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