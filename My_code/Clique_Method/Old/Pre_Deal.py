
class Tree:
    inner_points = []
    count = 0
    father_point = []
    child_point = []

    def __init__(self):
        self.inner_points = []
        self.count = 0
        self.father_point = []
        self.child_point = []

    def append(self, point):
        if point.__class__ is int:
            self.inner_points.append(point)
        if point.__class__ is list:
            for i in range(0, len(point)):
                self.inner_points.append(point[i])

    def print(self):
        print(self.inner_points)


# 处理度数为0的点
def deal_0_degree(sign, points_set):
    delete_set = []
    for index, value in points_set.items():
        if value.children_num == 0:
            '''
            a = MBS()
            a.inner_points.append(index)
            a.count += 1
            result.append(a)
            '''
            delete_set.append(int(index))

    for item in delete_set:
        del points_set[str(item)]

    # print("下面是0处理后删除的点")
    # print(delete_set)
    if delete_set:
        sign = 1

    return sign, delete_set


# 处理度数为1的点
def deal_1_degree(sign, points_set):
    # 遍历点集合中所有的点
    delete_set = []  # 删除节点的集合
    tree_set = []  # 树结构的集合
    father_list = []  # 父节点的集合
    for index, value in points_set.items():
        if value.children_num == 1:
            # father_index为无向图中，目前选取的点所唯一连接的父节点点的编号
            father_index = value.children_info[0]
            father_list.append(father_index)

    father_list = list(set(father_list))
    # print("下面是father集合的内容")
    # print(father_list)
    if father_list:
        sign = 1

    for item in father_list:
        # 将父节点添加到树结构中
        try:
            points_set[str(item)]
        except KeyError:
            continue
        tree = Tree()
        # 将父节点添加到树结构中
        tree.append(item)
        # father_ne是父节点所连接的节点
        father_ne = points_set[str(item)].children_info
        # 遍历父节点中所有的相邻点，处理其中度数为1的点
        for element in father_ne:
            # 如果该点是度数为1的地点
            # 存储进入树结构，然后删去
            if points_set[str(element)].children_num == 1:
                tree.append(item)
                del points_set[str(element)]
                delete_set.append(element)
            # 如果该点不是度数为1的地点
            # 则度数减1，并且将父节点的相邻信息删去
            else:
                points_set[str(element)].children_num -= 1
                points_set[str(element)].children_info.remove(item)
        # tree.print()
        # print("---")
        # 将树结构存储起来
        tree_set.append(tree)

        del points_set[str(item)]
        delete_set.append(item)

    return sign, delete_set


# 预处理的程序
def pre_deal_points(points_set):
    delete_set = []

    while True:
        sign = 0

        # print("下面是开始前点集合的内容")
        # print(points_set.keys())
        sign, temp_delete_set = deal_0_degree(sign, points_set)
        delete_set.extend(temp_delete_set)
        # print("下面是0处理后sign的值")
        # print(sign)
        sign, temp_delete_set = deal_1_degree(sign, points_set)

        # print("下面是1处理后sign的值")
        # print(sign)
        # print("下面是结束后点集合的内容")
        # print(points_set.keys())
        delete_set.extend(temp_delete_set)

        # print("下面是删除的内容")
        # print(delete_set)

        if sign == 0:
            break

    return delete_set, points_set


def update_adjacency_matrix(adjacency_matrix, delete_set):
    for i in delete_set:
        adjacency_matrix[i, :] = -1
        adjacency_matrix[:, i] = -1

    return adjacency_matrix


def update_degree(adjacency_matrix):
    degree = {}
    """本代码段根据点和点之间的相邻信息更新所有点的度数信息"""
    for row_index, row_item in enumerate(adjacency_matrix):
        degree[str(row_index)] = 0
        for column_index, column_item in enumerate(row_item):
            if column_item == 1:
                degree[str(row_index)] = degree[str(row_index)] + 1
    # 删除已经被删除的点的信息
    for item in __delete_set:
        del degree[str(item)]
    return degree
