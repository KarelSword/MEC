"""
本程序是自己对吕老师算法的复现

"""
from scipy.spatial import ConvexHull
import numpy as np
from My_code.Spiral_Mbs.Old import My_LocalCover_Raw
import time


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
            uav_position = boundarySetInOrder[0]

        # 求当前UAV的位置，以便在进入下一个迭代的时候按逆时针方向选择其第一个最近的未被覆盖的点
        # sec_set = list(set(seq).difference(set(uav_position)))
        current_uav_position = boundarySetInOrder.index(uav_position)
        convex_set.append(uav_position)
        boundarySetInOrderSize = len(boundarySetInOrder)
        nextUAVInd = (current_uav_position + 1) % boundarySetInOrderSize

        # 传入：基站的位置，基站半径，内部点，其他非边界点
        uav_position, prio_set, miniball_count = My_LocalCover_Raw.LocalCover(uav_position, radius, [uav_position], seq)
        count += miniball_count

        uav_position_set.append(uav_position)
        # 计算下一个迭代距离当前选择的UAV位置第一近的未被覆盖的点
        flag = True
        while nextUAVInd != current_uav_position:
            if boundarySetInOrder[nextUAVInd] not in prio_set:
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

    return uav_position_set, count, convex_set


def main():
    # start_time = time.time()
    RADIUS = 25

    data = np.random.uniform(-100, 100, size=(800, 2))
    # data = np.random.normal(-100, 100, size=(800, 2))
    # data = np.random.poisson(100, size=(400, 2))
    data = np.round(data, 3)
    data = list_to_tuple(data)
    # data = [(math.modf(dot[0])[1], math.modf(dot[1])[1]) for dot in data]
    data = list(set(data))

    #  使用K-means
    # UAV_positionSet = UAV_By_KMean.binarysearch(data, RADIUS)
    # print(*data,sep='\n')
    UAV_positionSet, ex_count, convex_points_set = execute(data, RADIUS)
    # print(len(UAV_positionSet))

    # end_time = time.time()
    # secs = end_time - start_time
    # print(" took", secs, "seconds")

    """
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
        theta = np.arange(0, 2 * np.pi, 0.01)
        x = UAV[0] + RADIUS * np.cos(theta)
        y = UAV[1] + RADIUS * np.sin(theta)
        plt.plot(x, y)

    x = [dot[0] for dot in convex_points_set]
    y = [dot[1] for dot in convex_points_set]
    # # plt.xlim(126880000000,126880000700)
    # # plt.ylim(126880000000,126880000700)
    # plt.scatter(x, y, color='red')
 
    plt.plot(x, y, color="red")
    plt.axis("equal")
    plt.show()
    """
    return len(UAV_positionSet), ex_count


if __name__ == "__main__":
    # main()
    # profile.run('main()')

    start_time = time.time()

    ave_mbs_num = 0
    min_mbs_num = 5000
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
    secs_ave = secs/number

    print(" took", secs, "seconds")
    print(" sec_ave", secs_ave)
    print(" sum_ex_count", sum_ex_count/number)







