# 研究使用K-Means的使用过程
import time

from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import profile


# 二分确定范围
def bi_search(this, this_flag, prio, front, back):
    difference = abs(this - prio)
    if difference == 1:
        if this_flag == 1:
            return this, 0
        if this_flag == 0:
            return prio, 0

    if difference != 1:
        if this_flag == 1:
            next_one = this - int(((this-front)/2)+0.5)
            return next_one, 1
        if this_flag == 0:
            next_one = this + int(((back-this)/2)+0.5)
            return next_one, -1


def main():
    start_time = time.time()

    # 生成的数据
    global index
    global POINT_COUNT
    name = 'C:/Users/74412/Desktop/article/Article_Data/Uniform/' + str(POINT_COUNT) + '_' + str(index) + '.csv'
    data = pd.read_csv(name, usecols=[1, 2]).values
    # data = pd.read_csv('F:/Scientific_Literature/写论文/test_data/Nicaragua - 3,496.csv', usecols=[1, 2]).values
    # data = np.random.uniform(-100, 100, size=(400, 2))
    # data = data.round(2)

    front = 0
    back = POINT_COUNT
    prio_success_flag = 0
    k = POINT_COUNT
    prio_index = 0
    mini_num = float("inf")

    # 初始情况下，分簇的总数从POINT_COUNT开始
    cluster_number = POINT_COUNT
    # 成功标志为一开始假设为失败
    success_flag = 0

    while True:

        cluster_number = k
        success_flag = 0

        for i in range(100):
            k_means = KMeans(n_clusters=cluster_number, init="k-means++").fit_transform(data)

            # 距离矩阵，每个点到最近MBS的距离
            distance_result = k_means

            closest_ball_dis = np.min(distance_result, axis=1)

            max_distance = np.max(closest_ball_dis)
            # print(max_distance)

            if max_distance <= RADIUS:
                success_flag = 1

                end_time = time.time()
                secs = end_time - start_time
                print(" took", secs, "seconds")

                break

        print(cluster_number)
        print(success_flag)
        print(max_distance)
        print("----------------------------------")

        # 如果成功了，并且目前的数目大于已知最优的，则重新执行
        if success_flag == 0 and cluster_number > mini_num:
            continue

        # 更新最优的数目
        if success_flag == 1 and mini_num > cluster_number:
            mini_num = cluster_number

        # flag:0代表二分搜索完毕
        k, flag = bi_search(cluster_number, success_flag, prio_index, front, back)

        # 如果二分搜索完毕，返回k个mini_mum最小值
        if flag == 0:
            if success_flag == 1:
                return k
            elif success_flag == 0:
                return mini_num
        # 如果二分搜索没完毕，收缩区间
        elif flag == -1:
            front = cluster_number
            prio_index = cluster_number
        elif flag == 1:
            back = cluster_number
            prio_index = cluster_number


if __name__ == "__main__":

    """
    # 用来单次执行，或者分析程序执行时间
    RADIUS = 50
    POINT_COUNT = 200
    index = 5

    profile.run('main()')
    """

    """
    # 用于单次执行
    start_time = time.time()

    count = 0
    for i in range(1):
        count += main()
    print(count/1)

    end_time = time.time()
    secs = end_time - start_time
    print(" took", secs, "seconds")
    """

    # 在已经生成的数据上，跑代码
    RADIUS = 50
    POINT_COUNT = 200

    start_time = time.time()

    ave_mbs_num = 0
    min_mbs_num = 10000
    number = 100
    sum_ex_count = 0

    for index in range(number):

        mbs_num = main()
        print("mbs_num")
        print(mbs_num)
        print("success------")
        ave_mbs_num += mbs_num
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

    """
    df = pd.DataFrame(opt_ball_center)
    df.to_csv('C:/Users/74412/Desktop/article/k_center/SDF_Rwanda.csv')
    """