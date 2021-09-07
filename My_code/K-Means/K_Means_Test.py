# 研究使用K-Means的使用过程
import time

from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import profile


# K-means 直接跑的方法,从第一个开始尝试
def main():
    start_time = time.time()

    cluster_number = 14

    # 生成的数据
    global index
    global POINT_COUNT
    # name = 'C:/Users/74412/Desktop/article/Article_Data/Uniform/' + str(POINT_COUNT) + '_' + str(index) + '.csv'
    # data = pd.read_csv(name, usecols=[1, 2]).values
    data = pd.read_csv('F:/Scientific_Literature/写论文/test_data/Uruguay - 734.csv', usecols=[1, 2]).values
    # data = np.random.uniform(-100, 100, size=(400, 2))
    # data = data.round(2)

    while True:

        cluster = {}
        cluster_number += 1

        for i in range(100):
            k_means = KMeans(n_clusters=cluster_number, init="k-means++").fit(data)

            # 距离矩阵，每个点到最近MBS的距离
            distance_result = k_means.transform(data)

            closest_ball_dis = np.min(distance_result, axis=1)

            max_distance = np.max(closest_ball_dis)
            print(max_distance)

            if max_distance <= RADIUS:

                fit_result = k_means
                predict_result = k_means.predict(data)

                for item in range(cluster_number):
                    index = np.where(predict_result == item)
                    cluster[item] = data[index]


                end_time = time.time()
                secs = end_time - start_time

                print(" took", secs, "seconds")

                for index, value in cluster.items():
                    plt.scatter(value[:, 0], value[:, 1], marker="p", color='#0099CC',
                                edgecolors="black", s=1, linewidths=0.02)
                    plt.scatter(fit_result.cluster_centers_[index][0], fit_result.cluster_centers_[index][1],
                                marker="*", color="#FF0066", edgecolors="black", s=50, linewidths=1, zorder = 100)

                    theta = np.arange(0, 2 * np.pi, 0.01)
                    x = fit_result.cluster_centers_[index][0] + RADIUS * np.cos(theta)
                    y = fit_result.cluster_centers_[index][1] + RADIUS * np.sin(theta)
                    plt.plot(x, y, linestyle="-", color="#008000", linewidth=1)

                plt.axis("equal")
                plt.show()
                print("cluster_number "+str(cluster_number)+" right")


                return cluster_number, fit_result.cluster_centers_
        """
        ax = plt.gca()
        n = np.arange(len(data))
        for i, txt in enumerate(n):
            ax.annotate(txt, (data[i][0], data[i][1]),
                        fontsize=10, xytext=(data[i][0] + 3, data[i][1] + 0))
        """
        """
        plt.scatter(data[:, 0], data[:, 1], marker="p", color="blue",
                    edgecolors="black", s=20, linewidths=1)
        """



if __name__ == "__main__":

    """
    # 测试K-Means
    RADIUS = 50
    POINT_COUNT = 200
    index = 5

    profile.run('main()')
    """

    """
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
    RADIUS = 800
    POINT_COUNT = 200

    start_time = time.time()

    ave_mbs_num = 0
    min_mbs_num = 10000
    number = 1

    for index in range(number):

        mbs_num, K_Means_center = main()
        print(mbs_num)
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
    df = pd.DataFrame(K_Means_center)
    df.to_csv('C:/Users/74412/Desktop/article/k_center/K-means_URA_3.csv')
    """