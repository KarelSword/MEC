from sklearn.cluster import KMeans
import numpy as np
import math


def calculate_distance(point1, point2):
    d_x = point1[0] - point2[0]
    d_y = point1[1] - point2[1]
    # 计算两点之间的距离
    distance = math.sqrt(d_x ** 2 + d_y ** 2)
    return distance


# 二分搜索
def binary_search(data, radius):
    # 簇的个数
    n_clusters = list(range(2, len(data) + 1))
    # 尝试簇的个数
    length = len(n_clusters)
    UAVPositionSet = []
    count = 0
    minSize = float('inf')
    aveUAV = 0

    while count < 5:
        low = 0
        high = length - 1
        UAVPosition = []
        # 不断尝试簇的个数
        while low <= high:
            cluster_result = {}

            # 从簇的中间开始尝试
            mid = int(low + ((high - low) / 2))  # 使用(low+high)/2会有整数溢出的问题
            kmeans = KMeans(n_clusters=n_clusters[mid], max_iter=1000)
            predicted = kmeans.fit_predict(data)
            distance = kmeans.fit_transform(data)
            # 记录簇结果
            # 将每个点与对应的簇对应起来

            for item in range(n_clusters[mid]):
                index = np.where(predicted == item)
                cluster_result[item] = data[index]

            """
            for i in range(n_clusters[mid]):
                cluster_result[i] = list()
            for i in range(len(predicted)):
                cluster_result[predicted[i]].append(data[i])
            """

            centroids = kmeans.cluster_centers_

            out_loop_Flag = True

            for i in range(n_clusters[mid]):
                inner_loop_flag = False
                for point in cluster_result[i]:
                    if calculate_distance(point, centroids[i]) > radius:
                        inner_loop_flag = True
                        break
                if inner_loop_flag:
                    low = mid + 1
                    out_loop_Flag = False
                    break

            if out_loop_Flag:
                high = mid - 1
                UAVPosition = centroids

        print("当前UAV的数量；", len(UAVPosition))
        count = count + 1
        aveUAV = aveUAV + len(UAVPosition)
        if len(UAVPosition) < minSize:
            minSize = len(UAVPosition)
            UAVPositionSet = UAVPosition

    return UAVPositionSet


if __name__ == "__main__":
    data = np.random.uniform(-100, 100, size=(400, 2))
    data = data.round(2)

    binary_search(data, 50)
