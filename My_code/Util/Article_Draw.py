import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform, cdist

RADIUS = 25
DIAMETER = 2 * RADIUS


# 绘制邻接的信息
def draw_plot(points, adjacency_matrix):
    tmp = np.where(adjacency_matrix == 1)
    plt.plot(points[tmp, 0], points[tmp, 1], color="red",
             linestyle="-", linewidth=0.8, zorder=0)
    plt.plot(points[0, 0], points[0, 1], color="red",label="EDGE",
            linestyle="-", linewidth=0.8, zorder=0)


# 绘制点，可以选择是否绘制点的信息
def show_scatter(points, flag, points_color):
    plt.scatter(points[:, 0], points[:, 1], marker="o", color=points_color,
                edgecolors="black", s=10, linewidths=0.8, label='GT', )
    if flag:
        ax = plt.gca()
        n = np.arange(len(points))
        for i, txt in enumerate(n):
            ax.annotate(txt, (points[i][0], points[i][1]),
                        fontsize=12, xytext=(points[i][0] + 4, points[i][1] + 0))


# 绘制出最终的mbs
def draw_mbs(mbs_set):
    for value in mbs_set.values():
        plt.scatter(value.center[0], value.center[1], marker="*", color="#FF0066",
                    edgecolors="black", s=80, linewidths=1)
        theta = np.arange(0, 2 * np.pi, 0.01)
        x = value.center[0] + RADIUS * np.cos(theta)
        y = value.center[1] + RADIUS * np.sin(theta)
        plt.plot(x, y, linestyle="-", color="#008000", linewidth=1)


# 绘制出最终的mbs
def draw_mbs_temp(mbs_set):
    theta = np.arange(0, 2 * np.pi, 0.01)
    x = mbs_set.center[0] + RADIUS * np.cos(theta)
    y = mbs_set.center[1] + RADIUS * np.sin(theta)
    plt.plot(x, y, "r--", color="blue")


# 为了绘制论文中的图片而写的函数
def draw_mbs_for_Show_off(mbs_set, points_set):
    i = 1
    ax = plt.gca()
    for value in mbs_set.values():
        print(value.center)
        plt.scatter(value.center[0], value.center[1], marker="*", color="#FF0066",
                    edgecolors="black", s=100, linewidths=1)
        theta = np.arange(0, 2 * np.pi, 0.01)
        x = value.center[0] + RADIUS * np.cos(theta)
        y = value.center[1] + RADIUS * np.sin(theta)
        plt.plot(x, y, linestyle="-", color="#008000", linewidth=1)
        ax.annotate("mbs" + str(i), (value.center[0], value.center[1]),
                    fontsize=10, xytext=(value.center[0] + 22, value.center[1] + 15))
        inner_points = value.inner_points
        print(inner_points)
        for item in inner_points:
            location = points_set[str(item)].location
            plt.scatter(location[0], location[1], marker="p", color="#33CCFF",
                        edgecolors="black", s=100, linewidths=1)
        i += 1
        if i > 3:
            break


# 创建无向图
def create_graph(points):
    """本代码段根据点集创建无向图，当任意两点之间的距离小于2r时，则认为他们是相邻的"""
    points_distance = squareform(pdist(points))
    points_distance = np.where(points_distance <= DIAMETER, 1, points_distance)
    points_distance = np.where(points_distance > DIAMETER, 0, points_distance)
    adjacency_matrix = points_distance
    np.fill_diagonal(adjacency_matrix, 0)

    return adjacency_matrix, points_distance


if __name__ == '__main__':
    # name = 'C:/Users/74412/Desktop/article/Article_Data/Article/Article.csv'
    # name = 'F:/Scientific_Literature/cow.wrl/cow.csv'
    name = 'F:/Scientific_Literature/写论文/test_data/Rwanda - 1,621.csv'
    points = pd.read_csv(name, usecols=[1, 2]).values

    # adjacency_matrix, points_distance = create_graph(points)
    # draw_plot(points, adjacency_matrix)

    show_scatter(points, 0, '#00CCFF')
    plt.legend()
    plt.axis("equal")
    plt.show()
