import matplotlib.pyplot as plt
import numpy as np
from My_code.Clique_Method import GDC_0411

RADIUS = 250


# 绘制出边的连接图
def draw_plot(points, adjacency_matrix):
    tmp = np.where(adjacency_matrix == 1)
    plt.plot(points[tmp, 0], points[tmp, 1], color="red",
             linestyle="-", linewidth=1, zorder=0)
    plt.plot(points[0, 0], points[0, 1], color="red", label="Edge",
            linestyle="-", linewidth=1, zorder=0)
    """
    for i, item in enumerate(adjacency_matrix):
        for j in range(i, len(points)):
            if adjacency_matrix[i][j] == 1:
                plt.plot((points[i][0], points[j][0]), (points[i][1], points[j][1]), color="#0099FF",
                         linestyle="--", linewidth=1)
    """


# 绘制出散点图
def show_scatter(points, flag, points_color):
    plt.scatter(points[:, 0], points[:, 1], marker="o", color=points_color,
                edgecolors="black", s=80, linewidths=1, label='Vertex', )
    """
    if flag:
        ax = plt.gca()
        n = np.arange(len(points))
        for i, txt in enumerate(n):
            ax.annotate(txt, (points[i][0], points[i][1]),
                        fontsize=11, xytext=(points[i][0] + 4, points[i][1] + 4))
    """

# 绘制出最终的mbs
def draw_mbs(mbs_set):
    # i = 0
    # ax = plt.gca()
    for value in mbs_set.values():
        plt.scatter(value.center[0], value.center[1], marker="*", color="#CC3399",
                    edgecolors="black", s=80, linewidths=0.8)
        theta = np.arange(0, 2 * np.pi, 0.01)
        x = value.center[0] + RADIUS * np.cos(theta)
        y = value.center[1] + RADIUS * np.sin(theta)
        plt.plot(x, y, linestyle="-", color="#008000", linewidth=1)  # color="#008000"

        # ax.annotate("$MBS$_"+str(i), (value.center[0], value.center[1]),
        #             fontsize=10, xytext=(value.center[0] + 7, value.center[1] + -28))
        # i += 1

    plt.scatter(mbs_set[str(0)].center[0], mbs_set[str(0)].center[1], label='Coverage disk', marker="*", color="#CC3399",
                edgecolors="black", s=80, linewidths=0.8)


# 调试BUG用的绘图工具
def show_scatter_debug(points, points_color):
    if len(points) > 2:
        plt.scatter(points[:, 0], points[:, 1], color=points_color, s=100)
    if len(points) == 2:
        plt.scatter(points[0], points[1], color=points_color, s=100)
        print("show_scatter_debug_1" + "len(points) == 2" + points_color)
    if len(points) == 1:
        plt.scatter(points[0][0], points[0][1], color=points_color, s=100)
        print("show_scatter_debug_1" + "len(points) == 1" + points_color)


# 调试BUG用的绘图工具
def show_scatter_debug_2(points, points_color):
    if len(points) >= 2:
        plt.scatter(points[:, 0], points[:, 1], color=points_color, s=100)
    if len(points) == 1:
        plt.scatter(points[0][0], points[0][1], color=points_color, s=100)


# 调试BUG用的绘图工具
def draw_txt(points):
    ax = plt.gca()
    n = np.arange(len(points))
    for i, txt in enumerate(n):
        ax.annotate(txt, (points[i][0], points[i][1]))


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
        plt.scatter(value.center[0], value.center[1], marker="*", color="#CC3399",
                    edgecolors="black", s=80, linewidths=0.8)
        theta = np.arange(0, 2 * np.pi, 0.01)
        x = value.center[0] + RADIUS * np.cos(theta)
        y = value.center[1] + RADIUS * np.sin(theta)
        plt.plot(x, y, linestyle="-", color="#008000", linewidth=1)
        ax.annotate("$MBS$_"+str(i), (value.center[0], value.center[1]),
                    fontsize=10, xytext=(value.center[0] + 5, value.center[1] - 30))
        """
        inner_points = value.inner_points
        print(inner_points)
        for item in inner_points:
            location = points_set[str(item)].location
            plt.scatter(location[0], location[1], marker="p", color="#33CCFF",
                        edgecolors="black", s=100, linewidths=1)
        """
        i += 1
        if i > 3:
            break

    plt.scatter(mbs_set[str(1)].center[0], mbs_set[str(1)].center[1], label='MBS', marker="*", color="#CC3399",
                edgecolors="black", s=80, linewidths=0.8)
