import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
import matplotlib

"""
对比图像的绘制
"""

RADIUS = 800
DIAMETER = 2 * RADIUS


# 绘制点，可以选择是否绘制点的信息
def show_scatter(points, flag, points_color):
    plt.scatter(points[:, 0], points[:, 1], marker="o", color=points_color,
                edgecolors="black", s=10, linewidths=0.5, label='地面终端')
    # plt.scatter(points[400, 0], points[400, 1], marker="o", color=points_color, zorder=0,
    #             edgecolors="black", s=10, linewidths=0.5, label='GT', )
    if flag:
        ax = plt.gca()
        n = np.arange(len(points))
        for i, txt in enumerate(n):
            ax.annotate(txt, (points[i][0], points[i][1]),
                        fontsize=12, xytext=(points[i][0] + 4, points[i][1] + 0))


# 绘制出MDP的mbs
def draw_mbs(mbs_set):
    for value in mbs_set:
        plt.scatter(value[0], value[1], marker="*", color="#33CC33",
                    edgecolors="black", s=80, linewidths=1, zorder=100)
        theta = np.arange(0, 2 * np.pi, 0.01)
        x = value[0] + RADIUS * np.cos(theta)
        y = value[1] + RADIUS * np.sin(theta)
        plt.plot(x, y, linestyle="-", color="#33CC33", linewidth=1.5)

    plt.scatter(value[0], value[1], marker="*", color="#33CC33",
                edgecolors="black", s=80, linewidths=1, label="(MBS)MDP")


# 绘制出其它mbs
def draw_mbs_other(mbs_set, style, name, color):
    for value in mbs_set:
        plt.scatter(value[0], value[1], marker=style, color=color,
                    edgecolors="black", s=80, linewidths=1, zorder=100)
        theta = np.arange(0, 2 * np.pi, 0.01)
        x = value[0] + RADIUS * np.cos(theta)
        y = value[1] + RADIUS * np.sin(theta)
        plt.plot(x, y, linestyle="-", color="#33CC33", linewidth=1.5)

    plt.scatter(value[0], value[1], label=name, marker=style, color=color,
                edgecolors="black", s=80, linewidths=1)


if __name__ == '__main__':

    # 设置中文字体和负号正常显示
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False

    name = 'F:/Scientific_Literature/写论文/test_data/Uruguay - 734.csv'
    # name = 'C:/Users/karel/Desktop/大论文/Data/50_0.csv'
    points = pd.read_csv(name, usecols=[1, 2]).values
    # hull = ConvexHull(points)
    # plt.plot(points[hull.vertices, 0], points[hull.vertices, 1], 'r--', lw=2)
    # plt.plot(points[hull.vertices, 0], points[hull.vertices, 1], 'ro')
    show_scatter(points, 0, '#00CCFF')  # 00CCFF # 0099CC

    """
    name = 'C:/Users/karel/Desktop/article/Article_Data/Fig_4/SDF_Uruguay.csv'
    sdf_mbs = pd.read_csv(name, usecols=[1, 2]).values
    draw_mbs_other(sdf_mbs, "*", "MBS(Core-sets)", "#33CC33")
    """

    """
    # name = 'C:/Users/74412/Desktop/article/Article_Data/Fig_4/MDP_Uruguay.csv'
    name = 'C:/Users/karel/Desktop/大论文/Data/MDP_VBS.csv'
    MDP_mbs = pd.read_csv(name, usecols=[1, 2]).values
    draw_mbs(MDP_mbs)
    """

    """
    # name = 'C:/Users/74412/Desktop/article/Article_Data/Fig_4/Spiral_Uruguay.csv'
    name = 'C:/Users/karel/Desktop/大论文/Data/Spiral_VBS.csv'
    spiral_mbs = pd.read_csv(name, usecols=[1, 2]).values
    x = [dot[0] for dot in spiral_mbs]
    y = [dot[1] for dot in spiral_mbs]
    # plt.plot(x, y, color="red", linewidth=1.5)
    draw_mbs_other(spiral_mbs, "*", "MBS(Spiral)", "#33CC33")
    """

    """
    name = 'C:/Users/karel/Desktop/大论文/Data/K-means_VBS.csv'
    k_means_mbs = pd.read_csv(name, usecols=[1, 2]).values
    draw_mbs_other(k_means_mbs, "*", "MBS(K-means)", "#33CC33")
    """


    # Edge-prior算法
    name = 'C:/Users/karel/Desktop/EDGE_VBS.csv'
    edge_mbs = pd.read_csv(name, usecols=[1, 2]).values
    draw_mbs_other(edge_mbs, "*", "基站(Edge-Prior)", "#33CC33")


    """
    # 遗传算法
    name = 'C:/Users/karel/Desktop/遗传_VBS.csv'
    edge_mbs = pd.read_csv(name, usecols=[1, 2]).values
    draw_mbs_other(edge_mbs, "*", "基站(遗传算法)", "#33CC33")
    """

    plt.legend()
    plt.axis("equal")
    plt.axis('off')

    plt.savefig("C:/Users/karel/Desktop/E.png", dpi=1000, bbox_inches = 'tight')
    plt.show()

