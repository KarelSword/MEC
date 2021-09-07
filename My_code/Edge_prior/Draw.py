import matplotlib.pyplot as plt
import numpy as np

RADIUS = 25


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
def draw_mbs(center):
    theta = np.arange(0, 2 * np.pi, 0.01)
    x = center[0] + RADIUS * np.cos(theta)
    y = center[1] + RADIUS * np.sin(theta)
    plt.plot(x, y, "r--", color="blue")


# 绘制出最终的mbs
def draw_mbs_set(center_set):
    for item in center_set:
        theta = np.arange(0, 2 * np.pi, 0.01)
        x = item[0] + RADIUS * np.cos(theta)
        y = item[1] + RADIUS * np.sin(theta)
        plt.plot(x, y, "r--", color="blue")

