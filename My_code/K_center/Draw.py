import matplotlib.pyplot as plt
import numpy as np


def show_scatter(points, flag, points_color):
    plt.scatter(points[:, 0], points[:, 1], marker="o", color=points_color,
                edgecolors="black", s=2, linewidths=0.1)
    if flag:
        ax = plt.gca()
        n = np.arange(len(points))
        for i, txt in enumerate(n):
            ax.annotate(txt, (points[i][0], points[i][1]),
                        fontsize=10, xytext=(points[i][0] + 3, points[i][1] + 0))


# 绘制出最终的mbs
def draw_mbs(ball_center, opt):
    for value in ball_center:
        plt.scatter(value[0], value[1], marker="*", color="red", # #FF0066
                    edgecolors="black", s=80, linewidths=1)
        theta = np.arange(0, 2 * np.pi, 0.00001)
        x = value[0] + opt * np.cos(theta)
        y = value[1] + opt * np.sin(theta)
        plt.plot(x, y, linestyle="-", color="red", linewidth=1) # "#008000"


