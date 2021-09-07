import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


if __name__ == '__main__':
    name = 'C:/Users/74412/Desktop/article/Article_Data/limit/limit_L.csv'
    points = pd.read_csv(name, usecols=[1, 2]).values
    line_R25_L10 = points[0:7]
    line_R25_L20 = points[7:14]
    line_R50_L10 = points[14:21]
    line_R50_L20 = points[21:28]

    plt.plot(line_R25_L10[:,0], line_R25_L10[:,1], linestyle="-.", marker=".", markersize=15,
             color="#33CC33", linewidth=1, label="r=25,L=10")

    plt.plot(line_R25_L20[:,0], line_R25_L20[:,1], linestyle="-.", marker=".", markersize=15,
             color="red", linewidth=1, label="r=25,L=20")

    plt.plot(line_R50_L10[:,0], line_R50_L10[:,1], linestyle="-.", marker=".", markersize=15,
             color="blue", linewidth=1, label="r=50,L=10")

    plt.plot(line_R50_L20[:,0], line_R50_L20[:,1], linestyle="-.", marker=".", markersize=15,
             color="magenta", linewidth=1, label="r=50,L=20")

    x = range(50, 550, 50)
    plt.xticks(x)
    plt.grid()
    plt.legend()
    # plt.xlim(0,550)
    # plt.ylim(5,55)
    plt.xlabel("the number of GTs")
    plt.ylabel("the number of required MBSs")
    plt.savefig("C:/Users/74412/Desktop/limit.png", dpi=1000, bbox_inches='tight')
    plt.show()

