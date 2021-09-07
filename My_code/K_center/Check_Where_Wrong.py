import SDF_K_center
import RDF_K_center_0424

import numpy as np
import pandas as pd


# 初始化的操作
def initialize(number):
    np.random.seed()
    points = np.random.uniform(-100, 100, size=(number, 2))
    # points = pd.read_csv('C:/Users/74412/Desktop/article/k_center/k_center_test_0425.csv', usecols=[1, 2]).values

    points = np.round(points, 2)
    df = pd.DataFrame(points)
    df.to_csv('C:/Users/74412/Desktop/article/k_center/k_center_test_0425.csv')

    return points


def main():
    __k = 3
    __number = 6  # 点的数目
    __points = initialize(__number)

    sdf__optimal, sdf__opt_ball_center = SDF_K_center.main(__k, __number, __points)
    rdf__optimal, rdf__opt_ball_center = RDF_K_center_0424.main(__k, __number, __points)

    if sdf__optimal != rdf__optimal:
        print(sdf__optimal)
        print(rdf__optimal)

        global i
        points = np.round(__points, 2)
        df = pd.DataFrame(points)
        name = 'C:/Users/74412/Desktop/article/k_center/k_center_test_0425'+str(i)+'.csv'
        df.to_csv(name)


if __name__ == "__main__":
    # __points = initialize(__number)

    for i in range(100):

        main()


