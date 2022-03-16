import math

import numpy as np
from scipy.spatial import distance
import miniball
import pandas as pd


def find_L_nearest(base, uncovered):
    wk = np.array([base])
    dist = distance.cdist(wk, uncovered, 'euclidean')
    # print("---------------------")
    # print(uncovered)
    # print(dist)
    arg_sort = np.argsort(dist[0], axis=0)
    # print(arg_sort)

    if L > len(uncovered):
        lens = len(uncovered)
    else:
        lens = L

    tmp_set = np.array([base])

    for i in range(0, lens):
        index_i = arg_sort[i]
        tmp_set = np.append(tmp_set, [uncovered[index_i]], axis=0)
        if len(tmp_set) == L:
            break


    final_ball = miniball.Miniball(tmp_set)
    center = final_ball.center()
    radius = math.sqrt(final_ball.squared_radius())
    ball_points = tmp_set

    # print("start----------------------")
    # print(tmp_set)
    # print(radius)

    radii = radius

    while radii >= R:

        mini_radii = float("inf")
        for i in range(len(tmp_set)-1, 0, -1):
            temp_miniball = miniball.Miniball(tmp_set)
            temp_radii = math.sqrt(temp_miniball.squared_radius())
            if mini_radii > temp_radii:
                mini_radii = temp_radii
                ball = temp_miniball

        if mini_radii >= R:
            if len(tmp_set) > 1:
                tmp_set = np.delete(tmp_set, -1, 0)
            if len(tmp_set) == 1:
                center = tmp_set[0]
                radius = R
                ball_points = tmp_set
                break
        else:
            center = ball.center()
            radius = math.sqrt(final_ball.squared_radius())
            ball_points = tmp_set
            break

    return center, ball_points


if __name__ == "__main__":
    L = 100000
    R = 800
    number = 100

    sum_MBS = 0

    for index in range(0, 1):

        ball_set = []

        # name = 'C:/Users/karel/Desktop/article/Article_Data/Uniform/' + str(number) + '_' + str(index) + '.csv'
        name = 'F:/Scientific_Literature/写论文/test_data/Uruguay - 734.csv'
        # name = 'C:/Users/karel/Desktop/article/Article_Data/Fig_4/Spiral_Uruguay.csv'
        data = pd.read_csv(name, usecols=[1, 2]).values

        points = data.copy()
        # print(data)

        while len(data) > 0:
            mean = np.mean(data, axis=0)
            # print(mean)

            dis = distance.cdist([mean], data, 'euclidean')
            # print(dis)

            arg_max = np.argmax(dis)
            max_point = data[arg_max]

            # Draw.show_scatter(data, 0, "green")
            data = np.delete(data, arg_max, 0)
            ball_center, points_set = find_L_nearest(max_point, data)
            # print(len(points_set))
            ball_set.append(ball_center)

            if len(points_set) != 1:
                for i in range(0, len(points_set)):
                    item = np.array([points_set[i]])[0]
                    element = np.where(data == points_set[i])[0]
                    index = 5000
                    # print("------------")
                    for j in element:
                        # print(element)
                        if data[j][0] == item[0] and data[j][1] == item[1]:
                            index = j
                            break

                    # print(item)
                    # print(index)
                    if index != 5000:
                        data = np.delete(data, index, 0)

            """
            mean = np.array([mean])
            Draw.show_scatter(mean, 0, "red")
            Draw.draw_mbs(ball_center)
    
            Draw.plt.axis("equal")
            Draw.plt.show()
            """

        # Draw.show_scatter(points, 0, "green")
        # Draw.draw_mbs_set(ball_set)
        # Draw.plt.axis("equal")
        # Draw.plt.show()

        # print("result--------------------")
        # print(len(ball_set))

        sum_MBS += len(ball_set)

    print(sum_MBS/1)
    print(ball_set)
    df = pd.DataFrame(ball_set)
    df.to_csv('C:/Users/karel/Desktop/EDGE_VBS.csv')
    """
    data = np.array([[0,0], [1, 1], [3, 3]])
    print(data)

    mean = np.mean(data, axis=0)
    print(mean)

    dis = distance.cdist([mean], data, 'euclidean')
    print(dis)

    arg_max = np.argmax(dis)
    print(arg_max)

    arg_sort = np.argsort(dis)[0]
    print(arg_sort)
    print(np.where(arg_sort == 1)[0])

    data = np.append(data, [[2, 2]], axis=0)
    print(data)
    """