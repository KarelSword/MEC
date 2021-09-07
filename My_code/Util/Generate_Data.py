import numpy as np
import pandas as pd
"""
生成实验所需要的数据
"""


# 初始化的操作
def initialize(number, index):
    np.random.seed()
    points = np.random.uniform(-100, 100, size=(number, 2))
    # points = np.random.normal(0, 50, size=(number, 2))

    points = np.round(points, 2)
    df = pd.DataFrame(points)
    name = 'C:/Users/74412/Desktop/article/Article_Data/Uniform/'+str(number)+'_'+str(index)+'.csv'
    df.to_csv(name)

    return points


if __name__ == '__main__':
    for index in range(0, 100):
        initialize(300, index)