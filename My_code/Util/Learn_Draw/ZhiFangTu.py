import matplotlib.pyplot as plt
import matplotlib
import numpy as np

if __name__ == '__main__':
    # 设置中文字体和负号正常显示
    # matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    # matplotlib.rcParams['axes.unicode_minus'] = False

    label_list = ['20', '30', '50']  # 横坐标刻度显示值
    num_list1 = [6.73, 6.58, 6.44]  # 纵坐标值1
    num_list2 = [6.99, 6.93, 6.93]  # 纵坐标值2
    num_list3 = [10.42, 7.98, 7.27]  # 纵坐标值2
    num_list4 = [10.43, 8.07, 7.97]  # 纵坐标值2
    x = [0, 1, 2]

    """
    绘制条形图
    left:长条形中点横坐标
    height:长条形高度
    width:长条形宽度，默认值0.8
    label:为后面设置legend准备
    """

    rects1 = plt.bar(x, height=num_list1, width=0.2, alpha=0.8,
                     color='#0099FF', linewidth=1, label="100 GTs, MDP")
    rects2 = plt.bar([i + 0.2 for i in x], height=num_list2, width=0.2,
                     alpha=0.8,  color='#FF5050', linewidth=1, label="100 GTs, EDGE")
    rects3 = plt.bar([i + 0.4 for i in x], height=num_list3, width=0.2,
                     alpha=0.8, color='#006699', linewidth=1, label="200 GTs, MDP")
    rects4 = plt.bar([i + 0.6 for i in x], height=num_list4, width=0.2,
                     alpha=0.8, color='#FF3399',linewidth=1,  label="200 GTs, EDGE")
    plt.ylim(6, 11)  # y轴取值范围
    plt.ylabel("the number of MBSs")

    """
    设置x轴刻度显示值
    参数一：中点坐标
    参数二：显示值
    """

    plt.xticks([index + 0.3 for index in x], label_list)
    plt.xlabel("L, the capacities of MBSs")
    # plt.title("某某公司")
    plt.legend()  # 设置题注
    """
    # 编辑文本
    for rect in rects1:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height + 0.1, str(height), ha="center", va="bottom")
    for rect in rects2:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height + 0.1, str(height), ha="center", va="bottom")
    for rect in rects3:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height + 0.05, str(height), ha="center", va="bottom")
    for rect in rects4:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height + 0.2, str(height), ha="center", va="bottom")
    """
    plt.grid()
    plt.savefig("C:/Users/74412/Desktop/bar.png", dpi=1000, bbox_inches='tight')
    plt.show()
