import matplotlib.pyplot as plt
import matplotlib
import numpy as np

if __name__ == '__main__':
    # 设置中文字体和负号正常显示
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False

    label_list = ['2.5', '3.5', '4.5']  # 横坐标刻度显示值
    num_list1 = [2.25, 2.31, 2.42]  # 纵坐标值1
    num_list2 = [2.17, 2.24, 2.33]  # 纵坐标值2
    num_list3 = [1.89, 1.9, 2.13]  # 纵坐标值1
    num_list4 = [1.88, 1.97, 2.03]  # 纵坐标值2
    x = [0, 1.5, 3]

    """
    绘制条形图
    left:长条形中点横坐标
    height:长条形高度
    width:长条形宽度，默认值0.8
    label:为后面设置legend准备
    """

    rects1 = plt.bar(x, height=num_list1, width=0.2, alpha=0.8,
                     color='#00CCFF', linewidth=1, label="0%再生料")
    rects2 = plt.bar([i + 0.2 for i in x], height=num_list2, width=0.2,
                     alpha=0.8,  color='#e300a6', linewidth=1, label="45%再生料")
    rects3 = plt.bar([i + 0.4 for i in x], height=num_list3, width=0.2, alpha=0.8,
                     color='#0054ff', linewidth=1, label="75%再生料")
    rects4 = plt.bar([i + 0.6 for i in x], height=num_list4, width=0.2,
                     alpha=0.8,  color='#ff003c', linewidth=1, label="100%再生料")
    plt.ylim(1.5, 3)  # y轴取值范围
    plt.ylabel("最大干密度$(g/cm^3)$")

    """
    设置x轴刻度显示值
    参数一：中点坐标
    参数二：显示值
    """

    plt.xticks([index + 0.45 for index in x], label_list)
    plt.xlabel("水泥剂量")
    # plt.title("某某公司")
    plt.legend()  # 设置题注

    # 编辑文本
    for rect in rects1:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height + 0.02, str(height), ha="center", va="bottom")
    for rect in rects2:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height + 0.02, str(height), ha="center", va="bottom")
    for rect in rects3:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height + 0.03, str(height), ha="center", va="bottom")
    for rect in rects4:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height + 0.02, str(height), ha="center", va="bottom")

    plt.grid()
    plt.savefig("C:/Users/karel/Desktop/bar.png", dpi=1000, bbox_inches='tight')
    plt.show()
