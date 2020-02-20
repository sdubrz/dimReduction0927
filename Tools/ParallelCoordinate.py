import numpy as np
import matplotlib.pyplot as plt
from Main import Preprocess


def parallel_coordinate(data, label, linewidth=0.7):
    """
    绘制平行坐标系，假设数据已经normalize到(0, 1)
    :param data:
    :param label:
    :param linewidth:
    :return:
    """

    (n, m) = data.shape
    colors = ['r', 'g', 'b', 'orange', 'm', 'k', 'c', 'yellow']

    for i in range(0, n):
        if label[i] != 3:
            continue
        c = colors[label[i] % len(colors)]
        for j in range(0, m-1):
            plt.plot([j, j+1], [data[i, j], data[i, j+1]], c=c, alpha=0.4, linewidth=linewidth)

    for i in range(0, m):
        plt.plot([i, i], [0, 1], c='k')

    plt.show()


def run1():
    # path = "E:\\Project\\result2019\\result1026without_straighten\\datasets\\Wine\\"
    path = "E:\\文件\\IRC\\特征向量散点图项目\\result2020\\result0119\\datasets\\Wine\\"
    data = np.loadtxt(path + "data.csv", dtype=np.float, delimiter=",")
    data = Preprocess.normalize(data, 0, 1)
    label = np.loadtxt(path + "label.csv", dtype=np.int, delimiter=",")

    parallel_coordinate(data, label)


def select_data_show():
    """
    指定两部分数据，用平行坐标系进行比较
    :return:
    """
    path = "E:\\文件\\IRC\\特征向量散点图项目\\result2020\\result0119\\datasets\\Iris3\\"
    data = np.loadtxt(path + "data.csv", dtype=np.float, delimiter=",")
    data = Preprocess.normalize(data, 0, 1)
    (n, m) = data.shape

    index1 = [109, 128, 106, 124, 121, 134]
    index2 = [135, 114, 142, 141, 104, 143, 127, 145]

    n1 = len(index1)
    n2 = len(index2)

    X = np.zeros((n1+n2, m))
    label = []
    for i in range(0, n1):
        X[i, :] = data[index1[i]-1, :]
        label.append(0)
    for i in range(0, n2):
        X[i+n1, :] = data[index2[i]-1, :]
        label.append(1)

    parallel_coordinate(X, label, linewidth=1.0)


def highlight_point():
    """
    对某一些点高亮
    :return:
    """
    path = "E:\\文件\\IRC\\特征向量散点图项目\\result2020\\result0119\\datasets\\Wine\\"
    data = np.loadtxt(path + "data.csv", dtype=np.float, delimiter=",")
    data = Preprocess.normalize(data, 0, 1)
    label = np.loadtxt(path+"label.csv", dtype=np.int, delimiter=",")

    (n, m) = data.shape

    colors = ['orange', 'r', 'g', 'b', 'm', 'k', 'c', 'yellow']

    for i in range(0, n):
        if label[i] != 3:
            continue
        c = colors[label[i] % len(colors)]
        for j in range(0, m - 1):
            plt.plot([j, j + 1], [data[i, j], data[i, j + 1]], c=c, alpha=0.4, linewidth=0.7)

    # 画坐标轴
    for i in range(0, m):
        plt.plot([i, i], [0, 1], c='k')

    # 高亮部分点
    indexs = [69]  # 因为系统中是从1开始计数的，所以最终使用的时候要减一
    for index in indexs:
        i = index - 1
        c = colors[label[i] % len(colors)]
        for j in range(0, m - 1):
            plt.plot([j, j + 1], [data[i, j], data[i, j + 1]], c=c, alpha=0.9, linewidth=4.0)

    plt.show()


if __name__ == '__main__':
    # select_data_show()
    run1()
    # highlight_point()
