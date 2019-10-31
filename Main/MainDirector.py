# 在散点图上画主方向的小线段
import numpy as np
import matplotlib.pyplot as plt


def draw_main_director(path):
    """
    在散点图上画主方向的小线段
    :param path: 读取计算结果的文件目录
    :return:
    """
    Y = np.loadtxt(path+"y.csv", dtype=np.float, delimiter=',')
    Y1 = np.loadtxt(path+"y_add_1.csv", dtype=np.float, delimiter=',')
    Y2 = np.loadtxt(path + "y_sub_1.csv", dtype=np.float, delimiter=',')
    label = np.loadtxt(path+"label.csv", dtype=np.int, delimiter=',')
    (n, m) = Y.shape

    colors = ['r', 'g', 'b', 'm', 'yellow', 'k', 'c']
    for i in range(0, n):
        plt.scatter(Y[i, 0], Y[i, 1], marker='o', c=colors[label[i] % len(colors)], alpha=0.9)
        plt.plot([Y[i, 0], Y1[i, 0]], [Y[i, 1], Y1[i, 1]], linewidth=0.7, c=colors[label[i] % len(colors)], alpha=0.7)
        plt.plot([Y[i, 0], Y2[i, 0]], [Y[i, 1], Y2[i, 1]], linewidth=0.7, c=colors[label[i] % len(colors)], alpha=0.7)

    ax = plt.gca()
    ax.set_aspect(1)
    plt.show()


def test():
    path = "E:\\Project\\result2019\\result0927\\MDS\\seeds\\yita(0.3)nbrs_k(20)method_k(70)numbers(4)_b-spline_weighted\\"
    draw_main_director(path)


if __name__ == '__main__':
    test()
