# 在散点图上画主方向的小线段
import numpy as np
import matplotlib.pyplot as plt


def draw_main_director(path, normalize=False, line_length=0.03):
    """
    在散点图上画主方向的小线段
    :param path: 读取计算结果的文件目录
    :param normalize: 是否在图中将所有线段的长度设置为相同
    :param line_length: 统一设置得线段长度，仅在 normalize 为True的时候有效
    :return:
    """
    Y = np.loadtxt(path+"y.csv", dtype=np.float, delimiter=',')
    Y1 = np.loadtxt(path+"y_add_1.csv", dtype=np.float, delimiter=',')
    Y2 = np.loadtxt(path + "y_sub_1.csv", dtype=np.float, delimiter=',')
    label = np.loadtxt(path+"label.csv", dtype=np.int, delimiter=',')
    (n, m) = Y.shape

    if normalize:
        # 进行归一化，统一设置线段的长度
        for i in range(0, n):
            norm1 = np.linalg.norm(Y[i, :] - Y1[i, :])
            norm2 = np.linalg.norm(Y[i, :] - Y2[i, :])
            if norm1 > 0:
                d_x = (Y1[i, 0] - Y[i, 0]) * line_length / norm1
                d_y = (Y1[i, 1] - Y[i, 1]) * line_length / norm1
                Y1[i, 0] = Y[i, 0] + d_x
                Y1[i, 1] = Y[i, 1] + d_y
            if norm2 > 0:
                d_x = (Y2[i, 0] - Y[i, 0]) * line_length / norm2
                d_y = (Y2[i, 1] - Y[i, 1]) * line_length / norm2
                Y2[i, 0] = Y[i, 0] + d_x
                Y2[i, 1] = Y[i, 1] + d_y

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
