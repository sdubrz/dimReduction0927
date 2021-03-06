import numpy as np
import matplotlib.pyplot as plt

path = "E:\\ChinaGraph\\期中考核\\figures\\"


def scatter_plot2():
    # 不旋转的版本
    Y = np.loadtxt(path+"y.csv", dtype=np.float, delimiter=",")
    (n, m) = Y.shape
    # label = np.loadtxt(path+"label.csv", dtype=np.int, delimiter=",")
    label = np.ones((n, 1))

    (n, m) = Y.shape
    for i in range(0, n):
        c = 'b'
        if label[i] == 2:
            c = 'g'
        elif label[i] == 3:
            c = 'b'
        plt.scatter(Y[i, 0], Y[i, 1], c=c, alpha=0.6, marker='.')

    # ax = plt.gca()
    # ax.set_aspect(1)
    # plt.show()

    temp_y1 = np.loadtxt(path + "y_add_1.csv", dtype=np.float, delimiter=",")
    temp_y2 = np.loadtxt(path + "y_add_2.csv", dtype=np.float, delimiter=",")
    temp_y3 = np.loadtxt(path + "y_add_3.csv", dtype=np.float, delimiter=",")
    temp_y4 = np.loadtxt(path + "y_add_4.csv", dtype=np.float, delimiter=",")

    dY1 = temp_y1 - Y
    dY2 = temp_y2 - Y
    dY3 = temp_y3 - Y
    dY4 = temp_y4 - Y

    dy_list = [dY1, dY2, dY3, dY4]
    eta = 0.7  # 控制线的长度
    for dY in dy_list:
        y_add = Y + eta * dY
        y_sub = Y - eta * dY
        for i in range(0, n):
            plt.plot([y_add[i, 0], y_sub[i, 0]], [y_add[i, 1], y_sub[i, 1]], c='b', linewidth=1.2, alpha=0.7)

    ax = plt.gca()
    ax.set_aspect(1)
    plt.show()


if __name__ == '__main__':
    scatter_plot2()

