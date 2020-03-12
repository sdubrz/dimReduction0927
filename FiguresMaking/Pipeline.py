# Pipeline的相关支持代码
import numpy as np
import matplotlib.pyplot as plt


path = "E:\\文件\\IRC\\特征向量散点图项目\\result2020\\result0119\\MDS\\Iris3\\yita(0.102003062)nbrs_k(20)method_k(90)numbers(4)_b-spline_weighted\\"


def scatter_plot():
    temp_y = np.loadtxt(path+"y.csv", dtype=np.float, delimiter=",")
    label = np.loadtxt(path+"label.csv", dtype=np.int, delimiter=",")
    Y = np.zeros(temp_y.shape)
    Y[:, 0] = -1 * temp_y[:, 1]
    Y[:, 1] = temp_y[:, 0]

    (n, m) = Y.shape
    for i in range(0, n):
        c = 'r'
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
    dY1 = np.zeros((n, 2))
    dY2 = np.zeros((n, 2))
    dY3 = np.zeros((n, 2))
    dY4 = np.zeros((n, 2))

    dY1[:, 0] = -1 * temp_y1[:, 1] - Y[:, 0]
    dY1[:, 1] = temp_y1[:, 0] - Y[:, 1]
    dY2[:, 0] = -1 * temp_y2[:, 1] - Y[:, 0]
    dY2[:, 1] = temp_y2[:, 0] - Y[:, 1]
    dY3[:, 0] = -1 * temp_y3[:, 1] - Y[:, 0]
    dY3[:, 1] = temp_y3[:, 0] - Y[:, 1]
    dY4[:, 0] = -1 * temp_y4[:, 1] - Y[:, 0]
    dY4[:, 1] = temp_y4[:, 0] - Y[:, 1]

    dy_list = [dY1, dY2, dY3, dY4]
    eta = 1.2  # 控制线的长度
    for dY in dy_list:
        y_add = Y + eta * dY
        y_sub = Y - eta * dY
        for i in range(0, n):
            c = 'r'
            if label[i] == 2:
                c = 'g'
            elif label[i] == 3:
                c = 'b'
            plt.plot([y_add[i, 0], y_sub[i, 0]], [y_add[i, 1], y_sub[i, 1]], c=c, linewidth=1.2, alpha=0.7)

    ax = plt.gca()
    ax.set_aspect(1)
    plt.show()


if __name__ == '__main__':
    scatter_plot()
