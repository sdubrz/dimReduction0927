# 生成一个蓝噪声的平面数据，并画出它们local PCA中的主特征向量与次特征向量
# 用来阐释，为什么不能只投影一个特征向量
import numpy as np
import matplotlib.pyplot as plt
from Main import Preprocess
from Main import LocalPCA
import random


def all_far(points, point, radius=0.01):
    """
    判断新加入的点是否满足距离points中的每一个点的距离都大于 radius
    :param points: 当前的点集
    :param point: 新的点
    :param radius: 半径
    :return:
    """
    if len(points) == 0:
        return True
    for p in points:
        s = 0
        for i in range(0, len(point)):
            s = s + (p[i] - point[i]) * (p[i] - point[i])
        if s < radius * radius:
            return False

    return True


def darts():
    """
    darts
    by Cook
    :return:
    """
    path = "E:\\Project\\result2019\\samplingTest\\"
    max_fail = 3000  # 最大失败次数
    points = []

    loop_count = 0
    while loop_count < max_fail:
        temp_x = random.uniform(0, 1)
        temp_y = random.uniform(0, 1)
        p = [temp_x, temp_y]
        if all_far(points, p, radius=0.05):
            points.append(p)
            loop_count = 0
            if len(points) % 1000 == 0:
                print(len(points))
        else:
            loop_count = loop_count + 1

    print("公共的点数是： ", len(points))

    X = np.array(points)
    # np.savetxt(path + "blue_noise.csv", X, fmt="%f", delimiter=",")
    # plt.scatter(X[:, 0], X[:, 1])
    # ax = plt.gca()
    # ax.set_aspect(1)
    # plt.show()

    return X


def draw_local_pca():
    X = darts()
    (n, m) = X.shape
    k = 10
    eta = 0.06
    knn = Preprocess.knn(X, k)

    vectors1 = np.zeros((n, m))  # first eigenvector for each point
    vectors2 = np.zeros((n, m))  # second eigenvector for each point
    values = np.zeros((n, m))  # eigenvalues for each point

    for i in range(0, n):
        local_data = np.zeros((k, m))
        for j in range(0, k):
            local_data[j, :] = X[knn[i, j], :]

        current_vectors, values[i, :] = LocalPCA.local_pca_dn(local_data)
        vectors1[i, :] = current_vectors[0, :]
        vectors2[i, :] = current_vectors[1, :]

    weights1 = np.zeros((n, m))
    for i in range(0, n):
        weights1[i, :] = values[i, 0] / (values[i, 0] + values[i, 1])

    weights2 = np.ones((n, m)) - weights1

    dX1 = vectors1 * weights1 * eta
    dX2 = vectors2 * weights2 * eta

    plt.plot(weights1[:, 0])
    plt.plot(weights2[:, 1])
    plt.show()

    # 画图
    alpha = 0.8
    plt.scatter(X[:, 0], X[:, 1])
    for i in range(0, n):
        plt.plot([X[i, 0], X[i, 0] + dX1[i, 0]], [X[i, 1], X[i, 1] + dX1[i, 1]], linewidth=1.5, c='r', alpha=alpha)
        plt.plot([X[i, 0], X[i, 0] - dX1[i, 0]], [X[i, 1], X[i, 1] - dX1[i, 1]], linewidth=1.5, c='r', alpha=alpha)
        plt.plot([X[i, 0], X[i, 0] + dX2[i, 0]], [X[i, 1], X[i, 1] + dX2[i, 1]], linewidth=1.5, c='b', alpha=alpha)
        plt.plot([X[i, 0], X[i, 0] - dX2[i, 0]], [X[i, 1], X[i, 1] - dX2[i, 1]], linewidth=1.5, c='b', alpha=alpha)

    ax = plt.gca()
    ax.set_aspect(1)
    plt.show()


if __name__ == '__main__':
    draw_local_pca()
