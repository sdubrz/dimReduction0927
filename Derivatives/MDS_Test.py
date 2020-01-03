import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import euclidean_distances
import math

from Main import Preprocess
from sklearn.manifold import MDS


def hessian_y(Dx, Dy, Y):
    """
    计算 MDS的目标函数对 Y的二阶导数，即 Hessian 矩阵
    假设 Y 是 n×m的矩阵，则返回的结果是 (n×m)×(n×m)的
    :param Dx: 高维数据的欧氏距离矩阵
    :param Dy: MDS降维所得的低维数据的欧氏距离矩阵
    :param Y: 低维数据矩阵
    :return:
    """
    (n, m) = Y.shape
    H = np.zeros((n*m, n*m))

    # 重新组织
    for i in range(0, n*m):
        b = i % m
        a = i // m
        for j in range(0, n*m):
            c = j // m
            d = j % m

            if a == c and b == d:
                s = 0
                for k in range(0, n):
                    if k == a:
                        continue
                    if Dy[a, k] == 0:
                        print("warning:\n有重合的点导致不可导", (a, k))
                        continue
                    s = s + 2 - 2 * Dx[a, k] / Dy[a, k] + 2 * Dx[a, k] * math.pow(Y[a, b] - Y[k, b], 2) / math.pow(
                        Dy[a, k], 3)
                H[i, j] = s
            elif a == c and b != d:
                s = 0
                for k in range(0, n):
                    if k == a:
                        continue
                    if Dy[a, k] == 0:
                        print("warning:\t有重合的点导致不可导", (a, k))
                        continue
                    s = 2*Dx[a, k] * (Y[a, b]-Y[k, b])*(Y[a, d]-Y[k, d]) / math.pow(Dy[a, k], 3)
                H[i, j] = s
            elif a != c and b == d:
                if Dy[a, c] == 0:
                    print("warning:\t有重合的点导致不可导", (a, c))
                    continue
                s = -2 + 2*Dx[a, c]/Dy[a, c] - 2*Dx[a, c]*math.pow(Y[a, b]-Y[c, b], 2)/math.pow(Dy[a, c], 3)
                H[i, j] = s
            else:  # a != c and b != d
                if Dy[a, c] == 0:
                    print("warning:\t有重合的点导致不可导", (a, c))
                    continue
                s = -2*Dx[a, c]*(Y[a, b]-Y[c, b])*(Y[a, d]-Y[c, d]) / math.pow(Dy[a, c], 3)

    return H


def derivative_X(Dx, Dy, X, Y):
    """
    计算目标函数对 Y的一阶导数对 X计算导数的结果
    :param Dx: 高维数据的距离矩阵
    :param Dy: MDS 降维结果的距离矩阵
    :param X: 高维数据矩阵
    :param Y: 降维结果的数据矩阵
    :return:
    """
    (n, d) = X.shape
    (n2, m) = Y.shape

    J = np.zeros((n*m, n*d))
    for row in range(0, n*m):
        a = row // m
        i = row % m
        for col in range(0, n*d):
            b = col // d
            j = col % d

            if a == b:
                s = 0
                for k in range(0, n):
                    if k == a:
                        continue
                    if Dy[a, k] == 0 or Dx[a, k] == 0:
                        print("warning:\t有点重合导致不可导", (a, k))
                        continue
                    s = s - 2*(Y[a, i]-Y[k, i])/Dy[a, k] * (X[a, j]-X[k, j])/Dx[a, k]
                J[row, col] = s

            else:
                if Dy[a, b] == 0 or Dx[a, b] == 0:
                    print("warning:\t有点重合导致不可导", (a, b))
                    continue
                J[row, col] = 2*(Y[a, i]-Y[b, i])/Dy[a, b] * (X[a, j]-X[b, j])/Dx[a, b]

    return J


def Jyx(H, J):
    """
    计算 Y对 X求导结果
    :param H: 目标函数对 Y的二阶导
    :param J: 目标函数对 Y 求导然后对 X求导
    :return:
    """
    H_ = np.linalg.inv(H)
    P = -1 * np.dot(H_, J)

    return P


def run1():
    path = "E:\\Project\\result2019\\DerivationTest\\Wine\\"
    data = np.loadtxt(path+"data.csv", dtype=np.float, delimiter=",")
    label = np.loadtxt(path+"label.csv", dtype=np.int, delimiter=",")
    X = Preprocess.normalize(data)
    (n, d) = X.shape

    mds = MDS(n_components=2)
    Y = mds.fit_transform(X)

    Dx = euclidean_distances(X)
    Dy = euclidean_distances(Y)

    np.savetxt(path + "X.csv", X, fmt='%f', delimiter=",")
    np.savetxt(path + "Y.csv", Y, fmt='%f', delimiter=",")

    print("计算二阶导......")
    H = hessian_y(Dx, Dy, Y)
    print("计算X的一阶导......")
    J = derivative_X(Dx, Dy, X, Y)

    np.savetxt(path+"H.csv", H, fmt='%f', delimiter=",")
    np.savetxt(path+"J.csv", J, fmt='%f', delimiter=",")

    print("计算Y对X的导数......")
    P = Jyx(H, J)
    np.savetxt(path+"P.csv", P, fmt='%f', delimiter=",")

    print("计算增量......")
    X0 = X.reshape((n*d, 1))
    dX = np.zeros((n, d))
    dX[0, 0] = 0.2
    dX_ = dX.reshape((n*d, 1))

    dY = np.matmul(P, dX_)

    np.savetxt(path + "dY.csv", dY, fmt='%f', delimiter=",")
    Y2 = Y + dY.reshape((n, 2))
    Y3 = Y - dY.reshape((n, 2))

    np.savetxt(path+"dX.csv", dX, fmt='%f', delimiter=",")
    np.savetxt(path+"dX_.csv", dX_, fmt='%f', delimiter=",")

    np.savetxt(path+"Y+.csv", Y2, fmt='%f', delimiter=",")
    np.savetxt(path+"Y-.csv", Y3, fmt='%f', delimiter=",")

    plt.scatter(Y[:, 0], Y[:, 1], c=label)
    for i in range(0, n):
        plt.plot([Y[i, 0], Y2[i, 0]], [Y[i, 1], Y2[i, 1]], c='deepskyblue', alpha=0.7, linewidth=0.8)
        plt.plot([Y[i, 0], Y3[i, 0]], [Y[i, 1], Y3[i, 1]], c='deepskyblue', alpha=0.7, linewidth=0.8)
    ax = plt.gca()
    ax.set_aspect(1)
    plt.show()


def run2():
    """
    直接读取已经计算过的 P矩阵进行实验
    :return:
    """
    path = "E:\\Project\\result2019\\DerivationTest\\Wine\\"
    data = np.loadtxt(path + "data.csv", dtype=np.float, delimiter=",")
    label = np.loadtxt(path + "label.csv", dtype=np.int, delimiter=",")
    X = Preprocess.normalize(data)
    (n, d) = X.shape
    Y = np.loadtxt(path+"Y.csv", dtype=np.float, delimiter=",")
    P = np.loadtxt(path+"P.csv", dtype=np.float, delimiter=",")

    dX = np.zeros((n, d))
    index = 170
    dX[:, 5] = 0.2
    dX_ = dX.reshape((n * d, 1))

    dY = np.matmul(P, dX_)

    np.savetxt(path + "dY.csv", dY, fmt='%f', delimiter=",")
    Y2 = Y + dY.reshape((n, 2))
    # Y3 = Y - dY.reshape((n, 2))

    np.savetxt(path + "dX.csv", dX, fmt='%f', delimiter=",")
    np.savetxt(path + "dX_.csv", dX_, fmt='%f', delimiter=",")

    np.savetxt(path + "Y+.csv", Y2, fmt='%f', delimiter=",")
    # np.savetxt(path + "Y-.csv", Y3, fmt='%f', delimiter=",")

    plt.scatter(Y[:, 0], Y[:, 1], c=label)
    # plt.scatter(Y[index, 0], Y[index, 1], marker='p', c='orange')
    for i in range(0, n):
        plt.plot([Y[i, 0], Y2[i, 0]], [Y[i, 1], Y2[i, 1]], c='deepskyblue')
        # plt.plot([Y[i, 0], Y3[i, 0]], [Y[i, 1], Y3[i, 1]], c='deepskyblue', alpha=0.7, linewidth=0.8)
    ax = plt.gca()
    ax.set_aspect(1)
    plt.show()


def dim_perturb_run():
    """
    用某一个维度进行扰动
    :return:
    """
    path = "E:\\Project\\result2019\\DerivationTest\\Iris\\"
    data = np.loadtxt(path + "data.csv", dtype=np.float, delimiter=",")
    label = np.loadtxt(path + "label.csv", dtype=np.int, delimiter=",")
    X = Preprocess.normalize(data)
    (n, d) = X.shape
    Y = np.loadtxt(path + "Y.csv", dtype=np.float, delimiter=",")
    P = np.loadtxt(path + "P.csv", dtype=np.float, delimiter=",")

    Y2 = np.zeros((n, 2))
    dim = 3  # 扰动的属性号
    for i in range(0, n):
        dX = np.zeros((n, d))
        dX[i, dim] = 0.1
        dX_ = dX.reshape((n * d, 1))
        dY = np.matmul(P, dX_)
        temp_y = Y + dY.reshape((n, 2))
        Y2[i, :] = temp_y[i, :]

    plt.scatter(Y[:, 0], Y[:, 1], c=label)
    for i in range(0, n):
        plt.plot([Y[i, 0], Y2[i, 0]], [Y[i, 1], Y2[i, 1]], c='deepskyblue')
    ax = plt.gca()
    ax.set_aspect(1)
    plt.title(str(dim))
    plt.show()


def vectors_perturb_run():
    """
    特征向量作为扰动向量
    :return:
    """
    path = "E:\\Project\\result2019\\DerivationTest\\Iris\\"
    data = np.loadtxt(path + "data.csv", dtype=np.float, delimiter=",")
    label = np.loadtxt(path + "label.csv", dtype=np.int, delimiter=",")
    X = Preprocess.normalize(data)
    (n, d) = X.shape
    Y = np.loadtxt(path + "Y.csv", dtype=np.float, delimiter=",")
    P = np.loadtxt(path + "P.csv", dtype=np.float, delimiter=",")
    vectors = np.loadtxt(path+"【weighted】eigenvectors0.csv", dtype=np.float, delimiter=",")

    Y2 = np.zeros((n, 2))
    Y3 = np.zeros((n, 2))
    for i in range(0, n):
        dX = np.zeros((n, d))
        dX[i, :] = vectors[i, :] * 0.04
        dX_ = dX.reshape((n * d, 1))
        dY = np.matmul(P, dX_)
        temp_y = Y + dY.reshape((n, 2))
        temp_y3 = Y - dY.reshape((n, 2))
        Y2[i, :] = temp_y[i, :]
        Y3[i, :] = temp_y3[i, :]

    plt.scatter(Y[:, 0], Y[:, 1], c=label)
    for i in range(0, n):
        plt.plot([Y[i, 0], Y2[i, 0]], [Y[i, 1], Y2[i, 1]], c='deepskyblue')
        plt.plot([Y[i, 0], Y3[i, 0]], [Y[i, 1], Y3[i, 1]], c='deepskyblue')
    ax = plt.gca()
    ax.set_aspect(1)
    plt.show()


if __name__ == '__main__':
    # run2()
    # shape_test()
    # dim_perturb_run()
    vectors_perturb_run()
