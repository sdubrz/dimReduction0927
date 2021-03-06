import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import euclidean_distances
import math

from Main import Preprocess
from sklearn.manifold import MDS
from Derivatives import MDS_Derivative
from Derivatives import MDS_DerivativeSecond
import time


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
                H[i, j] = s

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


def time_test():
    """
    比较不同实现方式的速度
    :return:
    """
    path = "E:\\Project\\result2019\\DerivationTest\\MDS\\Iris3\\"
    data = np.loadtxt(path + "data.csv", dtype=np.float, delimiter=",")
    label = np.loadtxt(path + "label.csv", dtype=np.int, delimiter=",")
    X = Preprocess.normalize(data)
    (n, d) = X.shape
    print((n, d))

    mds_start = time.time()
    mds = MDS(n_components=2)
    Y = mds.fit_transform(X)
    mds_finish = time.time()
    print("MDS用时 ", mds_finish-mds_start)

    Dx = euclidean_distances(X)
    Dy = euclidean_distances(Y)

    np.savetxt(path + "X.csv", X, fmt='%f', delimiter=",")
    np.savetxt(path + "Y.csv", Y, fmt='%f', delimiter=",")

    H_number = MDS_Derivative.hessian_y(Dx, Dy, Y)
    H_matrix = MDS_Derivative.hessian_y_matrix(Dx, Dy, Y)
    dH = H_matrix - H_number
    print("max dH = ", np.max(dH))

    H2 = np.linalg.inv(H_matrix)
    H3 = np.matmul(H2, H_matrix)
    print("H*H-1 = \n", H3)

    J_number = MDS_Derivative.derivative_X(Dx, Dy, X, Y)
    J_matrix = MDS_Derivative.derivative_X_matrix(Dx, Dy, X, Y)
    dJ = J_matrix - J_number
    print("max dJ = ", np.max(dJ))

    np.savetxt(path+"H_number.csv", H_number, fmt='%f', delimiter=",")
    np.savetxt(path+"H_matrix.csv", H_matrix, fmt='%f', delimiter=",")
    np.savetxt(path+"dH.csv", dH, fmt='%f', delimiter=",")
    np.savetxt(path + "J_number.csv", J_number, fmt='%f', delimiter=",")
    np.savetxt(path + "J_matrix.csv", J_matrix, fmt='%f', delimiter=",")
    np.savetxt(path + "dJ.csv", dJ, fmt='%f', delimiter=",")
    np.savetxt(path+"Dx.csv", Dx, fmt='%f', delimiter=",")
    np.savetxt(path+"Dy.csv", Dy, fmt='%f', delimiter=",")


def number_test():
    # path = "E:\\Project\\result2019\\DerivationTest\\MDS\\Wine\\"  # 华硕
    path = "E:\\文件\\IRC\\特征向量散点图项目\\result2020\\DerivativeTest\\MDS\\Iris3\\"  # XPS
    data = np.loadtxt(path + "data.csv", dtype=np.float, delimiter=",")
    label = np.loadtxt(path + "label.csv", dtype=np.int, delimiter=",")
    X = Preprocess.normalize(data)
    (n, d) = X.shape
    print((n, d))

    mds_start = time.time()
    mds = MDS(n_components=2)
    Y = mds.fit_transform(X)
    mds_finish = time.time()
    print("MDS用时 ", mds_finish - mds_start)

    plt.scatter(Y[:, 0], Y[:, 1], c=label)
    plt.show()

    Dx = euclidean_distances(X)
    Dy = euclidean_distances(Y)

    np.savetxt(path + "X.csv", X, fmt='%f', delimiter=",")
    np.savetxt(path + "Y.csv", Y, fmt='%f', delimiter=",")

    H = MDS_Derivative.hessian_y(Dx, Dy, Y)
    H2 = np.linalg.inv(H)
    H3 = np.matmul(H, H2)
    H4 = np.matmul(H2, H)
    print(H3)
    print(H3[0, 0])
    np.savetxt(path+"HH-1.csv", H3, fmt='%f', delimiter=",")
    np.savetxt(path+"H-1H.csv", H4, fmt='%f', delimiter=",")

    J = MDS_Derivative.derivative_X_matrix(Dx, Dy, X, Y)
    P = (-1) * np.matmul(H2, J)

    print(H3[0, 0])
    print("H的行列式 = ", np.linalg.det(H))

    np.savetxt(path+"H.csv", H, fmt='%s', delimiter=",")
    np.savetxt(path + "H0.csv", H, fmt='%.18f', delimiter=",")
    np.savetxt(path+"Hinv.csv", H2, fmt='%.18f', delimiter=",")
    np.savetxt(path+"J.csv", J, fmt='%s', delimiter=",")
    np.savetxt(path+"P.csv", P, fmt='%s', delimiter=",")
    np.savetxt(path+"Pstring.csv", P, fmt='%s', delimiter=",")


def time_part_test():
    """
    分析时间都消耗到了哪里
    :return:
    """
    path = "E:\\文件\\IRC\\特征向量散点图项目\\result2020\\timeTest\\Wine\\"
    data = np.loadtxt(path + "data.csv", dtype=np.float, delimiter=",")
    X = Preprocess.normalize(data, -1, 1)
    label = np.loadtxt(path + "label.csv", dtype=np.float, delimiter=",")

    time0 = time.time()
    mds = MDS(n_components=2)
    Y = mds.fit_transform(X)
    time1 = time.time()
    print("降维耗时 ", time1-time0)
    plt.scatter(Y[:, 0], Y[:, 1], marker='o', c=label)
    plt.show()

    Dx = euclidean_distances(X)
    Dy = euclidean_distances(Y)

    # H1 = MDS_Derivative.hessian_y_matrix(Dx, Dy, Y)
    # H2 = MDS_Derivative.hessian_y_matrix_fast(Dx, Dy, Y)
    # np.savetxt(path+"MDS_H1.csv", H1, fmt='%f', delimiter=",")
    # np.savetxt(path+"MDS_H2.csv", H2, fmt='%f', delimiter=",")
    # np.savetxt(path+"MDS_dH.csv", H1-H2, fmt='%f', delimiter=",")

    J1 = MDS_Derivative.derivative_X_matrix(Dx, Dy, X, Y)
    J2 = MDS_Derivative.derivative_X_matrix_fast(Dx, Dy, X, Y)
    dJ = J1 - J2
    np.savetxt(path+"MDS_J1.csv", J1, fmt='%f', delimiter=",")
    np.savetxt(path+"MDS_J2.csv", J2, fmt='%f', delimiter=",")
    np.savetxt(path+"MDS_dJ.csv", dJ, fmt='%f', delimiter=",")
    print("sum J = ", np.sum(J2))


def second_test():
    """
    测试MDS的二阶导，
    每一个dn×dn的模块是不是对称的
    :return:
    """
    path = "E:\\文件\\IRC\\特征向量散点图项目\\DataLab\\MDSsecond\\Iris3\\"
    X = np.loadtxt(path+"data.csv", dtype=np.float, delimiter=",")
    X = Preprocess.normalize(X, -1, 1)
    (n, d) = X.shape
    label = np.loadtxt(path+"label.csv", dtype=np.int, delimiter=",")

    mds = MDS(n_components=2)
    Y = mds.fit_transform(X)

    np.savetxt(path+"Y.csv", Y, fmt="%f", delimiter=",")
    plt.scatter(Y[:, 0], Y[:, 1], c=label)
    plt.show()

    der = MDS_Derivative.MDS_Derivative()
    J = der.getP(X, Y)
    np.savetxt(path+"J.csv", J, fmt='%f', delimiter=",")
    J.tofile(path+"J.txt")

    Dx = der.Dx
    Dy = der.Dy
    H = der.H
    Hinv = np.linalg.pinv(H)

    A = MDS_DerivativeSecond.yDx2(X, Y, Dx, Dy, J, Hinv)
    A.tofile(path+"A.txt")
    path2 = path + "second\\"
    for i in range(0, 2*n):
        np.savetxt(path2+str(i)+".csv", A[i, :, :], fmt='%f', delimiter=",")
    print("存储完成")


if __name__ == '__main__':
    # run2()
    # shape_test()
    # dim_perturb_run()
    # vectors_perturb_run()
    # time_test()
    # number_test()
    # time_part_test()
    second_test()
