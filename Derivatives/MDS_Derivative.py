# 计算MDS的导数
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import euclidean_distances
import math


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
                        print("warning:\t有重合的点导致不可导", (a, k))
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


class MDS_Derivative:
    def __init__(self):
        self.H = None
        self.J_yx = None
        self.P = None

    def getP(self, X, Y):
        """
        计算Y关于X的导数
        :param Dx: 高维数据的距离矩阵
        :param Dy: 低维数据的距离矩阵
        :param X: 高维数据矩阵，每一行是一个数据
        :param Y: 低维数据矩阵，每一行是一个数据
        :return:
        """
        Dx = euclidean_distances(X)
        Dy = euclidean_distances(Y)
        self.H = hessian_y(Dx, Dy, Y)
        self.J_yx = derivative_X(Dx, Dy, X, Y)
        self.P = Jyx(self.H, self.J_yx)

        return self.P
