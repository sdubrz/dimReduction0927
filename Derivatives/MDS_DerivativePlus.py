# 计算MDS的导数，专门针对较大数据的版本 2020.07.21
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import euclidean_distances
import math
import time


def hessian_y_matrix_fast(Dx, Dy, Y):
    """
    使用矩阵方式对 hessian_y 函数加速的版本
    目前最快的版本 2020.01.27
    :param Dx: 高维数据的欧氏距离矩阵
    :param Dy: 降维结果的欧氏距离矩阵
    :param Y: 降维结果矩阵，每一行是一个样本
    :return:
    """
    begin_time = time.time()
    (n, m) = Y.shape
    H = np.zeros((n*m, n*m))
    Dy2 = Dy.copy()
    Dy2[range(n), range(n)] = 1.0

    for a in range(0, n):  # n
        dY = np.tile(Y[a, :], (n, 1)) - Y
        W = np.tile(2 * Dx[a, :] / (Dy2[a, :] ** 3), (m, 1)).T
        for c in range(0, n):
            H_sub = np.zeros((m, m))
            if a == c:
                H_sub = np.matmul(dY.T, W * dY)
                dH = np.eye(m)*(2*n-2-2*np.sum(Dx[a, :]/Dy2[a, :]))
                H_sub = H_sub + dH
            else:
                left_sub = (-2+2*Dx[a, c]/Dy2[a, c]) * np.eye(m)
                right_sub = W[c, 0] * np.outer(dY[c, :], dY[c, :])
                H_sub = left_sub - right_sub
            H[a*m:a*m+m, c*m:c*m+m] = H_sub[:, :]
        # if a % 100 == 0:
        #     print(a)
    finish_time = time.time()
    # print("计算 Hessian耗时 ", finish_time-begin_time)
    return H


def derivative_X_matrix_fast(Dx, Dy, X, Y):
    """
    计算目标函数对 Y的一阶导数对 X计算导数的结果 用矩阵方式进行加速
    更改权重矩阵，进行加速，目前最快的实现方式 2020.01.27
    :param Dx: 高维数据的距离矩阵
    :param Dy: MDS 降维结果的距离矩阵
    :param X: 高维数据矩阵
    :param Y: 降维结果的数据矩阵
    :return:
    """
    begin_time = time.time()
    (n, d) = X.shape
    (n2, m) = Y.shape
    Dx2 = Dx.copy()
    Dy2 = Dy.copy()
    Dx2[range(n), range(n)] = 1.0
    Dy2[range(n), range(n)] = 1.0

    J = np.zeros((n * m, n * d))
    for a in range(0, n):
        Wy = np.tile(1.0 / Dy2[a, :], (m, 1)).T
        Wx = np.tile(1.0 / Dx2[a, :], (d, 1)).T
        dY = np.tile(Y[a, :], (n, 1)) - Y
        dX = np.tile(X[a, :], (n, 1)) - X
        for b in range(0, n):
            H_sub = np.zeros((m, d))
            if a == b:
                H_sub = -2 * np.matmul((Wy * dY).T, Wx * dX)
            else:
                H_sub = 2 * Wy[b, 0] * Wx[b, 0] * np.outer(dY[b, :], dX[b, :])
            J[a*m:a*m+m, b*d:b*d+d] = H_sub[:, :]
        # if a % 100 == 0:
        #     print(a)
    finish_time = time.time()
    # print("计算 J 耗时 ", finish_time-begin_time)

    return J


def Jyx_Plus_memory(Dx, Dy, X, Y, H):
    """
    计算Y对X的导数，本实现方式对空间的要求更低，有利于较大数据的计算
    理论上16GB的内存可以支持约 20000个点的数据
    :param Dx: 高维数据的欧氏距离矩阵
    :param Dy: MDS降维结果的欧氏距离矩阵
    :param X: 高维数据矩阵
    :param Y: 降维结果矩阵
    :param H: 目标函数对Y的二阶导矩阵
    :return:
    """
    begin_time = time.time()
    (n, d) = X.shape
    (n2, m) = Y.shape
    P = np.zeros((n*2, d))  # 最终的导数矩阵
    H = np.linalg.pinv(H)
    Dx2 = Dx.copy()
    Dy2 = Dy.copy()
    Dx2[range(n), range(n)] = 1.0
    Dy2[range(n), range(n)] = 1.0

    print("节省内存的版本，需要更多的时间")
    for b in range(0, n):
        Jb = np.zeros((m*n, d))
        for a in range(0, n):
            Wy = np.tile(1.0 / Dy2[a, :], (m, 1)).T
            Wx = np.tile(1.0 / Dx2[a, :], (d, 1)).T
            dY = np.tile(Y[a, :], (n, 1)) - Y
            dX = np.tile(X[a, :], (n, 1)) - X
            if a == b:
                H_sub = -2 * np.matmul((Wy * dY).T, Wx * dX)
            else:
                H_sub = 2 * Wy[b, 0] * Wx[b, 0] * np.outer(dY[b, :], dX[b, :])
            Jb[a * m:a * m + m, :] = H_sub[:, :]
        P[b*m:b*m+m, :] = np.matmul(H[b*m:b*m+m, :], Jb)
    P = -1 * P
    finish_time = time.time()
    print("求导时间", finish_time-begin_time)
    return P


def Jyx_Plus(H, J):
    H = np.linalg.pinv(H)
    (Jn, Jm) = J.shape
    n = Jn // 2
    m = Jm // n

    P = np.zeros((2*n, m))
    for i in range(0, n):
        P[i*2:i*2+2, :] = np.matmul(H[i*2:i*2+2, :], J[:, i*m:i*m+m])

    P = (-1) * P
    return P


class MDS_Derivative_Plus:
    def __init__(self):
        self.H = None
        self.J_yx = None
        self.P = None
        self.Dx = None
        self.Dy = None

    def getP(self, X, Y):
        """
        计算Y关于X的导数
        :param X: 高维数据矩阵，每一行是一个数据
        :param Y: 低维数据矩阵，每一行是一个数据
        :return:
        """
        Dx = euclidean_distances(X)
        Dy = euclidean_distances(Y)
        H = hessian_y_matrix_fast(Dx, Dy, Y)
        J_yx = derivative_X_matrix_fast(Dx, Dy, X, Y)
        self.P = Jyx_Plus(H, J_yx)

        return self.P

    def getP_memory(self, X, Y):
        """
        计算Y关于X的导数，节省内存版，需要更多的时间
        :param X:
        :param Y:
        :return:
        """
        Dx = euclidean_distances(X)
        Dy = euclidean_distances(Y)
        H = hessian_y_matrix_fast(Dx, Dy, Y)
        self.P = Jyx_Plus_memory(Dx, Dy, X, Y, H)

        return self.P