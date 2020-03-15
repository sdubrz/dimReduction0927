# 计算MDS的导数
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import euclidean_distances
import math
import time

test_path = "E:\\Project\\result2019\\DerivationTest\\MDS\\Wine\\"


def hessian_y(Dx, Dy, Y):
    """
    计算 MDS的目标函数对 Y的二阶导数，即 Hessian 矩阵
    假设 Y 是 n×m的矩阵，则返回的结果是 (n×m)×(n×m)的
    :param Dx: 高维数据的欧氏距离矩阵
    :param Dy: MDS降维所得的低维数据的欧氏距离矩阵
    :param Y: 低维数据矩阵
    :return:
    """
    begin_time = time.time()
    (n, m) = Y.shape
    H = np.zeros((n*m, n*m))

    # 重新组织
    for i in range(0, n*m):  # n*m
        b = i % m
        a = i // m
        for j in range(0, n*m):
            c = j // m
            d = j % m

            if a == c and b == d:
                s = 0
                ds = 0
                ds2 = 0
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
                    s = s + 2*Dx[a, k] * (Y[a, b]-Y[k, b])*(Y[a, d]-Y[k, d]) / math.pow(Dy[a, k], 3)
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
    finish_time = time.time()
    print("完全标量版本的Hessian矩阵耗时 ", finish_time-begin_time)
    return H


def hessian_y_matrix(Dx, Dy, Y):
    """
    使用矩阵方式对 hessian_y 函数加速的版本
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
        W = np.zeros((n, n))
        W[range(n), range(n)] = 2 * Dx[a, :] / (Dy2[a, :] ** 3)
        for c in range(0, n):
            H_sub = np.zeros((m, m))
            if a == c:
                H_sub = np.matmul(dY.T, np.matmul(W, dY))
                dH = np.eye(m)*(2*n-2-2*np.sum(Dx[a, :]/Dy2[a, :]))
                H_sub = H_sub + dH
            else:
                left_sub = (-2+2*Dx[a, c]/Dy2[a, c]) * np.eye(m)
                right_sub = W[c, c] * np.outer(dY[c, :], dY[c, :])
                H_sub = left_sub - right_sub
            H[a*m:a*m+m, c*m:c*m+m] = H_sub[:, :]
        if a % 100 == 0:
            print(a)
    finish_time = time.time()
    print("计算 Hessian耗时 ", finish_time-begin_time)
    return H


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


def derivative_X(Dx, Dy, X, Y):
    """
    计算目标函数对 Y的一阶导数对 X计算导数的结果
    :param Dx: 高维数据的距离矩阵
    :param Dy: MDS 降维结果的距离矩阵
    :param X: 高维数据矩阵
    :param Y: 降维结果的数据矩阵
    :return:
    """
    begin_time = time.time()
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
    finish_time = time.time()
    print("标量形式的 J 耗时 ", finish_time-begin_time)

    return J


def derivative_X_matrix(Dx, Dy, X, Y):
    """
    计算目标函数对 Y的一阶导数对 X计算导数的结果 用矩阵方式进行加速
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
        Wy = np.zeros((n, n))
        Wx = np.zeros((n, n))
        Wy[range(n), range(n)] = 1.0 / Dy2[a, :]
        Wx[range(n), range(n)] = 1.0 / Dx2[a, :]
        dY = np.tile(Y[a, :], (n, 1)) - Y
        dX = np.tile(X[a, :], (n, 1)) - X
        for b in range(0, n):
            H_sub = np.zeros((m, d))
            if a == b:
                H_sub = -2 * np.matmul(np.matmul(Wy, dY).T, np.matmul(Wx, dX))
            else:
                H_sub = 2 * Wy[b, b] * Wx[b, b] * np.outer(dY[b, :], dX[b, :])
            J[a*m:a*m+m, b*d:b*d+d] = H_sub[:, :]
        if a % 100 == 0:
            print(a)
    finish_time = time.time()
    print("计算 J 耗时 ", finish_time-begin_time)

    return J


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


def Jyx(H, J):
    """
    计算 Y对 X求导结果
    :param H: 目标函数对 Y的二阶导
    :param J: 目标函数对 Y 求导然后对 X求导
    :return:
    """
    begin_time = time.time()
    H_ = np.linalg.pinv(H)  # inv
    P = (-1) * np.matmul(H_, J)
    end_time = time.time()
    print("计算最终导数矩阵用时 ", end_time-begin_time)

    return P


def Jyx_2(H, J):
    """
    计算 Y 对 X 的求导结果
    与 Jyx 的不同是，这里在计算H的逆的时候先进行了 Jacobi 缩放处理
    :param H:
    :param J:
    :return:
    """
    (n, m) = H.shape  # n == m should be true
    A = np.zeros((n, n))
    for i in range(0, n):
        A[i, i] = 1.0 / H[i, i]  # 这里本应该要求 H[i, i]!=0 的，但是还没有去想 H[i, i]==0 时应该怎么处理
    H2 = np.matmul(np.linalg.inv(np.matmul(A, H)), A)
    P = (-1) * np.matmul(H2, J)

    return P


class MDS_Derivative:
    def __init__(self):
        self.H = None
        self.J_yx = None
        self.P = None
        self.Dx = None
        self.Dy = None

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
        self.Dx = Dx
        self.Dy = Dy
        self.H = hessian_y_matrix_fast(Dx, Dy, Y)
        self.J_yx = derivative_X_matrix_fast(Dx, Dy, X, Y)
        self.P = Jyx(self.H, self.J_yx)

        return self.P
