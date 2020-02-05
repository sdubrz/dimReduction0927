# 计算MDS中Y对X的二阶导
import numpy as np
import time

from Derivatives import MDS_Derivative
from sklearn.metrics import euclidean_distances


def dY_dX2(X, Y, Dx, Dy):
    """
    目标函数对Y的偏导数再对X求二阶偏导数，结果是一个2n×dn×dn的
    :param X: 高维数据矩阵
    :param Y: 降维后的结果矩阵
    :param Dx: 高维空间的距离矩阵
    :param Dy: 低维空间的距离矩阵
    :return:
    """
    print("目标函数对Y求偏导，然后对X求二阶偏导...")
    t1 = time.time()
    (n, dim) = X.shape
    A = np.zeros((2*n, dim*n, dim*n))  # 结果矩阵

    Dx2 = Dx.copy() + np.eye(n)
    Dy2 = Dy.copy() + np.eye(n)
    Ex = 1 / Dx2
    Ey = 1 / Dy2

    for a in range(0, n):
        dY = (np.tile(Y[a, :], (n, 1)) - Y).T
        dX = np.tile(X[a, :], (n, 1)) - X
        Wa = Ex[a, :] * Ey[a, :]
        Wy1 = dY[0, :] / Wa
        Wy2 = dY[1, :] / Wa

        for b in range(0, n):
            if b == a:
                continue
            s = (np.eye(dim) - Ex[a, b]*Ex[a, b]*np.outer(dX[b, :], dX[b, :]))
            sub1 = 2 * Wy1[b] * s
            sub2 = 2 * Wy2[b] * s

            A[2*a, dim*a:dim*a+dim, b*dim:b*dim+dim] = sub1[:, :]  # yaxaxb
            A[2 * a+1, dim * a:dim * a + dim, b * dim:b * dim + dim] = sub2[:, :]  # yaxaxb
            A[2 * a, dim * a:dim * a + dim, a * dim:a * dim + dim] = A[2 * a, dim * a:dim * a + dim, a * dim:a * dim + dim] - sub1[:, :]  # 主对角线上的元素
            A[2 * a + 1, dim * a:dim * a + dim, a * dim:a * dim + dim] = A[2 * a + 1, dim * a:dim * a + dim, a * dim:a * dim + dim] - sub2[:, :]  # 主对角线上的元素

            A[2*a, dim*b:dim*b+dim, dim*a:dim*a+dim] = sub1[:, :]  # yaxbxa
            A[2*a+1, dim*b:dim*b+dim, dim*a:dim*a+dim] = sub2[:, :]  # yaxbxa

            A[2*a, dim*b:dim*b+dim, dim*b:dim*b+dim] = (-1) * sub1[:, :]  # yaxbxb
            A[2 * a+1, dim * b:dim * b + dim, dim * b:dim * b + dim] = (-1) * sub2[:, :]  # yaxbxb

    t2 = time.time()
    print("用时 ", t2 - t1)
    return A


def dY2_dX(X, Y, Dx, Dy):
    """
    计算MDS的目标函数对Y求二阶偏导之后再对X求偏导的结果
    :param X: 高维数据矩阵
    :param Y: 降维结果矩阵
    :param Dx: 高维空间中的距离矩阵
    :param Dy: 低维空间中的距离矩阵
    :return:
    """
    print("目标函数对Y求二阶偏导，然后对X求偏导...")
    t1 = time.time()
    (n, dim) = X.shape
    Dx2 = Dx.copy() + np.eye(n)
    Dy2 = Dy.copy() + np.eye(n)

    A = np.zeros((2*n, 2*n, dim*n))  # 结果矩阵
    Ex = 1 / Dx2
    Ey = 1 / Dy2

    for a in range(0, n):
        dY = np.tile(Y[a, :], (n, 1)) - Y
        dX = (np.tile(X[a, :], (n, 1)) - X)

        for c in range(0, n):
            if a == c:
                continue

            s = (Ey[a, c] * np.eye(2) - Ey[a, c]*Ey[a, c]*Ey[a, c]*np.outer(dY[c, :], dY[c, :])) * 2 * Ex[a, c]
            for i in range(0, dim):
                sub = s * dX[c, i]

                A[a*2:a*2+2, a*2:a*2+2, c*dim+i] = sub[:, :]
                A[a*2:a*2+2, c*2:c*2+2, a*dim+i] = sub[:, :]
                A[a*2:a*2+2, c*2:c*2+2, c*dim+i] = (-1) * sub[:, :]
                A[a*2:a*2+2, a*2:a*2+2, a*dim+i] = A[a*2:a*2+2, a*2:a*2+2, a*dim+i] - sub

    t2 = time.time()
    print("用时 ", t2 - t1)
    return A


def dYdYdY(Y, Dx, Dy):
    """
    计算MDS的目标函数对Y的三阶导
    暂时没有想出合理的矩阵方式，暂时用标量形式实现，尽量做到重用部分计算
    :param Y: 降维结果矩阵
    :param Dx: 高维空间中的距离关系矩阵
    :param Dy: 低维空间中的距离关系矩阵
    :return:
    """
    print("目标函数对Y求三阶偏导数...")
    t1 = time.time()
    (n, m) = Y.shape
    Dy2 = Dy.copy() + np.eye(n)

    A = np.zeros((2*n, 2*n, 2*n))  # 结果矩阵
    E = 1 / Dy2
    E3 = E**3
    E2 = E**2

    for a in range(0, n):
        dY = np.tile(Y[a, :], (n, 1)) - Y

        for c in range(0, n):
            if a == c:
                continue

            # 先试着算一下aac
            for b in range(0, 2):
                d = 1 - b
                w = 2*Dx[a, c]*E3[a, c]
                s1 = w * ((-3)*dY[c, b] + 3*(dY[c, b]**3)*E2[a, c])
                s2 = w * ((-1)*dY[c, d] + 3*(dY[c, b]**2)*dY[c, d]*E2[a, c])
                s3 = s2
                s4 = w * ((-1)*dY[c, b] + 3*(dY[c, d]**2)*dY[c, b]*E2[a, c])

                # 计算索引号
                ab = a*2+b
                ad = a*2+d
                cb = c*2+b
                cd = c*2+d

                # 给aac赋值
                A[ab, ab, cb] = s1
                A[ab, ab, cd] = s2
                A[ab, ad, cb] = s3
                A[ab, ad, cd] = s4

                # 给aca赋值
                A[ab, cb, ab] = s1
                A[ab, cb, ad] = s2
                A[ab, cd, ab] = s3
                A[ab, cd, ad] = s4

                # 给acc赋值
                A[ab, cb, cb] = (-1)*s1
                A[ab, cb, cd] = (-1)*s2
                A[ab, cd, cb] = (-1)*s3
                A[ab, cd, cd] = (-1)*s4

                # 给aaa加值
                A[ab, ab, ab] = A[ab, ab, ab] - s1
                A[ab, ab, ad] = A[ab, ab, ad] - s2
                A[ab, ad, ab] = A[ab, ad, ab] - s3
                A[ab, ad, ad] = A[ab, ad, ad] - s4

    t2 = time.time()
    print("用时 ", t2 - t1)
    return A


def yDx2(X, Y, Dx, Dy, J, Hinv):
    """
    用隐函数的性质，计算MDS中Y对X的二阶导
    :param X: 高维数据矩阵
    :param Y: 降维结果矩阵
    :param Dx: 高维空间中的距离矩阵
    :param Dy: 低维空间中的距离矩阵
    :param J: Y对X的一阶导数矩阵
    :param Hinv: MDS的目标函数对Y的二阶导的逆
    :return: 结果应该是一个
    """
    print("计算二阶导的主函数...")
    t1 = time.time()
    (n, d) = X.shape

    M1 = dY_dX2(X, Y, Dx, Dy)  # 2n×dn×dn
    S2 = dY2_dX(X, Y, Dx, Dy)  # 2n×2n×dn
    S3 = dYdYdY(Y, Dx, Dy)  # 2n×2n×2n

    print("M2...")
    M2 = np.zeros((2*n, d*n, d*n))
    for i in range(0, 2*n):
        M2[:, :] = 2 * np.matmul(S2[i, :, :].T, J)

    print("S4...")
    S4 = np.zeros((2*n, 2*n, d*n))
    for i in range(0, 2*n):
        S4[i, :, :] = np.matmul(S3[i, :, :], J)
    print("M3...")
    M3 = np.zeros((2*n, d*n, d*n))
    for i in range(0, 2*n):
        M3[i, :, :] = np.matmul(S4[i, :, :].T, J)

    right = M1 + M2 + M3

    A = np.zeros((2*n, d*n, d*n))  # 结果矩阵
    # 尚未想出这种乘法是否是正确的，这一步目前是最耗时的
    print("A...")
    for i in range(0, d*n):
        A[:, :, i] = (-1)*np.matmul(Hinv, right[:, :, i])

    t2 = time.time()
    print("用时 ", t2 - t1)
    return A


class MDS_SecondDerivative:
    """
    用于同时计算MDS中Y对X的一阶导和二阶导的类
    """
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.J = None
        self.yHx = None
        self.Jacobi1 = None
        self.Hessian1 = None

    def getH(self):
        Dx = euclidean_distances(self.X)
        Dy = euclidean_distances(self.Y)
        Hy = MDS_Derivative.hessian_y_matrix_fast(Dx, Dy, self.Y)
        J_yx = MDS_Derivative.derivative_X_matrix_fast(Dx, Dy, self.X, self.Y)
        Hinv = np.linalg.pinv(Hy)
        J = (-1) * np.matmul(Hinv, J_yx)  # Y 对X的一阶导

        yHx = yDx2(self.X, self.Y, Dx, Dy, J, Hinv)

        self.J = J
        self.yHx = yHx
        self.Hessian1 = Hy
        self.Jacobi1 = J_yx
        return yHx



