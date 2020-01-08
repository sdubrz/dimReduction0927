import numpy as np
from sklearn.metrics import euclidean_distances
import math

from MyDR import cTSNE
import matplotlib.pyplot as plt


def hessian_y_sub1(Dy, P, Q, Y, a, b):
    """
    计算Hessian矩阵时出现的第一种情况，即 a==c and b==d
    :param Dy: 降维之后的距离矩阵
    :param P: 高维空间中的概率矩阵
    :param Q: 低维空间中的概率矩阵
    :param Y: 降维之后的数据坐标矩阵
    :param a:
    :param b:
    :return:
    """
    (n, m) = Y.shape

    dq_right = 0  # dq 的右端项，但是在计算过程中并没有用到 k ，所以放到了外面
    for j in range(0, n):
        if j == a:
            continue
        dq_right = dq_right + Q[a, j] * math.pow(1 + Dy[a, j] * Dy[a, j], -1) * (Y[a, b] - Y[j, b])
    dq_right = dq_right * 4

    dC = 0
    for k in range(0, n):
        if k == a:
            continue
        y_ab = Y[a, b]-Y[k, b]
        e_ak = 1 + Dy[a, k] * Dy[a, k]
        dq = Q[a, k] * (dq_right - 2 / e_ak * y_ab)

        d_phi = (-1) * dq * y_ab / e_ak
        d_phi = d_phi + (P[a, k]-Q[a, k]) * (1/e_ak - 2*y_ab*y_ab/(e_ak*e_ak))
        dC = dC + d_phi

    dC = dC * 4
    return dC


def hessian_y_sub2(Dy, P, Q, Y, a, b, d):
    """
    计算 Hessian 矩阵的第二种情况，即 a==c and b!=d
    :param Dy: 降维之后的距离矩阵
    :param P: 高维空间中的概率矩阵
    :param Q: 低维中的概率矩阵
    :param Y: 降维结果矩阵
    :param a:
    :param b:
    :param d:
    :return:
    """
    (n, m) = Y.shape

    dq_right = 0
    for j in range(0, n):
        if j == a:
            continue
        dq_right = dq_right + Q[a, j]*(Y[a, d]-Y[j, d])/(1+Dy[a, j]*Dy[a, j])

    dC = 0
    for k in range(0, n):
        if k == a:
            continue
        y_d = Y[a, d] - Y[k, d]

        e_ak = 1/(1+Dy[a, k]*Dy[a, k])
        dq = 4*Q[a, k]*dq_right - 2*Q[a, k]*e_ak*y_d

        d_phi = (-1)*e_ak*(Y[a, b]-Y[k, b]) * (dq + 2*(P[a, k]-Q[a, k])*y_d*e_ak)
        dC = dC + d_phi
    dC = dC * 4

    return dC


def hessian_y_sub3(Dy, P, Q, Y, a, b, c):
    """
    计算 Hessian 矩阵的第三种情况，即 a!=c and b==d
    :param Dy: 降维之后的距离矩阵
    :param P: 高维空间的概率矩阵
    :param Q: 低维空间的概率矩阵
    :param Y: 降维结果矩阵
    :param a:
    :param b:
    :param c:
    :return:
    """
    (n, m) = Y.shape

    dq_right = 0
    for j in range(0, n):
        if j == c:
            continue
        dq_right = dq_right + Q[c, j] * (Y[c, b]-Y[j, b]) / (1+Dy[c, j]*Dy[c, j])
    dq_right = 4 * Q[a, c] * dq_right

    y_b = Y[a, b] - Y[c, b]
    e_ac = 1 / (1+Dy[a, c]*Dy[a, c])
    dq = dq_right + 2*Q[a, c] * e_ac * y_b
    d_phi = (-1)*dq*e_ac*y_b + (P[a, c]-Q[a, c]) * (2*e_ac*e_ac*y_b*y_b-e_ac)

    dC = 4*d_phi
    return dC


def hessian_y_sub4(Dy, P, Q, Y, a, b, c, d):
    """
    计算 Hessian 矩阵的第四种情况，即 a!=c and b!=d
    :param Dy: 降维之后的距离矩阵
    :param P: 高维空间中的概率矩阵
    :param Q: 低维空间中的概率矩阵
    :param Y: 降维结果矩阵
    :param a:
    :param b:
    :param c:
    :param d:
    :return:
    """
    (n, m) = Y.shape

    dq_right = 0
    for j in range(0, n):
        if j == c:
            continue
        dq_right = dq_right + Q[c, j]*(Y[c, d]-Y[j, d])/(1+Dy[a, j]*Dy[a, j])

    y_d = Y[a, d] - Y[c, d]
    e_ac = 1 / (1+Dy[a, c]*Dy[a, c])
    y_b = Y[a, b] - Y[c, b]

    dq = 2*Q[a, c]*e_ac*y_d + 4*Q[a, c]*dq_right

    d_phi = (-1)*e_ac*y_b*dq + 2*(P[a, c]-Q[a, c])*y_b*y_d*e_ac*e_ac
    dC = 4*d_phi

    return dC


def hessian_y(Dy, P, Q, Y):
    """
    计算目标函数对Y的二阶导数
    :param Dy: 降维之后的距离矩阵
    :param P: 高维空间的概率矩阵
    :param Q: 低维空间的概率矩阵
    :param Y: 降维之后的数据坐标矩阵
    :return:
    """
    (n, m) = Y.shape
    H = np.zeros((n*m, n*m))

    for row in range(0, n*m):
        b = row % m
        a = row // m
        for column in range(0, n*m):
            d = column % m
            c = column // m

            if a == c and b == d:  # 对同一个元素求二阶导
                H[row, column] = hessian_y_sub1(Dy, P, Q, Y, a, b)
            elif a == c and b != d:
                H[row, column] = hessian_y_sub2(Dy, P, Q, Y, a, b, d)
            elif a != c and b == d:
                H[row, column] = hessian_y_sub3(Dy, P, Q, Y, a, b, c)
            else:
                H[row, column] = hessian_y_sub4(Dy, P, Q, Y, a, b, c, d)

    return H


def derivative_X(X, Y, Dy, beta, P0):
    """
    计算目标函数对Y的导数再对X求导
    :param X: 高维数据矩阵
    :param Y: 降维结果矩阵
    :param Dy: 降维结果的距离矩阵
    :param beta: t-SNE计算高维概率时所用到的方差，注意搞清楚一个2倍的关系
    :param P0: 没有进行对称化的高维概率矩阵
    :return:
    """
    (n, dim) = X.shape
    (n_, m) = Y.shape

    J = np.zeros((n*m, n*dim))
    for row in range(0, n*m):
        a = row // m
        b = row % m
        for column in range(0, n*dim):
            c = column // dim
            d = column % dim

            if a == c:  # 同一个点的情况
                dC = 0
                for k in range(0, n):
                    if k == a:
                        continue
                    dp_ak = (X[a, d]-X[k, d])*(P0[k, a]-1)*P0[k, a]*beta[k]*2  # P0[k, a]即为以k为中心时的概率
                    dp_ka_right = 0
                    for j in range(0, n):
                        if j == a:
                            continue
                        dp_ka_right = dp_ka_right + X[j, d]*P0[a, j]
                    dp_ka = P0[a, k]*(X[k, d]-dp_ka_right)*beta[a]*2
                    dp = (dp_ka + dp_ak) / (2*n)
                    d_phi = (Y[a, b]-Y[k, b])*dp/(1+Dy[a, k]*Dy[a, k])
                    dC = dC + d_phi
                J[row, column] = dC*4
            else:  # 不同点的情况
                dp_ac_right = 0
                for j in range(0, n):
                    if j == c:
                        continue
                    dp_ac_right = dp_ac_right + P0[c, j]*X[j, d]
                dp_ac = P0[c, a]*(X[a, d]-dp_ac_right)*beta[c]*2
                dp_ca = P0[a, c]*(P0[a, c]-1)*(X[c, d]-X[a, d])*beta[a]*2
                dp = (dp_ac+dp_ca)/(2*n)
                dC = (Y[a, b]-Y[c, b])*dp/(1+Dy[a, c]*Dy[a, c])
                J[row, column] = dC*4
    return J


def Jxy(H, J):
    """
    计算Y对X求导的结果
    :param H:
    :param J:
    :return:
    """
    H_ = np.linalg.inv(H)
    P = -1 * np.dot(H_, J)

    return P


class TSNE_Derivative:
    def __init__(self):
        self.H = None
        self.J = None
        self.P = None

    def getP(self, X, Y, P, Q, P0, beta):
        """
        计算 dY/dX
        :param X: 高维数据矩阵
        :param Y: 降维结果矩阵
        :param P: 对称的高维概率矩阵
        :param Q: 低维概率矩阵
        :param P0: 不对称的高维概率矩阵
        :param beta: 高维中每个点的方差
        :return:
        """
        Dy = euclidean_distances(Y)
        H = hessian_y(Dy, P, Q, Y)
        J = derivative_X(X, Y, Dy, beta, P0)
        self.H = H
        self.J = J
        Pxy = Jxy(H, J)
        self.P = Pxy

        return Pxy


def run1():
    """
    测试运行，这只是个没有用的测试函数
    :return:
    """
    path = "E:\\Project\\result2019\\DerivationTest\\tsne\\Iris\\"
    X = np.loadtxt(path+"x.csv", dtype=np.float, delimiter=",")
    label = np.loadtxt(path+"label.csv", dtype=np.int, delimiter=",")

    print("t-SNE...")
    t_sne = cTSNE.cTSNE(n_component=2, perplexity=20.0)
    Y = t_sne.fit_transform(X)

    Dy = euclidean_distances(Y)
    P = t_sne.P
    Q = t_sne.Q

    print("Hessian...")
    H = hessian_y(Dy, P, Q, Y)

    print("Jacobi...")
    P0 = t_sne.P0
    beta = t_sne.beta
    J = derivative_X(X, Y, Dy, beta, P0)

    print("Jyx...")
    J2 = Jxy(H, J)

    np.savetxt(path+"Y.csv", Y, fmt='%f', delimiter=",")
    np.savetxt(path+"P.csv", P, fmt='%f', delimiter=",")
    np.savetxt(path+"Q.csv", Q, fmt='%f', delimiter=",")
    np.savetxt(path+"H.csv", H, fmt='%f', delimiter=",")
    np.savetxt(path+"J.csv", J, fmt='%f', delimiter=",")
    np.savetxt(path+"Jxy.csv", J2, fmt='%f', delimiter=",")

    plt.scatter(Y[:, 0], Y[:, 1], c=label)
    plt.show()


if __name__ == '__main__':
    run1()



