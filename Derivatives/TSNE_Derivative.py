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

    # 之前漏掉的部分 鸿武八年一月十日
    k_right = 0
    for j in range(0, n):
        if j == c:
            continue
        k_right = k_right + (Y[c, b]-Y[j, b]) / math.pow(1+Dy[c, j]*Dy[c, j], 2)
    for k in range(0, n):
        if k == a or k == c:
            continue
        d_phi_k = (-4)*Q[a, k]*Q[a, k]*(Y[a, b]-Y[k, b])*k_right
        dC = dC + 4*d_phi_k

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
        dq_right = dq_right + Q[c, j]*(Y[c, d]-Y[j, d])/(1+Dy[c, j]*Dy[c, j])

    y_d = Y[a, d] - Y[c, d]
    e_ac = 1 / (1+Dy[a, c]*Dy[a, c])
    y_b = Y[a, b] - Y[c, b]

    dq = 2*Q[a, c]*e_ac*y_d + 4*Q[a, c]*dq_right

    d_phi = (-1)*e_ac*y_b*dq + 2*(P[a, c]-Q[a, c])*y_b*y_d*e_ac*e_ac
    dC = 4*d_phi

    # 之前漏掉的部分  鸿武八年一月十日
    k_right = 0
    for j in range(0, n):
        if j == c:
            continue
        k_right = k_right + (Y[c, d]-Y[j, d]) / math.pow(1+Dy[c, j]*Dy[c, j], 2)

    for k in range(0, n):
        if k == a or k == c:
            continue
        d_phi_k = (-4)*Q[a, k]*Q[a, k]*(Y[a, b]-Y[k, b])*k_right
        dC = dC + 4*d_phi_k

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

    for row in range(0, n*m):  # n*m
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


def hessian_y_matrix(Dy, P, Q, Y):
    """
    计算目标函数对Y的二阶导数 矩阵方式实现
    :param Dy: 降维之后的距离矩阵
    :param P: 高维空间的概率矩阵
    :param Q: 低维空间的概率矩阵
    :param Y: 降维之后的数据坐标矩阵
    :return:
    """
    path = "E:\\Project\\result2019\\DerivationTest\\tsne\\Wine\\"
    (n, m) = Y.shape
    H = np.zeros((n*m, n*m))
    PQ = P - Q
    E = 1.0 / (1 + Dy**2)
    np.savetxt(path+"E.csv", E, fmt='%f', delimiter=",")

    for a in range(0, n):
        dY = np.tile(Y[a, :], (n, 1)) - Y
        Wq = np.zeros((n, n))
        Wq[range(n), range(n)] = Q[a, :]
        Wd = np.zeros((n, n))
        Wd[range(n), range(n)] = E[a, :]
        Wp = np.zeros((n, n))
        Wp[range(n), range(n)] = PQ[a, :]
        wY = np.matmul(Wd, dY)
        wqY = np.matmul(Wq, wY)

        for c in range(0, n):
            H_sub = np.zeros((m, m))
            if a == c:  # 与标量形式的实现有一个很小的差别，尚未找到原因 鸿武八年一月十四日
                H_sub1 = (-2)*np.matmul(np.matmul(Wp, wY).T, wY)
                H_sub2 = np.dot(PQ[a, :], E[a, :]) * np.eye(m)
                dY_in2 = 4 * np.matmul(Q[a, :], wY)
                dY_in = (-2) * wY + 4*dY_in2
                H_sub3 = (-1) * np.matmul(wY.T, np.matmul(Wq, dY_in))
                H_sub = (H_sub1 + H_sub2 + H_sub3)*4
            else:
                dYc = np.tile(Y[c, :], (n, 1)) - Y
                Wdc = np.zeros((n, n))
                Wdc[range(n), range(n)] = E[c, :]**2
                H_sub2 = (-4)*np.matmul(np.matmul(Wq**2, dY).T, np.matmul(Wdc, dYc))
                H_sub3 = 2 * PQ[a, c] * (E[a, c]**2) * np.outer(dY[c, :], dY[c, :]) - PQ[a, c]*E[a, c]*np.eye(m)
                # H_sub1 = (-1)*np.outer(wY[c, :], 2*wqY[c, :]+4*Q[a, c]*np.matmul(Q[c, :]*E[c, :], dYc))
                H_sub1 = (-2)*Q[a, c]*E[a, c]*E[a, c]*np.outer(dY[c, :], dY[c, :])
                H_sub = (H_sub1 + H_sub2 + H_sub3)*4

            H[a*m:a*m+m, c*m:c*m+m] = H_sub[:, :]
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
                dC = 0
                for j in range(0, n):
                    if j == c:
                        continue
                    dp_ac_right = dp_ac_right + P0[c, j]*X[j, d]
                dp_ac = P0[c, a]*(X[a, d]-dp_ac_right)*beta[c]*2
                dp_ca = P0[a, c]*(P0[a, c]-1)*(X[c, d]-X[a, d])*beta[a]*2
                dp = (dp_ac+dp_ca)/(2*n)
                dC_left = (Y[a, b]-Y[c, b])*dp/(1+Dy[a, c]*Dy[a, c]) * 4

                dC_right = 0
                for k in range(0, n):
                    if k == a or k == c:
                        continue
                    dp_right = P0[k, a]*P0[k, c]*(X[c, d]-X[k, d])*beta[k]*2 + P0[a, k]*P0[a, c]*(X[c, d]-X[a, d])*beta[a]*2
                    d_phi_k = (Y[a, b]-Y[k, b])*dp_right/(1+Dy[a, k]*Dy[a, k])
                    dC_right = dC_right + d_phi_k
                dC_right = dC_right*2/n

                J[row, column] = dC_left + dC_right
    return J


def derivative_X_matrix(X, Y, Dy, beta, P0):
    """
    计算目标函数对Y的导数再对X求导 矩阵实现方式
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
    E = 1.0 / (1+Dy**2)
    Wbeta = np.zeros((n, n))
    for i in range(0, n):
        Wbeta[i, i] = 2*beta[i]

    for a in range(0, n):
        dY = np.tile(Y[a, :], (n, 1)) - Y
        Wd = np.zeros((n, n))
        Wd[range(n), range(n)] = E[a, :]
        wY = np.matmul(Wd, dY).T

        dX = np.tile(X[a, :], (n, 1)) - X
        Wp2 = np.zeros((n, n))
        Wp2[range(n), range(n)] = P0[:, a]
        Wp1 = np.zeros((n, n))
        Wp1[range(n), range(n)] = P0[a, :]

        for c in range(0, n):

            J_sub = np.zeros((m, dim))
            if a == c:
                M1 = np.matmul(Wp2*(Wp2-1)*Wbeta, dX)
                M2 = np.matmul(Wp1/beta[a], X-np.matmul(P0[a, :], X))
                J_sub = 2/n * np.matmul(wY, M1+M2)
            else:
                dXc = np.tile(X[c, :], (n, 1)) - X
                Wp3 = np.zeros((n, n))
                Wp3[range(n), range(n)] = P0[:, c]
                M1 = E[a, c] * np.outer(dY[c, :], P0[c, a]*Wbeta[c, c]*(X[a, :]-np.matmul(P0[c, :], X))+Wbeta[a, a]*dX[c, :])
                M2 = np.matmul(wY, np.matmul(Wp3*Wp2*Wbeta, dXc))
                M3 = np.outer(np.matmul(wY, P0[a, :].T), P0[a, c]*Wbeta[a, a]*dXc[a, :])
                J_sub = 2/n * (M1 + M2 + M3)
            J[a*m:a*m+m, c*dim:c*dim+dim] = J_sub[:, :]

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
        print("Hessian...")
        H = hessian_y(Dy, P, Q, Y)
        print("J...")
        J = derivative_X(X, Y, Dy, beta, P0)
        self.H = H
        self.J = J
        print("Pxy...")
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



