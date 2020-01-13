"""
t-sne的测试代码
"""
import numpy as np
from sklearn.metrics import euclidean_distances
import math

from Derivatives import TSNE_Derivative
from MyDR import cTSNE
import matplotlib.pyplot as plt
import time


def run1():
    path = "E:\\Project\\result2019\\DerivationTest\\tsne\\Wine\\"
    X = np.loadtxt(path + "x.csv", dtype=np.float, delimiter=",")
    label = np.loadtxt(path + "label.csv", dtype=np.int, delimiter=",")
    (n, d) = X.shape

    print("t-SNE...")
    t_sne = cTSNE.cTSNE(n_component=2, perplexity=20.0)
    Y = t_sne.fit_transform(X)
    np.savetxt(path+"Y.csv", Y, fmt='%f', delimiter=",")

    # Dy = euclidean_distances(Y)
    P = t_sne.P
    Q = t_sne.Q
    P0 = t_sne.P0
    beta = t_sne.beta

    print('Pxy...')
    derivative = TSNE_Derivative.TSNE_Derivative()
    Pxy = derivative.getP(X, Y, P, Q, P0, beta)

    np.savetxt(path+"Pxy.csv", Pxy, fmt='%f', delimiter=",")
    np.savetxt(path+"P.csv", P, fmt='%f', delimiter=",")
    np.savetxt(path+"Q.csv", Q, fmt='%f', delimiter=",")
    np.savetxt(path+"P0.csv", P0, fmt='%f', delimiter=",")
    np.savetxt(path+"beta.csv", beta, fmt='%f', delimiter=",")
    np.savetxt(path+"H.csv", derivative.H, fmt='%f', delimiter=",")
    np.savetxt(path+"J.csv", derivative.J, fmt='%f', delimiter=",")

    plt.scatter(Y[:, 0], Y[:, 1], c=label)
    plt.show()


def run2():
    """
    根据现有的求导结果计算扰动
    :return:
    """
    path = "E:\\Project\\result2019\\DerivationTest\\tsne\\Iris\\"
    X = np.loadtxt(path + "x.csv", dtype=np.float, delimiter=",")
    label = np.loadtxt(path + "label.csv", dtype=np.int, delimiter=",")
    (n, d) = X.shape
    Y = np.loadtxt(path + "Y.csv", dtype=np.float, delimiter=",")
    P = np.loadtxt(path + "Pxy.csv", dtype=np.float, delimiter=",")

    dX = np.zeros((n, d))
    index = 111
    dX[:, 1] = 0.2
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


def check_P():
    """
    检查高维概率的计算过程与公式描述是否有偏差
    :return:
    """
    path = "E:\\Project\\result2019\\DerivationTest\\tsne\\Iris\\"
    X = np.loadtxt(path+"x.csv", dtype=np.float, delimiter=",")
    beta = np.loadtxt(path+"beta.csv", dtype=np.float, delimiter=",")

    (n, d) = X.shape
    P = np.zeros((n, n))
    D = euclidean_distances(X)
    D2 = D**2

    for i in range(0, n):
        for j in range(0, n):
            if i == j:
                continue
            P[i, j] = np.exp(-1*D2[i, j]*beta[i])
        s = np.sum(P[i, :])
        P[i, :] = P[i, :] / s

    np.savetxt(path+"checkP\\checkP.csv", P, fmt='%f', delimiter=",")


def just_test():
    A = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])
    A2 = A**2
    print(A2)
    A[0, :] = A[0, :] / 2
    print(A)


def vectors_perturb_run():
    """
    特征向量作为扰动向量
    :return:
    """
    path = "E:\\Project\\result2019\\DerivationTest\\tsne\\coil\\"
    X = np.loadtxt(path + "x.csv", dtype=np.float, delimiter=",")
    label = np.loadtxt(path + "label.csv", dtype=np.int, delimiter=",")
    (n, d) = X.shape
    Y = np.loadtxt(path + "Y.csv", dtype=np.float, delimiter=",")
    P = np.loadtxt(path + "Pxy.csv", dtype=np.float, delimiter=",")
    vectors = np.loadtxt(path+"【weighted】eigenvectors0.csv", dtype=np.float, delimiter=",")

    Y2 = np.zeros((n, 2))
    Y3 = np.zeros((n, 2))
    for i in range(0, n):
        dX = np.zeros((n, d))
        dX[i, :] = vectors[i, :] * 0.5
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
    比较不同实现方式的时间差异
    :return:
    """
    path = "E:\\Project\\result2019\\DerivationTest\\tsne\\Wine\\"
    X = np.loadtxt(path + "x.csv", dtype=np.float, delimiter=",")
    label = np.loadtxt(path + "label.csv", dtype=np.int, delimiter=",")
    (n, d) = X.shape

    print("t-SNE...")
    time1 = time.time()
    t_sne = cTSNE.cTSNE(n_component=2, perplexity=20.0)
    Y = t_sne.fit_transform(X)
    time2 = time.time()
    print("t-SNE降维花费的时间 ", time2-time1)
    np.savetxt(path + "Y.csv", Y, fmt='%f', delimiter=",")

    # Dy = euclidean_distances(Y)
    P = t_sne.P
    Q = t_sne.Q
    P0 = t_sne.P0
    beta = t_sne.beta

    plt.scatter(Y[:, 0], Y[:, 1], c=label)
    plt.show()

    Dy = euclidean_distances(Y)
    Dx = euclidean_distances(X)

    time3 = time.time()
    H_number = TSNE_Derivative.hessian_y(Dy, P, Q, Y)
    time4 = time.time()
    print("标量形式的Hessian耗时 ", time4-time3)
    H_matrix = TSNE_Derivative.hessian_y_matrix(Dy, P, Q, Y)
    time5 = time.time()
    print("矩阵形式的Hessian耗时 ", time5-time4)

    dH = H_matrix - H_number
    print("max dH = ", np.max(dH))

    np.savetxt(path+"H_number.csv", H_number, fmt='%f', delimiter=",")
    np.savetxt(path+"H_matrix.csv", H_matrix, fmt='%f', delimiter=",")
    np.savetxt(path+"dH.csv", dH, fmt='%f', delimiter=",")
    np.savetxt(path+"Dx.csv", Dx, fmt='%f', delimiter=",")
    np.savetxt(path+"Dy.csv", Dy, fmt='%f', delimiter=",")
    np.savetxt(path + "P.csv", P, fmt='%f', delimiter=",")
    np.savetxt(path + "Q.csv", Q, fmt='%f', delimiter=",")


if __name__ == '__main__':
    # run1()
    # run2()
    # just_test()
    # check_P()
    # vectors_perturb_run()
    time_test()
