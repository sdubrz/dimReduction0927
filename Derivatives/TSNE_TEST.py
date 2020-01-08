"""
t-sne的测试代码
"""
import numpy as np
from sklearn.metrics import euclidean_distances
import math

from Derivatives import TSNE_Derivative
from MyDR import cTSNE
import matplotlib.pyplot as plt


def run1():
    path = "E:\\Project\\result2019\\DerivationTest\\tsne\\Iris\\"
    X = np.loadtxt(path + "x.csv", dtype=np.float, delimiter=",")
    label = np.loadtxt(path + "label.csv", dtype=np.int, delimiter=",")
    (n, d) = X.shape

    print("t-SNE...")
    t_sne = cTSNE.cTSNE(n_component=2, perplexity=20.0)
    Y = t_sne.fit_transform(X)

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


if __name__ == '__main__':
    # run1()
    run2()
