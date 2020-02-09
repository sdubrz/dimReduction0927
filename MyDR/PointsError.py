# 计算每个点造成的降维误差
import numpy as np
from sklearn.metrics import euclidean_distances


def pca_error(X, Y):
    """
    计算每个高维中的点到投影平面的距离
    因为PCA有中心化的处理，所以可以使用勾股定理
    :param X: 高维数据矩阵
    :param Y: 低维数据矩阵
    :return:
    """
    (n, m) = X.shape
    X2 = X - np.mean(X, axis=0)

    distance = np.zeros((n, 1))
    for i in range(0, n):
        distance[i] = np.linalg.norm(X2[i, :])**2 - np.linalg.norm(Y[i, :])**2

    return distance


def mds_stress(X, Y):
    """
    计算MDS中每个点造成的 stress
    :param X:
    :param Y:
    :return:
    """
    (n, m) = X.shape
    Dx = euclidean_distances(X)
    Dy = euclidean_distances(Y)

    D = 0.5 * (Dx - Dy) ** 2
    stress = np.sum(D, axis=1)

    return stress.reshape((n, 1))


def tsne_kl(P, Q):
    """
    计算每个点造成的 KL散度
    :param P: 高维空间中的概率矩阵
    :param Q: 低维空间中的概率矩阵
    :return:
    """
    (n, m) = P.shape
    kl = np.zeros((n, 1))

    # 因为P和Q的对角线上是0，所以需要做一下处理
    P2 = np.maximum(P, 1e-12)
    Q2 = np.maximum(Q, 1e-12)

    M = P * np.log(P2/Q2)
    for i in range(0, n):
        kl[i] = np.sum(M[i, :])

    return kl

