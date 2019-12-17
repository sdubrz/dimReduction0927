# 计算流形上点与点之间的距离
# 用测地线距离来代替
import numpy as np
from sklearn.neighbors import NearestNeighbors
from Tools import Floyed
import matplotlib.pyplot as plt
path = "E:\\Project\\result2019\\result1026without_straighten\\PCA\\Wine\\yita(0.1)nbrs_k(45)method_k(20)numbers(4)_b-spline_weighted\\"


def manifold_distance(X, k=5):
    """
    计算流形上点与点之间的距离矩阵
    :param X: 数据矩阵，每一行是一个点
    :param k:
    :return:
    """
    (n, m) = X.shape

    nbr_s = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(X)
    knn_distance, knn = nbr_s.kneighbors(X)
    # np.savetxt(path+"yknnD.csv", knn_distance, fmt='%f', delimiter=",")

    D = np.ones((n, n)) * float('inf')
    for i in range(0, n):
        for j in range(0, k):
            D[i, knn[i, j]] = knn_distance[i, j]
            D[knn[i, j], i] = knn_distance[i, j]

    D = Floyed.floyed(D)
    np.savetxt(path+"ymanifold_distance.csv", D, fmt='%f', delimiter=",")
    return D


if __name__ == '__main__':
    X = np.loadtxt(path+"y.csv", dtype=np.float, delimiter=",")
    print(X.shape)
    D = manifold_distance(X)
    plt.scatter(X[:, 0], X[:, 1], c=D[18, :])
    plt.colorbar()
    ax = plt.gca()
    ax.set_aspect(1)
    plt.show()
