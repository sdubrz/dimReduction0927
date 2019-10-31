# 研究 local PCA 的相似性等性质
import numpy as np
from Main import Preprocess
from Main import LocalPCA

import matplotlib.pyplot as plt


def spectral_norm(A=np.array([])):
    """
    计算矩阵的谱范数的平方
    :param A:
    :return:
    """
    (n, m) = A.shape

    AA = np.dot(A.T, A)
    (l, M) = np.linalg.eig(AA)

    return max(l)


def local_pca_distance(X, k):
    """
    计算 local-PCA的距离矩阵，具体计算方式是计算每个点的局部协方差矩阵之间的谱范数
    :param X: 数据矩阵
    :param k: 所考虑的邻域大小
    :return:
    """
    (n, m) = X.shape

    knn = Preprocess.knn(X, k)
    COV = LocalPCA.local_cov(X, knn)

    D = np.zeros((n, n))
    for i in range(0, n-1):
        for j in range(i+1, n):
            d = spectral_norm(COV[i] - COV[j])
            D[i, j] = d
            D[j, i] = d

    return D


def test():
    a = np.zeros((3, 3, 3))
    print(a[1, :, :])
    print(a[1])


def run_test():
    path = "E:\\Project\\result2019\\result1026without_straighten\\datasets\\coil20obj_16_5class\\"
    X = np.loadtxt(path+"data.csv", dtype=np.float, delimiter=",")

    X = Preprocess.normalize(X, -1, 1)
    D = local_pca_distance(X, 30)

    np.savetxt(path+"D.csv", D, fmt='%f', delimiter=",")


if __name__ == '__main__':
    run_test()
