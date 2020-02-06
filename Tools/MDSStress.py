# 计算MDS降维之后，每个点的Stress值
import numpy as np
from sklearn.metrics import euclidean_distances


def stress_value(X, Y):
    """
    计算MDS降维结束后，每个点贡献的stress值
    :param X: 高维数据矩阵
    :param Y: 降维结果矩阵
    :return:
    """
    (n, m) = X.shape
    Dx = euclidean_distances(X)
    Dy = euclidean_distances(Y)

    D = 0.5 * (Dx - Dy)**2
    stress = np.sum(D, axis=1)

    return stress.reshape((n, 1))







