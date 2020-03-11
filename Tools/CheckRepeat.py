# 检查是否有重复的点
import numpy as np
from sklearn.metrics import euclidean_distances


def has_repeat(X):
    """
    检查X中是否有重复的点
    :param X: 数据矩阵
    :return:
    """
    (n, m) = X.shape
    D = euclidean_distances(X)

    for i in range(0, n):
        for j in range(0, n):
            if i == j:
                continue
            else:
                if D[i, j] == 0:
                    print("有重复的点 ", (i, j))
                    return True
    return False

