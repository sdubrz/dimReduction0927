# 计算二维数据的 local PCA
import numpy as np
import matplotlib.pyplot as plt
from Main import Preprocess
from Main import LocalPCA


def local_pca_2dim(Y, k):
    """
    计算二维数据的 local PCA
    :param Y: 数据矩阵，每一行是一个二维的点
    :param k: 邻域大小
    :return:
    """
    (n, dim) = Y.shape
    nbrs_index = Preprocess.knn(Y, k)

    local_eigenvalues = np.zeros((n, dim))
    first_eigenvectors = np.zeros((n, dim))
    second_eigenvectors = np.zeros((n, dim))

    for i in range(0, n):
        local_data = np.zeros((k, dim))
        for j in range(0, k):
            local_data[j, :] = Y[nbrs_index[i, j], :]
        temp_vectors, local_eigenvalues[i, :] = LocalPCA.local_pca_dn(local_data)
        first_eigenvectors[i, :] = temp_vectors[0, :]
        second_eigenvectors[i, :] = temp_vectors[1, :]

    return first_eigenvectors, second_eigenvectors, local_eigenvalues


def local_pca_2dim_file(path='', k=10):
    """
    从文件中读取数据，并计算二维的 local PCA
    :param path: 文件夹目录
    :param k: 邻居数
    :return:
    """
    y_reader = np.loadtxt(path+"y.csv", dtype=np.str, delimiter=',')
    Y = y_reader[:, :].astype(np.float)

    return local_pca_2dim(Y, k)

