# 支持高维数据的PCA实现
import numpy as np
import matplotlib.pyplot as plt


class MyPCA:
    def __init__(self, n_components=2):
        """
        init function
        :param n_components: 主成分个数
        """
        self.n_components = n_components
        self.values = None
        self.components = None

    def fit(self, data):
        mean_x = np.mean(data, axis=0)
        X = data - mean_x
        C = np.matmul(np.transpose(X), X)
        values, vectors = np.linalg.eig(C)


