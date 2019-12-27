# LLE 降维方法添加扰动
import numpy as np
from MyDR import LLE
import matplotlib.pyplot as plt


class LLE_Perturb:

    def __init__(self, X, method_k):
        self.X = X
        self.method_k = method_k
        self.W = None
        self.Y = None
        self.n_samples = X.shape[0]
        self.y_add_list = []
        self.y_sub_list = []
        self.lle = None
        self.init_y(X, method_k)

    def init_y(self, X, method_k):
        self.lle = LLE.LocallyLinearEmbedding(n_components=2, n_neighbors=method_k)
        self.Y = self.lle.fit_transform(X)

    def perturb(self, vectors):
        """
        执行一次扰动计算
        :param vectors: 扰动向量
        :return:
        """
        n = self.n_samples
        X = self.X.copy()
        Y2 = np.zeros((n, 2))

        for i in range(0, n):
            X[i, :] = X[i, :] + vectors[i, :]
            weights, neighbours = LLE.barycenter_weights_i(X, n_neighbors=self.method_k, index=i)
            for j in range(0, self.method_k):
                Y2[i, :] = Y2[i, :] + weights[j]*self.Y[neighbours[j], :]

            X[i, :] = self.X[i, :]
        return Y2

    def perturb_all(self, vector_list, weights):
        """
        对所有的特征向量按照权重进行正向和负向的扰动
        :param vector_list: 扰动向量矩阵列表
        :param weights: 各特征值的权重
        :return:
        """
        eigen_number = len(vector_list)
        y_add_list = []
        y_sub_list = []
        n = self.n_samples

        for loop_index in range(0, eigen_number):
            vectors = vector_list[loop_index].copy()
            for i in range(0, n):
                vectors[i, :] = weights[i, loop_index] * vectors[i, :]
            y_add = self.perturb(vectors)
            y_sub = self.perturb(-1*vectors)
            y_add_list.append(y_add)
            y_sub_list.append(y_sub)

        self.y_add_list = y_add_list
        self.y_sub_list = y_sub_list
        return y_add_list, y_sub_list

    def evaluation(self):
        """
        评估降维降维的质量，主要是检查根据邻居加权重构的结果与结果之间的差距
        :return:
        """
        print("evaluation尚未实现")




