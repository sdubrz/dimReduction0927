# 同时使用一阶导和二阶导的计算扰动的方法
import numpy as np


class VectorPerturbSecond:
    """
    用于计算 local PCA在某次降维结果中的投影
    """
    def __init__(self, Y, P, H):
        """
        初始化函数
        :param Y: 降维结果矩阵
        :param P: 根据隐函数理论计算出的Y关于X的导数
        """
        self.Y = Y
        self.P = P
        self.H = H
        self.y_add_list = []
        self.y_sub_list = []
        (self.n_samples, m) = Y.shape

    def perturb(self, vectors):
        """
        依次使用vectors中的向量作为扰动向量，计算扰动向量的投影
        :param vectors: 每一行是一个样本的一个local 特征向量
        :return:
        """
        (n, d) = vectors.shape
        Y2 = np.zeros((n, 2))

        for i in range(0, n):
            dX = np.zeros((n, d))
            dX[i, :] = vectors[i, :]
            dX_ = dX.reshape((n * d, 1))

            dY = np.matmul(self.P, dX_)  # 一阶导产生的增量

            temp_y = self.Y + dY.reshape((n, 2))
            Y2[i, :] = temp_y[i, :]

            H0 = self.H[i*2, :, :]
            H1 = self.H[i*2+1, :, :]
            d0 = 0.5 * np.matmul(np.matmul(dX_.T, H0), dX_)
            d1 = 0.5 * np.matmul(np.matmul(dX_.T, H1), dX_)

            Y2[i, 0] = Y2[i, 0] + d0
            Y2[i, 1] = Y2[i, 1] + d1

        return Y2

    def perturb_all(self, vector_list, weights):
        """
        对 vector_list 中的向量矩阵分别做正向和反向的投影
        :param vector_list: 存放着若干个特征向量矩阵，第 i 个特征向量矩阵存放着所有点的第 i 个特征向量
        :param weights: 每个点的 local PCA中各个特征值所占的比重 × eta
        :return:
        """
        eigen_number = len(vector_list)
        self.y_add_list = []
        self.y_sub_list = []

        for loop_index in range(0, eigen_number):
            vectors = vector_list[loop_index].copy()
            (n, d) = vectors.shape
            for i in range(0, n):
                vectors[i, :] = vectors[i, :] * weights[i, loop_index]
            y_add = self.perturb(vectors)
            y_sub = self.perturb(-1*vectors)
            self.y_add_list.append(y_add)
            self.y_sub_list.append(y_sub)

        return self.y_add_list, self.y_sub_list



