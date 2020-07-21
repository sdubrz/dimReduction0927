# 计算向量的投影，针对较大数据的版本 2020.07.21
import numpy as np


class VectorPerturbPlus:
    """
    用于计算 local PCA在某次降维结果中的投影
    """
    def __init__(self, Y, P):
        """
        初始化函数
        :param Y: 降维结果矩阵
        :param P: 根据隐函数理论计算出的Y关于X的导数，这里是压缩版的矩阵，只有正常版本的对角线部分，即 2n×m的矩阵
        """
        self.Y = Y
        self.P = P
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
            Y2[i, 0] = np.inner(self.P[i*2, :], vectors[i, :])
            Y2[i, 1] = np.inner(self.P[i*2+1, :], vectors[i, :])

        Y2 = Y2 + self.Y

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

