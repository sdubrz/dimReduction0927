# PCA算法的扰动实现方式
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


class PCA_Perturb:
    def __init__(self, X):
        """
        数据矩阵，假设已经经过了必要的预处理过程
        :param X:
        """
        self.X = X
        self.n_samples = X.shape[0]
        self.Y = None
        self.y_add_list = []
        self.y_sub_list = []
        self.init_y()
        self.add_influence = None
        self.sub_influence = None

    def init_y(self):
        pca = PCA(n_components=2, copy=True)
        self.Y = pca.fit_transform(self.X)

    def y_similar(self, Y):
        """
        计算 Y 与 self.Y 的相似度
        :param Y:
        :return:
        """
        dY = self.Y - Y
        n = self.n_samples
        s = 0
        for i in range(0, n):
            s = s + np.linalg.norm(dY[i, :])
        s = s / n
        return s

    def rotate(self, Y):
        """
        对 Y 进行颠倒尝试，使之与 self.Y 更加相似
        :param Y: 某次扰动之后得到的新的降维结果
        :return:
        """
        Y1 = Y.copy()  # 用于横坐标翻转，纵坐标不变
        Y2 = Y.copy()  # 用于纵坐标翻转，横坐标不变
        Y3 = Y * -1  # 用于横纵坐标同时翻转
        Y1[:, 0] = Y1[:, 0] * -1
        Y2[:, 1] = Y2[:, 1] * -1

        s0 = self.y_similar(Y)
        s1 = self.y_similar(Y1)
        s2 = self.y_similar(Y2)
        s3 = self.y_similar(Y3)
        min_s = min([s0, s1, s2, s3])
        if min_s == s0:
            return Y
        elif min_s == s1:
            return Y1
        elif min_s == s2:
            return Y2
        else:
            return Y3

    def influence(self, Y, index, eta):
        """
        计算第 index 个点的影响力
        :param Y: 对第 index 个点扰动之后的降维结果
        :param index: 当前点的索引号
        :param eta: 扰动向量的长度
        :return:
        """
        if eta == 0:
            return 0
        s = 0
        n = self.n_samples
        dY = Y - self.Y
        for i in range(0, n):
            if i == index:
                continue
            s = s + np.linalg.norm(dY[i, :])
        s = s / (n-1)
        return s/eta

    def perturb(self, vectors):
        """
        计算一波儿扰动
        :param vectors: 扰动向量矩阵，每一行是对每一个点的扰动向量
        :return:
        """
        n = self.n_samples
        Y = np.zeros((n, 2))
        influence = np.zeros((n, 1))  # 每个点的影响力

        for i in range(0, n):
            v_matrix = np.zeros(vectors.shape)
            v_matrix[i, :] = vectors[i, :]
            X = self.X + v_matrix
            pca = PCA(n_components=2, copy=True)
            temp_y = self.rotate(pca.fit_transform(X))
            influence[i] = self.influence(temp_y, i, np.linalg.norm(vectors[i, :]))
            Y[i, :] = temp_y[i, :]

        return Y, influence

    def perturb_all(self, vector_list, weights):
        """
        使用所有的特征向量作为扰动，尚未完成
        :param vector_list:
        :param weights:
        :return:
        """


if __name__ == '__main__':
    from Main import Preprocess
    path = "E:\\Project\\result2019\\result1224\\datasets\\Wine\\"
    X = np.loadtxt(path+"data.csv", dtype=np.float, delimiter=",")
    label = np.loadtxt(path+"label.csv", dtype=np.int, delimiter=",")
    X = Preprocess.normalize(X)

    pca_perturb = PCA_Perturb(X)
    Y = pca_perturb.Y
    plt.scatter(Y[:, 0], Y[:, 1], c=label)
    plt.show()
