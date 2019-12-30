# MDS的扰动方式实现
import numpy as np
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
from Main import Preprocess
from Main import LocalPCA
from Main import processData as pD


class MDSPerturb:
    def __init__(self, X):
        self.X = X
        self.n_samples = X.shape[0]
        self.Y = None
        self.y_add_list = []
        self.y_sub_list = []
        self.influence_add = None
        self.influence_sub = None
        self.relative_influence_add = None
        self.relative_influence_sub = None
        self.init_y()

    def init_y(self):
        n_inters = 3000
        mds = MDS(n_components=2, max_iter=n_inters)
        Y0 = mds.fit_transform(self.X)
        Y1 = mds.fit_transform(self.X, init=Y0)

        total_inters = n_inters
        while not self.convergence_screen(Y0, Y1):
            Y0 = mds.fit_transform(self.X, init=Y1)
            Y1 = mds.fit_transform(self.X, init=Y0)
            total_inters += n_inters
            if total_inters >= 10000:
                break
        if not self.convergence_screen(Y0, Y1):
            print("[MDS Perturb]:\tinit_y最终未能打到收敛精度")
        self.Y = Y0

    def convergence_screen(self, Y0, Y1, quality=1000):
        """
        判断降维结果是否达到一定的收敛精度
        :param Y0:
        :param Y1:
        :return:
        """
        (n, m) = Y0.shape
        dx = np.max(Y0[:, 0]) - np.min(Y0[:, 0])
        dy = np.max(Y0[:, 1]) - np.min(Y0[:, 1])
        d_screen = max(dx, dy)

        d_norm = np.zeros((n, 1))
        for i in range(0, n):
            d_norm[i] = np.linalg.norm(Y0[i, :] - Y1[i, :])
        d_mean = np.mean(d_norm)
        if dx * dy != 0:
            print("\t", d_mean / dx, d_mean / dy)
        if dx >= d_mean * 1000 or dy >= d_mean * 1000:
            return True
        else:
            return False

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

    def relative_influence(self, Y, index):
        """
        计算某个点的相对影响力，计算方式为 其余的点的平均降维改变量除以当前点的降维改变量
        :param Y: 某一次扰动后的降维结果
        :param index: 被扰动的点的索引号
        :return:
        """
        n = self.n_samples
        dY = Y - self.Y

        d_i = np.linalg.norm(dY[index, :])
        if d_i == 0:
            return 0

        s = 0
        for i in range(0, n):
            if i == index:
                continue
            s = s + np.linalg.norm(dY[i, :])
        s = s / (n-1)
        s = s / d_i
        return s

    def perturb(self, vectors):
        """
        进行一波儿扰动计算
        :param vectors: 扰动向量矩阵，每一行是对每一个点的扰动向量
        :return:
        """
        n = self.n_samples
        Y = np.zeros((n, 2))
        influence = np.zeros((n, 1))  # 每个点的影响力
        relative_influence = np.zeros((n, 1))  # 每个点的相对影响力

        for i in range(0, n):
            X = self.X.copy()
            X[i, :] = X[i, :] + vectors[i, :]
            mds = MDS(n_components=2, n_init=1)
            temp_y = mds.fit_transform(X, init=self.Y)
            Y[i, :] = temp_y[i, :]
            influence[i] = self.influence(temp_y, i, np.linalg.norm(vectors[i, :]))
            relative_influence[i] = self.relative_influence(temp_y, i)

        return Y, influence, relative_influence

    def perturb_all(self, vector_list, weights):
        """
        使用所有的特征向量作为扰动
        :param vector_list:
        :param weights:
        :return:
        """
        eigen_number = len(vector_list)
        n = self.n_samples
        self.influence_add = np.zeros((n, eigen_number))
        self.influence_sub = np.zeros((n, eigen_number))
        self.relative_influence_add = np.zeros((n, eigen_number))
        self.relative_influence_sub = np.zeros((n, eigen_number))
        y_add_list = []
        y_sub_list = []

        for loop_index in range(0, eigen_number):
            vectors = vector_list[loop_index].copy()
            for i in range(0, n):
                vectors[i, :] = weights[i, loop_index] * vectors[i, :]
            y_add, influence1, relative_influence1 = self.perturb(vectors)
            y_sub, influence2, relative_influence2 = self.perturb(-1*vectors)
            y_add_list.append(y_add)
            y_sub_list.append(y_sub)
            self.influence_add[:, loop_index] = influence1[:, 0]
            self.influence_sub[:, loop_index] = influence2[:, 0]
            self.relative_influence_add[:, loop_index] = relative_influence1[:, 0]
            self.relative_influence_sub[:, loop_index] = relative_influence2[:, 0]

        self.y_add_list = y_add_list
        self.y_sub_list = y_sub_list
        return y_add_list, y_sub_list


def perturb_mds_one_by_one(data, nbrs_k, y_init, method_k=30, MAX_EIGEN_COUNT=5, method_name="MDS",
                 yita=0.1, save_path="", weighted=True, label=None, y_precomputed=False):
    """
        一个点一个点地添加扰动，不同的特征向量需要根据它们的特征值分配权重。该方法只适用于某些非线性降维方法。
        该方法目前只支持新的MDS方法，即 method=="MDS"
        :param data:经过normalize之后的原始数据矩阵，每一行是一个样本
        :param nbrs_k:计算 local PCA的 k 值
        :param y_init:某些降维方法所需的初始随机矩阵
        :param method_k:某些降维方法所需要使用的k值
        :param MAX_EIGEN_COUNT:最多使用的特征值数目
        :param method_name:所使用的降维方法
        :param yita:扰动所乘的系数
        :param save_path:存储中间结果的路径
        :param weighted:特征向量作为扰动时是否按照其所对应的特征值分配权重
        :param label:数据的分类标签
        :param y_precomputed: y是否已经提前计算好，如果是，则直接从文件中读取
        :return:
        """
    print("one by one")
    data_shape = data.shape
    n = data_shape[0]
    dim = data_shape[1]

    # 检查method_k的值是否合理
    if method_k <= dim:
        method_k = dim + 1
    elif method_k > n:
        method_k = n

    save_path0 = save_path  # 原来本版的save_path
    if weighted:
        save_path = save_path + "【weighted】"

    # 计算每个点的邻域
    knn = Preprocess.knn(data, nbrs_k)
    np.savetxt(save_path + "knn.csv", knn, fmt="%d", delimiter=",")
    Preprocess.knn_radius(data, knn, save_path=save_path)

    eigen_vectors_list = []  # 存储的元素是特征向量矩阵，第i个元素里面存放的是每个点的第i个特征向量
    eigen_values = np.zeros((n, dim))  # 存储对每个点的localPCA所得的特征值
    eigen_weights = np.ones((n, dim))  # 计算每个特征值占所有特征值和的比重

    for i in range(0, MAX_EIGEN_COUNT):
        eigen_vectors_list.append(np.zeros((n, dim)))

    for i in range(0, n):
        local_data = np.zeros((nbrs_k, dim))
        for j in range(0, nbrs_k):
            local_data[j, :] = data[knn[i, j], :]
        temp_vectors, eigen_values[i, :] = LocalPCA.local_pca_dn(local_data)

        for j in range(0, MAX_EIGEN_COUNT):
            eigenvectors = eigen_vectors_list[j]
            eigenvectors[i, :] = temp_vectors[j, :]

        if weighted:  # 判断是否需要分配权重
            temp_eigen_sum = sum(eigen_values[i, :])
            for j in range(0, dim):
                eigen_weights[i, j] = eigen_values[i, j] / temp_eigen_sum

    eigen1_div_2 = pD.eigen1_divide_eigen2(eigen_values)
    np.savetxt(save_path + "eigen1_div_eigen2_original.csv", eigen1_div_2, fmt="%f", delimiter=",")

    np.savetxt(save_path + "eigenvalues.csv", eigen_values, fmt="%f", delimiter=",")
    np.savetxt(save_path + "eigenweights.csv", eigen_weights, fmt="%f", delimiter=",")

    mean_weight = np.mean(eigen_weights[:, 0])
    print("平均的扰动权重是 ", mean_weight * yita)

    if not method_name == "MDS":
        print("该方法只支持 MDS 降维方法")

    mds_perturb = MDSPerturb(data)
    y = mds_perturb.Y
    y_add_list, y_sub_list = mds_perturb.perturb_all(eigen_vectors_list, yita*eigen_weights)

    np.savetxt(save_path0+"influence_add.csv", mds_perturb.influence_add, fmt='%f', delimiter=",")
    np.savetxt(save_path0+"influence_sub.csv", mds_perturb.influence_sub, fmt='%f', delimiter=",")

    influence = mds_perturb.influence_add[:, 0]
    plt.hist(influence)
    plt.title("influence")
    plt.savefig(save_path0+"influcen1+.png")
    plt.close()

    return y, y_add_list, y_sub_list



