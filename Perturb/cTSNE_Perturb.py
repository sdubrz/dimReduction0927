# cTSNE 的扰动实现
import numpy as np
import matplotlib.pyplot as plt
from MyDR import cTSNE
from Main import Preprocess
from Main import LocalPCA
from Main import processData as pD
from Perturb import PointsInfluence


class cTSNEPerturb:
    def __init__(self, X, k):
        self.X = X
        self.k = k
        self.n_samples = X.shape[0]
        self.Y = None
        self.y_add_list = []
        self.y_sub_list = []
        self.influence_add = None
        self.influence_sub = None
        self.relative_influence_add = None
        self.relative_influence_sub = None
        self.n_iters_add = None
        self.n_iters_sub = None
        self.init_kl = None
        self.init_y()

    def init_y(self):
        t_sne = cTSNE.cTSNE(n_component=2, perplexity=self.k/3.0)
        self.Y = t_sne.fit_transform(self.X, max_iter=1000)
        self.init_kl = t_sne.final_kl

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
        n_iters = np.zeros((n, 1))

        for i in range(0, n):
            # print(i)
            X = self.X.copy()
            X[i, :] = X[i, :] + vectors[i, :]
            t_sne = cTSNE.cTSNE(n_component=2, perplexity=self.k/3.0)
            # temp_y = t_sne.fit_transform(X, y_random=self.Y, early_exaggerate=False, max_iter=20)
            temp_y = t_sne.fit_transform(X, y_random=self.Y, early_exaggerate=False, max_iter=50, min_kl=self.init_kl)
            Y[i, :] = temp_y[i, :]
            influence[i] = PointsInfluence.influence(self.Y, temp_y, i, np.linalg.norm(vectors[i, :]))
            relative_influence[i] = PointsInfluence.relative_influence(self.Y, temp_y, i)
            n_iters[i] = t_sne.final_iter

        return Y, influence, relative_influence, n_iters

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
        self.n_iters_add = np.zeros((n, eigen_number))
        self.n_iters_sub = np.zeros((n, eigen_number))
        y_add_list = []
        y_sub_list = []

        for loop_index in range(0, eigen_number):
            print("eigen index:", loop_index)
            vectors = vector_list[loop_index].copy()
            for i in range(0, n):
                vectors[i, :] = weights[i, loop_index] * vectors[i, :]
            y_add, influence1, relative_influence1, n_iters1 = self.perturb(vectors)
            y_sub, influence2, relative_influence2, n_iters2 = self.perturb(-1*vectors)
            y_add_list.append(y_add)
            y_sub_list.append(y_sub)
            self.influence_add[:, loop_index] = influence1[:, 0]
            self.influence_sub[:, loop_index] = influence2[:, 0]
            self.relative_influence_add[:, loop_index] = relative_influence1[:, 0]
            self.relative_influence_sub[:, loop_index] = relative_influence2[:, 0]
            self.n_iters_add[:, loop_index] = n_iters1[:, 0]
            self.n_iters_sub[:, loop_index] = n_iters2[:, 0]

        self.y_add_list = y_add_list
        self.y_sub_list = y_sub_list
        return y_add_list, y_sub_list


def perturb_tsne_one_by_one(data, nbrs_k, y_init, method_k=30, MAX_EIGEN_COUNT=5, method_name="cTSNE",
                 yita=0.1, save_path="", weighted=True, label=None, y_precomputed=False):
    """
        一个点一个点地添加扰动，不同的特征向量需要根据它们的特征值分配权重。该方法只适用于某些非线性降维方法。
        该方法目前只支持新的cTSNE方法，即 method=="cTSNE"
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
    print("t-SNE one by one")
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

    if not method_name == "cTSNE":
        print("该方法只支持 cTSNE 降维方法")

    tsne_perturb = cTSNEPerturb(data, method_k)
    y = tsne_perturb.Y
    y_add_list, y_sub_list = tsne_perturb.perturb_all(eigen_vectors_list, yita*eigen_weights)

    np.savetxt(save_path0+"influence_add.csv", tsne_perturb.influence_add, fmt='%f', delimiter=",")
    np.savetxt(save_path0+"influence_sub.csv", tsne_perturb.influence_sub, fmt='%f', delimiter=",")
    np.savetxt(save_path0+"relative_influence_add.csv", tsne_perturb.relative_influence_add, fmt='%f', delimiter=",")
    np.savetxt(save_path0+"relative_influence_sub.csv", tsne_perturb.relative_influence_sub, fmt='%f', delimiter=",")
    np.savetxt(save_path0+"n_iters_add.csv", tsne_perturb.n_iters_add, fmt='%d', delimiter=",")
    np.savetxt(save_path0+"n_iters_sub.csv", tsne_perturb.n_iters_sub, fmt='%d', delimiter=",")

    influence = tsne_perturb.influence_add[:, 0]
    plt.hist(influence)
    plt.title("influence")
    plt.savefig(save_path0+"influcen1+.png")
    plt.close()
    relative_influence = tsne_perturb.relative_influence_add[:, 0]
    plt.hist(relative_influence)
    plt.title("relative influence")
    plt.savefig(save_path0+"relative_influence1+.png")
    plt.close()
    plt.hist(tsne_perturb.n_iters_add[:, 0])
    plt.title("n_iters")
    plt.savefig(save_path0 + "n_iters1+.png")
    plt.close()

    return y, y_add_list, y_sub_list
