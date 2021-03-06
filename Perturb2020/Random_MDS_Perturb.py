# 用DimReader的random方法计算扰动
import numpy as np
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
from Main import Preprocess
from Main import LocalPCA
from Main import processData as pD
from sklearn.metrics import euclidean_distances
import time
from MyDR import PointsError
from Main import LocalLDA
from Main import LocalLPP
from Perturb2020 import RandomPerturb


class MDSPerturb:
    def __init__(self, X, y_random):
        self.X = X
        self.n_samples = X.shape[0]
        self.Y = None
        self.y_add_list = []
        self.y_sub_list = []
        self.y_random = y_random
        self.gradient = None
        self.init_y()
        self.first_derivative()

    def init_y(self):
        time1 = time.time()
        if self.y_random is None:
            mds = MDS(n_components=2, max_iter=1000, eps=-1.0)  # 这样应该可以限制死执行次数
            Y = mds.fit_transform(self.X)
        else:
            mds = MDS(n_components=2, max_iter=1000, eps=-1.0, n_init=1)
            Y = mds.fit_transform(self.X, init=self.y_random)
        self.Y = Y
        time2 = time.time()
        print("初始降维用时为, ", time2-time1)
        print("总共迭代的次数为 ", mds.n_iter_)

    def first_derivative(self):
        """
        计算所求结果处的一阶导数
        :return:
        """
        (n, m) = self.X.shape
        Dx = euclidean_distances(self.X) + np.eye(n)
        Dy = euclidean_distances(self.Y) + np.eye(n)
        np.savetxt("F:\\Dy.csv", Dy, fmt='%.18e', delimiter=",")

        # 计算每个点产生的误差
        dD = (Dx - Dy)**2
        dD = 0.5*dD
        error = np.sum(dD, axis=1)

        first = np.zeros((n, 2))  # 每个点处的一阶导
        for i in range(0, n):
            dY = np.tile(self.Y[i, :], (n, 1)) - self.Y
            w = 1 - Dx[i, :] / Dy[i, :]
            W = np.tile(w, (2, 1)).T
            first[i, :] = np.sum(W*dY, axis=0)

        self.gradient = first
        plt.subplot(131)
        plt.plot(error)
        plt.title("error of each point")

        plt.subplot(132)
        plt.plot(first[:, 0])
        plt.title("first derivative 1")

        plt.subplot(133)
        plt.plot(first[:, 1])
        plt.title("first derivative 2")

        plt.show()

    def perturb(self, vectors_list, weights):
        """
        依次计算vectors_list中特征向量的投影
        :param vectors_list:
        :param weights:
        :return:
        """
        time2 = time.time()
        self.y_add_list, self.y_sub_list = RandomPerturb.random_perturb_all(self.X, self.Y, "MDS", vectors_list,
                                                                            weights)
        time3 = time.time()
        print("扰动已经计算完成，用时 ", time3 - time2)

        return self.y_add_list, self.y_sub_list


def perturb_mds_one_by_one(data, nbrs_k, y_init=None, method_k=30, MAX_EIGEN_COUNT=5, method_name="MDS",
                 yita=0.1, save_path="", weighted=True, label=None, y_precomputed=False, local_struct="pca"):
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
        :param local_struct: 要投影的local structure
        :return:
        """
    print("MDS one by one")
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
        if local_struct == "pca":
            temp_vectors, eigen_values[i, :] = LocalPCA.local_pca_dn(local_data)
        elif local_struct == "lpp":
            temp_vectors, eigen_values[i, :] = LocalLPP.local_lpp(local_data)
        else:
            print("暂不支持该local structure")

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
    np.savetxt(save_path0 + "eigenvectors1.csv", eigen_vectors_list[0], fmt='%f', delimiter=",")

    mean_weight = np.mean(eigen_weights[:, 0])
    print("平均的扰动权重是 ", mean_weight * yita)

    if not method_name == "MDS_random":
        print("该方法只支持 MDS 降维方法")

    mds_perturb = MDSPerturb(data, y_init)
    y = mds_perturb.Y
    print("初次降维已经计算完毕")
    y_add_list, y_sub_list = mds_perturb.perturb(eigen_vectors_list, yita*eigen_weights)

    points_error = PointsError.mds_stress(data, y)

    np.savetxt(save_path0+"gradient.csv", mds_perturb.gradient, fmt='%.18e', delimiter=",")
    np.savetxt(save_path0+"error.csv", points_error, fmt='%.18e', delimiter=",")

    return y, y_add_list, y_sub_list

