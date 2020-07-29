# 用求导的方式计算 local PCA 在t-SNE的投影
# 节省内存的版本 2020.07.22
import numpy as np
# from sklearn.manifold import MDS
import matplotlib.pyplot as plt
from MyDR import cTSNE
from Main import Preprocess
from Main import LocalPCA
from Main import processData as pD
from Derivatives.TSNE_DerivativePlus import TSNE_DerivativePlus
from Derivatives.VectorPerturbPlus import VectorPerturbPlus
from sklearn.metrics import euclidean_distances
import time
from MyDR import PointsError
from Main import LocalLDA
from Main import LocalLPP


class TSNEPerturbPlus:
    def __init__(self, X, n_nbrs, Y0=None):
        self.X = X
        self.n_samples = X.shape[0]
        self.n_nbrs = n_nbrs
        self.Y = None
        self.y_add_list = []
        self.y_sub_list = []
        self.P = None
        self.Px0 = None  # 没有对称化的高维概率矩阵
        self.Px = None  # 对称化后的高维概率矩阵
        self.Q = None  # 低维概率矩阵
        self.beta = None  # 计算高维概率矩阵时用的方差
        self.init_y(Y0)

    def init_y(self, Y0):
        time1 = time.time()
        t_sne = cTSNE.cTSNE(n_component=2, perplexity=self.n_nbrs / 3.0)
        if Y0 is None:
            Y = t_sne.fit_transform(self.X, max_iter=30000)
        else:
            Y = t_sne.fit_transform(self.X, max_iter=600, early_exaggerate=False, y_random=Y0, follow_gradient=False)
        self.Y = Y
        self.beta = t_sne.beta
        self.Px0 = t_sne.P0
        self.Px = t_sne.P
        self.Q = t_sne.Q
        time2 = time.time()
        print("初始降维用时为, ", time2-time1)

    def first_derivative(self):
        """
        计算目标函数对Y的一阶导数
        :return:
        """
        X = self.X
        Y = self.Y
        P = self.Px
        Q = self.Q
        (n, m) = X.shape
        first_derivative = np.zeros((n, 2))

        Dy = euclidean_distances(Y)
        D = 1 / (1+Dy**2)
        PQ = P - Q

        for i in range(0, n):
            dY = np.tile(Y[i, :], (n, 1)) - Y
            w = PQ[i, :] * D[i, :]
            W = np.tile(w, (2, 1)).T
            first_derivative[i, :] = np.sum(W*dY, axis=0)

        first_derivative = first_derivative * 4
        self.gradient = first_derivative

        plt.subplot(121)
        plt.plot(first_derivative[:, 0])
        plt.title("first derivative 1")

        plt.subplot(122)
        plt.plot(first_derivative[:, 1])
        plt.title("first derivative 2")

        plt.show()

    def perturb(self, vectors_list, weights):
        """
        依次计算vectors_list中特征向量的投影
        :param vectors_list:
        :param weights:
        :return:
        """
        time1 = time.time()
        derivative = TSNE_DerivativePlus()
        self.P = derivative.getP(self.X, self.Y, self.Px, self.Q, self.Px0, self.beta)  # 较快的版本
        # self.P = derivative.getP_memory(self.X, self.Y, self.Px, self.Q, self.Px0, self.beta)  # 节省内存的版本
        time2 = time.time()
        print("导数矩阵已经计算完成，用时为 ", time2 - time1)
        vector_perturb = VectorPerturbPlus(self.Y, self.P)
        self.y_add_list, self.y_sub_list = vector_perturb.perturb_all(vectors_list, weights)
        time3 = time.time()
        print("扰动已经计算完成，用时 ", time3 - time2)

        return self.y_add_list, self.y_sub_list


def perturb_tsne_one_by_one(data, nbrs_k, y_init, method_k=30, MAX_EIGEN_COUNT=5, method_name="cTSNE",
                 yita=0.1, save_path="", weighted=True, label=None, y_precomputed=False, local_struct='pca'):
    """
        一个点一个点地添加扰动，不同的特征向量需要根据它们的特征值分配权重。该方法只适用于某些非线性降维方法。
        该方法目前只支持新的MDS方法，即 method=="cTSNE"
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
        :param local_struct:要投影的local structure
        :return:
        """
    print("cTSNE one by one")
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
        if local_struct == 'pca':
            temp_vectors, eigen_values[i, :] = LocalPCA.local_pca_dn(local_data)
        elif local_struct == 'lpp':
            temp_vectors, eigen_values[i, :] = LocalLPP.local_lpp(local_data)
        else:
            print("暂不支持该local structure的投影")
            return

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

    if not (method_name == "cTSNE" or method_name == "cTSNEPlus"):
        print("该方法只支持 cTSNE 降维方法")

    Y0 = None
    if y_precomputed:
        Y0 = np.loadtxt(save_path0+"Y.csv", dtype=np.float, delimiter=",")
    tsne_perturb = TSNEPerturbPlus(data, method_k, Y0)
    y = tsne_perturb.Y
    print("初次降维已经计算完毕")
    y_add_list, y_sub_list = tsne_perturb.perturb(eigen_vectors_list, yita*eigen_weights)

    points_error = PointsError.tsne_kl(tsne_perturb.Px, tsne_perturb.Q)

    # np.savetxt(save_path0+"gradient.csv", tsne_perturb.gradient, fmt='%.18e', delimiter=",")
    np.savetxt(save_path0+"Px.csv", tsne_perturb.Px, fmt='%.18e', delimiter=",")
    np.savetxt(save_path0+"Q.csv", tsne_perturb.Q, fmt='%.18e', delimiter=",")
    np.savetxt(save_path0+"error.csv", points_error, fmt='%.18e', delimiter=",")
    np.savetxt(save_path0+"cTSNE_Pxy.csv", tsne_perturb.P, fmt='%.18e', delimiter=",")

    return y, y_add_list, y_sub_list

