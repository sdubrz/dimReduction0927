# 扰动的计算
# 七年九月二十七日，从run中独立出来
import numpy as np
from Main import DimReduce
from Main import LocalPCA
from Main import Preprocess
from Main import processData as pD
from Main.LDA import LDA
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from Tools import SymbolAdjust


def perturb_once_weighted(data, nbrs_k, y_init, method_k=30, MAX_EIGEN_COUNT=5, method_name="MDS",
                 yita=0.1, save_path="", weighted=True, P_matrix=None, label=None, MIN_EIGEN_NUMBER=2, min_proportion=0.9, min_good_points=0.9):
    """
    一次性对所有的点添加扰动，是之前使用过的方法
    这里各个特征向量的扰动按照特征值的比重添加权重
    :param data: 经过normalize之后的原始数据矩阵
    :param nbrs_k: 计算 local PCA的 k 值
    :param y_init: 某些降维方法所需的初始随机矩阵
    :param enough: 是否达到给定的阈值要求
    :param method_k: 某些降维方法所需要使用的k值
    :param MAX_EIGEN_COUNT: 最多使用的特征值数目，暂时不用
    :param method_name: 所使用的降维方法
    :param yita: 扰动所乘的系数
    :param save_path: 存储中间结果的路径
    :param weighted: 特征向量作为扰动时是否按照其所对应的特征值分配权重
    :param P_matrix: 一个 dim × 2 的矩阵，直接观测数据中的两个维度，可以看做是一种线性降维方法
    :param label: 数据的分类标签
    :param MIN_EIGEN_NUMBER: 最少使用的特征向量个数
    :param min_proportion: 每个点的主特征值占所有特征值的占比
    :param min_good_points: 主特征值占比达到要求的点占所有的点的比重
    :return:
    """
    data_shape = data.shape
    n = data_shape[0]
    dim = data_shape[1]

    # 检查method_k的值是否合理
    if method_k <= dim:
        method_k = dim+1
    elif method_k > n:
        method_k = n

    if weighted:
        save_path = save_path + "【weighted】"

    # 计算每个点的邻域
    knn = Preprocess.knn(data, nbrs_k)
    np.savetxt(save_path+"knn.csv", knn, fmt="%d", delimiter=",")

    y_list_add = []  # 储存的元素是矩阵，把多次降维投影的结果矩阵存储起来
    y_list_sub = []

    eigen_vectors_list = []  # 存储的元素是特征向量矩阵，第i个元素里面存放的是每个点的第i个特征向量
    eigen_values = np.zeros((n, dim))  # 存储对每个点的localPCA所得的特征值

    eigen_weights = np.ones((n, dim))  # 计算每个特征值占所有特征值和的比重

    MAX_EIGEN_COUNT = LocalPCA.eigen_number(data, knn, proportion=min_proportion, good_points=min_good_points, min_number=MIN_EIGEN_NUMBER)
    print("使用的特征向量个数为：", MAX_EIGEN_COUNT)

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
                eigen_weights[i, j] = eigen_values[i, j]/temp_eigen_sum

    eigen1_div_2 = pD.eigen1_divide_eigen2(eigen_values)
    np.savetxt(save_path + "eigen1_div_eigen2_original.csv", eigen1_div_2, fmt="%f", delimiter=",")

    np.savetxt(save_path+"eigenvalues.csv", eigen_values, fmt="%f", delimiter=",")
    np.savetxt(save_path + "eigenweights.csv", eigen_weights, fmt="%f", delimiter=",")

    # 开始进行降维
    y = np.zeros((n, 2))
    if method_name == "pca" or method_name == "PCA":
        print('当前使用PCA方法')
        pca = PCA(n_components=2, copy=True, whiten=True)
        pca.fit(data)
        P_ = pca.components_
        P = np.transpose(P_)
        y = np.matmul(data, P)
        np.savetxt(save_path+"P.csv", P, fmt="%f", delimiter=",")
    elif method_name == 'LDA' or method_name == 'lda':
        print('当前使用 LDA 方法')
        lda = LDA(n_component=2)
        y = lda.fit_transform(data, label)
        P = lda.P
        np.savetxt(save_path + "P.csv", P, fmt="%f", delimiter=",")
    elif method_name == "P_matrix" and not (P_matrix is None):
        print("当前使用普通的线性降维方法")
        y = np.matmul(data, P_matrix)
        np.savetxt(save_path + "P_matrix.csv", P_matrix, fmt="%f", delimiter=",")
    elif method_name == "tsne2" or method_name == "t-SNE2":
        print('当前使用比较稳定的t-SNE方法')
        tsne = TSNE(n_components=2, perplexity=method_k / 3, init=y_init)
        y = tsne.fit_transform(data)
    # elif method_name == "tsne" or method_name == "t-SNE":
    #     # 通过实验来看t-SNE只执行一次的话结果不是很稳定 2019.12.13 实验证明并不怎么起作用
    #     t_sne = TSNE(n_components=2, n_iter=5000, perplexity=method_k / 3, init=np.random.random((n, 2)))
    #     y0 = t_sne.fit_transform(data)
    #     t_sne2 = TSNE(n_components=2, n_iter=5000, perplexity=method_k / 3, init=y0)
    #     y = t_sne2.fit_transform(data)
    else:
        y = DimReduce.dim_reduce(data, method=method_name, method_k=method_k, n_iters=30000)  # 第一次降维不需要设置初始的随机矩阵，以保证获得更好的结果
        # y = DimReduce.dim_reduce(data, method=method_name, method_k=method_k, y_random=y_init)

    # 开始执行扰动计算
    for loop_index in range(0, MAX_EIGEN_COUNT):
        eigenvectors = eigen_vectors_list[loop_index]
        np.savetxt(save_path+"eigenvectors"+str(loop_index)+".csv", eigenvectors, fmt="%f", delimiter=",")
        x_add_v = np.zeros((n, dim))
        x_sub_v = np.zeros((n, dim))
        for i in range(0, n):
            x_add_v[i, :] = data[i, :] + yita*eigen_weights[i, loop_index]*eigenvectors[i, :]
            x_sub_v[i, :] = data[i, :] - yita*eigen_weights[i, loop_index]*eigenvectors[i, :]

        np.savetxt(save_path+"x_add_v"+str(loop_index)+".csv", x_add_v, fmt="%f", delimiter=",")
        np.savetxt(save_path + "x_sub_v" + str(loop_index) + ".csv", x_sub_v, fmt="%f", delimiter=",")

        y_add_v = np.zeros((n, 2))
        y_sub_v = np.zeros((n, 2))
        if method_name == "pca" or method_name == "PCA" or method_name == 'LDA' or method_name == 'lda':
            print('当前使用PCA方法')
            y_add_v = np.matmul(x_add_v, P)
            y_sub_v = np.matmul(x_sub_v, P)
        elif method_name == "P_matrix" and not (P_matrix is None):
            print("当前使用普通的线性降维方法")
            y_add_v = np.matmul(x_add_v, P_matrix)
            y_sub_v = np.matmul(x_sub_v, P_matrix)
        elif method_name == "tsne2" or method_name == "t-SNE2":
            print('当前使用比较稳定的t-SNE方法')
            tsne = TSNE(n_components=2, n_iter=1, perplexity=method_k / 3, init=y)
            y_add_v = tsne.fit_transform(x_add_v)
            y_sub_v = tsne.fit_transform(x_sub_v)
        else:
            y_add_v = DimReduce.dim_reduce(x_add_v, method=method_name, method_k=method_k, y_random=y, n_iters=1500, c_early_exage=False)
            # y_sub_v = 2*y-y_add_v  # 胡乱加的，要改回去
            y_sub_v = DimReduce.dim_reduce(x_sub_v, method=method_name, method_k=method_k, y_random=y, n_iters=1500, c_early_exage=False)

        y_add_v = SymbolAdjust.symbol_adjust(y, y_add_v)  # 这个是防止翻转的那种情况发生的。
        y_sub_v = SymbolAdjust.symbol_adjust(y, y_sub_v)

        y_list_add.append(y_add_v)
        y_list_sub.append(y_sub_v)

    if weighted:
        mean_first_weight = np.mean(eigen_weights[:, 0])
        print('第一个特征值平均占比为： ', mean_first_weight)

    return y, y_list_add, y_list_sub


def perturb_together(data, prefect_k, eigen_numbers, y_init, MAX_K, method_k=30, MAX_EIGEN_COUNT=4, method_name="MDS",
                 yita=0.1, save_path=""):
    """
    将原始数据与使用不同特征向量扰动的所有扰动数据放到一起，进行降维
    写程序的时候需要注意数据结构与之前的程序兼容的问题
    时七年四月二十二日
    :param data: 原始的高维数据矩阵，每一行是一个样本
    :param prefect_k: 记录了每个样本点的最佳k值
    :param eigen_numbers: 每个样本点所使用的特征向量个数
    :param y_init: 初始的y矩阵
    :param MAX_K: 所能使用的最大k值
    :param method_k: 某些降维方法所使用的k值
    :param MAX_EIGEN_COUNT: 所能使用的最多的特征向量个数
    :param method_name: 降维方法的名称
    :param yita: 扰动率
    :param save_path: 储存中间结果的路径
    :return:
    """

    """目前该方法仅支持MDS方法  时七年四月二十二日"""
    data_shape = data.shape
    n = data_shape[0]
    dim = data_shape[1]

    # 检查method_k的值是否合理
    if method_k <= dim:
        method_k = dim + 1
    elif method_k > n:
        method_k = n

    save_path = save_path + "【weighted】"
    knn = Preprocess.knn(data, MAX_K)
    np.savetxt(save_path+"knn.csv", knn, fmt="%d", delimiter=",")

    y_list_add = []  # 储存的元素是矩阵
    y_list_sub = []

    eigen_vectors_list = []  # 储存的元素是特征向量矩阵，第i个元素里面存放的是每个点的第i个特征向量
    eigen_values = np.zeros((n, dim))  # 存储每个点的localPCA所得的特征值

    eigen_weights = np.zeros((n, dim))  # 计算每个特征值占所有特征值的比重

    for i in range(0, MAX_EIGEN_COUNT):
        eigen_vectors_list.append(np.zeros((n, dim)))

    for i in range(0, n):
        local_data = np.zeros((int(prefect_k[i]), dim))
        for j in range(0, int(prefect_k[i])):
            local_data[j, :] = data[knn[i, j], :]
        temp_vectors, eigen_values[i, :] = LocalPCA.local_pca_dn(local_data)

        for j in range(0, MAX_EIGEN_COUNT):
            eigenvectors = eigen_vectors_list[j]
            eigenvectors[i, :] = temp_vectors[j, :]

        temp_eigen_sum = sum(eigen_values[i, :])
        for j in range(0, dim):
            eigen_weights[i, j] = eigen_values[i, j]/temp_eigen_sum

    eigen1_div_2 = pD.eigen1_divide_eigen2(eigen_values)
    np.savetxt(save_path + "eigen1_div_eigen2_original.csv", eigen1_div_2, fmt="%f", delimiter=",")

    np.savetxt(save_path + "eigenvalues.csv", eigen_values, fmt="%f", delimiter=",")
    np.savetxt(save_path + "eigenweights.csv", eigen_weights, fmt="%f", delimiter=",")

    # 计算高维中第一特征向量与第二特征向量的角度
    eigen_vectors1 = eigen_vectors_list[0]
    eigen_vectors2 = eigen_vectors_list[1]

    angles_v1_v2 = pD.angle_v1_v2(eigen_vectors1, eigen_vectors2)
    np.savetxt(save_path + "angles_v1_v2_original.csv", angles_v1_v2, fmt="%f", delimiter=",")

    # 准备降维数据
    x_merge = np.zeros((n*(2*MAX_EIGEN_COUNT+1), dim))  # 合并版的高维数据集，包含了所有的扰动情况
    x_merge[0:n, :] = data[0:n, :]

    for loop_index in range(0, MAX_EIGEN_COUNT):
        eigenvectors = eigen_vectors_list[loop_index]
        np.savetxt(save_path+"eigenvectors"+str(loop_index)+".csv", eigenvectors, fmt="%f", delimiter=",")
        x_add_v = np.zeros((n, dim))
        x_sub_v = np.zeros((n, dim))

        for i in range(0, n):
            if eigen_numbers[i] > loop_index:
                x_add_v[i, :] = data[i, :] + yita*eigen_weights[i, loop_index]*eigenvectors[i, :]
                x_sub_v[i, :] = data[i, :] - yita*eigen_weights[i, loop_index]*eigenvectors[i, :]
            else:
                x_add_v[i, :] = data[i, :]  # 控制使用特征向量的个数，修改一下可以改成都使用相同数目的特征向量
                x_sub_v[i, :] = data[i, :]
        np.savetxt(save_path+"x_add_v"+str(loop_index)+".csv", x_add_v, fmt="%f", delimiter=",")
        np.savetxt(save_path+"x_sub_v"+str(loop_index)+".csv", x_sub_v, fmt="%f", delimiter=",")

        x_merge[((loop_index+1)*2-1)*n:(loop_index+1)*2*n, :] = x_add_v[0:n, :]
        x_merge[(loop_index+1)*2*n:((loop_index+1)*2+1)*n, :] = x_sub_v[0:n, :]

    np.savetxt(save_path+"x_merge.csv", x_merge, fmt="%f", delimiter=",")

    # 执行降维
    if method_name == "newMDS":
        print("当前使用合并执行的MDS方法")
        y = DimReduce.dim_reduce(data, method="MDS")
        y_merge = DimReduce.dim_reduce(x_merge, method="MDS")
        np.savetxt(save_path+"y_merge.csv", y_merge, fmt="%f", delimiter=",")
        y_new, y_list_add, y_list_sub = divide_y(y_merge, MAX_EIGEN_COUNT, n)
    else:
        print("没有匹配到合适的降维方法")
        return

    return y_new, y_list_add, y_list_sub


def divide_y(y_merge, MAX_EIGEN_COUNT, n):
    """
    将合并执行所得到的降维结果拆分开来，还原为原来老版本中的数据结构
    时七年四月二十二日
    :param y_merge: 合并执行所得到的降维结果
    :param MAX_EIGEN_COUNT: 最多使用的特征向量个数
    :param n: 原始数据集的样本个数
    :return:
    """
    y_list_add = []
    y_list_sub = []

    y = np.zeros((n, 2))
    y[0:n, :] = y_merge[0:n, :]

    for loop_index in range(0, MAX_EIGEN_COUNT):
        y_add = np.zeros((n, 2))
        y_sub = np.zeros((n, 2))
        y_add[0:n, :] = y_merge[(2*(loop_index+1)-1)*n:2*(loop_index+1)*n, :]
        y_sub[0:n, :] = y_merge[2*(loop_index+1)*n:(2*(loop_index+1)+1)*n, :]

        y_list_add.append(y_add)
        y_list_sub.append(y_sub)
    return y, y_list_add, y_list_sub
