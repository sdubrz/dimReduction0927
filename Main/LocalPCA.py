from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import numpy as np
from Main import Preprocess

# import matplotlib.pyplot as plt


"""
与计算localPCA有关的相关功能实现
"""


def local_pca_dn(data):
    """
    计算localPCA，需要把所有的特征值都返回
    :param data: 局部数据
    :return:
    """
    data_shape = data.shape
    local_pca = PCA(n_components=data_shape[1], copy=True, whiten=True)
    local_pca.fit(data)
    vectors = local_pca.components_  # 所有的特征向量
    values = local_pca.explained_variance_  # 所有的特征值
    return vectors, values


def how_many_eigens(data, k, threshold=0.8):
    """
    对data计算local-PCA，计算最少需要多少个特征值，才能使得占特征值总和的比重超过threshold
    :param data: 数据，应该是规范化之后的数据
    :param k: 当前所取的k值
    :param threshold: 占比的阈值，默认是80%
    :return:
    """
    data_shape = data.shape
    n = data_shape[0]
    dim = data_shape[1]

    nbr_s = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(data)
    distance, knn = nbr_s.kneighbors(data)

    eigen_count = np.zeros((n, 1))  # 记录每个点邻域local_PCA所需的特征值数量

    for i in range(0, n):
        local_data = np.zeros((k, dim))
        for j in range(0, k):
            local_data[j, :] = data[knn[i, j], :]

        vectors, values = local_pca_dn(local_data)

        for j in range(0, dim):
            values[j] = np.abs(values[j])

        sum_values = np.sum(values)

        temp_sum = 0
        for j in range(0, j):
            temp_sum = temp_sum + values[j]
            if temp_sum > sum_values*threshold:
                eigen_count[i] = j + 1
                break

    return eigen_count


def how_many_k(data, MAX_K=70, eigen_numbers=1):
    """
    对每个数据点计算取不同的k值时最大的特征值占所有的特征值和的比重，此处假设数据data已经进行过预处理
    :param data: 要处理的数据
    :param MAX_K: 允许计算的最大k值，如果k值过大，就无法代表局部信息了
    :param eigen_numbers: 所使用的特征值数，默认为1，也就是在计算时只看最大特征值所占的比重
    :return: 每个数据在不同的k值的情况下最大特征值占所有特征值和的比重矩阵，每一行是一条数据的记录
    """
    data_shape = data.shape
    n = data_shape[0]
    dim = data_shape[1]

    k_start = dim+1  # 最小的k值应该大于数据的维度
    if MAX_K > n-1:
        MAX_K = n-1
    k_end = MAX_K  # 最大的k值计算范围不能超过样本数据

    eigen_count = np.zeros((n, k_end-k_start+1))

    for k in range(k_start, k_end+1):

        nbr_s = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(data)
        distance, knn = nbr_s.kneighbors(data)

        for i in range(0, n):
            local_data = np.zeros((k, dim))
            for j in range(0, k):
                local_data[j, :] = data[knn[i, j], :]

            vectors, values = local_pca_dn(local_data)
            for j in range(0, dim):
                values[j] = np.abs(values[j])

            temp_sum = 0
            for j in range(0, eigen_numbers):
                temp_sum = temp_sum + values[j]
            eigen_count[i, k-k_start] = temp_sum/np.sum(values)   # 在这里进行修改可以调整所选的特征值数目

    return eigen_count, k_start


def find_max_k(eigen_count, k0, low_k, up_k):
    """
    寻找最大特征值占比最大的k值，要求k必须在[low_k, up_k]中
    :param eigen_count: 不同k值情况下，最大特征值占所有特征值和的比重矩阵
    :param k0: 矩阵中第一列所对应的k
    :param low_k:
    :param up_k:
    :return:
    """
    # print(eigen_count)
    data_shape = eigen_count.shape
    n = data_shape[0]

    result = np.zeros((n, 1))
    max_eigens = np.zeros((n, 1))
    for i in range(0, n):
        max_index = low_k
        for j in range(low_k, up_k+1):
            if eigen_count[i, j-k0] > eigen_count[i, max_index-k0]:
                max_index = j
        result[i] = max_index
        max_eigens[i] = eigen_count[i, max_index-k0]

    return result, max_eigens


def find_max_k_adaptive(eigen_count, k0, low_k, up_k, threshold=0):
    """
    使用一种设置阈值的方法来寻找 k 值，先选择前几个特征值占比最大的k值，
    然后找出所有阈值内的k值，选择那个最大的k值作为目标结果
    :param eigen_count: 不同k值情况下，最大特征值占所有特征值和的比重
    :param k0: 矩阵中第一列所对应的k值
    :param low_k:
    :param up_k:
    :param threshold:
    :return:  result: 特征值占比最大的k值，是一个n维的向量
            max_eigens: 最大的占比
    """
    data_shape = eigen_count.shape
    n = data_shape[0]

    result = np.zeros((n, 1))
    max_eigens = np.zeros((n, 1))
    for i in range(0, n):
        max_index = low_k
        for j in range(low_k, up_k + 1):
            if eigen_count[i, j - k0] > eigen_count[i, max_index - k0]:
                max_index = j
        index = up_k
        temp_result = max_index
        while index > max_index:
            if eigen_count[i, max_index-k0] - eigen_count[i, index-k0] < threshold:
                temp_result = index
                break
            index = index - 1
        result[i] = temp_result
        max_eigens[i] = eigen_count[i, temp_result - k0]

    return result, max_eigens


def robust_k(data, min_k, max_k):
    """ 寻找稳定的k值
        在[min_k, max_k]范围内寻找特征向量最稳定的k值，通过特征向量来进行判定。
        特征向量方向变化比较小的。

        :param data: 高维数据矩阵，应该是经历过预处理之后的
        :param min_k: 最小的k值
        :param max_k: 最大的k值
        :return:
    """
    """ 首先统计与使用上一个k值计算出来的特征向量的夹角，画一个变化曲线 """
    data_shape = data.shape
    n = data_shape[0]
    m = data_shape[1]

    nbr_s = NearestNeighbors(n_neighbors=max_k, algorithm='ball_tree').fit(data)
    distance, knn = nbr_s.kneighbors(data)

    # 计算特征向量
    eigen_vector_list = []  # 存放每个点的特征向量，里面每个元素是一个矩阵，里面存放的是这个点使用不同k值所求得的特征向量
    k_num = max_k - min_k + 1
    for i in range(0, n):
        vectors = np.zeros((k_num, m))
        local_data = np.zeros((max_k, m))

        for j in range(0, max_k):
            local_data[j, :] = data[knn[i, j], :]

        for k in range(min_k, max_k+1):
            temp_vectors, temp_values = local_pca_dn(local_data[0:k, :])
            vectors[k_num-k-1, :] = temp_vectors[0, :]
        eigen_vector_list.append(vectors)

    print("计算特征向量完毕")

    angle_change = np.zeros((n, k_num))
    for i in range(0, n):
        vectors = eigen_vector_list[i]
        for j in range(1, k_num):
            dot_value = np.matmul(vectors[j, :], np.transpose(vectors[j-1, :]))
            angle = np.arccos(dot_value)/np.pi * 180
            if angle > 90:
                angle = 180 - angle
            angle_change[i, j] = angle

    """ 目前的版本中只是计算出了k值增大时角度的变化量 """

    return angle_change


def prefect_local_pca(data, min_k, max_k):
    """
    对数据选择第一特征值占比最大的k值，计算该k值情况下的localPCA
    要求k值应该在[min_k, max_k]中
    :param data: 原始的高维数据矩阵
    :param min_k: 最小的k值，一般设置为data的维度数加一
    :param max_k: 最大的k值，一般不得超过data样本数减一
    :return:
    """
    data_shape = data.shape
    n = data_shape[0]
    dim = data_shape[1]

    if min_k < dim+1:
        print("[LocalPCA]\tprefect_local_pca: 输入的min_k值过小")
        min_k = dim + 1

    if max_k > n-1:
        print("[LocalPCA]\tprefect_local_pca：输入的max_k值过大")
        max_k = n-1

    eigen_count, k_start = how_many_k(data, MAX_K=max_k)
    prefect_k, max_eigens = find_max_k(eigen_count, k_start, min_k, max_k)

    eig_vectors1 = np.zeros(data_shape)  # 存放最大的特征值所对应的特征向量
    eig_values = np.zeros(data_shape)  # 存放特征值

    for i in range(0, n):
        nbr_s = NearestNeighbors(n_neighbors=prefect_k[i], algorithm='ball_tree').fit(data)
        distance, indexs = nbr_s.kneighbors(data)

        local_data = np.zeros((prefect_k[i], dim))
        for j in range(0, prefect_k[i]):
            local_data[j, :] = data[indexs[i, j], :]
        eig_vectors, eig_values[i, :] = local_pca_dn(local_data)
        eig_vectors1[i, :] = eig_vectors[0, :]

    return eig_vectors1, eig_values


def test():
    data_name = "Iris"
    main_path = "F:\\result2019\\result0116\\"
    path = main_path + "datasets\\" + data_name + "\\data.csv"
    data_file = np.loadtxt(path, dtype=np.str, delimiter=',')
    data = data_file[:, :].astype(np.float)
    data = Preprocess.normalize(data)
    data_shape = data.shape
    n = data_shape[0]
    m = data_shape[1]

    # eigen_count = how_many_eigens(data, 30, threshold=0.8)
    # print(eigen_count)
    eigen_count, k_start = how_many_k(data, MAX_K=n - 1, eigen_numbers=1)

    temp_path = main_path + "k_proportion\\" + data_name + "_kstart" + str(k_start) + "eigennumber_1" + ".csv"
    np.savetxt(temp_path, eigen_count, fmt="%f", delimiter=",")
    print("保存成功")

    # k_values = np.zeros((1, n-m-1))
    # for i in range(0, n-m-1):
    #     k_values[0, i] = k_start+i
    #
    # for i in range(0, n):
    #     plt.plot(k_values, eigen_count[i, :])
    #     plt.savefig(main_path+"k_proportion\\"+data_name+"\\"+str(i)+".png")
    #     plt.close()
    # print("画图保存完毕")

    # print(eigen_count)
    # result, max_eigens = find_max_k(eigen_count, 5, 5, 50)
    # print(result)


def k_angle_test():
    main_path = "F:\\result2019\\result0116\\datasets\\Iris\\data.csv"
    data_reader = np.loadtxt(main_path, dtype=np.str, delimiter=",")
    data = data_reader[:, :].astype(np.float)

    data_shape = data.shape
    n = data_shape[0]
    m = data_shape[1]
    data = Preprocess.normalize(data)

    angles = robust_k(data, 15, 149)
    np.savetxt("F:\\angles.csv", angles, fmt="%f", delimiter=",")


def reconfirm_k_range(k_list, difference, number, dim, MAX_K):
    """
    重新确定k值的范围，用于在保证相邻的点的k值差别不要过大的时候的一种修正方法
    这是一个类似于OJ上的题的东西
    :param k_list: 某个点当前的k个邻居的k值
    :param difference: 所允许与相邻点的k值方面的最大差别
    :param number: 所允许的差别过大的邻居点的个数上限
    :param dim: 维度数目
    :param MAX_K: 所允许使用的最大k值
    :return: 新的k值选择范围
    """
    range_list = []

    low = -1
    up = -1

    for i in range(dim+1, MAX_K+1):
        count = 0
        for k_value in k_list:
            if np.abs(i-k_value) > difference:
                count = count+1
        if count < number:
            if low < 0:
                low = i
                up = i
            else:
                up = i
        else:
            if low < 0:
                continue
            else:
                a_range = (low, up)
                range_list.append(a_range)

    if range_list.__len__() == 0:  # 这种情况下没有任何可以满足要求的k值范围
        range_list.append((dim+1, MAX_K))

    return range_list


def best_k_one_point(data, index, range_list, eigen_number, MAX_K):
    """
    寻找一个点的特征值占比最大的k值，结果的k值必须在range_list所确定的范围之内
    :param data: 经过预处理之后的数据矩阵，每一行是一个高维数据点
    :param index: 所要重新计算k值的点的索引号
    :param range_list: 一系列的k值范围，每一个元素是一个二元组，二元组中第一项是范围下界，第二项是范围上界。list中可能有多个断断续续的范围元组
    :param eigen_number: 这个点所使用的特征向量个数
    :param MAX_K: 所允许使用的最大的k值
    :return:
    """
    nbr_s = NearestNeighbors(n_neighbors=MAX_K, algorithm='ball_tree').fit(data)
    distance, nbrs = nbr_s.kneighbors(data)

    data_shape = data.shape
    n = data_shape[0]
    dim = data_shape[1]

    best_k = dim+1
    best_proportion = 0

    local_data = np.zeros((MAX_K, dim))
    for i in range(0, MAX_K):
        local_data[i, :] = data[nbrs[index, i], :]

    for a_range in range_list:
        for k in range(a_range[0], a_range[1]+1):
            temp_vectors, temp_values = local_pca_dn(local_data[0:k, :])
            temp_proportion = np.sum(temp_values[0:eigen_number]) / np.sum(temp_values)
            if temp_proportion > best_proportion:
                best_k = k
                best_proportion = temp_proportion

    return best_k, best_proportion


def loop_test():
    low = 0
    up = 0
    list = []
    for i in range(0, 5):
        low = i
        up = i
        a = (low, up)
        list.append(a)
    print(list)


if __name__ == "__main__":
    loop_test()
