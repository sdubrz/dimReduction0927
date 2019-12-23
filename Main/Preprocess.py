import numpy as np
import os
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt


"""
对数据进行预处理的一些函数功能
"""


def normalize(x, low=-1, up=1):
    """
    将数据各个维度都均匀地缩放到[-1, 1]
    :param x: 数据矩阵，每一行代表一条数据记录
    :param low: 处理之后每个维度的最小值，默认是 -1
    :param up: 处理之后每隔维度的最大值，默认值是 1
    :return:
    """
    data_shape = x.shape
    n = data_shape[0]
    dim = data_shape[1]
    new_x = np.zeros(data_shape)
    min_v = np.zeros((1, dim))
    max_v = np.zeros((1, dim))

    for i in range(0, dim):
        min_v[0, i] = min(x[:, i])
        max_v[0, i] = max(x[:, i])
    for i in range(0, n):
        for j in range(0, dim):
            if min_v[0, j] == max_v[0, j]:
                new_x[i, j] = 0
                continue
            new_x[i, j] = (x[i, j]-min_v[0, j])/(max_v[0, j]-min_v[0, j])*(up-low)+low

    return new_x


def knn_by_distance(data, distance):
    """
    通过距离来计算近邻，目前的算法复杂度比较高
    这样会有一个问题，就是设定一个距离半径之后，有些点的邻居数可能会小于数据维度，这种情况计算PCA会出现问题
    :param data: 数据矩阵
    :param distance: 距离半径
    :return:
    """
    nbrs = []
    data_shape = data.shape
    n = data_shape[0]
    dim = data_shape[1]

    for i in range(0, n):
        a_nbrs = []
        # for j in range()


def knn(data, k):
    """
    计算k近邻, 返回的结果中第一个邻居，也就是最近的邻居，其实是数据点本身
    :param data: 数据矩阵，每一行是一条数据记录
    :param k:
    :return:
    """
    nbr_s = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(data)
    distance, index = nbr_s.kneighbors(data)
    return index


def knn_radius(data, knn, save_path=None):
    """
    计算KNN的半径范围
    :param data: 数据矩阵，每一行是一个点
    :param knn: K近邻关系矩阵
    :param save_path: 储存结果的路径
    :return:
    """
    (n, k) = knn.shape
    radius = np.zeros((n, 1))
    for i in range(0, n):
        radius[i] = np.linalg.norm(data[i, :] - data[knn[i, k-1], :])

    print("KNN的平均半径是 ", np.mean(radius))
    if not save_path is None:
        np.savetxt(save_path+"knn_radius.csv", radius, fmt='%f', delimiter=",")
        plt.hist(radius)
        plt.savefig(save_path+"knn_radius.png")
        plt.close()


def check_filepath(path):
    """
    通过向一个路径下存放一个很小的文件来检查该文件夹是否存在
    避免出现存储结果时报错，以至于浪费时间的情况
    :param path: 要检查的文件路径
    :return:
    """
    if not os.path.exists(path):
        os.makedirs(path)
    data = np.array([[1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9]])
    np.savetxt(path + "check_path.csv", data, fmt="%d", delimiter=",")
