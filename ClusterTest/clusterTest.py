# 用聚类来进行测试
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN


COLORS = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange', 'deepskyblue']


def k_means_data(path):
    """
    对数据进行k_means聚类
    :param path: 数据存放的目录
    :return:
    """
    X = np.loadtxt(path+"x.csv", dtype=np.float, delimiter=",")
    Y = np.loadtxt(path+"y.csv", dtype=np.float, delimiter=",")
    (n, m) = X.shape
    kmeans = KMeans(n_clusters=9).fit(X)
    label = kmeans.labels_
    for i in range(0, n):
        c_index = label[i] % len(COLORS)
        plt.scatter(Y[i, 0], Y[i, 1], c=COLORS[c_index])
    ax = plt.gca()
    ax.set_aspect(1)
    plt.show()


def dbscan_data(path):
    """
    对数据使用DBSCAN聚类
    :param path: 数据存放的目录
    :return:
    """
    X = np.loadtxt(path + "x.csv", dtype=np.float, delimiter=",")
    Y = np.loadtxt(path + "y.csv", dtype=np.float, delimiter=",")
    (n, m) = X.shape
    


if __name__ == '__main__':
    path = "E:\\Project\\result2019\\result1026without_straighten\\PCA\\Wine\\yita(0.05)nbrs_k(30)method_k(30)numbers(4)_b-spline_weighted\\"
    olive_path = "E:\\Project\\result2019\\result1026without_straighten\\PCA\\olive\\yita(0.03)nbrs_k(45)method_k(20)numbers(4)_b-spline_weighted\\"
    k_means_data(olive_path)
