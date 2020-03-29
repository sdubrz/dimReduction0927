# 画KNN树
import numpy as np
import matplotlib.pyplot as plt


def draw_knn_tree(path, n_node):
    """

    :param path: 数据存放目录
    :param n_node: 近邻数
    :return:
    """
    Y = np.loadtxt(path+"y.csv", dtype=np.float, delimiter=",")
    (n, m) = Y.shape
    KNN = np.loadtxt(path+"【weighted】knn.csv", dtype=np.int, delimiter=",")

    plt.scatter(Y[:, 0], Y[:, 1], marker='o', c='y')
    for i in range(0, n):
        for j in range(1, n_node+1):
            plt.plot([Y[i, 0], Y[KNN[i, j], 0]], [Y[i, 1], Y[KNN[i, j], 1]], c='deepskyblue', linewidth=0.9, alpha=0.7)

    ax = plt.gca()
    ax.set_aspect(1)
    plt.show()


if __name__ == '__main__':
    path = "E:\\文件\\IRC\\特征向量散点图项目\\result2020\\result0119_withoutnormalize\\cTSNE\\fashion50mclass6_251\\yita(100.20200306222)nbrs_k(40)method_k(60)numbers(4)_b-spline_weighted\\"
    path_mds = "E:\\文件\\IRC\\特征向量散点图项目\\result2020\\result0119_withoutnormalize\\PCA\\fashion50mclass6_251\\yita(100.20200306222)nbrs_k(40)method_k(60)numbers(4)_b-spline_weighted\\"
    path = "E:\\文件\\IRC\\特征向量散点图项目\\result2020\\result0119_withoutnormalize\\cTSNE\\fashion50mclass568\\yita(50.202002172)nbrs_k(51)method_k(90)numbers(4)_b-spline_weighted\\"
    draw_knn_tree(path, n_node=2)
