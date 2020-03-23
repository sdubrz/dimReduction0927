# 显示KNN信息
import numpy as np
import matplotlib.pyplot as plt


def show_knn(index, path):
    """
    显示某个点的KNN
    :param index: 要显示的点
    :param path:
    :return:
    """
    Y = np.loadtxt(path+"y.csv", dtype=np.float, delimiter=",")
    KNN = np.loadtxt(path+"【weighted】knn.csv", dtype=np.int, delimiter=",")

    (n, k) = KNN.shape
    print((n, k))

    plt.scatter(Y[:, 0], Y[:, 1], marker='o', c='y')

    for i in range(1, k):
        plt.scatter(Y[KNN[index, i], 0], Y[KNN[index, i], 1], c='r')
    plt.scatter(Y[index, 0], Y[index, 1], marker='p', c='k')

    ax = plt.gca()
    ax.set_aspect(1)
    plt.show()


if __name__ == '__main__':
    index = 538
    path = "E:\\文件\\IRC\\特征向量散点图项目\\result2020\\result0119_withoutnormalize\\cTSNE\\fashion50mclass568\\yita(50.202002172)nbrs_k(51)method_k(90)numbers(4)_b-spline_weighted\\"
    show_knn(index-1, path)

