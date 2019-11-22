# 画KNN图
import numpy as np
import matplotlib.pyplot as plt

from Main import Preprocess


def draw_knn(path="", k=3):
    """
    draw knn graph
    :param path:
    :param k:
    :return:
    """
    X = np.loadtxt(path+"x.csv", dtype=np.float, delimiter=",")
    Y = np.loadtxt(path+"y.csv", dtype=np.float, delimiter=",")
    label = np.loadtxt(path+"label.csv", dtype=np.int, delimiter=",")

    (n, m) = X.shape
    X = Preprocess.normalize(X)
    knn = Preprocess.knn(X, k)

    plt.scatter(Y[:, 0], Y[:, 1], c=label)
    for i in range(0, n):
        for j in range(1, k):
            plt.plot([Y[i, 0], Y[knn[i, j], 0]], [Y[i, 1], Y[knn[i, j], 1]], linewidth=0.8, c='deepskyblue', alpha=0.7)
    ax = plt.gca()
    ax.set_aspect(1)
    plt.show()


if __name__ == '__main__':
    path = "E:\\Project\\DataLab\\wineQuality\\"
    draw_knn(path, k=3)

