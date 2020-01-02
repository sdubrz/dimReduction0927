# 测试MDS的稳定性
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from Main import Preprocess
from Main import DimReduce


def evaluation(Y0, Y1):
    """
    评测收敛的效果
    :param Y0:
    :param Y1:
    :return:
    """
    (n, m) = Y0.shape
    dx = np.max(Y0[:, 0]) - np.min(Y0[:, 0])
    dy = np.max(Y0[:, 1] - np.min(Y0[:, 1]))
    dd = max(dx, dy)

    radius = np.zeros((n, 1))
    for i in range(0, n):
        radius = np.linalg.norm(Y0[i, :] - Y1[i, :])
    r = np.mean(radius)

    return r/dd


def run():
    path = "E:\\Project\\result2019\\result1026without_straighten\\datasets\\digits5_8\\"
    X = np.loadtxt(path+"data.csv", dtype=np.float, delimiter=",")
    X = Preprocess.normalize(X)

    mds = MDS(n_components=2, max_iter=3000)
    Y = mds.fit_transform(X)

    mds2 = MDS(n_components=2, max_iter=3000)
    Y2 = mds2.fit_transform(X, init=Y)

    print(evaluation(Y, Y2))

    plt.scatter(Y[:, 0], Y[:, 1], c='r')
    plt.scatter(Y2[:, 0], Y[:, 1], c='b')
    plt.show()


def run2():
    path = "E:\\Project\\result2019\\result1026without_straighten\\datasets\\Wine\\"
    X = np.loadtxt(path + "data.csv", dtype=np.float, delimiter=",")
    X = Preprocess.normalize(X)

    Y = DimReduce.dim_reduce_convergence(X, method="MDS", n_iter_init=1000)

    Y2 = DimReduce.dim_reduce_convergence(X, method="MDS", n_iter_init=300, y_random=Y)

    print(evaluation(Y, Y2))

    plt.scatter(Y[:, 0], Y[:, 1], c='r')
    plt.scatter(Y2[:, 0], Y[:, 1], c='b')
    plt.show()


def run3():
    path = "E:\\Project\\DataLab\\t-SNETest\\Wine\\"
    X = np.loadtxt(path + "x.csv", dtype=np.float, delimiter=",")
    (n, m) = X.shape
    n_iter = 10000

    mds = MDS(n_components=2, max_iter=n_iter)
    Y = mds.fit_transform(X)

    vectors = np.loadtxt(path + "【weighted】eigenvectors0.csv", dtype=np.float, delimiter=",")
    weights = np.loadtxt(path + "【weighted】eigenweights.csv", dtype=np.float, delimiter=",")
    eta = 0.8

    index = 100
    perturb_iter = 10000
    X2 = X.copy()
    X2[index, :] = X2[index, :] + eta * weights[index, 0] * vectors[index, :]

    mds2 = MDS(n_components=2, n_init=1, max_iter=perturb_iter)
    Y2 = mds2.fit_transform(X2, init=Y)

    plt.scatter(Y[:, 0], Y[:, 1], c='r')
    plt.scatter(Y2[:, 0], Y2[:, 1], c='b')
    plt.scatter(Y[index, 0], Y[index, 1], marker='p', c='yellow')
    for i in range(0, n):
        plt.plot([Y[i, 0], Y2[i, 0]], [Y[i, 1], Y2[i, 1]], c='deepskyblue', alpha=0.7, linewidth=0.8)
    plt.show()


if __name__ == '__main__':
    run3()

