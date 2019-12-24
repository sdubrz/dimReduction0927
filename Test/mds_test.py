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


if __name__ == '__main__':
    run2()

