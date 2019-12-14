# 测试 LTSA 降维方法
import numpy as np
import matplotlib.pyplot as plt
from Main import DimReduce
from Main import Preprocess
from sklearn.manifold import LocallyLinearEmbedding


def run():
    path = "E:\\Project\\result2019\\result1026without_straighten\\datasets\\band3d\\"
    data = np.loadtxt(path+"data.csv", dtype=np.float, delimiter=",")
    label = np.loadtxt(path+"label.csv", dtype=np.int, delimiter=",")
    X = Preprocess.normalize(data, -1, 1)

    Y = DimReduce.dim_reduce(X, method='LTSA', method_k=60)
    plt.scatter(Y[:, 0], Y[:, 1], c=label)
    plt.colorbar()
    ax = plt.gca()
    ax.set_aspect(1)
    plt.show()


def run2():
    path = "E:\\Project\\result2019\\result1026without_straighten\\datasets\\band3d\\"
    data = np.loadtxt(path + "data.csv", dtype=np.float, delimiter=",")
    label = np.loadtxt(path + "label.csv", dtype=np.int, delimiter=",")
    X = Preprocess.normalize(data, -1, 1)
    (n, m) = X.shape

    ltsa = LocallyLinearEmbedding(n_neighbors=20, n_components=1, method='ltsa')
    Y = ltsa.fit_transform(X)
    plt.scatter(range(n), Y)
    # plt.colorbar()
    # ax = plt.gca()
    # ax.set_aspect(1)
    # plt.plot(Y)
    plt.show()


if __name__ == '__main__':
    run2()
