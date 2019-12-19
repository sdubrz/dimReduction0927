import numpy as np
from MyDR import t_sne
from Main import Preprocess
import matplotlib.pyplot as plt


def run_test():
    path = "E:\\Project\\result2019\\result1026without_straighten\\datasets\\Iris\\"
    data = np.loadtxt(path+"data.csv", dtype=np.float, delimiter=",")
    data = Preprocess.normalize(data)
    label = np.loadtxt(path+"label.csv", dtype=np.int, delimiter=",")
    dr = t_sne.TSNE(n_components=2)
    Y = dr.fit_transform(data)

    plt.scatter(Y[:, 0], Y[:, 1], c=label)
    plt.show()


def run_test2():
    # path = "E:\\Project\\result2019\\result1026without_straighten\\datasets\\Iris\\"  # 华硕
    path = "E:\\文件\\IRC\\特征向量散点图项目\\result2019\\result1219without_straighten\\PCA\\Iris\\yita(0.1)nbrs_k(30)method_k(30)numbers(3)_b-spline_weighted\\"  # XPS
    data = np.loadtxt(path+"x.csv", dtype=np.float, delimiter=",")
    # data = Preprocess.normalize(data)
    label = np.loadtxt(path+"label.csv", dtype=np.int, delimiter=",")
    (n, m) = data.shape

    # 扰动向量
    vectors = np.loadtxt(path+"【weighted】eigenvectors0.csv", dtype=np.float, delimiter=",")
    weights = np.loadtxt(path+"【weighted】eigenweights.csv", dtype=np.float, delimiter=",")
    for i in range(0, n):
        vectors[i, :] = vectors[i, :] * weights[i, 0]
    eta = 0.001

    y_random = np.random.random((n, 2))
    dr = t_sne.TSNE(n_components=2, n_iter=10000, init=y_random)
    Y = dr.fit_transform(data)

    # dr4 = t_sne.TSNE(n_components=2, n_iter=10000, init=y_random)
    # Y4 = dr4.fit_transform(data+eta*vectors)

    dr2 = t_sne.TSNE(n_components=2, n_iter=1000, init=Y, early_exaggeration=1.0, learning_rate=10.0)
    Y2 = dr2.fit_transform(data)

    dr3 = t_sne.TSNE(n_components=2, n_iter=1500, init=Y, early_exaggeration=1.0, learning_rate=10.0)
    Y3 = dr3.fit_transform(data)

    plt.scatter(Y[:, 0], Y[:, 1], c='r')
    plt.scatter(Y2[:, 0], Y2[:, 1], c='g')
    plt.scatter(Y3[:, 0], Y3[:, 1], c='b')
    # plt.scatter(Y4[:, 0], Y4[:, 1], c='k')
    plt.show()


if __name__ == '__main__':
    run_test2()
