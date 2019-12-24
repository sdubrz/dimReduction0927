import numpy as np
from MyDR import t_sne
from Main import Preprocess
import matplotlib.pyplot as plt
from MyDR import cTSNE


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
    path = "E:\\Project\\result2019\\result1026without_straighten\\datasets\\Iris\\"
    data = np.loadtxt(path+"data.csv", dtype=np.float, delimiter=",")
    data = Preprocess.normalize(data)
    label = np.loadtxt(path+"label.csv", dtype=np.int, delimiter=",")
    dr = t_sne.TSNE(n_components=2, n_iter=100000)
    Y = dr.fit_transform(data)

    dr2 = t_sne.TSNE(n_components=2, n_iter=1000, init=Y, early_exaggeration=1.0)
    Y2 = dr2.fit_transform(data)

    dr3 = t_sne.TSNE(n_components=2, n_iter=1500, init=Y, early_exaggeration=1.0)
    Y3 = dr3.fit_transform(data)

    plt.scatter(Y[:, 0], Y[:, 1], c='r')
    plt.scatter(Y2[:, 0], Y2[:, 1], c='g')
    plt.scatter(Y3[:, 0], Y3[:, 1], c='b')
    plt.show()


def run_test3():
    """
    对cTSNE一次只扰动一个点
    :return:
    """
    # path = "E:\\Project\\result2019\\result1026without_straighten\\datasets\\Iris\\"
    path = "E:\\Project\\result2019\\result1026without_straighten\\cTSNE\\coil20obj_16_3class\\yita(0.4)nbrs_k(24)method_k(24)numbers(4)_b-spline_weighted\\"
    X = np.loadtxt(path + "x.csv", dtype=np.float, delimiter=",")
    Y = np.loadtxt(path+"y.csv", dtype=np.float, delimiter=",")
    vectors = np.loadtxt(path+"【weighted】eigenvectors0.csv", dtype=np.float, delimiter=",")
    weights = np.loadtxt(path+"【weighted】eigenweights.csv", dtype=np.float, delimiter=",")

    (n, m) = X.shape
    for i in range(0, n):
        vectors[i, :] = vectors[i, :] * weights[i, 0]

    eta = 0.8
    t_sne = cTSNE.cTSNE(n_component=2, perplexity=8.0)
    Y2 = np.zeros((n, 2))
    Y3 = np.zeros((n, 2))
    for i in range(0, n):
        X2 = X.copy()
        X3 = X.copy()
        X2[i, :] = X2[i, :] + eta * vectors[i, :]
        X3[i, :] = X3[i, :] - eta * vectors[i, :]
        temp_Y = t_sne.fit_transform_i(X2, i, max_iter=200, y_random=Y)
        temp_Y3 = t_sne.fit_transform_i(X3, i, max_iter=200, y_random=Y)
        Y2[i, :] = temp_Y[i, :]
        Y3[i, :] = temp_Y3[i, :]
        if i % 10 == 0:
            print(i)

    print(evaluation(Y, Y2))
    print(evaluation(Y, Y3))

    plt.scatter(Y[:, 0], Y[:, 1], c='r')
    # plt.scatter(Y2[:, 0], Y2[:, 1], c='b')
    for i in range(0, n):
        plt.plot([Y[i, 0], Y2[i, 0]], [Y[i, 1], Y2[i, 1]], c='deepskyblue', linewidth=0.7, alpha=0.7)
        plt.plot([Y[i, 0], Y3[i, 0]], [Y[i, 1], Y3[i, 1]], c='deepskyblue', linewidth=0.7, alpha=0.7)
    plt.show()


if __name__ == '__main__':
    run_test3()
