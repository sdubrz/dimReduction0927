# 对导数的分析
import numpy as np
import matplotlib.pyplot as plt
import os


def attr_influence(P):
    """
    根据导数计算每个属性的影响力
    :param P:
    :return:
    """
    (n_p, m_p) = P.shape
    n = n_p // 2
    m = m_p // n

    changes = np.zeros((n, m))
    for i in range(0, n):
        Pi = P[i*2:i*2+2, m*i:m*i+m]
        changes[i, :] = np.sum(Pi**2, axis=0)

    return changes


def max_index(X):
    """
    计算X的每行中第几个元素最大
    :param X: 矩阵
    :return:
    """
    (n, m) = X.shape
    indexs = np.zeros((n, 1))

    for i in range(0, n):
        index = 0
        for j in range(1, m):
            if X[i, index] < X[i, j]:
                index = j
        indexs[i] = index

    return indexs


def find_max_attr(path):
    """
    计算每个点的降维结果对哪个属性最敏感
    :param path:
    :return:
    """
    P = None
    if "MDS" in path and not("cTSNE" in path):
        P = np.loadtxt(path+"MDS_Pxy.csv", dtype=np.float, delimiter=",")
        print("MDS")
    elif "cTSNE" in path:
        P = np.loadtxt(path + "cTSNE_Pxy.csv", dtype=np.float, delimiter=",")
    changes = attr_influence(P)
    np.savetxt(path + "attr_influence.csv", changes, fmt='%f', delimiter=",")
    max_attr = max_index(changes)
    np.savetxt(path + "max_attr.csv", max_attr, fmt='%d', delimiter=",")

    (n, ) = max_attr.shape
    max_list = []
    for i in range(0, n):
        max_list.append(max_attr[i, 0])

    return max_list


def run():
    # path = "E:\\文件\\IRC\\特征向量散点图项目\\result2020\\result0119\\MDS\\Iris3\\yita(0.20200214)nbrs_k(20)method_k(90)numbers(3)_b-spline_weighted\\"
    path = "E:\\文件\\IRC\\特征向量散点图项目\\result2020\\result0119\\cTSNE\\Iris3\\yita(0.20200222)nbrs_k(20)method_k(90)numbers(3)_b-spline_weighted\\"
    P = np.loadtxt(path+"cTSNE_Pxy.csv", dtype=np.float, delimiter=",")
    changes = attr_influence(P)
    np.savetxt(path+"attr_influence.csv", changes, fmt='%f', delimiter=",")
    changes_mean = np.mean(changes, axis=0)
    print(changes_mean)
    (n, m) = changes.shape

    max_attr = max_index(changes)
    np.savetxt(path+"max_attr.csv", max_attr, fmt='%d', delimiter=",")
    plt.plot(max_attr)
    plt.show()

    color_label = []
    for i in range(0, n):
        color_label.append(max_attr[i, 0]+1)

    Y = np.loadtxt(path+"y.csv", dtype=np.float, delimiter=",")
    plt.scatter(Y[:, 0], Y[:, 1], c=color_label)
    plt.colorbar()
    ax = plt.gca()
    ax.set_aspect(1)
    plt.show()

    for attr in range(0, m):
        plt.scatter(Y[:, 0], Y[:, 1], c=changes[:, attr])
        plt.colorbar()
        ax = plt.gca()
        ax.set_aspect(1)
        plt.title(attr+1)
        plt.show()


if __name__ == '__main__':
    # run()
    path = "E:\\文件\\IRC\\特征向量散点图项目\\result2020\\result0119\\MDS\\Iris3\\yita(0.20200214)nbrs_k(20)method_k(90)numbers(3)_b-spline_weighted\\"
    a = find_max_attr(path)
