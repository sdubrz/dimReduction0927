# local LPP的计算
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import euclidean_distances


def max_indexs(a_list0, num_head=2):
    """获得前几个大的数的索引号"""
    k_list = []
    a_list = []
    for i in a_list0:
        a_list.append(i.real)

    n = len(a_list)
    for i in range(0, n):
        if len(k_list) < num_head:
            k_list.append(i)
            index = len(k_list) - 1
            while index > 0:
                if a_list[k_list[index]] > a_list[k_list[index - 1]]:
                    temp = k_list[index]
                    k_list[index] = k_list[index - 1]
                    k_list[index - 1] = temp
                    index = index - 1
                else:
                    break
        else:
            if a_list[k_list[num_head - 1]] < a_list[i]:
                k_list[num_head - 1] = i
                index = len(k_list) - 1
                while index > 0:
                    if a_list[k_list[index]] > a_list[k_list[index - 1]]:
                        temp = k_list[index]
                        k_list[index] = k_list[index - 1]
                        k_list[index - 1] = temp
                        index = index - 1
                    else:
                        break
    return k_list


def local_lpp(data):
    """
    计算local LPP， LPP本意是挑选特征值最小的几个特征向量
    :param data: local data，每一行是一个数据
    :return:
    """
    print("lpp-")
    (n, m) = data.shape
    distance = euclidean_distances(data)
    W = np.exp(-1 * distance**2) - np.eye(n)
    D = np.zeros((n, n))
    for i in range(0, n):
        D[i, i] = np.sum(W[i, :])
    L = D - W

    M1 = np.matmul(np.matmul(data.T, D), data)
    M2 = np.matmul(np.matmul(data.T, L), data)
    M = np.matmul(np.linalg.inv(M1), M2)

    eigenvalues, eigenvectors = np.linalg.eig(M)
    # 暂时取个倒数
    eigenvalues = 1 / eigenvalues
    sort_index = max_indexs(eigenvalues, m)
    eigenvectors = eigenvectors.T

    values = np.zeros((1, m))
    vectors = np.zeros((m, m))
    for i in range(0, m):
        values[0, i] = eigenvalues[sort_index[i]]
        vectors[i, :] = eigenvectors[sort_index[i], :]

    return vectors, values


def test():
    path = "E:\\文件\\IRC\\特征向量散点图项目\\result2020\\locallda\\datasets\\Iris3\\"
    data = np.loadtxt(path+"data.csv", dtype=np.float, delimiter=",")
    from Main import Preprocess
    X = Preprocess.normalize(data, -1, 1)

    vectors, values = local_lpp(X)
    print(vectors)
    print(values)

    P = vectors[0:2, :]
    Y = np.matmul(X, P.T)
    plt.scatter(Y[:, 0], Y[:, 1])
    plt.show()


if __name__ == '__main__':
    test()





