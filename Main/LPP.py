# Locality Preserving Projections(LPP)降维方法实现
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import euclidean_distances
from sklearn.neighbors import NearestNeighbors


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


class LPP:
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.Y = None
        self.vectors = None
        self.values = None

    def fit_transform(self, x, n_nbrs=15, weight_function='gauss'):
        """

        :param x:
        :param n_nbrs:
        :param weight_function: 权重函数，'gauss'则用高斯函数， 'one'直接赋1
        :return:
        """
        (n, m) = x.shape
        nbr_s = NearestNeighbors(n_neighbors=n_nbrs, algorithm='ball_tree').fit(x)
        distance, index = nbr_s.kneighbors(x)

        W = np.zeros((n, n))

        if weight_function == 'one':
            for i in range(0, n):
                for j in range(1, n_nbrs):
                    i2 = index[i, j]
                    W[i, i2] = 1
                    W[i2, i] = 1
        elif weight_function == 'gauss':
            for i in range(0, n):
                for j in range(1, n_nbrs):
                    i2 = index[i, j]
                    w = np.exp(-1 * distance[i, j]**2)
                    W[i, i2] = w
                    W[i2, i] = w

        D = np.zeros((n, n))
        for i in range(0, n):
            D[i, i] = np.sum(W[i, :])

        L = D - W
        M1 = np.matmul(np.matmul(x.T, D), x)
        M2 = np.matmul(np.matmul(x.T, L), x)
        M = np.matmul(np.linalg.inv(M1), M2)

        eigenvalues, eigenvectors = np.linalg.eig(M)
        eigenvectors = eigenvectors.T
        sort_indexs = max_indexs(eigenvalues, m)
        self.values = np.zeros((1, m))
        self.vectors = np.zeros((m, m))

        P = np.zeros((self.n_components, m))
        for i in range(0, m):
            self.values[0, m-i-1] = eigenvalues[sort_indexs[i]]
            self.vectors[m-i-1, :] = eigenvectors[sort_indexs[i], :]
        for i in range(0, self.n_components):
            P[i, :] = eigenvectors[m-1-sort_indexs[i], :]

        self.Y = np.matmul(x, P.T)
        return self.Y


def test():
    path = "E:\\文件\\IRC\\特征向量散点图项目\\result2020\\locallpp\\datasets\\Wine\\"
    from Main import Preprocess

    data = np.loadtxt(path+"data.csv", dtype=np.float, delimiter=",")
    label = np.loadtxt(path+"label.csv", dtype=np.int, delimiter=",")
    X = Preprocess.normalize(data, -1, 1)

    lpp = LPP(n_components=2)
    Y = lpp.fit_transform(X, n_nbrs=20, weight_function="one")

    plt.scatter(Y[:, 0], Y[:, 1], c=label)
    ax = plt.gca()
    ax.set_aspect(1)
    plt.show()


if __name__ == '__main__':
    test()

