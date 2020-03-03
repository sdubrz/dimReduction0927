# 测试PCA
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from Main import Preprocess


def test():
    """

    :return:
    """
    path = "E:\\文件\\IRC\\特征向量散点图项目\\result2020\\result0119\\datasets\\Iris3\\"
    data = np.loadtxt(path+"data.csv", dtype=np.float, delimiter=",")
    X = Preprocess.normalize(data, -1, 1)

    pca = PCA(n_components=2)
    Y1 = pca.fit_transform(X)
    P_ = pca.components_
    P = np.transpose(P_)
    Y2 = np.matmul(X, P)

    plt.scatter(Y1[:, 0], Y1[:, 1], c='r')
    plt.scatter(Y2[:, 0], Y2[:, 1], c='b')

    plt.show()


if __name__ == '__main__':
    test()

