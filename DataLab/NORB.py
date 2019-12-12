import numpy as np
from Main import DimReduce
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os
from Tools import ImageScatter
from Main import Preprocess


def sub_sampling():
    path = "E:\\Project\\特征向量散点图项目\\MATLAB\\NORB\\1000\\"
    data = np.loadtxt(path+"origin.csv", dtype=np.int, delimiter=",")
    print(data.shape)
    (n, m) = data.shape

    X = np.zeros((n, 256))
    for i in range(0, n):
        face = data[i].reshape(96, 96)
        sub_face = np.zeros((16, 16))
        for row in range(0, 16):
            for column in range(0, 16):
                s = 0
                for j1 in range(row*6-6, row*6):
                    for j2 in range(column*6-6, column*6):
                        s = s + face[j1, j2]
                sub_face[row, column] = s / 36
        X[i, :] = sub_face.reshape(1, 256)
    np.savetxt(path+"data.csv", X, fmt='%f', delimiter=",")


def pca_research():
    """
    进行PCA变换，观察需要用多少个特征向量来表示原来的数据
    :return:
    """
    path = "E:\\Project\\DataLab\\NORB\\"
    data = np.loadtxt(path + "data.csv", dtype=np.float, delimiter=",")
    (n, m) = data.shape
    label = np.loadtxt(path + "label.csv", dtype=np.int, delimiter=",")

    pca = PCA(n_components=m)
    pca.fit(data)
    vectors = pca.components_  # 所有的特征向量
    values = pca.explained_variance_  # 所有的特征值
    np.savetxt(path+"eigenvectors.csv", vectors, fmt='%f', delimiter=",")
    np.savetxt(path+"eigenvalues.csv", values, fmt='%f', delimiter=",")
    print("finished")


def pre_pca():
    """
    用PCA方法进行预降维
    :return:
    """
    path = "E:\\Project\\DataLab\\NORB\\"
    data = np.loadtxt(path + "data.csv", dtype=np.float, delimiter=",")
    (n, m) = data.shape

    pca = PCA(n_components=30)
    Y = pca.fit_transform(data)
    np.savetxt(path+"pca_data.csv", Y, fmt='%f', delimiter=",")
    np.savetxt(path+"small_pca.csv", Y[0:1000, :], fmt='%f', delimiter=",")


def tsne_data():
    path = "E:\\Project\\DataLab\\NORB\\"
    data = np.loadtxt(path + "data.csv", dtype=np.float, delimiter=",")
    (n, m) = data.shape
    label = np.loadtxt(path + "label.csv", dtype=np.int, delimiter=",")
    info = np.loadtxt(path + "info.csv", dtype=np.int, delimiter=",")

    t_sne = TSNE(n_components=2)
    # t_sne = PCA(n_components=2)
    Y = t_sne.fit_transform(data)
    plt.scatter(Y[:, 0], Y[:, 1], c=label)
    ax = plt.gca()
    ax.set_aspect(1)
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    # sub_sampling()
    # pca_research()
    # pre_pca()
    tsne_data()
