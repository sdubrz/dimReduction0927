# 检查亚采样的效果
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.manifold import TSNE


def test():
    path = "E:\\文件\\IRC\\特征向量散点图项目\\DataLab\\IsomapFace\\"
    data0 = np.loadtxt(path+"origin.csv", dtype=np.int, delimiter=",")
    data = np.loadtxt(path+"data.csv", dtype=np.float, delimiter=",")

    index = 0

    X0 = np.reshape(data0[index, :], (64, 64))
    X = np.reshape(data[index, :], (8, 8))

    plt.subplot(121)
    plt.imshow(X0)
    plt.subplot(122)
    plt.imshow(X)

    plt.show()


def test2():
    path = "E:\\文件\\IRC\\特征向量散点图项目\\DataLab\\coil20obj\\"
    data0 = np.loadtxt(path + "origin.csv", dtype=np.int, delimiter=",")
    data = np.loadtxt(path + "data.csv", dtype=np.float, delimiter=",")

    index = 0

    X0 = np.reshape(data0[index, :], (128, 128))
    X = np.reshape(data[index, :], (8, 8))

    plt.subplot(121)
    plt.imshow(X0)
    plt.subplot(122)
    plt.imshow(X)

    plt.show()


def dr_test():
    path = "E:\\文件\\IRC\\特征向量散点图项目\\DataLab\\IsomapFace\\"
    data0 = np.loadtxt(path + "origin.csv", dtype=np.int, delimiter=",")
    data = np.loadtxt(path + "data.csv", dtype=np.float, delimiter=",")
    print(data0.shape)
    print(data.shape)

    pca0 = TSNE(n_components=2)
    Y0 = pca0.fit_transform(data0)

    pca = TSNE(n_components=2)
    Y = pca.fit_transform(data)

    plt.subplot(121)
    plt.scatter(Y0[:, 0], Y0[:, 1])
    plt.title("origin t-sne")
    plt.subplot(122)
    plt.scatter(Y[:, 0], Y[:, 1])
    plt.title("down sampling t-sne")
    plt.show()


def dr_test_coil():
    path = "E:\\文件\\IRC\\特征向量散点图项目\\DataLab\\coil20obj\\"
    data0 = np.loadtxt(path + "origin.csv", dtype=np.int, delimiter=",")
    data = np.loadtxt(path + "Y16.csv", dtype=np.float, delimiter=",")
    X0 = data0[0:72*3, :]
    X = data[0:216, :]
    print(X0.shape)
    print(X.shape)

    label = []
    for i in range(0, 72):
        label.append(1)
    for i in range(0, 72):
        label.append(2)
    for i in range(0, 72):
        label.append(3)

    pca0 = TSNE(n_components=2)
    Y0 = pca0.fit_transform(X0)

    pca = TSNE(n_components=2)
    Y = pca.fit_transform(X)

    plt.subplot(121)
    plt.scatter(Y0[:, 0], Y0[:, 1], c=label)
    plt.title("origin TSNE")
    plt.subplot(122)
    plt.scatter(Y[:, 0], Y[:, 1], c=label)
    plt.title("down sampling TSNE")
    plt.show()


if __name__ == '__main__':
    # test()
    # test2()
    dr_test_coil()


