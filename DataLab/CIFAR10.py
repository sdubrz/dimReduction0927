import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict0 = pickle.load(fo, encoding='bytes')

    # keys = dict0.keys()
    # print(keys)
    # data = dict0[keys[2]]
    # print(data)
    return dict0


def see_data():
    path = "E:\\Project\\DataLab\\CIFAR\\"
    dict0 = unpickle(path+"data_batch_1")
    keys = list(dict0.keys())
    label = dict0.get(keys[1])
    data = dict0[keys[2]]
    print(label)
    print(data)
    np.savetxt(path+"label.csv", label, fmt='%d', delimiter=",")
    np.savetxt(path+"data.csv", data, fmt="%d", delimiter=",")
    np.savetxt(path+"red.csv", data[:, 0:1024], fmt="%d", delimiter=",")
    np.savetxt(path + "blue.csv", data[:, 1025:2048], fmt="%d", delimiter=",")
    np.savetxt(path + "green.csv", data[:, 2049:3072], fmt="%d", delimiter=",")


def pre_pca():
    path = "E:\\Project\\DataLab\\CIFAR\\"
    data = np.loadtxt(path+"data.csv", dtype=np.int, delimiter=",")
    label = np.loadtxt(path+"label.csv", dtype=np.int, delimiter=",")
    (n, m) = data.shape
    pca = PCA(n_components=2)
    Y = pca.fit_transform(data[:, 1024:2048])
    plt.scatter(Y[0:1000, 0], Y[0:1000, 1], c=label[0:1000])
    plt.colorbar()
    ax = plt.gca()
    ax.set_aspect(1)
    plt.show()


def pca_research():
    """
    进行PCA变换，观察需要用多少个特征向量来表示原来的数据
    :return:
    """
    path = "E:\\Project\\DataLab\\CIFAR\\"
    data = np.loadtxt(path + "blue.csv", dtype=np.float, delimiter=",")
    (n, m) = data.shape
    # label = np.loadtxt(path + "label.csv", dtype=np.int, delimiter=",")

    pca = PCA(n_components=m)
    pca.fit(data)
    vectors = pca.components_  # 所有的特征向量
    values = pca.explained_variance_  # 所有的特征值
    np.savetxt(path+"green_eigenvectors.csv", vectors, fmt='%f', delimiter=",")
    np.savetxt(path+"green_eigenvalues.csv", values, fmt='%f', delimiter=",")
    print("finished")


def t_sne_data():
    path = "E:\\Project\\DataLab\\CIFAR\\"
    data = np.loadtxt(path + "green.csv", dtype=np.float, delimiter=",")
    (n, m) = data.shape
    label = np.loadtxt(path + "label.csv", dtype=np.int, delimiter=",")

    pca = PCA(n_components=64)
    pca_y = pca.fit_transform(data)
    t_sne = TSNE(n_components=2)
    Y = t_sne.fit_transform(pca_y)

    plt.scatter(Y[:, 0], Y[:, 1], c=label)
    plt.colorbar()
    ax = plt.gca()
    ax.set_aspect(1)
    plt.show()


if __name__ == '__main__':
    # see_data()
    # pre_pca()
    # pca_research()
    t_sne_data()
