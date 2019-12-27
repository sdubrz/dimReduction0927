# 测试LLE
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import LocallyLinearEmbedding
from Main import Preprocess
from MyDR import LLE
from Main import LLE_Perturb


def run():
    path = "E:\\Project\\result2019\\result1224\\datasets\\coil20obj_16_3class\\"
    data = np.loadtxt(path+"data.csv", dtype=np.float, delimiter=",")
    label = np.loadtxt(path+"label.csv", dtype=np.int, delimiter=",")
    X = Preprocess.normalize(data)

    lle = LocallyLinearEmbedding(n_neighbors=20, n_components=2)
    Y = lle.fit_transform(X)
    np.savetxt(path+"lle.csv", Y, fmt='%f', delimiter=",")
    plt.scatter(Y[:, 0], Y[:, 1], c=label)
    plt.show()


def run2():
    path = "E:\\Project\\result2019\\result1224\\datasets\\coil20obj_16_3class\\"
    data = np.loadtxt(path + "data.csv", dtype=np.float, delimiter=",")
    label = np.loadtxt(path + "label.csv", dtype=np.int, delimiter=",")
    X = Preprocess.normalize(data)

    lle = LLE.LocallyLinearEmbedding(n_neighbors=20, n_components=2)
    Y = lle.fit_transform(X)
    np.savetxt(path + "lle.csv", Y, fmt='%f', delimiter=",")
    plt.scatter(Y[:, 0], Y[:, 1], c=label)
    plt.show()


def run3():
    path = "E:\\Project\\result2019\\result1224\\datasets\\coil20obj_16_3class\\"
    data = np.loadtxt(path + "data.csv", dtype=np.float, delimiter=",")
    label = np.loadtxt(path + "label.csv", dtype=np.int, delimiter=",")
    X = Preprocess.normalize(data)

    lle_p = LLE_Perturb.LLE_Perturb(X, method_k=20)
    Y = lle_p.Y
    Y2 = lle_p.perturb(0.1*np.ones(X.shape))
    plt.scatter(Y[:, 0], Y[:, 1], c='r')
    plt.scatter(Y2[:, 0], Y2[:, 1], c='b')
    plt.show()


if __name__ == '__main__':
    run3()

