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


if __name__ == '__main__':
    run_test2()
