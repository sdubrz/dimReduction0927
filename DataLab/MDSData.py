# 有一部分MDS效果较好的数据，或许应该从MDS与Isomap结果很不一样的那种找起
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.manifold import Isomap
from Main import Preprocess


def mds_isomap():
    """
    比较 MDS 与 Isomap 降维的区别
    :return:
    """
    path = "E:\\文件\\IRC\\特征向量散点图项目\\result2020\\result0119\\datasets\\olive\\"
    data = np.loadtxt(path+"data.csv", dtype=np.float, delimiter=",")
    label = np.loadtxt(path+"label.csv", dtype=np.int, delimiter=",")

    # X = Preprocess.normalize(data, -1, 1)
    X = data
    mds = MDS(n_components=2)
    Y1 = mds.fit_transform(X)

    k = 10
    iso = Isomap(n_neighbors=k, n_components=2)
    Y2 = iso.fit_transform(X)

    plt.subplot(121)
    plt.scatter(Y1[:, 0], Y1[:, 1], c=label)
    plt.title("MDS")
    ax = plt.gca()
    ax.set_aspect(1)

    plt.subplot(122)
    plt.scatter(Y2[:, 0], Y2[:, 1], c=label)
    plt.title("Isomap k="+str(k))

    ax = plt.gca()
    ax.set_aspect(1)
    plt.show()


if __name__ == '__main__':
    mds_isomap()


