# 研究olive数据
import numpy as np
import matplotlib.pyplot as plt


def small_class():
    path = "E:\\文件\\IRC\\特征向量散点图项目\\DataLab\\olive\\"
    Y = np.loadtxt(path+"MDS.csv", dtype=np.float, delimiter=",")

    label2 = np.loadtxt(path+"label2.csv", dtype=np.int, delimiter=",")

    plt.scatter(Y[:, 0], Y[:, 1], c=label2)
    plt.colorbar()

    ax = plt.gca()
    ax.set_aspect(1)
    plt.show()


if __name__ == '__main__':
    small_class()

