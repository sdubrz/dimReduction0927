# 在降维结果中，画某种属性的colormap
import numpy as np
import matplotlib.pyplot as plt


def draw_colormap():
    """
    根据降维结果，画colormap
    :return:
    """
    path = "E:\\文件\\IRC\\特征向量散点图项目\\result2020\\result0119\\datasets\\Iris3\\"
    data = np.loadtxt(path+"data.csv", dtype=np.float, delimiter=",")
    Y = np.loadtxt(path+"MDS.csv", dtype=np.float, delimiter=",")

    attr = 3
    plt.scatter(Y[:, 0], Y[:, 1], c=data[:, attr])
    plt.title(str(attr))
    plt.colorbar()

    ax = plt.gca()
    ax.set_aspect(1)
    plt.show()


if __name__ == '__main__':
    draw_colormap()


