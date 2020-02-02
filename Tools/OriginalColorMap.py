# 根据数据原始的维度属性值，在降维结果散点图中用colormap表示数据的某个原始维度的值的分布情况
import numpy as np
import matplotlib.pyplot as plt


def color_scatter():
    """
    用颜色表示维度值
    :return:
    """
    # path = "E:\\文件\\IRC\\特征向量散点图项目\\result2020\\result0119_withoutnormalize\\datasets\\olive\\"
    path = "E:\\文件\\IRC\\特征向量散点图项目\\result2020\\result0119\\datasets\\seeds\\"
    attr = 5  # 第几个维度

    data = np.loadtxt(path+"data.csv", dtype=np.float, delimiter=",")
    Y = np.loadtxt(path+"PCA.csv", dtype=np.float, delimiter=",")  # 文件名由降维方法确定

    plt.scatter(Y[:, 0], Y[:, 1], c=data[:, attr])
    plt.title(str(attr))
    plt.colorbar()
    ax = plt.gca()
    ax.set_aspect(1)
    plt.show()


if __name__ == "__main__":
    color_scatter()


