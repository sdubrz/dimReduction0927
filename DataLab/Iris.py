# 对Iris数据集的操作
import numpy as np
from Main import Preprocess


def image_data():
    """
    生成一个图片数据
    :return:
    """
    path = "E:\\文件\\IRC\\特征向量散点图项目\\result2020\\result0119\\datasets\\Iris3\\"
    data = np.loadtxt(path + "data.csv", dtype=np.float, delimiter=",")
    data = Preprocess.normalize(data, 0, 255)
    (n, m) = data.shape

    origin = np.zeros((n, 784))
    for i in range(0, n):
        a = np.zeros((28, 28))
        a[0:14, 0:14] = data[i, 0]
        a[0:14, 14:28] = data[i, 1]
        a[14:28, 0:14] = data[i, 2]
        a[14:28, 14:28] = data[i, 3]
        origin[i, :] = a.reshape((1, 784))

    np.savetxt(path+"origin.csv", origin, fmt='%f', delimiter=",")


if __name__ == '__main__':
    image_data()

