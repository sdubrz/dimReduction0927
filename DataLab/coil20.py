# 对coil20数据进行处理
import numpy as np


def run1():
    """
    把预处理做得更狠一点
    :return:
    """
    path = "E:\\文件\\IRC\\特征向量散点图项目\\DataLab\\coil20obj_16_3class\\"
    data = np.loadtxt(path+"data.csv", dtype=np.float, delimiter=",")

    data2 = data[:, 0:10]
    np.savetxt(path+"data2.csv", data2, fmt='%f', delimiter=",")


if __name__ == '__main__':
    run1()

