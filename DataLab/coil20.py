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


def select3():
    """
    从数据中抽选出3类
    :return:
    """
    path1 = "E:\\文件\\IRC\\特征向量散点图项目\\DataLab\\coil20obj\\"
    path2 = "E:\\文件\\IRC\\特征向量散点图项目\\DataLab\\coil20obj\\coil16m1413\\"
    data = np.loadtxt(path1+"Y16.csv", dtype=np.float, delimiter=",")
    origin = np.loadtxt(path1+"origin.csv", dtype=np.int, delimiter=",")
    (n, m) = data.shape
    (n, m2) = origin.shape

    indexs = [1, 4, 13]
    data2 = np.zeros((72*len(indexs), m))
    label2 = np.zeros((72*len(indexs), 1))
    origin2 = np.zeros((72*len(indexs), m2))

    for i in range(0, len(indexs)):
        obj = indexs[i]
        data2[i*72:i*72+72, :] = data[obj*72-72:obj*72, :]
        label2[i*72:i*72+72] = i+1
        origin2[i*72:i*72+72, :] = origin[obj*72-72:obj*72, :]

    np.savetxt(path2+"data.csv", data2, fmt='%f', delimiter=",")
    np.savetxt(path2+"label.csv", label2, fmt='%d', delimiter=",")
    np.savetxt(path2+"origin.csv", origin2, fmt='%d', delimiter=",")

    print("finished")


if __name__ == '__main__':
    # run1()
    select3()

