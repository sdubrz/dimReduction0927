# 用亚采样的方法处理MNIST-digit数据
import numpy as np
import os

def sub_sample():
    """
    用亚采样的方法处理MNIST-digit数据
    :return:
    """
    path = "E:\\文件\\IRC\\特征向量散点图项目\\DataLab\MNIST\\"
    total = np.loadtxt(path+"origin\\data.csv", dtype=np.int, delimiter=",")
    label = np.loadtxt(path+"origin\\label.csv", dtype=np.int, delimiter=",")

    (n, m) = total.shape
    data = np.zeros((n, 49))
    for i in range(0, n):
        current = np.reshape(total[i, :], (28, 28))
        image = np.zeros((7, 7))
        for row in range(0, 7):
            for col in range(0, 7):
                temp = current[row*4:row*4+4, col*4:col*4+4]
                image[row, col] = np.mean(temp[1:3, 1:3])

        data[i, :] = np.reshape(image, (1, 49))

    np.savetxt(path+"total.csv", data, fmt='%d', delimiter=",")


def classification():
    """
    按照类别进行分类
    :return:
    """
    path = "E:\\文件\\IRC\\特征向量散点图项目\\DataLab\\MNIST\\"
    total = np.loadtxt(path+"total.csv", dtype=np.int, delimiter=",")
    label = np.loadtxt(path+"label.csv", dtype=np.int, delimiter=",")
    origin = np.loadtxt(path+"origin\\data.csv", dtype=np.int, delimiter=",")
    (n, m) = total.shape

    origin_list = []
    data_list = []
    for i in range(0, 10):
        origin_list.append([])
        data_list.append([])

    for i in range(0, n):
        origin_list[label[i]].append(origin[i, :].tolist())
        data_list[label[i]].append(total[i, :].tolist())

    for i in range(0, 10):
        current_origin = origin_list[i]
        current_data = data_list[i]
        n1 = len(current_data)

        small_data, small_origin = down_sampling(np.array(current_origin), np.array(current_data))
        (n2, m2) = small_data.shape
        path1 = path+"classification\\mnist49m"+str(i)+"\\"
        path2 = path+"classmini\\mnist49m"+str(i)+"\\"

        if not os._exists(path1):
            os.makedirs(path1)
        if not os._exists(path2):
            os.makedirs(path2)

        np.savetxt(path1+"data.csv", np.array(current_data), fmt='%d', delimiter=",")
        np.savetxt(path1+"origin.csv", np.array(current_origin), fmt='%d', delimiter=",")
        np.savetxt(path1+"label.csv", np.ones((n1, 1))*i, fmt='%d', delimiter=",")

        np.savetxt(path2+"data.csv", small_data, fmt='%d', delimiter=",")
        np.savetxt(path2+"origin.csv", small_origin, fmt='%d', delimiter=",")
        np.savetxt(path2+"label.csv", np.ones((n2, 1))*i, fmt='%d', delimiter=",")
        print(i)


def down_sampling(origin, data):
    """
    均匀采样出部分数据
    :return:
    """
    (n, m) = data.shape
    data_list = data.tolist()
    origin_list = origin.tolist()

    small_data = []
    small_origin = []

    for i in range(0, n):
        if i % 13 != 0:
            continue
        small_data.append(data_list[i])
        small_origin.append(origin_list[i])

    return np.array(small_data), np.array(small_origin)


if __name__ == '__main__':
    # sub_sample()
    classification()
