# 删除数据中重复的数据
import numpy as np
from sklearn.metrics import euclidean_distances


def remove_repeat(path):
    """
    检查数据中是否有重复的数据，如果有，删掉重复的数据，生成一个新的数据
    :param path: 存放数据的文件目录
    :return:
    """
    data = np.loadtxt(path+"data.csv", dtype=np.float, delimiter=",")
    label = np.loadtxt(path+"label.csv", dtype=np.int, delimiter=",")

    (n, m) = data.shape
    D = euclidean_distances(data)
    repeat = []

    for i in range(0, n-1):
        for j in range(i+1, n):
            if D[i, j] == 0.0 and not(j in repeat):
                print((i, j))
                repeat.append(j)

    if len(repeat) == 0:
        print("没有重复数据")
        return
    else:
        print("共有 "+str(len(repeat))+" 个重复数据")

    data2 = np.zeros((n-len(repeat), m))
    label2 = np.zeros((n-len(repeat), 1))
    index = 0
    for i in range(0, n):
        if i in repeat:
            continue
        else:
            data2[index, :] = data[i, :]
            label2[index] = label[i]
            index = index + 1

    np.savetxt(path+"data2.csv", data2, fmt='%f', delimiter=",")
    np.savetxt(path+"label2.csv", label2, fmt='%d', delimiter=",")


def run_test():
    main_path = "E:\\文件\\IRC\\特征向量散点图项目\\result2020\\result0119\\datasets\\"
    data_name = "ImageSegmentation_repeat"
    path = main_path + data_name + "\\"
    remove_repeat(path)


if __name__ == '__main__':
    run_test()
