# 检查是否有重复的点
import numpy as np
from sklearn.metrics import euclidean_distances


def has_repeat(X, label, path=None):
    """
    检查X中是否有重复的点
    :param X: 数据矩阵
    :param label: 数据的标签
    :return:
    """
    (n, m) = X.shape
    D = euclidean_distances(X)

    repeat = False
    repeat_index = []
    for i in range(0, n):
        for j in range(0, n):
            if i == j:
                continue
            else:
                if D[i, j] == 0:
                    repeat_index.append(max(i, j))
                    repeat = True

    if repeat:
        print("repeat index ", repeat_index)
        count = np.zeros((n, 1))
        for index in repeat_index:
            count[index] = 1
        number = int(np.sum(count))

        data2 = np.zeros((n-number, m))
        label2 = np.zeros((n-number, 1))
        j = 0
        for i in range(0, n):
            if count[i] == 0:
                data2[j, :] = X[i, :]
                label2[j] = label[i]
                j += 1

        if not path is None:
            np.savetxt(path+"data2.csv", data2, fmt='%f', delimiter=",")
            np.savetxt(path+"label2.csv", label2, fmt='%d', delimiter=",")

    return repeat


def check_repeat2(path):
    data = np.loadtxt(path+"data.csv", dtype=np.str, delimiter=",")
    (n, m) = data.shape

    for i in range(0, n-1):
        for j in range(i+1, n):
            temp = True
            for k in range(0, m):
                if data[i, k] != data[j, k]:
                    temp = False
                    break
            if temp:
                print((i, j))


if __name__ == '__main__':
    path = "E:\\文件\\IRC\\特征向量散点图项目\\result2020\\result0119_withoutnormalize\\datasets\\TravelReviews\\"
    check_repeat2(path)

