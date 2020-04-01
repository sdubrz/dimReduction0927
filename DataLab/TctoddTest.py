import numpy as np
import matplotlib.pyplot as plt


def test1():
    path = "E:\\文件\\IRC\\特征向量散点图项目\\DataLab\\AustralianSignLanguage\\tctodd1\\"
    data1 = np.loadtxt(path+"alive-1.tsd", dtype=np.float, delimiter="\t")
    data2 = np.loadtxt(path+"alive-2.tsd", dtype=np.float, delimiter="\t")
    data3 = np.loadtxt(path+"alive-3.tsd", dtype=np.float, delimiter="\t")
    (n1, m1) = data1.shape
    (n2, m2) = data2.shape
    (n3, m3) = data3.shape

    data = np.zeros((n1+n2+n3, m1))
    data[0:n1, :] = data1[:, :]
    data[n1:n1+n2, :] = data2[:, :]
    data[n1+n2:n1+n2+n3, :] = data3[:, :]

    label = []
    for i in range(0, n1):
        label.append(1)
    for i in range(0, n2):
        label.append(2)
    for i in range(0, n3):
        label.append(3)

    np.savetxt(path+"data.csv", data, fmt='%f', delimiter=",")
    np.savetxt(path+"label.csv", label, fmt='%d', delimiter=",")


if __name__ == '__main__':
    test1()
