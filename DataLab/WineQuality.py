import numpy as np
import matplotlib.pyplot as plt


def red_data():
    path = "E:\\文件\\IRC\\特征向量散点图项目\\DataLab\\Winequality\\"
    origin = np.loadtxt(path+"winequality-white.csv", dtype=np.float, delimiter=";")
    (n, m) = origin.shape
    path = path + "white\\"

    data = origin[:, 0:m-1]

    X = []
    label = []
    for i in range(0, n):
        if i % 11 == 0:
            X.append(data[i, :].tolist())
            label.append(origin[i, m-1])

    np.savetxt(path+"data.csv", np.array(X), fmt='%f', delimiter=",")
    np.savetxt(path+"label.csv", label, fmt='%d', delimiter=",")


if __name__ == '__main__':
    red_data()

