# 处理IsomapFace数据集
import numpy as np


def sub_sample():
    """
    将图片变为8×8的
    :return:
    """
    path = "E:\\文件\\IRC\\特征向量散点图项目\\DataLab\\IsomapFace\\"
    data0 = np.loadtxt(path+"data0.csv", dtype=np.float, delimiter=",")
    origin = data0.T
    (n, m) = origin.shape

    np.savetxt(path+"origin.csv", origin, fmt='%f', delimiter=",")
    data = np.zeros((n, 64))

    # 将64×64变为8×8
    # 相当于把原来的8×8变为1个像素，每个8×8的网格中间的4个取平均
    for i in range(0, n):
        face = np.reshape(origin[i, :], (64, 64))
        face2 = np.zeros((8, 8))

        for row in range(0, 8):
            for col in range(0, 8):
                mat = face[8*row:8*row+8, 8*col:8*col+8]
                face2[row, col] = np.mean(mat[3:5, 3:5])

        data[i, :] = np.reshape(face2, (1, 64))

    np.savetxt(path+"data.csv", data, fmt="%f", delimiter=",")
    np.savetxt(path+"label.csv", np.ones((n, 1)), fmt='%d', delimiter=",")


def origin2():
    path = "E:\\文件\\IRC\\特征向量散点图项目\\DataLab\\IsomapFace\\"
    origin0 = np.loadtxt(path+"origin.csv", dtype=np.float, delimiter=",")
    np.savetxt(path+"origin2.csv", 255*origin0, fmt='%d', delimiter=",")


def test():
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(a)
    b = np.reshape(a, (1, 9))
    print(b)
    c = np.reshape(b, (3, 3))
    print(c)
    print(np.mean(a))


if __name__ == '__main__':
    # test()
    # sub_sample()
    origin2()
