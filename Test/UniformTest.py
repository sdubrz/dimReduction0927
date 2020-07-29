# 统计一些数字 2020.07.28
import numpy as np
import matplotlib.pyplot as plt


# path = "E:\\文件\\IRC\\特征向量散点图项目\\result2020\\result0119_withoutnormalize\\MDS\\plane\\yita(0.20200727)nbrs_k(9)method_k(70)numbers(3)_b-spline_weighted\\"
path = "E:\\文件\\IRC\\特征向量散点图项目\\result2020\\result0119_withoutnormalize\\MDS_random\\plane\\yita(0.20200727)nbrs_k(9)method_k(70)numbers(3)_b-spline_weighted\\"


def vector_length():
    Y = np.loadtxt(path+"y.csv", dtype=np.float, delimiter=",")
    Y1 = np.loadtxt(path+"y_add_1.csv", dtype=np.float, delimiter=",")
    Y2 = np.loadtxt(path+"y_add_2.csv", dtype=np.float, delimiter=",")
    X = np.loadtxt(path+"x.csv", dtype=np.float, delimiter=",")
    (n, m) = Y.shape

    dY1 = Y1 - Y
    dY2 = Y2 - Y

    border = np.zeros((n, 1))  # 记录每一个点是否是边缘上的点
    for i in range(0, n):
        if X[i, 0] > 9.5 or X[i, 0] < -9.5 or X[i, 1] > 9.5 or X[i, 1] < -9.5:
            border[i] = 1

    border_count = np.sum(border)
    print("总共有 " + str(n) + " 个点， 其中 " + str(border_count) + " 个边缘点")
    n_center = int(n - border_count)  # 非边缘点的个数

    length1 = np.zeros((n_center, 1))
    length2 = np.zeros((n_center, 1))
    temp_index = 0
    for i in range(0, n):
        if border[i] == 0:
            length1[temp_index] = np.linalg.norm(dY1[i, :])
            length2[temp_index] = np.linalg.norm(dY2[i, :])
            temp_index = temp_index + 1

    print("第一个轴的平均长度 ", np.mean(length1))
    print("第二个轴的平均长度 ", np.mean(length2))

    plt.hist(length1)
    plt.title("first vector-projected length")
    plt.show()

    plt.hist(length2)
    plt.title("second vector-projected length")
    plt.show()

    angles0 = np.loadtxt(path+"angles_v1_v2_projected.csv", dtype=np.float, delimiter=",")
    angles = np.zeros((n_center, 1))
    temp_index = 0
    for i in range(0, n):
        if border[i] == 0:
            angles[temp_index] = angles0[i]
            temp_index = temp_index + 1

    print("平均的夹角 ", np.mean(angles))
    plt.hist(angles)
    plt.title("angles between two vectors")
    plt.show()

    mean_angle = np.mean(angles)
    temp = 0
    for i in range(0, n_center):
        temp = temp + (angles[i] - mean_angle)**2
    temp = temp / n_center
    print("夹角的方差为 ", temp)


if __name__ == '__main__':
    vector_length()
