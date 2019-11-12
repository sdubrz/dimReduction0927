# 采样算法研究
# 目前主要针对蓝噪声采样
import numpy as np
import matplotlib.pyplot as plt
import random
from JSON_Data import Json_2d
from Dim2LocalPCA import LocalPCA_2dim


def random_sample():
    """
    直接用随机数生成器
    虽然说是均匀的，但是看生成的效果来看更倾向于随机
    :return:
    """
    path = "E:\\Project\\result2019\\samplingTest\\"
    n = 3495
    y = np.zeros((n, 2))
    for i in range(0, n):
        y[i, 0] = random.uniform(0, 1)
        y[i, 1] = random.uniform(0, 1)

    np.savetxt(path+"random_data.csv", y, fmt="%f", delimiter=",")

    plt.scatter(y[:, 0], y[:, 1])
    ax = plt.gca()
    ax.set_aspect(1)
    plt.show()


def all_far(points, point, radius=0.01):
    """
    判断新加入的点是否满足距离points中的每一个点的距离都大于 radius
    :param points: 当前的点集
    :param point: 新的点
    :param radius: 半径
    :return:
    """
    if len(points) == 0:
        return True
    for p in points:
        s = 0
        for i in range(0, len(point)):
            s = s + (p[i] - point[i]) * (p[i] - point[i])
        if s < radius * radius:
            return False

    return True


def darts():
    """
    darts
    by Cook
    :return:
    """
    path = "E:\\Project\\result2019\\samplingTest\\"
    max_fail = 3000  # 最大失败次数
    points = []

    loop_count = 0
    while loop_count < max_fail:
        temp_x = random.uniform(0, 1)
        temp_y = random.uniform(0, 1)
        p = [temp_x, temp_y]
        if all_far(points, p, radius=0.02):
            points.append(p)
            loop_count = 0
            if len(points) % 1000 == 0:
                print(len(points))
        else:
            loop_count = loop_count + 1

    print("公共的点数是： ", len(points))

    X = np.array(points)
    np.savetxt(path + "blue_noise.csv", X, fmt="%f", delimiter=",")
    plt.scatter(X[:, 0], X[:, 1])
    ax = plt.gca()
    ax.set_aspect(1)
    plt.show()


def dart_two_plane():
    """
    用飞镖法生成两个平面的数据
    :return:
    """
    path = "E:\\Project\\result2019\\samplingTest\\darts_2plane\\"
    max_fail = 3000  # 最大失败次数
    points = []

    loop_count = 0
    while loop_count < max_fail:
        temp_x = random.uniform(0, 1)
        temp_y = random.uniform(0, 1)
        p = [temp_x, temp_y]
        if all_far(points, p, radius=0.02):
            points.append(p)
            loop_count = 0
            if len(points) % 1000 == 0:
                print(len(points))
        else:
            loop_count = loop_count + 1

    print("第一个平面： ", len(points))

    points2 = []
    loop_count = 0
    while loop_count < max_fail:
        temp_x = random.uniform(0, 1)
        temp_y = random.uniform(0, 1)
        p = [temp_x, temp_y]
        if all_far(points2, p, radius=0.02):
            points2.append(p)
            loop_count = 0
            if len(points2) % 1000 == 0:
                print(len(points2))
        else:
            loop_count = loop_count + 1

    print("第二个平面： ", len(points2))

    n = len(points) + len(points2)
    X = np.zeros((n, 3))
    X1 = np.array(points)
    X2 = np.array(points2)

    X[0:len(points), 0:2] = X1[:, :]
    X[len(points):n, 1:3] = X2[:, :]
    X[0:len(points), 2] = 0.5
    X[len(points):n, 0] = 0.5
    np.savetxt(path+"data.csv", X, fmt="%f", delimiter=",")
    label = np.ones((n, 1))
    label[len(points):n, 0] = 2
    np.savetxt(path+"label.csv", label, fmt="%d", delimiter=",")


def compare_pca():
    """
    比较二者的local PCA
    :return:
    """
    path1 = "E:\\Project\\result2019\\samplingTest\\blue_noise\\"
    path2 = "E:\\Project\\result2019\\samplingTest\\random_data\\"

    Json_2d.draw_b_spline(path=path2, k=60, line_length=0.015, draw=True)
    # Json_2d.draw_oval(path=path1, k=15, line_length=0.015, draw=True)


if __name__ == '__main__':
    # random_sample()
    # darts()
    # compare_pca()
    dart_two_plane()