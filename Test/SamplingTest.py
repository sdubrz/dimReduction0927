# 采样算法研究
# 目前主要针对蓝噪声采样
import numpy as np
import matplotlib.pyplot as plt
import random


def random_sample():
    """
    直接用随机数生成器
    虽然说是均匀的，但是看生成的效果来看更倾向于随机
    :return:
    """
    n = 1000
    y = np.zeros((2, n))
    for i in range(0, n):
        y[0, i] = random.uniform(0, 1)
        y[1, i] = random.uniform(0, 1)

    plt.scatter(y[0, :], y[1, :])
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
    max_fail = 1000  # 最大失败次数
    points = []

    loop_count = 0
    while loop_count < max_fail:
        temp_x = random.uniform(0, 1)
        temp_y = random.uniform(0, 1)
        p = [temp_x, temp_y]
        if all_far(points, p, radius=0.013):
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


if __name__ == '__main__':
    # random_sample()
    darts()
