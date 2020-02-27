# 两个相交的平面  作为一个三维的例子
# 投影之后使投影平面更靠近其中一个平面
import numpy as np
import random


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


def make_data():
    """
    制作两个平面相交的数据
    :return:
    """
    path = "E:\\文件\\IRC\\特征向量散点图项目\\做图\\VisFigures\\CrossPlanes\\"
    max_fail = 3000  # 最大失败次数
    points1 = []
    points2 = []

    loop_count = 0
    while loop_count < max_fail:
        temp_x = random.uniform(-0.5, 0.5)
        temp_y = random.uniform(-0.5, 0.5)
        p = [temp_x, temp_y]
        if all_far(points1, p, radius=0.05):
            points1.append(p)
            loop_count = 0
            if len(points1) % 1000 == 0:
                print(len(points1))
        else:
            loop_count = loop_count + 1

    n1 = len(points1)
    print("第一个平面中的点数为", n1)
    data1 = np.array(points1)

    loop_count = 0
    while loop_count < max_fail:
        temp_x = random.uniform(-0.5, 0.5)
        temp_y = random.uniform(-0.5, 0.5)
        p = [temp_x, temp_y]
        if all_far(points2, p, radius=0.05):
            points2.append(p)
            loop_count = 0
            if len(points2) % 1000 == 0:
                print(len(points2))
        else:
            loop_count = loop_count + 1
    n2 = len(points2)
    print("第二个平面中的点数为", n2)
    data2 = np.array(points2)

    X0 = np.zeros((n1+n2, 3))  # 用于画原始数据的
    X0[0:n1, 0:2] = data1[:, :]
    # X0[0:n1, 2] = 0.5
    X0[n1:n1+n2, 1:3] = data2[:, :]
    # X0[n1:n1 + n2, 0] = 0.5
    np.savetxt(path+"X0.csv", X0, fmt='%f', delimiter=",")

    label = np.ones((n1+n2, 1))
    label[n1:n1+n2] = 2
    np.savetxt(path+"label.csv", label, fmt='%d', delimiter=",")

    # 进行旋转
    alpha = np.pi / 11
    cos0 = np.cos(alpha)
    sin0 = np.sin(alpha)
    X = np.zeros((n1+n2, 3))
    X[0:n1, 0] = X0[0:n1, 0] * cos0
    X[0:n1, 1] = X0[0:n1, 1]
    X[0:n1, 2] = X0[0:n1, 0] * sin0
    X[n1:n1+n2, 0] = X0[n1:n1+n2, 2] * sin0 * (-1)
    X[n1:n1+n2, 1] = X0[n1:n1+n2, 1]
    X[n1:n1+n2, 2] = X0[n1:n1+n2, 2] * cos0

    np.savetxt(path+"data.csv", X, fmt='%f', delimiter=",")


if __name__ == '__main__':
    make_data()
