# Monkey Saddle dataset
# http://mathworld.wolfram.com/MonkeySaddle.html
import numpy as np
import matplotlib.pyplot as plt
import random
import math


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


def monkey_saddle():
    path = "E:\\Project\\DataLab\\MonekySaddle\\"

    max_fail = 8000  # 最大失败次数
    points = []
    max_d = 0.12

    loop_count = 0
    while loop_count < max_fail:
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        z = math.pow(x, 3) - 3*x*y*y
        temp_point = [x, y, z]
        if all_far(points, temp_point, radius=max_d):
            points.append(temp_point)
            loop_count = 0
            if len(points) % 1000 == 0:
                print(len(points))
        else:
            loop_count += 1

    n = len(points)
    print(n)
    np.savetxt(path+"data.csv", np.array(points), fmt='%f', delimiter=",")
    np.savetxt(path+"label.csv", np.ones((n, 1)), fmt='%d', delimiter=",")


if __name__ == '__main__':
    monkey_saddle()
