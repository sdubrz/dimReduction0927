# Möbius Strip dataset
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


def mobius_strip():
    """
    制造符合蓝噪声采样的莫比乌斯环
    方法参见 http://mathworld.wolfram.com/MoebiusStrip.html
    """
    path = "E:\\Project\\DataLab\\MobiusStrip\\"
    w = 1
    R = 2

    max_fail = 8000  # 最大失败次数
    points = []
    max_d = 0.12

    loop_count = 0
    while loop_count < max_fail:
        s = (random.uniform(0, 1) - 0.5) * w
        t = random.uniform(0, 1) * 2 * math.pi
        temp_x = (R + s*math.cos(t/2)) * math.cos(t)
        temp_y = (R + s*math.cos(t/2)) * math.sin(t)
        temp_z = s * math.sin(t/2)
        temp_point = [temp_x, temp_y, temp_z]

        if all_far(points, temp_point, radius=max_d):
            points.append(temp_point)
            loop_count = 0
            if len(points) % 1000 == 0:
                print(len(points))
        else:
            loop_count += 1

    np.savetxt(path+"data.csv", np.array(points), fmt='%f', delimiter=",")
    print(len(points))


if __name__ == '__main__':
    mobius_strip()
