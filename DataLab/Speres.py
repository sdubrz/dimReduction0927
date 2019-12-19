import numpy as np
import matplotlib.pyplot as plt
import random
import math


"""
" 制造三维球形数据
"""


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


def sphere():
    """
    用飞镖法制造符合蓝噪声采样的球形数据
    :return:
    """
    r = 0.7
    center = [0, 0, 0]
    path = "E:\\Project\\DataLab\\twoSpheres\\"

    max_fail = 8000  # 最大失败次数
    points = []
    max_d = 0.11

    loop_count = 0
    while loop_count < max_fail:
        sita = 2*math.pi*random.uniform(0, 1)
        alpha = (random.uniform(0, 1)-0.5)*math.pi
        temp_x = r * math.cos(alpha) * math.cos(sita)
        temp_y = r * math.cos(alpha) * math.sin(sita)
        temp_z = r * math.sin(alpha)
        temp_point = [temp_x, temp_y, temp_z]

        if all_far(points, temp_point, radius=max_d):
            points.append(temp_point)
            loop_count = 0
            if len(points) % 1000 == 0:
                print(len(points))
        else:
            loop_count += 1

    np.savetxt(path+"smalldata.csv", np.array(points), fmt='%f', delimiter=",")
    print(len(points))


if __name__ == '__main__':
    sphere()
