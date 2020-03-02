# 计算B-Spline闭合曲线的长和宽
import numpy as np
import math


def spline_length_width(spline_list):
    """
    计算每个闭合B-Spline曲线的长度，宽度，以及长宽比
    :param spline_list:
    :return:
    """
    n = len(spline_list)
    lengths = np.zeros((n, 1))  # 每个曲线的长度
    widths = np.zeros((n, 1))  # 每个曲线的宽度
    radios = np.ones((n, 1))  # 每个曲线的宽度除以长度

    index = 0
    for current in spline_list:
        n_points = len(current)
        length_list = []
        half = n_points // 2
        for i in range(0, half):
            point1 = current[i]
            point2 = current[i+half]
            dx = point1[0] - point2[0]
            dy = point1[1] - point2[1]
            d = math.sqrt(dx*dx + dy*dy)
            length_list.append(d)

        current_length = max(length_list)
        current_width = min(length_list)

        lengths[index] = current_length
        widths[index] = current_width
        if current_length != 0:
            radios[index] = current_length / current_width

        index = index + 1

    return lengths, widths, radios



