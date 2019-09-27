import numpy as np

"""
    对LLE等算法出现的扰动后横纵坐标反向的问题进行校正
    @author: sdu_brz
    @date: 2019/03/03
"""


def symbol_adjust(y0, y1):
    """
    以y0为基准对y1进行校正
    因为添加扰动之后使用LLE等方法可能会出现横坐标或纵坐标出现反转的情况
    这里使用计算距离的方式进行校正
    :param y0: 原始数据的降维结果
    :param y1: 添加扰动之后的降维结果
    :return:
    """
    p2 = np.array([[-1, 0],
                   [0, 1]])
    p3 = np.array([[1, 0],
                   [0, -1]])
    p4 = np.array([[-1, 0],
                   [0, -1]])

    y2 = np.matmul(y1, p2)
    y3 = np.matmul(y1, p3)
    y4 = np.matmul(y1, p4)

    distance1 = average_distance(y0, y1)
    distance2 = average_distance(y0, y2)
    distance3 = average_distance(y0, y3)
    distance4 = average_distance(y0, y4)

    if distance1 <= distance2 and distance1 <= distance3 and distance1 <= distance4:
        return y1
    if distance2 <= distance1 and distance2 <= distance3 and distance2 <= distance4:
        return y2
    if distance3 <= distance1 and distance3 <= distance2 and distance3 <= distance4:
        return y3
    return y4


def average_distance(y0, y1):
    """
    计算两个矩阵的对应的行向量的距离平均值
    :param y0:
    :param y1:
    :return:
    """
    dy = y1 - y0
    data_shape = y0.shape
    n = data_shape[0]

    distance = np.zeros((n, 1))
    for i in range(0, n):
        distance[i] = np.linalg.norm(dy[i, :])
    return np.mean(distance)

