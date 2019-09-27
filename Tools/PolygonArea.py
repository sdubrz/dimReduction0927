import numpy as np


"""
    计算多边形的面积，包括非凸多边形的情况，
    要求输入的顶点是按照逆时针顺序顺次排列的。
    所要求解面积的多边形即为这些顶点一个一个地相连而成
    E = {<v0, v1>, <v1, v2>, <v2, v3>,...,<vn-2, vn-1>, <vn-1, v0>}
    @author: sdu_brz
    @date: 2019/02/18
"""


def coss_multi(v1, v2):
    """
    计算两个向量的叉乘
    :param v1:
    :param v2:
    :return:
    """
    return v1[0]*v2[1] - v1[1]*v2[0]


def polygon_area(polygon):
    """
    计算多边形的面积，支持非凸情况
    :param polygon: 多边形顶点，已经进行顺次逆时针排序
    :return: 该多边形的面积
    """
    n = len(polygon)

    if n < 3:
        return 0

    vectors = np.zeros((n, 2))
    for i in range(0, n):
        vectors[i, :] = polygon[i, :] - polygon[0, :]

    area = 0
    for i in range(1, n):
        area = area + coss_multi(vectors[i-1, :], vectors[i, :]) / 2

    return area


if __name__ == "__main__":
    """测试"""
    polygon1 = np.array([[0, 0],
                         [1, 0],
                         [1, 1],
                         [0, 1]])
    print(polygon_area(polygon1))
    polygon2 = np.array([[0, 0],
                         [5, 0],
                         [5, 4],
                         [4, 4],
                         [4, 1],
                         [1, 1],
                         [1, 7],
                         [0, 7]])
    print(polygon_area(polygon2))
