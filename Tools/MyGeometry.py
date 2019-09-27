import numpy as np


"""
    一些几何的算法，包括：
    （1）根据三角形三个顶点的坐标，计算三角形的面积；
    （2）根据凸多边形的周围顶点和凸多边形内部的一个顶点计算其面积；
    （3）计算不规则的多边形的面积，也就是star_shape

    @author: sdu_brz
    @date: 2019/01/30
"""


def triangle_area(point_a, point_b, point_c):
    """
    根据三角形三个顶点计算三角形的面积
    :param point_a:
    :param point_b:
    :param point_c:
    :return:
    """
    temp1 = point_a[0]*(point_b[1]-point_c[1])
    temp2 = point_b[0]*(point_c[1]-point_a[1])
    temp3 = point_c[0]*(point_a[1]-point_b[1])

    s = np.abs(0.5*(temp1+temp2+temp3))
    return s


def convex_area(convex):
    """
    根据凸多边形的顶点以及内部的一个点，计算其面积。
    使用的方法是将其分割成一个一个的小三角形
    要求这个多边形的顶点是按照顺时针或逆时针顺序拍好了的
    :param convex: 凸多边形的顶点坐标
    :return: 凸多边形的面积
    """
    s = 0
    data_shape = convex.shape
    n = data_shape[0]

    if n < 3:
        return 0

    for i in range(1, n-1):
        s = s + triangle_area(convex[i, :], convex[i+1, :], convex[0, :])

    return s


def centrosymmetry_area(polygon, center):
    """
    计算中心对称的非凸多边形的面积
    也可以兼容凸多边形的情况
    :param polygon: 非凸多边形的顶点坐标，是按照顺时针或逆时针顺序排好了的
    :param center: 中心点，多边形关于该点中心对称
    :return:
    """
    s = 0
    data_shape = polygon.shape
    n = data_shape[0]

    if n < 3:
        print("小于3")
        return 0

    for i in range(0, n-1):
        s = s + triangle_area(polygon[i, :], polygon[i+1, :], center)

    s = s + triangle_area(polygon[n-1, :], polygon[0, :], center)

    return s


if __name__ == "__main__":
    point1 = [0, 1]
    point2 = [1, 0]
    point3 = [0, 0]
    s = triangle_area(point1, point2, point3)
    print(s)

    convex_list = [[0, 0],
                   [0, 1],
                   [1, 1],
                   [1, 0]]
    convex = np.array(convex_list)
    s2 = convex_area(convex)
    print(s2)
