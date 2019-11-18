# 对每个维度都要生成一个 Heatmap
# 实际上是一个对空间的划分，然后插值

import numpy as np
import matplotlib.pyplot as plt

""""
    所有需要计算的属性有
        Heatmap区域左上方格子的左下顶点坐标
        每个格子的长度
        横向的格子数
        纵向的格子数
        对应着m个属性的m个二维数组

"""


def square_size(Y):
    """
    计算Heatmap方格的大小，横纵向网格数，以及左下方网格的左下方顶点坐标
    我们的系统中，图的横向宽度是确定的，所以可以先按照横向来划分一个。
    :param Y: 降维后的二维坐标数据
    :return:
    """
    (n_points, dim) = Y.shape
    x_max = np.max(Y[:, 0])
    x_min = np.min(Y[:, 0])
    y_max = np.max(Y[:, 1])
    y_min = np.min(Y[:, 1])

    x_length = (x_max - x_min) * 1.2
    y_length = (y_max - y_min) * 1.2

    y_left_low = y_min - 0.1 * (y_max - y_min)
    x_left_low = x_min - 0.1 * (x_max - x_min)
    left_low = (x_left_low, y_left_low)

    m = 25  # 横向的格子数
    r = y_length / m
    n = x_length / r + 1  # 纵向的格子数，因为一般都不会被整除，所以加一
    square_number = (n, m)

    return left_low, square_number, r


def heatmap(path=""):
    """
    计算各个属性的 heatmap
    :param path: 读取数据的文件路径
    :return:
    """
    X = np.loadtxt(path+"x.csv", dtype=np.float, delimiter=",")
    Y = np.loadtxt(path+"y.csv", dtype=np.float, delimiter=",")
    (n_points, dim) = X.shape

    left_low, square_number, r = square_size(Y)

    # 判断每个方格内都有哪些点


def run_test():
    path = "E:\\Project\\result2019\\result1026without_straighten\\PCA\\Wine\\yita(0.05)nbrs_k(40)method_k(40)numbers(4)_b-spline_weighted\\"


def run_test2():
    n = 5
    m = 6


if __name__ == '__main__':
    run_test2()
