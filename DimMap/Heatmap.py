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

    m = 25  # 格子的列数
    r = y_length / m
    n = int(x_length / r + 1)  # 格子的行数，因为一般都不会被整除，所以加一
    square_number = (n, m)

    # 一下部分用于 debug
    # y_right_up = y_max + 0.1 * (y_max - y_min)
    # x_right_up = x_max + 0.1 * (x_max - x_min)
    # plt.scatter(Y[:, 0], Y[:, 1])
    # for i in range(0, m):
    #     plt.plot([x_left_low, x_right_up], [y_left_low+i*r, y_left_low+i*r], c='c')
    # for i in range(0, n):
    #     plt.plot([x_left_low+i*r, x_left_low+i*r], [y_left_low, y_right_up], c='c')
    # ax = plt.gca()
    # ax.set_aspect(1)
    # plt.show()

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

    # 计算每个方格对应的点
    # 每个方格对应这个一个 list， list中是这个方格对应的点的索引号
    # 如果这个list中第一个数是-1，表示这个格子中没有点，需要寻找距离这个格子最近的 3个点，取加权平均

    square_points = []  # 存储每个点中都有哪些点
    for i in range(0, square_number[0]):
        temp_list = []
        for j in range(0, square_number[1]):
            temp_list.append([-1])
        square_points.append(temp_list)

    # 计算每个点属于哪个格子
    for i in range(0, n_points):
        dx = Y[i, 0] - left_low[0]
        dy = Y[i, 1] - left_low[1]
        row_index = int(dx / r)
        column_index = int(dy / r)
        temp_list = square_points[row_index][column_index]
        if temp_list[0] == -1:
            temp_list[0] = i
        else:
            temp_list.append(i)

    # 如果某个格子中没有数据，需要寻找离它最近的 3个点，然后根据距离取加权平均
    # 寻找最近的 3个点，使用八连通向外逐步扩张的方式
    # 至少 3个，有可能会超过 3个
    for i in range(0, square_number[0]):
        for j in range(0, square_number[1]):
            current_list = square_points[i][j]
            if current_list[0] != -1:
                continue
            d_index = 1
            while len(current_list) < 4:
                row_start = max(i-d_index, 0)
                row_end = min(i+d_index, square_number[0]-1)
                column_start = max(j-d_index, 0)
                column_end = min(j+d_index, square_number[1]-1)
                temp_list2 = []
                for i2 in range(row_start, row_end+1):  # 这里有错误
                    for j2 in range(column_start, column_end+1):
                        neighbor_list = square_points[i2][j2]
                        if neighbor_list[0] != -1:
                            for point in neighbor_list:
                                temp_list2.append(point)
                if len(temp_list2) >= 3:
                    for point in temp_list2:
                        current_list.append(point)
                else:
                    d_index += 1
                    temp_list2 = []

    return square_points


def debug_square_points():
    path = "E:\\Project\\result2019\\result1026without_straighten\\PCA\\Wine\\yita(0.05)nbrs_k(40)method_k(40)numbers(4)_b-spline_weighted\\"
    square_points = heatmap(path)
    file_writer = open(path+"square_points.csv", 'w', encoding="UTF-8")
    n = len(square_points)
    m = len(square_points[0])

    for i in range(0, n):
        for j in range(0, m):
            temp_list = square_points[i][j]
            line = "(" + str(i) + ", " + str(j) + "):\t"
            for index in temp_list:
                line = line + str(index) + ", "
            line = line + '\n'
            file_writer.write(line)
    file_writer.close()
    print('存储完毕')


def run_test2():
    n = 5
    m = 6
    A = []
    for i in range(0, n):
        temp_list = []
        for j in range(0, m):
            temp_list.append([])
        A.append(temp_list)

    print(len(A))
    print(len(A[0]))

    a_list = A[0][1]
    print(a_list)
    print(len(a_list))
    a_list.append(2)
    print(A[0][1])
    print(len(a_list))


if __name__ == '__main__':
    run_test2()
    debug_square_points()
