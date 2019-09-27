import numpy as np
import numpy.linalg as LA


"""
自己实现的一些小的排序方法
"""


def matrix_insert_sort(data):
    """
    一个关于矩阵的插入排序，防止计算的k近邻不是严格按照距离递增的顺序排列的
    因为已经基本上按照距离递增的顺序排好序，这个排序只是起一个保险的作用
    插入排序在最好的情况下的时间复杂度为O(n)，所以这种情况下，插入排序比较适合
    本函数会修改输入的数据矩阵
    :param data: 对矩阵的每一行按照从小到大的顺序继续排序
    :return:
    """
    data_shape = data.shape
    n = data_shape[0]
    dim = data_shape[1]

    for row_index in range(0, n):
        for i in range(1, dim):
            j = i
            while j > 0:
                if data[row_index, j] < data[row_index, j-1]:
                    temp = data[row_index, j]
                    data[row_index, j] = data[row_index, j-1]
                    data[row_index, j-1] = temp
                    j = j-1
                else:
                    break
    return data


def matrix_descend_insert_sort(data):
    """
    对矩阵的每一行进行降序排序， 在原始的数组上进行排序
    对于已经差不多排好序的输入而言，插入排序的耗时是 O(n) 的，性能比较快
    :param data: 待排序的数组
    :return:
    """
    data_shape = data.shape
    n = data_shape[0]
    dim = data_shape[1]

    for row_index in range(0, n):
        for i in range(1, dim):
            j = i
            while j > 0:
                if data[row_index, j] > data[row_index, j-1]:
                    temp = data[row_index, j]
                    data[row_index, j] = data[row_index, j-1]
                    data[row_index, j-1] = temp
                    j = j-1
                else:
                    break
    return data


def points_sort(points):
    """
    将points中的点按照顺时针或逆时针的顺序调整，points的最后一个点是中心点
    用角度的方式来进行排序
    :param points: 需要排序的点，最后一个点是中心点，不需要进行改变，里面存储的是二维的点
    :return:
    """
    n = len(points)
    center_point = points[n-1, :]
    vectors = np.zeros((n-1, 2))
    rank = []
    angles = []
    for i in range(0, n-1):
        vectors[i, :] = points[i, :]-center_point
        rank.append(i)
        if LA.norm(vectors[i, :]) == 0:
            angles.append(0)
            continue

        temp_cos = vectors[i, 0] / LA.norm(vectors[i, :])
        if vectors[i, 1] >= 0:
            angles.append(np.arccos(temp_cos))
        else:
            angles.append(2*np.pi-np.arccos(temp_cos))

    for i in range(1, n-1):  # 这里n-1就是angles的长度
        index = i
        # print("i = ", i)
        while index > 0:
            # print("index= ", index)
            if angles[index] < angles[index-1]:
                temp = angles[index]
                temp_rank = rank[index]
                angles[index] = angles[index-1]
                rank[index] = rank[index-1]
                angles[index-1] = temp
                rank[index-1] = temp_rank
                index = index-1
            else:
                break

    sorted_points = np.zeros((n-1, 2))
    for i in range(0, n-1):
        sorted_points[i, :] = vectors[rank[i], :] + center_point

    return sorted_points


if __name__ == '__main__':
    points = np.array([[4, 6],
                       [2, 5],
                       [6, 5],
                       [7, 3],
                       [6, 2],
                       [2, 1],
                       [5, 3]])
    vectors = points_sort(points)
    print(vectors)
