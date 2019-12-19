import numpy as np

"""
    使用Floyed-Warshall算法计算最短路径距离
    输入是一个初始的距离矩阵
    @author: sdu_brz
    @date: 2019/02/27
"""


def floyed(distance0):
    """
    计算所有顶点对的最短路径距离
    :param distance0: 初始的距离矩阵
    :return: 保存了最短路径长度的矩阵
    """
    data_shape = distance0.shape
    n = data_shape[0]

    distance = distance0.copy()

    for index in range(0, n):
        for i in range(0, n):
            for j in range(0, n):
                if distance[i, index] < float('inf') and distance[index, j] < float('inf') and distance[i, index] + \
                        distance[index, j] < distance[i, j]:
                    distance[i, j] = distance[i, index] + distance[index, j]

    return distance


def run_test():
    M = np.array([[0, 3, 8, float('inf'), -4],
                  [float('inf'), 0, float('inf'), 1, 7],
                  [float('inf'), 4, 0, float('inf'), float('inf')],
                  [2, float('inf'), -5, 0, float('inf')],
                  [float('inf'), float('inf'), float('inf'), 6, 0]])
    distance = floyed(M)
    print(distance)


if __name__ == '__main__':
    run_test()
