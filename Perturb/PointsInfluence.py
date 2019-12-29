# 计算点对降维算法的影响力和相对影响力
import numpy as np
import matplotlib.pyplot as plt


def influence(Y0, Y, index, eta):
    """
    计算某个点的影响力
    具体计算方式是 其他所有点的平均降维结果改变量，除以这个点的扰动向量长度
    :param Y0: 没有扰动的降维结果矩阵
    :param Y: 对index 这个点进行扰动后的降维结果
    :param index: 被扰动的点的索引号
    :param eta: 对 index 这个点所加的扰动向量的长度
    :return:
    """
    if eta == 0:
        return 0
    (n, m) = Y.shape
    dY = Y - Y0
    s = 0

    for i in range(0, n):
        if i == index:
            continue
        s = s + np.linalg.norm(dY[i, :])

    s = s/(n-1)
    s = s / eta
    return s


def relative_influence(Y0, Y, index):
    """
    计算某个点的相对影响力
    具体计算方式为其他所有点的降维结果改变量除以当前点的降维结果改变量
    :param Y0: 没有扰动时的降维结果
    :param Y: 对 index 添加扰动后的降维结果
    :param index: 被扰动的点的索引号
    :return:
    """
    (n, m) = Y.shape
    dY = Y - Y0

    d_i = np.linalg.norm(dY[index, :])
    if d_i == 0:
        return 0

    s = 0
    for i in range(0, n):
        if i == index:
            continue
        s = s + np.linalg.norm(dY[i, :])
    s = s/(n-1)
    s = s / d_i

    return s
