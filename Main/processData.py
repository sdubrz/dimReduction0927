import numpy as np
from numpy import linalg as LA


"""
计算一些中间结果数据
"""


def distance_change(x, x1, x2, y, y1, y2):
    """
    计算降维前后距离的变化情况
    :param x: 原始的高维数据
    :param x1: 添加了第一特征向量扰动之后的高维数据
    :param x2: 添加了第二特征向量扰动之后的高维数据
    :param y: 原始高维数据降维所得的二维数据
    :param y1: 添加了第一特征向量扰动之后降维所得的二维数据
    :param y2: 添加了第二特征向量扰动之后降维所得的二维数据
    :return:
    """
    high_v1 = x1 - x
    high_v2 = x2 - x
    low_v1 = y1 - y
    low_v2 = y2 - y

    data_shape = high_v1.shape
    n = data_shape[0]
    high_distance1 = np.zeros((n, 1))
    high_distance2 = np.zeros((n, 1))
    low_distance1 = np.zeros((n, 1))
    low_distance2 = np.zeros((n, 1))

    for i in range(0, n):
        high_distance1[i] = LA.norm(high_v1[i, :])
        high_distance2[i] = LA.norm(high_v2[i, :])
        low_distance1[i] = LA.norm(low_v1[i, :])
        low_distance2[i] = LA.norm(low_v2[i, :])

    high_di = np.zeros((n, 1))  # 高维的距离倍数矩阵
    low_di = np.zeros((n, 1))  # 低维的距离倍数矩阵
    change = 0
    error_count = 0  # 统计需要忽略的数据的数量
    for i in range(0, n):
        # 如果有距离变化为0者，需要忽略这个点
        if high_distance1[i] * low_distance1[i] * high_distance2[i] * low_distance2[i] == 0:
            error_count = error_count + 1
            continue
        high_di[i] = high_distance2[i]/high_distance1[i]
        low_di[i] = low_distance2[i]/low_distance1[i]
        change = change + low_di[i]/high_di[i]
    change = change/n

    every_distance_change = np.ones((n, 1))  # 低维的比值除以高维的比值
    for i in range(0, n):
        every_distance_change[i] = low_di[i]/high_di[i]

    print('统计距离变化倍数结束，总共被忽略的点数是', error_count)
    # 返回平均的变化倍数、高维的距离倍数和低维的距离变化倍数
    return change, high_di, low_di, every_distance_change


def angle_change(x, x1, x2, y, y1, y2):
    """
    统计降维前后沿两个特征向量扰动，角度的变化
    :param x: 原始的高维数据
    :param x1: 加了第一特征向量扰动之后的高维数据
    :param x2: 加了第二特征向量扰动之后的高维数据
    :param y: 原始数据降维之后的数据
    :param y1: 增加了第一特征向量扰动之后降维的数据
    :param y2: 增加了第二特征向量扰动之后的降维数据
    :return: 平均角度变化量
    """
    data_shape = x.shape
    n = data_shape[0]
    high_v1 = x1-x
    high_v2 = x2-x
    low_v1 = y1-y
    low_v2 = y2-y

    high_v2t = np.transpose(high_v2)
    low_v2t = np.transpose(low_v2)

    high_angle = np.zeros((n, 1))  # 高维空间中两个扰动方向的角度
    low_angle = np.zeros((n, 1))  # 低维空间中两个扰动方向的角度

    # 计算角度
    for i in range(0, n):
        temp_high_cos = 0
        temp_low_cos = 0
        error_count = 0
        # 如果某一个模为0，可以假定这个角度没有发生变化
        if LA.norm(low_v1[i, :])*LA.norm(low_v2t[:, i]) == 0:
            temp_low_cos = 1
            error_count = error_count+1
        else:
            temp_low_cos = np.dot(low_v1[i, :], low_v2t[:, i]) / (LA.norm(low_v1[i, :]) * LA.norm(low_v2t[:, i]))
        if (LA.norm(high_v1[i, :])*LA.norm(high_v2t[:, i])) == 0:
            temp_high_cos = 1
        else:
            temp_high_cos = np.dot(high_v1[i, :], high_v2t[:, i])/(LA.norm(high_v1[i, :])*LA.norm(high_v2t[:, i]))

        high_angle[i] = np.arccos(temp_high_cos)  # 高维的角度
        low_angle[i] = np.arccos(temp_low_cos)  # 低维的角度

        high_angle[i] = high_angle[i]/np.pi*180  # 将弧度制转化成度数
        low_angle[i] = low_angle[i]/np.pi*180

        if high_angle[i] > 90:
            high_angle[i] = 180-high_angle[i]  # 方向应该是双向的，所以夹角应该在[0, 90]之间
        if low_angle[i] > 90:
            low_angle[i] = 180-low_angle[i]

    change_angle = low_angle-high_angle
    mean_change = np.mean(change_angle)

    print('角度变化计算结束，误差点的总数是', error_count)
    # 返回平均角度变化量、高维空间中的角度以及在低维空间中的角度
    return mean_change, high_angle, low_angle


def weight_of_max_eigenvalue2(eigenvalues):
    """
    计算最大特征值占前两个特征值之和的比重
    要考虑特征值有负数的情况，这时候需要用其绝对值
    :param eigenvalues: 存放所有数据的localPCA的特征值，是一个n*dim的矩阵
                        其中n为数据的个数，dim是数据的维度
    :return: 一个n*1的矩阵，其值为每个数据的最大特征值占前两个特征值之和的比重
    """
    data_shape = eigenvalues.shape
    n = data_shape[0]
    dim = data_shape[1]
    weights = np.zeros((n, 1))

    for i in range(0, n):
        eigen_sum = np.abs(eigenvalues[i, 0]) + np.abs(eigenvalues[i, 1])
        if eigen_sum != 0:
            weights[i] = np.abs(eigenvalues[i, 0]) / eigen_sum

    return weights


def eigen1_divide_eigen2(eigenvalues):
    """
    最大特征值除以第二特征值
    :return:
    """
    data_shape = eigenvalues.shape
    n = data_shape[0]
    dim = data_shape[1]

    result = np.zeros((n, 1))
    for i in range(0, n):
        if eigenvalues[i, 1] != 0:
            result[i] = eigenvalues[i, 0]/eigenvalues[i, 1]

    return result


def weight_of_max_eigenvalue(eigenvalues):
    """
    计算最大特征值占所有特征值之和的比重
    要考虑特征值有负数的情况，这时候需要用其绝对值
    :param eigenvalues: 存放所有数据的localPCA的特征值，是一个n*dim的矩阵
                        其中n为数据的个数，dim是数据的维度
    :return: 一个n*1的矩阵，其值为每个数据的最大特征值占所有特征值之和的比重
    """
    data_shape = eigenvalues.shape
    n = data_shape[0]
    dim = data_shape[1]
    weights = np.zeros((n, 1))

    for i in range(0, n):
        eigen_sum = 0
        for j in range(0, dim):
            eigen_sum = eigen_sum + np.abs(eigenvalues[i, j])
        if eigen_sum != 0:
            weights[i] = np.abs(eigenvalues[i, 0]) / eigen_sum

    return weights


def vector_angle_projected(y, y_add_v, y_sub_v):
    """
    计算降维之后特征向量正负方向在投影平面上的角度值
    这个角度在高维空间中是180°，但是在降维投影之后可能会发生变化
    :param y: 原始数据降维之后的投影坐标，是一个n*2的矩阵
    :param y_add_v: 原始数据增加上一个特征向量扰动之后降维的投影坐标
    :param y_sub_y: 原始数据减去一个特征向量扰动之后降维的投影坐标
    :return:
    """
    y_shape = y.shape
    n = y_shape[0]
    v1 = y_add_v - y
    v2 = y_sub_v - y
    v2_t = np.transpose(v2)
    angles = np.zeros((n, 1))

    for i in range(0, n):
        norm_product = LA.norm(v1[i, :]) * LA.norm(v2[i, :])
        # 假设存在某个扰动之后位置没有变，默认角度为0°
        if norm_product == 0:
            angles[i] = 0
            continue
        dot_product = np.dot(v1[i, :], v2_t[:, i])
        # 有时候会因为浮点误差之类的情况导致所求的cos值超出了[-1, 1]的范围
        cos_value = dot_product/norm_product
        if cos_value > 1:
            cos_value = 1
        if cos_value < -1:
            cos_value = -1
        angles[i] = np.arccos(cos_value) / np.pi * 180

    return angles


def eigen1_div_eigen2(eigenvalues):
    """
    第一个特征值除以第二个特征值
    也就是那个linearity
    :param eigenvalues: 特征值矩阵，每一行是一个点的特征值由大到小排列，这是一个numpy的矩阵
    :return: 一个n×1的向量
    """
    data_shape = eigenvalues.shape
    n = data_shape[0]
    m = data_shape[1]
    linearity = np.zeros((n, 1))

    for i in range(0, n):
        if eigenvalues[i, 1] == 0:
            linearity[i] = 10
            continue
        linearity[i] = eigenvalues[i, 0] / eigenvalues[i, 1]

    return linearity


def angle_v1_v2(y1, y2, y=None):
    """
    计算角y1yy2的角度，这个角度应该在0到90°之间
    :param y: 角的顶点的坐标矩阵
    :param y1: 角的第一个端点的坐标矩阵
    :param y2: 角的第二个端点的坐标矩阵
    :return:
    """
    data_shape = y1.shape
    n = data_shape[0]
    m = data_shape[1]

    if y is None:
        y = np.zeros(data_shape)
    v1 = y1 - y
    v2 = y2 - y

    angles = np.zeros((n, 1))
    zeros_count = 0
    for i in range(0, n):
        point1 = v1[i, :]
        point2 = v2[i, :]
        dot_value = np.matmul(point1, np.transpose(point2))

        temp_data = (LA.norm(point1) * LA.norm(point2))
        if temp_data == 0:
            angles[i] = 90
            zeros_count = zeros_count + 1
            continue

        cos_value = dot_value / temp_data
        angle = np.arccos(cos_value)
        angle = angle*180/np.pi
        if angle > 90:
            angle = 180 - angle

        angles[i] = angle

    if zeros_count > 1:
        print('[processData.angle_v1_v2]\t '+str(zeros_count)+' have no change')

    return angles


def length_radio(y1, y2, y=None):
    """
    计算两个向量长度的比值
    :param y1: 第一个向量矩阵
    :param y2: 第一个向量矩阵
    :param y: 原点矩阵
    :return:
    """
    data_shape = y1.shape
    n = data_shape[0]

    if y is None:
        y = np.zeros(data_shape)
    v1 = y1 - y
    v2 = y2 - y

    radios = np.zeros((n, 1))
    count = 0
    for i in range(0, n):
        temp = LA.norm(v2[i, :])
        if temp == 0:
            temp = 0.02
            count = count + 1
        radios[i] = LA.norm(v1[i, :]) / temp

    if count > 0:
        print('[warning processData.length_radio] divide by zero: '+str(count)+" , replaced by 0.02")

    return radios


def mds_stress(path=""):
    """
    计算 MDS 中的stress误差，只针对MDS方法有效
    :param path: 中间结果的存储路径
    :return:
    """
    X = np.loadtxt(path+"x.csv", dtype=np.float, delimiter=",")
    Y = np.loadtxt(path+"y.csv", dtype=np.float, delimiter=",")
    (n, m) = X.shape

    Dx = np.zeros((n, n))
    Dy = np.zeros((n, n))

    for i in range(0, n-1):
        for j in range(i+1, n):
            norm_x = np.linalg.norm(X[i, :] - X[j, :])
            Dx[i, j] = norm_x
            Dx[j, i] = norm_x
            norm_y = np.linalg.norm(Y[i, :] - Y[j, :])
            Dy[i, j] = norm_y
            Dy[j, i] = norm_y
    dD = Dy - Dx
    np.savetxt(path+"dD.csv", dD, fmt='%f', delimiter=",")
    return dD
