# 计算变形指标
import numpy as np


def angle1_2(save_path=''):
    # 计算第一特征向量和第二特征向量的夹角
    y_reader = np.loadtxt(save_path+'y.csv', dtype=np.str, delimiter=',')
    y1_reader = np.loadtxt(save_path+'y1+.csv', dtype=np.str, delimiter=',')
    y2_reader = np.loadtxt(save_path + 'y2+.csv', dtype=np.str, delimiter=',')
    y = y_reader[:, :].astype(np.float)
    y1 = y1_reader[:, :].astype(np.float)
    y2 = y2_reader[:, :].astype(np.float)

    (n, dim) = y.shape
    # 计算夹角的余弦值
    cos_list = np.zeros((n, 1))
    for i in range(0, n):
        v1 = y1[i, :] - y[i, :]
        v2 = y2[i, :] - y[i, :]
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 * norm2 == 0:
            cos_list[i] = 0
        else:
            cos_list[i] = np.dot(v1, v2) / (norm1*norm2)

    # 转化为正弦值，正弦值越大角度保持越好
    sin_list = np.zeros((n, 1))
    for i in range(0, n):
        sin_list[i] = np.sqrt(1-cos_list[i]*cos_list[i])

    np.savetxt(save_path+'sin_1_2.csv', sin_list, fmt='%f', delimiter=',')

    return sin_list


def angle_p_n(save_path='', vector_num=2):
    """
    计算每个特征向量正向和负向的夹角的余弦值，并映射到 [0, 1] 的区间上
    :param save_path: 读写数据的文件目录
    :param vector_num: 所要计算的特征向量数目
    :return:
    """
    y_reader = np.loadtxt(save_path + 'y.csv', dtype=np.str, delimiter=',')
    y = y_reader[:, :].astype(np.float)
    (n, dim) = y.shape
    value_list = []

    for index in range(1, vector_num+1):
        y_p_reader = np.loadtxt(save_path+'y'+str(index)+'+.csv', dtype=np.str, delimiter=',')
        y_n_reader = np.loadtxt(save_path + 'y' + str(index) + '-.csv', dtype=np.str, delimiter=',')
        y_p = y_p_reader[:, :].astype(np.float)
        y_n = y_n_reader[:, :].astype(np.float)

        cos_list = np.zeros((n, 1))
        for i in range(0, n):
            v1 = y_p[i, :] - y[i, :]
            v2 = y_n[i, :] - y[i, :]
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1*norm2 == 0:
                cos_list[i] = -1
            else:
                cos_list[i] = np.dot(v1, v2) / (norm1 * norm2)

        eve_list = np.zeros((n, 1))
        for i in range(0, n):
            eve_list[i] = (1-cos_list[i]) / 2

        np.savetxt(save_path+'angle_evaluation+-'+str(index)+'.csv', eve_list, fmt='%f', delimiter=',')

        value_list.append(eve_list)

    return value_list


def angle_p_n_weighted(save_path='', vector_num=2, weighted=True):
    """
    根据各个角度的正负角度得到一个加权的总的变形指标
    :param save_path: 进行文件读写的目录
    :param weighted: 是否按照特征值进行加权
    :return:
    """
    eigenvalue_reader = np.loadtxt(save_path+'【weighted】eigenvalues.csv', dtype=np.str, delimiter=',')
    eigenvalue = eigenvalue_reader[:, :].astype(np.float)
    (n, dim) = eigenvalue.shape

    value_list = angle_p_n(save_path, vector_num)
    k = len(value_list)
    sum_list = np.zeros((n, 1))
    for i in range(0, k):
        temp_list = value_list[i]
        w = 1 / k
        for j in range(0, n):
            if weighted:
                w = eigenvalue[j, i] / np.sum(eigenvalue[j, 0:k])
            sum_list[j] = sum_list[j] + w * temp_list[i]

    save_file = save_path + "angle_+-sum"
    if weighted:
        save_file = save_file + 'weighted'
    save_file = save_file + '.csv'

    np.savetxt(save_file, sum_list, fmt='%f', delimiter=',')

    return sum_list
