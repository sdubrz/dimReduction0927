# 在高维中计算KNN关系图，然后利用其邻居在二维平面中的坐标，计算local PCA
import numpy as np
import matplotlib.pyplot as plt
from BSpline import b_spline
from Dim2LocalPCA import LocalPCA_2dim

import json


def draw_b_spline(path="", line_length=0.1, draw=False):
    """
    根据二维的local PCA计算 B-Spline 曲线
    :param path: 文件操作的目录
    :param line_length: 控制 B-Spline 的大小
    :param draw: 是否画图
    :return:
    """
    knn = np.loadtxt(path+"【weighted】knn.csv", dtype=np.int, delimiter=",")
    Y = np.loadtxt(path+"y.csv", dtype=np.float, delimiter=",")
    (n, k) = knn.shape

    first_eigenvectors, second_eigenvectors, local_eigenvalues = LocalPCA_2dim.local_pca_2dim(Y, k, knn=knn, knn_precomputed=True)

    b_spline_list = []
    bad_count = 0
    for i in range(0, n):
        convex = np.zeros((4, 2))
        s = np.sum(local_eigenvalues[i, :])
        this_spline = []
        if s != 0:
            convex[0, :] = Y[i, :] + first_eigenvectors[i, :] * local_eigenvalues[i, 0] / s * line_length
            convex[1, :] = Y[i, :] + second_eigenvectors[i, :] * local_eigenvalues[i, 1] / s * line_length
            convex[2, :] = Y[i, :] - first_eigenvectors[i, :] * local_eigenvalues[i, 0] / s * line_length
            convex[3, :] = Y[i, :] - second_eigenvectors[i, :] * local_eigenvalues[i, 1] / s * line_length

            splines = b_spline.bspline(convex, n=100, degree=3, periodic=True)
            spline_x, spline_y = splines.T
            for j in range(0, len(spline_x)):
                this_spline.append([spline_x[j], spline_y[j]])
        else:
            bad_count += 1
            for j in range(0, 100):
                this_spline.append([Y[i, 0], Y[i, 1]])

        b_spline_list.append(np.array(this_spline))

    if draw:
        # 进行画图
        label_reader = np.loadtxt(path+"label.csv", dtype=np.str, delimiter=',')
        label = label_reader.astype(np.int)

        shapes = ['s', 'o', '^', '*', '.', '+', '>', '<', 'p', 'h', 'v']
        for i in range(0, n):
            this_spline = b_spline_list[i]
            plt.scatter(Y[i, 0], Y[i, 1], marker=shapes[label[i]], c='k')
            plt.plot(this_spline[:, 0], this_spline[:, 1], linewidth=0.6, c='deepskyblue', alpha=0.7)

        plt.show()

    if bad_count > 0:
        print('坏点个数：', bad_count)

    return b_spline_list


def create_json(path='', line_length=0.1, draw_spline=False):
    """
    生成可供系统使用的json文件，除了散点的坐标点以及B-spline图形之外的属性不保证准确
    :param path: 文件操作的目录
    :param line_length: 控制图形的大小，一般取0.1比较合适
    :parm draw_spline: 是否画出B-spline
    :return:
    """
    y_reader = np.loadtxt(path + "y.csv", dtype=np.str, delimiter=',')
    Y = y_reader[:, :].astype(np.float)
    (n, dim) = Y.shape
    label_reader = np.loadtxt(path + "label.csv", dtype=np.str, delimiter=',')
    label = label_reader.astype(np.int)
    knn = np.loadtxt(path + "【weighted】knn.csv", dtype=np.int, delimiter=",")
    (n, k) = knn.shape

    b_spline_list = draw_b_spline(path=path, line_length=line_length, draw=draw_spline)

    # 生成json数据
    jsonfile = open(path + "highKNN_2d_LocalPCA.json", 'w')
    jsonfile.write('[')

    for i in range(0, n):
        i_spline = b_spline_list[i]
        (i_n, i_m) = i_spline.shape
        line = "{\"x\": " + str(Y[i, 0]) + ",\"y\": " + str(Y[i, 1]) + ",\"class\": " + str(label[i]) + ","
        line = line + "\"dNum\": 2, \"hdata\": [" + str(Y[i, 0]) + "," + str(Y[i, 1]) + "], \"k\":" + str(k) + ","
        line = line + "\"pointsNum\": " + str(i_n) + ", \"polygon\": ["
        for j in range(0, i_n):
            line = line + "[" + str(i_spline[j, 0]) + "," + str(i_spline[j, 1]) + "]"
            if j != i_n-1:
                line = line + ","
        line = line + "], "
        line = line + "\"angles\": 90.0, \"eigenNumber\": 3, \"proportion\": 1.0,"
        line = line + "\"linearity\": 2.0, \"linearityEqualized\": 2.0,"  # 这个暂时是不准确的
        line = line + "\"linearProject\": 2.0, "  # 这个暂时也是不准确的
        line = line + "\"linearChange\": 0.0, \"angleAddSub\": 180.0,"
        line = line + "\"polygonSize\": 0.003, \"starPolygonSize\": 0.003, "  # 暂时不准确
        line = line + "\"y_add_points\": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], "  # 暂时不准确
        line = line + "\"y_sub_points\": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], "  # 暂时不准确
        line = line + "\"angleAddSub_basedcos\": 1.0, \"angleAddSub_cosweighted\": 1.0, \"angle12Sin\":1.0"
        line = line + "}"
        if i != n-1:
            line = line + ",\n"
        jsonfile.write(line)
    jsonfile.write(']')
    jsonfile.close()
    print('已成功生成高维邻域二维数据的json文件')


def create_json2(path='', line_length=0.1, draw_spline=False):
    """
    生成可供使用的json文件，除了B-Spline图形的数据之外，其余属性的值全部复制自对local PCA降维的版本的值
    :param path: 文件操作的主目录
    :param line_length: 控制图形的大小，一般取0.1
    :param draw_spline: 在计算完成之后，是否画出图形
    :return:
    """

    y_reader = np.loadtxt(path + "y.csv", dtype=np.str, delimiter=',')
    Y = y_reader[:, :].astype(np.float)
    (n, dim) = Y.shape
    label_reader = np.loadtxt(path + "label.csv", dtype=np.str, delimiter=',')
    label = label_reader.astype(np.int)
    knn = np.loadtxt(path + "【weighted】knn.csv", dtype=np.int, delimiter=",")
    (n, k) = knn.shape

    b_spline_list = draw_b_spline(path=path, line_length=line_length, draw=draw_spline)

    f = open(path + "removeStress.json", encoding='utf-8')
    data = json.load(f)
    index = 0
    for item in data:
        spline = b_spline_list[index]
        item['pointsNum'] = spline.shape[0]
        item['polygon'] = spline.tolist()
        index += 1

    out_file = open(path + "highKNN_2d_LocalPCA.json", "w")
    out_file.write('[')
    index = 0
    for item in data:
        line1 = str(item)
        line2 = ''
        for a_c in line1:
            if a_c == '\'':
                line2 = line2 + '\"'
            else:
                line2 = line2 + a_c
        out_file.write(line2)
        index += 1
        if index == n:
            out_file.write(']')
        else:
            out_file.write(',\n')
    out_file.close()
    print('已成功生成高维邻域二维数据的json文件')


def test():
    path = "E:\\Project\\result2019\\result1026without_straighten\\PCA\\Wine\\yita(0.1)nbrs_k(20)method_k(20)numbers(4)_b-spline_weighted\\"
    create_json(path, line_length=0.1, draw_spline=True)


if __name__ == '__main__':
    test()
