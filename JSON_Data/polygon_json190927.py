"""
    将结果合并成为前端可用的json数据

    @author: sdubrz
    @date: 2019.01.16

"""
import csv
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA

from Tools import HistogramEqualization as HistEqual
from Tools import MyGeometry
from Tools import PolygonArea
from Tools import VisualizationKNN


"""
本程序改自 polygon_json.py 
计算的属性种类有所改变
系统使用平滑过渡的color map 这样，数据就不用再去分组了
七年九月二十七日
"""


def dot_divide(A, B):
    """
    计算两个向量对应位置上的元素相除的结果，最后的结果是一个等大的向量
    :param A:
    :param B:
    :return:
    """
    A_shape = A.shape
    B_shape = B.shape

    if A_shape[0] != B_shape[0]:
        return None

    n = A_shape[0]
    # m = A_shape[1]
    C = np.zeros(A_shape)

    for i in range(0, n):
        if B[i] == 0:
            C[i] = float('inf')
            continue
        C[i] = A[i] / B[i]

    return C


def divide_group(values, group_num):
    """
        根据数值values将数据进行分组，通过将values的值域进行平均划分的方式
    :param values: 数值向量
    :param group_num: 所要分的组数
    :return:
    """
    data_shape = values.shape
    n = data_shape[0]

    groups = np.zeros((n, 1))

    min_value = np.min(values)
    max_value = np.max(values)

    step_length = (max_value - min_value) / group_num
    if step_length == 0:
        return groups

    for i in range(0, n):
        group = int((values[i]-min_value)/step_length)
        if group == group_num:
            group = group_num-1
        groups[i] = group

    return groups


def angles_p_n(y, y_add, y_sub):
    """
    计算正向扰动与负向扰动投影结果的角度
    :param y: 没有扰动的投影坐标
    :param y_add: 正向扰动的投影坐标
    :param y_sub: 负向扰动的投影坐标
    :return:
    """
    y_p = y_add - y
    y_n = y_sub - y

    data_shape = y.shape
    n = data_shape[0]
    angles = np.zeros((n, 1))

    for i in range(0, n):
        norm_p = LA.norm(y_p[i, :])
        norm_n = LA.norm(y_n[i, :])

        if norm_p*norm_n == 0:
            # 如果有一个方向上的扰动为0，按照2019年2月15日讨论的结果这个多边形可以不画
            # 退一步讲，说明这个点几乎不会发生变化，它的多边形面积应该会特别小，可以是看不见的
            continue

        cos_value = (y_p[i, 0]*y_n[i, 0] + y_p[i, 1]*y_n[i, 1]) / (norm_p*norm_n)
        if cos_value > 1:
            # print('不正常的cos值： ', cos_value)
            cos_value = 1
        if cos_value < -1:
            # print('不正常的cos值： ', cos_value)
            cos_value = -1
        angles[i] = np.arccos(cos_value)
        angles[i] = angles[i]*180/np.pi

    return angles


def polygon_area(polygon_list, y):
    """
    计算多边形的面积
    :param polygon_list: 就是文件中的那个convex_hull_list
    :param y: 原始数据降维后的投影坐标
    :return:
    """
    n = len(polygon_list)
    areas = []

    for i in range(0, n):
        this_polygon = polygon_list[i]
        points_num = int(len(this_polygon) / 2)
        points = np.zeros((points_num, 2))

        for j in range(0, points_num):
            points[j, 0] = this_polygon[2*j]
            points[j, 1] = this_polygon[2*j+1]

        area = MyGeometry.centrosymmetry_area(points, y[i, :])
        areas.append(area)

    print(len(areas))
    return areas


def star_polygon_area(polygons):
    """
    计算 star-shape polygon的面积
    :param polygons: 多边形顶点坐标
    :return:
    """
    n = len(polygons)
    areas = np.zeros((n, 1))

    for i in range(0, n):
        a_list = polygons[i]
        m = len(a_list)
        points_num = int(m/2)
        a_polygon = np.zeros((points_num, 2))
        for j in range(0, points_num):
            a_polygon[j, 0] = a_list[j*2]
            a_polygon[j, 1] = a_list[j*2+1]

        areas[i] = np.abs(PolygonArea.polygon_area(a_polygon))

    return areas


def linearity_change(linear1, linear0):
    """
    计算linearityChange，不再计算放大倍数，而是计算改变了多少倍
    正数表示放大了，负数表示缩小了。
    :param linear1: 投影之后的linearity
    :param linear0: 原始高维空间中的linearity
    :return:
    """
    data_shape = linear1.shape
    n = data_shape[0]

    change = np.zeros((n, 1))
    for i in range(0, n):
        if linear1[i] > linear0[i]:
            change[i] = np.log(linear1[i]/linear0[i])
        else:
            change[i] = np.log(linear0[i]/linear1[i])*-1

    return change


def points_proportion(read_path, eigen_number, weighted=True):
    """
    计算每个点使用的特征值占其特征值之和的比值
    :param read_path: 数据读写目录
    :param eigen_number: 所用的特征值的个数
    :param weighted: 在添加扰动的时候是否曾根据特征值的大小分配权重
    :return:
    """
    file_name = "【weighted】eigenvalues.csv"
    if not weighted:
        file_name = "eigenvalues.csv"
    eigenvalues_reader = np.loadtxt(read_path+file_name, dtype=np.str, delimiter=',')
    eigenvalues = eigenvalues_reader[:, :].astype(np.float)

    (n, dim) = eigenvalues.shape
    proportion = np.zeros((n, 1))
    for i in range(0, n):
        proportion[i] = np.sum(eigenvalues[i, 0:eigen_number]) / np.sum(eigenvalues[i, :])

    np.savetxt(read_path+"proportion.csv", proportion, fmt='%f', delimiter=',')

    return proportion


def how_many_eigens(read_path, proportion=0.8, weighted=True):
    """
    计算每个点需要多少个特征值才能达到要求的比重
    :param read_path: 数据读写目录
    :param proportion: 要求的比重
    :param weighted: 在计算扰动的时候是否分配了权重
    :return:
    """
    file_name = "【weighted】eigenvalues.csv"
    if not weighted:
        file_name = "eigenvalues.csv"

    eigenvalues_reader = np.loadtxt(read_path+file_name, dtype=np.str, delimiter=',')
    eigenvalues = eigenvalues_reader[:, :].astype(np.float)
    (n, dim) = eigenvalues.shape

    enough_number = np.zeros((n, 1))
    for i in range(0, n):
        temp_sum = 0
        total_sum = np.sum(eigenvalues[i, :])
        for j in range(0, dim):
            temp_sum = temp_sum + eigenvalues[i, j]
            if temp_sum >= total_sum * proportion:
                enough_number[i] = j+1
                break

    np.savetxt(read_path+'eigen_numbers.csv', enough_number, fmt='%f', delimiter=',')

    return enough_number


def merge_json(main_path, data_name, method_name, yita, method_k, nbrs_k, draw_kind=None,
               MAX_EIGEN_NUMBER=4, weighted=True, test_attr=None, notes=None, false_class=None):
    """
    构建json文件
    :param main_path:
    :param yita:
    :param method_k:
    :param max_eigen_numbers: 最多使用的特征向量个数
    :param weighted: 在使用特征向量作为扰动的时候是否根据特征值的大小分配了权重
    :param test_attr: 用于测试的冗余属性
    :param notes: 备注
    :param false_class: 假的类，离散的冗余属性
    :return:
    """

    # 根据是否分配权重选择部分所需的路径
    weight_str = ""
    if weighted:
        weight_str = "【weighted】"

    read_path = main_path + method_name + "\\" + data_name + "\\" + "yita("+str(yita) + ")nbrs_k(" + str(nbrs_k)
    read_path = read_path+")method_k("+str(method_k)+''+")numbers("+str(MAX_EIGEN_NUMBER) + ")"
    # print("【read_path】", read_path)
    read_path = read_path + "_" + draw_kind
    if weighted:
        read_path = read_path + "_weighted"
    else:
        read_path = read_path + "_withoutweight"

    read_path = read_path + "\\"

    y_reader = np.loadtxt(read_path+"y.csv", dtype=np.str, delimiter=",")  # 原始数据的降维结果
    y = y_reader[:, :].astype(np.float)
    (n, temp_dim) = y.shape
    knn_keep = VisualizationKNN.knn_keep(read_path)

    if notes is None:  # 默认的备注
        notes = []
        for i in range(0, n):
            notes.append(str(i))

    if test_attr is None:
        test_attr = np.random.random((n, 1))

    if false_class is None:
        false_class = np.ones((n, 1))

    y_add1_reader = np.loadtxt(read_path+"y1+.csv", dtype=np.str, delimiter=",")  # 添加第一特征向量作为扰动的投影结果
    y_add1 = y_add1_reader[:, :].astype(np.float)
    y_sub1_reader = np.loadtxt(read_path+"y1-.csv", dtype=np.str, delimiter=",")  # 减去第一特征向量的投影结果
    y_sub1 = y_sub1_reader[:, :].astype(np.float)

    angles_add_sub = angles_p_n(y, y_add1, y_sub1)
    np.savetxt(read_path + "angles_add_sub.csv", angles_add_sub, fmt="%f", delimiter=",")

    # 计算坐标的范围
    min_x = np.min(y[:, 0])
    max_x = np.max(y[:, 0])
    min_y = np.min(y[:, 1])
    max_y = np.max(y[:, 1])

    x_reader = np.loadtxt(read_path+"x.csv", dtype=np.str, delimiter=",")  # normalize之后的高维数据
    x = x_reader[:, :].astype(np.float)

    label_reader = np.loadtxt(read_path+"label.csv", dtype=np.str, delimiter=",")  # 数据的标签
    label = label_reader.astype(np.int)

    polygon_list = []  # 凸包的顶点坐标
    polygon_reader = csv.reader(open(read_path + "convex_hull_list.csv", encoding='UTF-8'))
    for row in polygon_reader:
        polygon = []
        for item in row:
            polygon.append(float(item))
        polygon_list.append(polygon)

    star_polygon_list = []  # star-shape polygon的顶点坐标
    star_polygon_reader = csv.reader(open(read_path + "final_star_polygon.csv", encoding='UTF-8'))
    for row in star_polygon_reader:
        polygon = []
        for item in row:
            polygon.append(float(item))
        star_polygon_list.append(polygon)

    polygon_areas = polygon_area(polygon_list, y)  # 每一个多边形的面积
    print(len(polygon_areas))
    np.savetxt(read_path+"polygon_areas.csv", polygon_areas, fmt="%f", delimiter=",")  # 保存每一个多边形的面积

    star_polygon_areas = star_polygon_area(star_polygon_list)
    np.savetxt(read_path+"star_polygon_area.csv", star_polygon_areas, fmt="%f", delimiter=",")  # 保存每一个star-shape polygon的面积

    for polygon in polygon_list:
        polygon_size = int(len(polygon)/2)
        for i in range(0, polygon_size):
            if min_x > polygon[i*2]:
                min_x = polygon[i*2]
            if max_x < polygon[i*2]:
                max_x = polygon[i*2]
            if min_y > polygon[i*2+1]:
                min_y = polygon[i*2+1]
            if max_y < polygon[i*2+1]:
                max_y = polygon[i*2+1]

    x_low = min_x - 0.1*(max_x-min_x)  # 绘图时的坐标范围
    x_up = max_x + 0.1*(max_x-min_x)
    y_low = min_y - 0.1*(max_y-min_y)
    y_up = max_y + 0.1*(max_y-min_y)

    scale_file = open(read_path+"scale.json", 'w')
    scale_file.write("{"+"\"min_x\":"+str(x_low)+", \"max_x\":"+str(x_up)+", \"min_y\":"+str(y_low)+", \"max_y\":"+str(y_up)+"}")
    scale_file.close()

    k_values = nbrs_k * np.ones((n, 1))

    angles_reader = np.loadtxt(read_path+"angles_v1_v2_projected.csv", dtype=np.str, delimiter=",")  # 第一特征向量与第二特征向量投影之后的角度
    angles = angles_reader.astype(np.float)

    eigen1_div_eigen2_reader = np.loadtxt(read_path+weight_str+"eigen1_div_eigen2_original.csv", dtype=np.str, delimiter=",")
    eigen1_div_eigen2 = eigen1_div_eigen2_reader.astype(np.float)  # 高维中第一特征值与第二特征值的比值
    linearityEqualized = HistEqual.hist_equalization(eigen1_div_eigen2, bins_num=100)

    eigen1_div_eigen2_project_reader = np.loadtxt(read_path + "eigen1_div_eigen2_projected.csv", dtype=np.str,
                                          delimiter=",")
    eigen1_div_eigen2_project = eigen1_div_eigen2_project_reader.astype(np.float)  # 第一特征向量与第二特征向量投影之后的长度比值

    # 每个点所使用的特征向量个数
    eigen_numbers = how_many_eigens(read_path, proportion=0.8, weighted=weighted)
    # eigen_numbers_reader = np.loadtxt(read_path+"eigen_numbers.csv", dtype=np.str, delimiter=",")
    # eigen_numbers = eigen_numbers_reader.astype(np.int)

    # 每个点的proportion
    proportions = points_proportion(read_path, MAX_EIGEN_NUMBER, weighted=weighted)
    # proportions_reader = np.loadtxt(read_path+"eigens_counts.csv", dtype=np.str, delimiter=",")
    # proportions = proportions_reader.astype(np.float)

    linearityChange = linearity_change(eigen1_div_eigen2_project, eigen1_div_eigen2)
    np.savetxt(read_path+"linearityChange.csv", linearityChange, fmt="%f", delimiter=",")

    # 2019.02.21测试，对右下方散点图做直方图均衡化处理，使之不在集中于左侧
    # linearityChange = HistEqual.hist_equalization(linearityChange0)
    # np.savetxt(read_path+"linearityChangeEqualized.csv", linearityChange, fmt="%f", delimiter=",")

    # 读取校直后的扰动结果，便于画图， 这一部分需要用到最多使用到的特征向量个数
    y_add_list = []  # 正向扰动投影结果
    y_sub_list = []  # 负向扰动投影结果

    for i in range(0, MAX_EIGEN_NUMBER):
        y_add_i_reader = np.loadtxt(read_path+"y_add_"+str(i+1)+".csv", dtype=np.str, delimiter=",")
        y_sub_i_reader = np.loadtxt(read_path+"y_sub_"+str(i+1)+".csv", dtype=np.str, delimiter=",")
        y_add_i = y_add_i_reader[:, :].astype(np.float)
        y_sub_i = y_sub_i_reader[:, :].astype(np.float)
        y_add_i = y_add_i - y
        y_sub_i = y_sub_i - y
        y_add_list.append(y_add_i)
        y_sub_list.append(y_sub_i)

    np.savetxt(read_path + "linearity_equalized.csv", linearityEqualized, fmt="%f", delimiter=",")

    # 将这些分组信息画图并保存起来
    plt.hist(linearityEqualized)
    plt.savefig(read_path + "linearityEqualized.png")
    plt.close()
    plt.hist(angles_add_sub)
    plt.savefig(read_path + "angles_add_sub.png")
    plt.close()
    plt.hist(angles)
    plt.savefig(read_path + "angles_v1_v2.png")
    plt.close()
    plt.hist(linearityChange)
    plt.savefig(read_path + "linearityChange.png")
    plt.close()

    angleAddSub_basedcos_reader = np.loadtxt(read_path+'angle_p_n_basecos.csv', dtype=np.str, delimiter=',')
    angleAddSub_basedcos = angleAddSub_basedcos_reader.astype(np.float)
    angleAddSub_cosweighted_reader = np.loadtxt(read_path+'angle_+-sumweighted.csv', dtype=np.str, delimiter=',')
    angleAddSub_cosweighted = angleAddSub_cosweighted_reader.astype(np.float)
    angle12Sin_reader = np.loadtxt(read_path+'sin_1_2.csv', dtype=np.str, delimiter=',')
    angle12Sin = angle12Sin_reader.astype(np.float)

    jsonfile = open(read_path+"temp_total.json", 'w')
    data_shape = x.shape
    n = data_shape[0]
    m = data_shape[1]

    jsonfile.write("[")
    for i in range(0, n-1):
        line = "{"
        line = line + "\"x\": "+str(y[i, 0]) + ",\"y\": "+str(y[i, 1]) + ","
        line = line + "\"class\": " + str(label[i]) + ","
        line = line + "\"dNum\": " + str(m) + ",\"hdata\": ["

        for j in range(0, m-1):
            line = line + str(x[i, j]) + ","
        line = line + str(x[i, m-1]) + "],"

        line = line + "\"k\": " + str(int(k_values[i])) + ","
        polygon = polygon_list[i]
        polygon_size = int(len(polygon)/2)
        line = line + "\"pointsNum\": " + str(polygon_size) + ","
        line = line + "\"polygon\": ["

        for j in range(0, polygon_size-1):
            line = line + "[" + str(polygon[j*2]) + ", "+str(polygon[j*2+1]) + "], "
        line = line + "[" + str(polygon[(polygon_size-1)*2]) + ", "+str(polygon[(polygon_size-1)*2+1]) + "]], "

        line = line + "\"angles\": " + str(angles[i]) + ", "
        line = line + "\"knn_keep\": " + str(knn_keep[i, 0]) + ","

        line = line + "\"eigenNumber\": " + str(int(eigen_numbers[i])) + ", "
        line = line + "\"proportion\": " + str(float(proportions[i])) + ", "

        line = line + "\"linearity\": " + str(eigen1_div_eigen2[i]) + ", "
        temp_num = linearityEqualized[i, 0] + 0.0
        line = line + "\"linearityEqualized\": " + str(temp_num) + ", "
        line = line + "\"linearProject\": " + str(eigen1_div_eigen2_project[i]) + ", "

        temp_num = linearityChange[i, 0] + 0.0
        line = line + "\"linearChange\": " + str(temp_num) + ", "

        temp_num = angles_add_sub[i, 0] + 0.0
        line = line + "\"angleAddSub\": " + str(temp_num) + ", "

        line = line + "\"polygonSize\": " + str(polygon_areas[i]) + ", "  # 多边形的面积大小
        temp_num = star_polygon_areas[i, 0] + 0.0
        line = line + "\"starPolygonSize\": " + str(temp_num) + ", "  # star-shape 多边形的面积大小

        # 校直后的正负扰动的结果
        line = line + "\"y_add_points\": ["
        for j in range(0, MAX_EIGEN_NUMBER-1):
            y_add_j = y_add_list[j]
            line = line + "[" + str(y_add_j[i, 0]) + ", " + str(y_add_j[i, 1]) + "], "
        y_add_jj = y_add_list[MAX_EIGEN_NUMBER-1]
        line = line + "[" + str(y_add_jj[i, 0]) + ", " + str(y_add_jj[i, 1]) + "]], "

        line = line + "\"y_sub_points\": ["
        for j in range(0, MAX_EIGEN_NUMBER - 1):
            y_sub_j = y_sub_list[j]
            line = line + "[" + str(y_sub_j[i, 0]) + ", " + str(y_sub_j[i, 1]) + "], "
        y_sub_jj = y_sub_list[MAX_EIGEN_NUMBER - 1]
        line = line + "[" + str(y_sub_jj[i, 0]) + ", " + str(y_sub_jj[i, 1]) + "]] ,"

        line = line + "\"angleAddSub_basedcos\": " + str(angleAddSub_basedcos[i]) + ","
        line = line + "\"angleAddSub_cosweighted\": " + str(angleAddSub_cosweighted[i]) + ","
        line = line + "\"angle12Sin\": " + str(angle12Sin[i]) + ","
        line = line + "\"test_attr\": " + str(test_attr[i]) + ","
        line = line + "\"int_attr\": " + str(false_class[i]) + ","
        line = line + "\"notes\": \"" + notes[i] + "\""

        line = line + "},\n"
        jsonfile.write(line)

    # 最后一行
    final_line = "{"
    final_line = final_line + "\"x\": " + str(y[n-1, 0]) + ",\"y\": " + str(y[n-1, 1]) + ","
    final_line = final_line + "\"class\": " + str(label[n-1]) + ","
    final_line = final_line + "\"dNum\": " + str(m) + ",\"hdata\": ["

    for j in range(0, m - 1):
        final_line = final_line + str(x[n-1, j]) + ","
    final_line = final_line + str(x[n-1, m - 1]) + "],"

    final_line = final_line + "\"k\": " + str(int(k_values[n-1])) + ","
    final_polygon = polygon_list[n-1]
    final_polygon_size = int(len(final_polygon) / 2)
    final_line = final_line + "\"pointsNum\": " + str(final_polygon_size) + ","
    final_line = final_line + "\"polygon\": ["

    for j in range(0, final_polygon_size - 1):
        final_line = final_line + "[" + str(final_polygon[j * 2]) + ", " + str(final_polygon[j * 2 + 1]) + "], "
    final_line = final_line + "[" + str(final_polygon[(final_polygon_size - 1) * 2]) + ", " + str(final_polygon[(final_polygon_size - 1) * 2 + 1]) + "]], "

    final_line = final_line + "\"angles\": " + str(angles[n-1]) + ", "
    final_line = final_line + "\"knn_keep\": " + str(knn_keep[n-1, 0]) + ","
    final_line = final_line + "\"eigenNumber\": " + str(int(eigen_numbers[n-1])) + ", "
    final_line = final_line + "\"proportion\": " + str(float(proportions[n-1])) + ", "

    final_line = final_line + "\"linearity\": " + str(eigen1_div_eigen2[n-1]) + ", "
    final_temp_num = linearityEqualized[n-1, 0]+0.0
    final_line = final_line + "\"linearityEqualized\": " + str(final_temp_num) + ", "
    final_line = final_line + "\"linearProject\": " + str(eigen1_div_eigen2_project[n-1]) + ","

    final_temp_num = linearityChange[n-1, 0] + 0.0
    final_line = final_line + "\"linearChange\": " + str(final_temp_num) + ","
    final_temp_num = angles_add_sub[n-1, 0] + 0.0
    final_line = final_line + "\"angleAddSub\": " + str(final_temp_num) + ", "

    final_line = final_line + "\"polygonSize\": " + str(polygon_areas[n-1]) + ", "
    final_temp_num = star_polygon_areas[n-1, 0] + 0.0
    final_line = final_line + "\"starPolygonSize\": " + str(final_temp_num) + ", "  # star-shape 多边形的面积大小

    # 校直后的正负扰动的结果
    final_line = final_line + "\"y_add_points\": ["
    for j in range(0, MAX_EIGEN_NUMBER - 1):
        final_y_add_j = y_add_list[j]
        final_line = final_line + "[" + str(final_y_add_j[n-1, 0]) + ", " + str(final_y_add_j[n-1, 1]) + "], "
    final_y_add_jj = y_add_list[MAX_EIGEN_NUMBER - 1]
    final_line = final_line + "[" + str(final_y_add_jj[n-1, 0]) + ", " + str(final_y_add_jj[n-1, 1]) + "]], "

    final_line = final_line + "\"y_sub_points\": ["
    for j in range(0, MAX_EIGEN_NUMBER - 1):
        final_y_sub_j = y_sub_list[j]
        final_line = final_line + "[" + str(final_y_sub_j[n-1, 0]) + ", " + str(final_y_sub_j[n-1, 1]) + "], "
    final_y_sub_jj = y_sub_list[MAX_EIGEN_NUMBER - 1]
    final_line = final_line + "[" + str(final_y_sub_jj[n-1, 0]) + ", " + str(final_y_sub_jj[n-1, 1]) + "]] ,"

    final_line = final_line + "\"angleAddSub_basedcos\": " + str(angleAddSub_basedcos[n-1]) + ","
    final_line = final_line + "\"angleAddSub_cosweighted\": " + str(angleAddSub_cosweighted[n-1]) + ","
    final_line = final_line + "\"angle12Sin\": " + str(angle12Sin[n-1]) + ","
    final_line = final_line + "\"test_attr\": " + str(test_attr[n-1]) + ","
    final_line = final_line + "\"int_attr\":" + str(false_class[n-1]) + ","
    final_line = final_line + "\"notes\": \"" + notes[n-1] + "\""

    final_line = final_line + "}"
    final_line = final_line + "]"
    jsonfile.write(final_line)
    jsonfile.close()

    print("已生成json文件")


def test():
    path = "E:\\特征向量散点图项目\\result2019\\0130\\datasets\\digits5_8\\label.csv"
    reader = np.loadtxt(path, dtype=np.str, delimiter=",")
    label = reader.astype(np.int)
    label_shape = label.shape
    n = label_shape[0]
    for i in range(0, n):
        if label[i] == 5:
            label[i] = 4
        if label[i] == 8:
            label[i] = 5

    np.savetxt(path, label, fmt="%d", delimiter=",")


if __name__ == "__main__":
    test()
