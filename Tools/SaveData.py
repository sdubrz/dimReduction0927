import numpy as np
import csv


"""
用于保存部分数据
"""


def merge_json(main_path=None, yita=0.1, k=30, method_k=30):
    """
    合并成json文件，这些参数用于自动生成源数据目录
    目前最终生成的json文件中的相关数据属性有
    x : 原始数据降维后的横坐标     y : 原始数据降维后的纵坐标
    cluster : 数据所属的类别标签
    x1 : 原始数据加特征向量扰动之后降维的横坐标    y1 : 原始数据加特征向量扰动之后降维的纵坐标
    x2 : 原始数据减特征向量扰动之后降维的横坐标    y2 : 原始数据减特征向量扰动之后降维的纵坐标
    dNum : 原始高维数据的维度数目
    d0 ... dn : 原始高维数据的各个维度的值
    isCircle : 原始数据以及增减特征向量扰动之后降维所得的三个投影点能否形成一个圆弧，也就是是不是不共线
    oX : 三个点所形成的圆的圆心横坐标     oY : 三个点所形成的圆的圆心纵坐标
    r : 三个点所形成的圆的半径
    angle : 原始数据降维后的投影点相对于圆心的角度，这个角度是在单位元上计算出来的。这个角度的取值范围是[0, 360°]
    angle1 : 原始数据加了特征向量扰动之后降维所得的投影点相对于圆心的角度
    angle2 : 原始数据减了特征向量扰动之后降维所得的投影点相对于圆心的角度

    下面是一个例子：
    {"x": -1.466145,"y": 0.181736,"cluster": 1.0,"x1": -1.490548,"y1": 0.077736,"x2": -1.423334,"y2": 0.307997,
        "dNum": 4.0,"hdata":{"d0": -0.555556,"d1": 0.25,"d2": -0.864407,"d3": -0.916667},
        "isCircle":1.0,"oX":-0.266662273237"oY":-0.154575940016,"r":1.24573839273,
        "angle":164.337527327"angle1":169.252249303"angle2":158.202691499,"eigen0":0.031578,"eigen1":0.012488},

    :return:
    """
    if main_path is None:
        print('merge_json:\t文件路径不正确')

    case_path = main_path + "yita" + str(yita) + "k" + str(k) + "[k2]" + str(method_k) + "\\"

    data_file = np.loadtxt(case_path+"x.csv", dtype=np.str, delimiter=',')
    data = data_file[:, :].astype(np.float)

    x_y_file = np.loadtxt(case_path + "y.csv", dtype=np.str, delimiter=',')
    x_y = x_y_file[:, :].astype(np.float)

    x1_y1_file = np.loadtxt(case_path + "y1.csv", dtype=np.str, delimiter=',')
    x1_y1 = x1_y1_file[:, :].astype(np.float)

    x2_y2_file = np.loadtxt(case_path + "y2.csv", dtype=np.str, delimiter=',')
    x2_y2 = x2_y2_file[:, :].astype(np.float)

    cluster_file = np.loadtxt(case_path + "cluster.csv", dtype=np.str, delimiter=',')  # 类别
    cluster = cluster_file.astype(np.int)

    r_file = np.loadtxt(case_path + "r.csv", dtype=np.str, delimiter=',')  # 半径
    r = r_file.astype(np.float)

    circles_file = np.loadtxt(case_path + "circles_param.csv", dtype=np.str, delimiter=',')  # 圆的相关参数
    circles_to_draw = circles_file.astype(np.float)

    is_circle_file = np.loadtxt(case_path + "is_circle.csv", dtype=np.str, delimiter=',')  # 是否是一个圆
    is_circle = is_circle_file.astype(np.float)

    eig_values_file = np.loadtxt(case_path + "eig_values.csv", dtype=np.str, delimiter=',')  # 特征值
    eig_values = eig_values_file.astype(np.float)

    data_shape = data.shape
    n = data_shape[0]
    dim = data_shape[1]

    before_hdata_dim = 8  # 高维数据开始的地方
    total_dim = before_hdata_dim + dim
    total_data = np.zeros((n, total_dim))
    for i in range(0, n):
        total_data[i, 0] = x_y[i, 0]
        total_data[i, 1] = x_y[i, 1]
        total_data[i, 2] = cluster[i]
        total_data[i, 3] = x1_y1[i, 0]
        total_data[i, 4] = x1_y1[i, 1]
        total_data[i, 5] = x2_y2[i, 0]
        total_data[i, 6] = x2_y2[i, 1]
        total_data[i, 7] = dim
        for j in range(0, dim):
            total_data[i, before_hdata_dim+j] = data[i, j]

    # np.savetxt(case_path+"total_data.csv", total_data, fmt="%f", delimiter=",")

    fieldlist = ["x", "y", "cluster", "x1", "y1", "x2", "y2", "dNum"]
    for i in range(0, dim):
        str_name = "d"+str(i)
        fieldlist.append(str_name)

    file_name = "yita " + str(yita) + " k " + str(k) + " method_k " + str(method_k) + ".json"
    jsonfile = open(case_path+file_name, 'w')

    for i in range(0, n-1):
        a_circle = circles_to_draw[i]
        row = "{"
        for j in range(0, total_dim - 1):
            if j == before_hdata_dim:
                row = row + "\"hdata\":{"
            row = row + "\"" + fieldlist[j] + "\": " + str(total_data[i, j]) + ","
        row = row + "\"" + fieldlist[total_dim-1] + "\": " + str(total_data[i, total_dim-1]) + "},"
        row = row + "\"isCircle\":" + str(is_circle[i]) + ","
        row = row + "\"oX\":" + str(a_circle[0]) + "\"oY\":" + str(a_circle[1]) + ","
        row = row + "\"r\":" + str(a_circle[2]) + ","
        row = row + "\"angle\":" + str(a_circle[3]) + "\"angle1\":" + str(a_circle[4]) + "\"angle2\":" + str(a_circle[5])
        row = row + ",\"eigen0\":" + str(eig_values[i, 0]) + ",\"eigen1\":" + str(eig_values[i, 1])
        row = row + "},\n"
        jsonfile.write(row)

    final_circle = circles_to_draw[n-1]
    final_row = "{"
    for j in range(0, total_dim - 1):
        if j == before_hdata_dim:
            final_row = final_row + "\"hdata\":{"
        final_row = final_row + "\"" + fieldlist[j] + "\": " + str(total_data[n-1, j]) + ","
    final_row = final_row + "\"" + fieldlist[total_dim - 1] + "\": " + str(total_data[n-1, total_dim - 1]) + "},"
    final_row = final_row + "\"isCircle\":" + str(is_circle[n-1]) + ","
    final_row = final_row + "\"oX\":" + str(final_circle[0]) + "\"oY\":" + str(final_circle[1]) + ","
    final_row = final_row + "\"r\":" + str(final_circle[2]) + ","
    final_row = final_row + "\"angle\":" + str(final_circle[3]) + "\"angle1\":" + str(final_circle[4]) + "\"angle2\":" + str(final_circle[5])
    final_row = final_row + ",\"eigen0\":" + str(eig_values[n-1, 0]) + ",\"eigen1\":" + str(eig_values[n-1, 1])
    final_row = final_row + "}\n"
    jsonfile.write(final_row)

    print('已经成功生成json文件')


def save_lists(data, path):
    """
    保存一个list的list的list
    :param data: 要保存的数据，这是一个list，其中的每一个元素也是一个list
    :param path: 要储存的文件路径
    :return:
    """
    csvfile = open(path, 'w', newline="")
    writer = csv.writer(csvfile)

    for row in data:
        new_row = []
        for element in row:
            new_row.append(element[0])
            new_row.append(element[1])

        writer.writerow(new_row)
    csvfile.close()
