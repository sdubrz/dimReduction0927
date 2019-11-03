import numpy as np
import matplotlib.pyplot as plt

from Main import Preprocess
from Main import processData as pD
from Tools import SaveData
from Main import OutShape
import os
from Convex_hull import GrahamScan
from BSpline import b_spline
from time import time
from Main import Preturb
from Main import CleanData

from Tools import MySort
from JSON_Data import polygon_json190927
from JSON_Data import Json_2d
from JSON_Data import CircleScatter
from JSON_Data import highKNN_2dPCA
from Main import MainDirector

""""
本程序是基于run190422.py修改的
在不更改前端代码的前提下，显示一些新的属性，只能李代桃僵
鸿武七年九月十七日
---------------------------------------------------------------------------------------------

程序再做修改
首先将扰动的方法再改回原来的多次分别降维的方式
其次将特征向量作为扰动时分配权重改为可以通过一个变量选择是否分配权重。
这次主要的改动部分在perturb_once_weighted函数中，虽然之前也曾经使用过一个没有扰动的版本，
但是因为那个函数长时间不使用，版本较老，因而直接在这个有权重版本上面进行修改，使之支持是否使用权重。
这是基于鸿武七年五月十三日上午第一节课上讨论所改
时七年五月一十三日

-----------------------------------------------------------------------------------------------

本程序是基于 run190216.py 改写的
主要的改进是针对降维方法的使用
在之前的版本中，如果使用4个特征向量，则计算扰动时需要执行8次降维过程
在新的版本中将把原始数据与所有的扰动数据放在一起，只执行1次降维过程
在临时的版本中只实现对PCA与MDS这2种降维算法的支持。
时七年四月二十二日也

------------------------------------------------------------------------------------------------

本程序是基于 runData1220.py 改写的
主要是把高维部分的计算分离出来，使得在换用不同的降维算法的时候不用不断地去重新计算k值
@author: sdu_brz
@date: 2019/02/16
下面的部分是原来版本中的注释内容

-------------------------------------------------------------------------------------------------
同时画多个特征向量扰动的结果
以占比75%为阈值，默认最多选4个特征向量，这两个参数都要设成可以调节的
这种情况下每个点的k值和特征值的个数就都不一样了，对于降维来讲会存在一些问题，以前的降维方法不能直接使用了
有两种选择：
    1.直接使用原来的方法，当选择的特征值个数较多的时候，就会存在有些点没有添加扰动的情况。但是其他的点有了扰动，所以这个点也会发生变化
        最后可能会需要将之强制恢复
    2.每次计算一个点的一个扰动，如果5个特征值的话，可能对于每个点需要计算10次扰动，时间复杂度会相当高，但是不会存在那些问题
"""


def k_density(knn):
    """
    计算每个样本点被多少个样本的邻域所包含
    这个结果是一种相当于密度的概念
    :param knn: 每一个元素存储的是每个点的近邻，每个点的近邻数不一定相等
    :return:
    """
    n = len(knn)
    density = np.zeros((n, 1))

    for nbrs in knn:
        length = len(nbrs)
        for i in range(0, length):
            density[nbrs[i]] = density[nbrs[i]] + 1

    return density


def local_pca_calculated(path):
    """
    判断当前参数所需的 localPCA 是否已经计算过

    :return:
    """

    exist = os.path.exists(path + "prefect_k.csv")

    return exist


def main_run(main_path, data_name, nbrs_k=30, yita=0.1, method_k=30, max_eigen_numbers=5, method="MDS",
        draw_kind="line", has_line=False, hasLabel=False, to_normalize=False, do_straight=False,
        weighted=True, P_matrix=None):
    """"

    :param main_path: 主文件目录
    :param data_name: 所使用的数据名称
    :param threshold: 阈值，要求所选用的特征值之和占所有特征值的和必须超过这个值
    :param yita: 控制扰动的大小
    :param method_k: 有些降维方法所需要使用的k值
    :param max_eigen_numbers: 允许使用的最多的特征值数目
    :param MAX_K: 最大允许的k值，在找最佳k值时的限定条件
    :param MDS: 所使用的降维方法
    :param draw_kind: 画图的方式
                        line: 简单地将中心点与各个扰动点连接， 已实现
                        convex_hull: 计算凸包，绘制凸包
                        star_shape: 严格地连接各个扰动的投影组成多边形，很有可能不是凸多边形
                        b-spline: 使用各个扰动点计算bezier曲线，画bezier曲线
    :param has_line: 在画凸包或星多边形的时候，是否将线也画上
    :param hasLabel: 数据是否是有标签的数据
    :param adaptive_threshold: 适应度值
    :param to_normalize: 是否需要对数据先做normalize，当使用一些比较常用的简单的三维数据时，如果使用normalize会与网上的一些操作不一致
                        因为它们大都没有进行normalize操作
    :param weighted: 在使用特征向量作为扰动的时候是否需要根据其特征值分配权重
    :param MAX_NK: 是一个元组，并且这个元组中的两个数都是0~1之间的值。第一个值控制k值差别大小，第二个值控制超过差别阈值的邻居个数
                    如果一个点的dim+1邻域中有超过(dim+1)*MAX_NK[1]个点与这个点的k值差别超过了(MAX_K-dim-1)*MAX_NK[0]，则我们需要考虑采取措施
    :param P_matrix: 普通的线性降维方法的投影矩阵

    @author subbrz
    2018年12月20日
    """
    data_path = main_path + "datasets\\" + data_name + "\\data.csv"
    label_path = main_path + "datasets\\" + data_name + "\\label.csv"
    y_random_path = main_path + "datasets\\" + data_name + "\\y_random.csv"

    data_reader = np.loadtxt(data_path, dtype=np.str, delimiter=",")
    data = data_reader[:, :].astype(np.float)
    data_shape = data.shape
    n = data_shape[0]
    dim = data_shape[1]
    print(data_shape)

    label = np.zeros((n, 1))
    if hasLabel:
        label_reader = np.loadtxt(label_path, dtype=np.str, delimiter=",")
        label = label_reader.astype(np.int)

    if not os.path.exists(y_random_path):
        print("还没有随机初始结果，现在生成")
        y_random0 = np.random.random((n, 2))
        np.savetxt(y_random_path, y_random0, fmt="%f", delimiter=",")

    y_random_reader = np.loadtxt(y_random_path, dtype=np.str, delimiter=",")
    y_random = y_random_reader[:, :].astype(np.float)

    save_path = main_path + method + "\\" + data_name + "\\yita(" + str(yita) + ")nbrs_k(" + str(nbrs_k)
    save_path = save_path + ")method_k(" + str(method_k) + ")numbers("+str(max_eigen_numbers) + ")"
    save_path = save_path + "_" + draw_kind
    if weighted:
        save_path = save_path + "_weighted"
    else:
        save_path = save_path + "_withoutweight"

    save_path = save_path + "\\"

    Preprocess.check_filepath(save_path)
    print(save_path)

    if to_normalize:
        print('进行normalize')
        x = Preprocess.normalize(data)
    else:
        print('不进行normalize')
        x = data
    np.savetxt(save_path + "x.csv", x, fmt="%f", delimiter=",")
    np.savetxt(save_path + "label.csv", label, fmt="%d", delimiter=",")

    if max_eigen_numbers > dim:
        max_eigen_numbers = dim
        print("所要求的的特征值数目过多")

    perturb_start = time()
    if method == "newMDS":  # 采用合并的降维方式
        # y, y_list_add, y_list_sub = perturb_together(x, prefect_k, eigen_numbers, y_random, MAX_K,
        #                                                   method_k=method_k,
        #                                                   MAX_EIGEN_COUNT=max_eigen_numbers, method_name=method,
        #                                                   yita=yita,
        #                                                   save_path=save_path)
        print('暂时不推荐使用这种方法')
        return
    else:
        y, y_list_add, y_list_sub = Preturb.perturb_once_weighted(x, nbrs_k=nbrs_k, y_init=y_random,
                                                          method_k=method_k,
                                                          MAX_EIGEN_COUNT=max_eigen_numbers, method_name=method,
                                                          yita=yita,
                                                          save_path=save_path, weighted=weighted, P_matrix=P_matrix)
    perturb_end = time()
    print("降维所花费的时间为\t", perturb_end-perturb_start)

    shapes = []
    colors = ['r', 'g', 'b', 'm', 'yellow', 'k', 'c']  # 鸿武七年三月一十八日临时改动
    for i in range(0, n):
        if label[i] == 1:
            shapes.append('o')
        elif label[i] == 2:
            shapes.append('^')
        else:
            shapes.append('s')

    np.savetxt(save_path + "y.csv", y, fmt="%f", delimiter=",")
    for i in range(0, n):
        plt.scatter(y[i, 0], y[i, 1], marker=shapes[i], c='k')
    plt.title("y")
    plt.savefig(save_path + "y.png")
    plt.close()

    # 校直
    y_list_add_adjust = []  # 校正后的扰动投影坐标
    y_list_sub_adjust = []
    for loop_index in range(0, max_eigen_numbers):
        y_add_v = y_list_add[loop_index]
        y_sub_v = y_list_sub[loop_index]
        y_add_v_adjust, y_sub_v_adjust = perturbation_adjust(y, y_add_v, y_sub_v)

        if do_straight:
            y_list_add_adjust.append(y_add_v_adjust)
            y_list_sub_adjust.append(y_sub_v_adjust)
        else:
            y_list_add_adjust.append(y_add_v)
            y_list_sub_adjust.append(y_sub_v)

        np.savetxt(save_path+"y"+str(loop_index+1)+"+.csv", y_add_v, fmt="%f", delimiter=",")
        np.savetxt(save_path+"y"+str(loop_index+1)+"-.csv", y_sub_v, fmt="%f", delimiter=",")

        for i in range(0, n):
            plt.scatter(y_sub_v[i, 0], y_sub_v[i, 1], marker=shapes[i], c='k')
        plt.title("y"+str(loop_index+1)+"-")
        plt.savefig(save_path + "y"+str(loop_index+1)+"-"+".png")
        plt.close()

        for i in range(0, n):
            plt.scatter(y_add_v[i, 0], y_add_v[i, 1], marker=shapes[i], c='k')
        plt.title("y"+str(loop_index+1)+"+")
        plt.savefig(save_path + "y"+str(loop_index+1)+"+"+".png")
        plt.close()

    # 计算投影之后第一特征向量与第二特征向量之间的角度
    angles_v1_v2_projected = pD.angle_v1_v2(y_list_add_adjust[0], y_list_add_adjust[1], y=y)
    np.savetxt(save_path+"angles_v1_v2_projected.csv", angles_v1_v2_projected, fmt="%f", delimiter=",")

    # 计算投影之后扰动的比值
    eigen1_div_eigen2_projected = pD.length_radio(y_list_add_adjust[0], y_list_add_adjust[1], y=y)
    np.savetxt(save_path+"eigen1_div_eigen2_projected.csv", eigen1_div_eigen2_projected, fmt="%f", delimiter=",")

    for loop_index in range(0, max_eigen_numbers):
        y_add_v = y_list_add_adjust[loop_index]
        y_sub_v = y_list_sub_adjust[loop_index]
        np.savetxt(save_path + "y_add_"+str(loop_index+1)+".csv", y_add_v, fmt="%f", delimiter=",")
        np.savetxt(save_path + "y_sub_" + str(loop_index+1) + ".csv", y_sub_v, fmt="%f", delimiter=",")

        for i in range(0, n):
            plt.scatter(y_add_v[i, 0], y_add_v[i, 1], marker=shapes[i], c='k')
        plt.title("y_add_v_"+str(loop_index+1))
        plt.savefig(save_path + "y_add_v_"+str(loop_index+1)+".png")
        plt.close()

        for i in range(0, n):
            plt.scatter(y_sub_v[i, 0], y_sub_v[i, 1], marker=shapes[i], c='k')
        plt.title("y_sub_v_"+str(loop_index+1))
        plt.savefig(save_path + "y_sub_v_"+str(loop_index+1)+".png")
        plt.close()

    convex_hull_list = []  # 存储每个样本点的凸包顶点

    print('开始画图')
    for i in range(0, n):
        plt.scatter(y[i, 0], y[i, 1], marker=shapes[i], c='k', alpha=0.6)

    if draw_kind == "line" or has_line:
        for j in range(0, max_eigen_numbers):
            y_add_v = y_list_add_adjust[j]
            y_sub_v = y_list_sub_adjust[j]

            for i in range(0, n):
                plt.plot([y[i, 0], y_add_v[i, 0]], [y[i, 1], y_add_v[i, 1]], linewidth=0.8, c=colors[j], alpha=0.9)
                plt.plot([y[i, 0], y_sub_v[i, 0]], [y[i, 1], y_sub_v[i, 1]], linewidth=0.8, c=colors[j], alpha=0.9)

    if draw_kind == "convex_hull" or draw_kind == "b-spline":
        print('使用凸包画法')
        total_list = []  # 存放所有数据的所有扰动投影点
        for i in range(0, n):
            temp_array = np.zeros((max_eigen_numbers*2, 2))
            temp_point = y[i, :]
            temp_array[(max_eigen_numbers-1)*2, :] = temp_point  # 最后存储的是原始数值降维的结果
            total_list.append(temp_array)

        for j in range(0, max_eigen_numbers):
            y_add_v = y_list_add_adjust[j]
            y_sub_v = y_list_sub_adjust[j]
            for i in range(0, n):
                temp_array = total_list[i]
                temp_points1 = y_add_v[i, :]
                temp_points2 = y_sub_v[i, :]
                temp_array[2*j, :] = temp_points1
                temp_array[2*j+1, :] = temp_points2

        for i in range(0, n):
            temp_array = total_list[i]
            if len(temp_array) < 4:  # 小于4个的就不要求凸包了，这个点的一个特征值所占的比重就已经超过所要求的的阈值了
                                    # 只使用一个特征向量的话，其长度应该是3.
                ttt = []
                ttt.append([temp_array[0, 0], temp_array[0, 1]])
                ttt.append([temp_array[1, 0], temp_array[1, 1]])
                convex_hull_list.append(ttt)
                continue
            temp_convex0 = GrahamScan.graham_scan(temp_array.tolist())
            convex_hull_list.append(temp_convex0)  # 存储这个凸包顶点信息
            temp_convex = np.array(temp_convex0)

            if draw_kind == "convex_hull":
                for j in range(0, len(temp_convex)-1):
                    plt.plot([temp_convex[j, 0], temp_convex[j+1, 0]], [temp_convex[j, 1], temp_convex[j+1, 1]], linewidth=0.6, c='deepskyblue', alpha=0.7)
                plt.plot([temp_convex[len(temp_convex)-1, 0], temp_convex[0, 0]], [temp_convex[len(temp_convex)-1, 1], temp_convex[0, 1]],
                         linewidth=0.6, c='deepskyblue', alpha=0.7)

            SaveData.save_lists(convex_hull_list, save_path+"convex_hull_list.csv")
        SaveData.save_lists(convex_hull_list, save_path + "real_convex_hull_list.csv")

    if draw_kind == "b-spline":
        print("使用B样条画法")
        total_list = []  # 存放所有数据的所有扰动投影点
        for i in range(0, n):
            temp_array = np.zeros((max_eigen_numbers*2, 2))
            temp_point = y[i, :]
            temp_array[(max_eigen_numbers-1)*2, :] = temp_point  # 最后存储的是不添加扰动降维的结果
            total_list.append(temp_array)
        print("total_list 0 ", len(total_list))

        for j in range(0, max_eigen_numbers):
            y_add_v = y_list_add_adjust[j]
            y_sub_v = y_list_sub_adjust[j]
            for i in range(0, n):
                temp_array = total_list[i]
                temp_points1 = y_add_v[i, :]
                temp_points2 = y_sub_v[i, :]
                temp_array[2 * j, :] = temp_points1
                temp_array[2 * j + 1, :] = temp_points2

        b_spline_list = []

        for i in range(0, n):
            temp_array = total_list[i]
            if len(temp_array) < 4:  # 小于4个的就不要求凸包了，这个点的一个特征值所占的比重就已经超过所要求的的阈值了
                # 只使用一个特征向量的话，其长度应该是3.
                ttt = []
                ttt.append([temp_array[0, 0], temp_array[0, 1]])
                ttt.append([temp_array[1, 0], temp_array[1, 1]])
                convex_hull_list.append(ttt)
                temp_convex = np.array(ttt)

            else:
                temp_convex0 = GrahamScan.graham_scan(temp_array.tolist())
                convex_hull_list.append(temp_convex0)  # 存储这个凸包顶点信息
                temp_convex = np.array(temp_convex0)

            # 计算B样条
            if len(temp_convex) >= 3:
                splines = b_spline.bspline(temp_convex, n=100, degree=2, periodic=True)
                spline_x, spline_y = splines.T
            else:
                spline_x = []
                spline_y = []
                for j in range(0, len(temp_convex)):
                    spline_x.append(temp_convex[j, 0])
                    spline_y.append(temp_convex[j, 1])

            if len(temp_convex) < 3:
                print('bad')
            plt.plot(spline_x, spline_y, linewidth=0.6, c='deepskyblue', alpha=0.7)

            this_spline = []
            for j in range(0, len(spline_x)):
                this_spline.append([spline_x[j], spline_y[j]])
            b_spline_list.append(this_spline)

        SaveData.save_lists(b_spline_list, save_path + "convex_hull_list.csv")

    if draw_kind == "star_shape":
        print('使用星多边形画法')
        total_list = []  # 存放所有数据的所有扰动投影点
        for i in range(0, n):
            temp_array = np.zeros((max_eigen_numbers*2, 2))
            temp_point = y[i, :]
            temp_array[(max_eigen_numbers-1)*2, :] = temp_point
            total_list.append(temp_array)

        for j in range(0, max_eigen_numbers):
            y_add_v = y_list_add[j]
            y_sub_v = y_list_sub[j]
            for i in range(0, n):
                temp_array = total_list[i]
                temp_points1 = y_add_v[i, :]
                temp_points2 = y_sub_v[i, :]
                temp_array[2*j, :] = temp_points1
                temp_array[2*j+1, :] = temp_points2

        star_shape_list = []  # 存放星型多边形顶点信息，方便进行存储
        for i in range(0, n):
            temp_points = total_list[i]
            if len(temp_points) == 3:  # 也就是只使用一个特征向量就满足了所要求的比例
                plt.plot([temp_points[2, 0], temp_points[0, 0]], [temp_points[2, 1], temp_points[0, 1]], linewidth=0.8, c='deepskyblue', alpha=0.7)
                plt.plot([temp_points[2, 0], temp_points[1, 0]], [temp_points[2, 1], temp_points[1, 1]], linewidth=0.8,
                         c='deepskyblue', alpha=0.7)
                # 这里是一种默认的添加，使之兼容只使用一个特征向量的情况。
                sorted_points0 = temp_points[0:2, :]
                star_shape_list.append(sorted_points0)
                continue

            sorted_points = MySort.points_sort(temp_points)
            star_shape_list.append(sorted_points)
            for j in range(0, len(sorted_points)-1):
                plt.plot([sorted_points[j, 0], sorted_points[j+1, 0]], [sorted_points[j, 1], sorted_points[j+1, 1]], linewidth=0.8, c='deepskyblue', alpha=0.7)
            plt.plot([sorted_points[0, 0], sorted_points[len(sorted_points)-1, 0]], [sorted_points[0, 1], sorted_points[len(sorted_points)-1, 1]],
                     linewidth=0.8, c='deepskyblue', alpha=0.7)

        SaveData.save_lists(star_shape_list, save_path + "convex_hull_list.csv")

    ax = plt.gca()
    ax.set_aspect(1)
    plt.show()

    # 无论使用什么画法都需要把最原始的star_shape_polygon信息保存下来
    final_star_shape_list = star_polygons(y, y_list_add, y_list_sub, max_eigen_numbers)
    SaveData.save_lists(final_star_shape_list, save_path+"final_star_polygon.csv")

    # 计算形变信息，鸿武七年九月十七日
    OutShape.angle1_2(save_path)
    OutShape.angle_p_n_weighted(save_path=save_path, vector_num=max_eigen_numbers)
    OutShape.angle_p_n_1(save_path=save_path)

    return save_path


def perturbation_adjust(y, y_add_v, y_sub_v):
    """
    对扰动之后的投影进行校正
    加某一扰动与减去某一扰动，结果的方向本应该是相反的，因为所添加的扰动比较小
    但是在具体的实现过程中往往并不是共线的。
    这时候需要进行一下方向的校正。我想到的一种方式是使用平行四边形的方式。
    :param y: 原始数据的降维结果
    :param y_add: 原始数据加上扰动之后的降维结果
    :param y_sub: 原始数据减去扰动之后的降维结果
    :return:
    """
    y_add = y_add_v - y
    y_sub = y_sub_v - y

    y1 = y_add - y_sub
    y2 = y_sub - y_add

    y1 = y1/2 + y
    y2 = y2/2 + y

    return y1, y2


def star_polygons(y, y_list_add, y_list_sub, max_eigen_numbers):
    """
    计算并保存star-shape的坐标
    :param y: 不加扰动的降维结果
    :param y_list_add: 正向扰动的降维结果集
    :param y_list_sub: 负向扰动的降维结果集
    :param max_eigen_numbers: 最多允许使用的特征向量个数
    :param eigen_numbers: 每个点具体使用的特征向量个数
    :return:
    """
    data_shape = y.shape
    n = data_shape[0]

    eigen_numbers = max_eigen_numbers*np.ones((n, 1))

    total_list = []  # 存放所有数据的所有扰动投影点
    for i in range(0, n):
        temp_array = np.zeros((int(eigen_numbers[i] * 2 + 1), 2))
        temp_point = y[i, :]
        temp_array[int(eigen_numbers[i] * 2), :] = temp_point
        total_list.append(temp_array)

    for j in range(0, max_eigen_numbers):
        y_add_v = y_list_add[j]
        y_sub_v = y_list_sub[j]
        for i in range(0, n):
            if eigen_numbers[i] > j:
                temp_array = total_list[i]
                temp_points1 = y_add_v[i, :]
                temp_points2 = y_sub_v[i, :]
                temp_array[2 * j, :] = temp_points1
                temp_array[2 * j + 1, :] = temp_points2

    star_shape_list = []  # 存放星型多边形顶点信息，方便进行存储
    for i in range(0, n):
        temp_points = total_list[i]
        if len(temp_points) == 3:  # 也就是只使用一个特征向量就满足了所要求的比例
            plt.plot([temp_points[2, 0], temp_points[0, 0]], [temp_points[2, 1], temp_points[0, 1]], linewidth=0.8,
                     c='deepskyblue', alpha=0.7)
            plt.plot([temp_points[2, 0], temp_points[1, 0]], [temp_points[2, 1], temp_points[1, 1]], linewidth=0.8,
                     c='deepskyblue', alpha=0.7)
            # 这里是一种默认的添加，使之兼容只使用一个特征向量的情况。
            sorted_points0 = temp_points[0:2, :]
            star_shape_list.append(sorted_points0)
            continue

        sorted_points = MySort.points_sort(temp_points)
        star_shape_list.append(sorted_points)

    return star_shape_list


def data_shape(main_path, data_name):
    """
    获取数据的维度数和样本数
    :param main_path: 主文件目录
    :param data_name: 数据集名称
    :return:
    """
    data = np.loadtxt(main_path+"datasets\\"+data_name+"\\data.csv", dtype=np.float, delimiter=",")
    return data.shape


def run_test(data_name0=None):
    """"
        画图方法：
            convex_hull
            line
            star_shape
            b-spline
        数据集：
            Iris
            Wine
            bostonHouse
            CCPP
            wdbc
            digits5_8
        """
    start_time = time()
    main_path_without_normalize = "F:\\result2019\\result0223without_normalize\\"
    main_path_without_straighten = "E:\\project\\result2019\\result1026without_straighten\\"  # 华硕
    # main_path = "F:\\result2019\\result0927\\"  # HP
    main_path = "E:\\Project\\result2019\\result0927\\"  # 华硕
    # main_path = 'D:\\文件\\IRC\\特征向量散点图项目\\result2019\\result0927\\'  # XPS

    data_name = "Wine"
    if data_name0 is None:
        pass
    else:
        data_name = data_name0

    method = "PCA"  # "PCA" "MDS" "P_matrix" "Isomap"
    yita = 0.1
    nbrs_k = 45
    method_k = 20
    eigen_numbers = 4
    draw_kind = "b-spline"
    normalize = True
    straighten = False  # 是否进行校直操作
    weighted = True  # 当使用特征向量作为扰动的时候是否添加权重
    P_matrix = None  # 普通的线性降维方法的投影矩阵

    # 默认是需要进行normalize的，如果不进行normalize需要更换主文件目录
    if not normalize:
        main_path = main_path_without_normalize

    if not straighten:
        main_path = main_path_without_straighten

    if (not normalize) and (not straighten):
        print("暂不支持，该组合形式")

    # 继续设置普通的线性降维方法的参数
    (n, m) = data_shape(main_path, data_name)
    if method == "P_matrix":
        P_matrix = np.zeros((m, 2))
        x_index = 0  # 第一个维度
        y_index = 2  # 第二个维度
        P_matrix[x_index, 0] = 1
        P_matrix[y_index, 1] = 1

    last_path = main_run(main_path, data_name, nbrs_k=nbrs_k, yita=yita, method_k=method_k, max_eigen_numbers=eigen_numbers,
        method=method, draw_kind=draw_kind, has_line=True, hasLabel=True, to_normalize=normalize,
        do_straight=straighten, weighted=weighted, P_matrix=P_matrix)

    json_start = time()
    # main_path2 = main_path + method + "\\" + data_name + "\\"
    polygon_json190927.merge_json(main_path, data_name, method, yita, method_k, nbrs_k, draw_kind,
                                  MAX_EIGEN_NUMBER=eigen_numbers,
                                  weighted=weighted)
    json_end = time()
    print("合成json文件的时间为\t", json_end - json_start)

    end_time = time()
    print("程序的总运行时间为\t", end_time - start_time)

    Json_2d.create_json2(last_path, k=nbrs_k, line_length=0.1, draw_spline=False)
    print("计算二维完成")

    CircleScatter.circle_json(last_path, r=0.05)
    print('生成散点json完成')

    highKNN_2dPCA.create_json2(last_path, line_length=0.1)

    # 画主成分的投影方向
    MainDirector.draw_main_director(last_path)

    return last_path, data_name, main_path, method


if __name__ == "__main__":
    last_path, data_name, main_path, method = run_test()

    do_remove = False  # 是否要做删除outlier操作
    attri_name = "angle_+-sumweighted.csv"  # "angle_+-sumweighted.csv"  'sin_1_2.csv'
    threshold = 0.2
    compare = 'less'  # 'bigger' or 'less'

    if do_remove:
        if compare == 'less':
            CleanData.clean_small_value(data_name, main_path=main_path, last_path=last_path, attri_file=attri_name,
                                        threshold=threshold, method=method)
            run_test(data_name0=data_name+"Clean " + method + " " + attri_name + " " + compare + " " + str(threshold))
        elif compare == 'bigger':
            CleanData.clean_big_value(data_name, main_path=main_path, last_path=last_path, attri_file=attri_name,
                                        threshold=threshold, method=method)
            run_test(data_name0=data_name + "Clean " + method + " " + attri_name + " " + compare + " " + str(threshold))
