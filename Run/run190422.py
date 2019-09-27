import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from numpy import linalg as LA
import csv
from Main import LocalPCA
from Main import Preprocess
from Main import processData as pD
from Tools import SaveData
from Main import DimReduce
from Main import OutShape
from sklearn.decomposition import PCA
import os

from Convex_hull import GrahamScan
from BSpline import b_spline

from time import time
from Tools import SymbolAdjust

# 以下为临时引用
from Tools import MySort
from JSON_Data import polygon_json

from sklearn.neighbors import NearestNeighbors
import math

""""

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


def perturb_once(data, prefect_k, eigen_numbers, y_init, method_k=30, MAX_EIGEN_COUNT=5, method_name="MDS",
                 yita=0.1, save_path=""):
    """
    一次性对所有的点添加扰动，是之前使用个的方法，有时候可能需要对投影结果进行一些强制恢复
    现在不提倡使用该方法 2019-02-03
    :param data: 经过normalize之后的原始数据矩阵
    :param prefect_k:  每一个点的最佳k值
    :param eigen_numbers: 每个点所需的特征值的数目
    :param y_init: 某些降维方法所需的初始随机矩阵
    :param enough: 是否达到给定的阈值要求
    :param label: 数据的label
    :param method_k: 某些降维方法所需要使用的k值
    :param MAX_EIGEN_COUNT: 最多使用的特征值数目
    :param method_name: 所使用的降维方法
    :param yita: 扰动所乘的系数
    :param save_path: 存储中间结果的路径
    :return:
    """
    data_shape = data.shape
    n = data_shape[0]
    dim = data_shape[1]

    # 检查method_k的值是否合理
    if method_k <= dim:
        method_k = dim+1
    elif method_k > n:
        method_k = n

    # max_prefect_k = 0
    # for i in range(0, n):
    #     if max_prefect_k < prefect_k[i]:
    #         max_prefect_k = prefect_k[i]
    knn = Preprocess.knn(data, n-1)
    colors = ['r', 'g', 'b', 'm', 'yellow', 'k', 'c']  # 所支持的颜色，用于区分不同的特征向量

    y_list_add = []  # 储存的元素是矩阵，把多次降维投影的结果矩阵存储起来
    y_list_sub = []

    eigen_vectors_list = []  # 存储的元素是特征向量矩阵，第i个元素里面存放的是每个点的第i个特征向量
    eigen_values = np.zeros((n, dim))  # 存储对每个点的localPCA所得的特征值

    for i in range(0, MAX_EIGEN_COUNT):
        eigen_vectors_list.append(np.zeros((n, dim)))

    for i in range(0, n):
        local_data = np.zeros((int(prefect_k[i]), dim))
        for j in range(0, int(prefect_k[i])):
            local_data[j, :] = data[knn[i, j], :]
        temp_vectors, eigen_values[i, :] = LocalPCA.local_pca_dn(local_data)

        for j in range(0, MAX_EIGEN_COUNT):
            eigenvectors = eigen_vectors_list[j]
            eigenvectors[i, :] = temp_vectors[j, :]

    np.savetxt(save_path+"eigenvalues.csv", eigen_values, fmt="%f", delimiter=",")

    # 开始进行降维
    y = np.zeros((n, 2))
    if method_name == "pca" or method_name == "PCA":
        print('当前使用PCA方法')
        pca = PCA(n_components=2, copy=True, whiten=True)
        pca.fit(data)
        P_ = pca.components_
        P = np.transpose(P_)
        y = np.matmul(data, P)
    else:
        y = DimReduce.dim_reduce(data, method=method_name, method_k=method_k, y_random=y_init)
    for loop_index in range(0, MAX_EIGEN_COUNT):
        eigenvectors = eigen_vectors_list[loop_index]
        np.savetxt(save_path+"eigenvectors"+str(loop_index)+".csv", eigenvectors, fmt="%f", delimiter=",")
        x_add_v = np.zeros((n, dim))
        x_sub_v = np.zeros((n, dim))
        for i in range(0, n):
            if eigen_numbers[i] > loop_index:
                x_add_v[i, :] = data[i, :] + yita*eigenvectors[i, :]
                x_sub_v[i, :] = data[i, :] - yita*eigenvectors[i, :]
            else:
                x_add_v[i, :] = data[i, :]
                x_sub_v[i, :] = data[i, :]

        y_add_v = np.zeros((n, 2))
        y_sub_v = np.zeros((n, 2))
        if method_name == "pca" or method_name == "PCA":
            print('当前使用PCA方法')
            y_add_v = np.matmul(x_add_v, P)
            y_sub_v = np.matmul(x_sub_v, P)
        else:
            y_add_v = DimReduce.dim_reduce(x_add_v, method=method_name, method_k=method_k, y_random=y)
            y_sub_v = DimReduce.dim_reduce(x_sub_v, method=method_name, method_k=method_k, y_random=y)
        y_list_add.append(y_add_v)
        y_list_sub.append(y_sub_v)

    return y, y_list_add, y_list_sub


def perturb_once_weighted(data, prefect_k, eigen_numbers, y_init, MAX_K, method_k=30, MAX_EIGEN_COUNT=5, method_name="MDS",
                 yita=0.1, save_path="", weighted=True):
    """
    一次性对所有的点添加扰动，是之前使用过的方法
    这里各个特征向量的扰动按照特征值的比重添加权重
    :param data: 经过normalize之后的原始数据矩阵
    :param prefect_k:  每一个点的最佳k值
    :param eigen_numbers: 每个点所需的特征值的数目
    :param y_init: 某些降维方法所需的初始随机矩阵
    :param enough: 是否达到给定的阈值要求
    :param label: 数据的label
    :param method_k: 某些降维方法所需要使用的k值
    :param MAX_EIGEN_COUNT: 最多使用的特征值数目
    :param method_name: 所使用的降维方法
    :param yita: 扰动所乘的系数
    :param save_path: 存储中间结果的路径
    :param weighted: 特征向量作为扰动时是否按照其所对应的特征值分配权重
    :return:
    """
    data_shape = data.shape
    n = data_shape[0]
    dim = data_shape[1]

    # 检查method_k的值是否合理
    if method_k <= dim:
        method_k = dim+1
    elif method_k > n:
        method_k = n

    if weighted:
        save_path = save_path + "【weighted】"
    # max_prefect_k = 0
    # for i in range(0, n):
    #     if max_prefect_k < prefect_k[i]:
    #         max_prefect_k = prefect_k[i]
    knn = Preprocess.knn(data, MAX_K)
    np.savetxt(save_path+"knn.csv", knn, fmt="%d", delimiter=",")

    # 计算每个点出现在多少个邻域中，这是一个类似于密度的概念
    knn_used = []
    for i in range(0, n):
        nbrs = knn[i, 0:int(prefect_k[i])]
        knn_used.append(nbrs)

    density = k_density(knn_used)
    np.savetxt(save_path+"k_density.csv", density, fmt="%d", delimiter=",")

    colors = ['r', 'g', 'b', 'm', 'yellow', 'k', 'c']  # 所支持的颜色，用于区分不同的特征向量

    y_list_add = []  # 储存的元素是矩阵，把多次降维投影的结果矩阵存储起来
    y_list_sub = []

    eigen_vectors_list = []  # 存储的元素是特征向量矩阵，第i个元素里面存放的是每个点的第i个特征向量
    eigen_values = np.zeros((n, dim))  # 存储对每个点的localPCA所得的特征值

    eigen_weights = np.ones((n, dim))  # 计算每个特征值占所有特征值和的比重

    for i in range(0, MAX_EIGEN_COUNT):
        eigen_vectors_list.append(np.zeros((n, dim)))

    for i in range(0, n):
        local_data = np.zeros((int(prefect_k[i]), dim))
        for j in range(0, int(prefect_k[i])):
            local_data[j, :] = data[knn[i, j], :]
        temp_vectors, eigen_values[i, :] = LocalPCA.local_pca_dn(local_data)

        for j in range(0, MAX_EIGEN_COUNT):
            eigenvectors = eigen_vectors_list[j]
            eigenvectors[i, :] = temp_vectors[j, :]

        if weighted:  # 判断是否需要分配权重
            temp_eigen_sum = sum(eigen_values[i, :])
            for j in range(0, dim):
                eigen_weights[i, j] = eigen_values[i, j]/temp_eigen_sum

    eigen1_div_2 = pD.eigen1_divide_eigen2(eigen_values)
    np.savetxt(save_path + "eigen1_div_eigen2_original.csv", eigen1_div_2, fmt="%f", delimiter=",")

    np.savetxt(save_path+"eigenvalues.csv", eigen_values, fmt="%f", delimiter=",")
    np.savetxt(save_path + "eigenweights.csv", eigen_weights, fmt="%f", delimiter=",")

    # 计算高维中第一特征向量与第二特征向量的角度
    eigen_vectors1 = eigen_vectors_list[0]
    eigen_vectors2 = eigen_vectors_list[1]

    angles_v1_v2 = pD.angle_v1_v2(eigen_vectors1, eigen_vectors2)
    np.savetxt(save_path+"angles_v1_v2_original.csv", angles_v1_v2, fmt="%f", delimiter=",")

    # 计算实际使用的特征值的占比
    # real_proportion = np.zeros((n, 1))
    # for i in range(0, n):
    #     temp =

    # 开始进行降维
    y = np.zeros((n, 2))
    if method_name == "pca" or method_name == "PCA":
        print('当前使用PCA方法')
        pca = PCA(n_components=2, copy=True, whiten=True)
        pca.fit(data)
        P_ = pca.components_
        P = np.transpose(P_)
        y = np.matmul(data, P)
        np.savetxt(save_path+"P.csv", P, fmt="%f", delimiter=",")
    elif method_name == "tsne2" or method_name == "t-SNE2":
        print('当前使用比较稳定的t-SNE方法')
        tsne = TSNE(n_components=2, perplexity=method_k / 3, init=y_init)
        y = tsne.fit_transform(data)
    else:
        y = DimReduce.dim_reduce(data, method=method_name, method_k=method_k, y_random=y_init)

    # 开始执行扰动计算
    for loop_index in range(0, MAX_EIGEN_COUNT):
        eigenvectors = eigen_vectors_list[loop_index]
        np.savetxt(save_path+"eigenvectors"+str(loop_index)+".csv", eigenvectors, fmt="%f", delimiter=",")
        x_add_v = np.zeros((n, dim))
        x_sub_v = np.zeros((n, dim))
        for i in range(0, n):
            if eigen_numbers[i] > loop_index:
                x_add_v[i, :] = data[i, :] + yita*eigen_weights[i, loop_index]*eigenvectors[i, :]
                x_sub_v[i, :] = data[i, :] - yita*eigen_weights[i, loop_index]*eigenvectors[i, :]
            else:
                x_add_v[i, :] = data[i, :]  # 在这里改动可以设置所有数据使用相同数目的特征向量
                x_sub_v[i, :] = data[i, :]
                # x_add_v[i, :] = data[i, :] + yita * eigen_weights[i, loop_index] * eigenvectors[i, :]
                # x_sub_v[i, :] = data[i, :] - yita * eigen_weights[i, loop_index] * eigenvectors[i, :]

        np.savetxt(save_path+"x_add_v"+str(loop_index)+".csv", x_add_v, fmt="%f", delimiter=",")
        np.savetxt(save_path + "x_sub_v" + str(loop_index) + ".csv", x_sub_v, fmt="%f", delimiter=",")

        y_add_v = np.zeros((n, 2))
        y_sub_v = np.zeros((n, 2))
        if method_name == "pca" or method_name == "PCA":
            print('当前使用PCA方法')
            y_add_v = np.matmul(x_add_v, P)
            y_sub_v = np.matmul(x_sub_v, P)
        elif method_name == "tsne2" or method_name == "t-SNE2":
            print('当前使用比较稳定的t-SNE方法')
            tsne = TSNE(n_components=2, n_iter=1, perplexity=method_k / 3, init=y)
            y_add_v = tsne.fit_transform(x_add_v)
            y_sub_v = tsne.fit_transform(x_sub_v)
        else:
            y_add_v = DimReduce.dim_reduce(x_add_v, method=method_name, method_k=method_k, y_random=y)
            # y_sub_v = 2*y-y_add_v  # 胡乱加的，要改回去
            y_sub_v = DimReduce.dim_reduce(x_sub_v, method=method_name, method_k=method_k, y_random=y)

        y_add_v = SymbolAdjust.symbol_adjust(y, y_add_v)
        y_sub_v = SymbolAdjust.symbol_adjust(y, y_sub_v)

        y_list_add.append(y_add_v)
        y_list_sub.append(y_sub_v)

    return y, y_list_add, y_list_sub


def perturb_together(data, prefect_k, eigen_numbers, y_init, MAX_K, method_k=30, MAX_EIGEN_COUNT=4, method_name="MDS",
                 yita=0.1, save_path=""):
    """
    将原始数据与使用不同特征向量扰动的所有扰动数据放到一起，进行降维
    写程序的时候需要注意数据结构与之前的程序兼容的问题
    时七年四月二十二日
    :param data: 原始的高维数据矩阵，每一行是一个样本
    :param prefect_k: 记录了每个样本点的最佳k值
    :param eigen_numbers: 每个样本点所使用的特征向量个数
    :param y_init: 初始的y矩阵
    :param MAX_K: 所能使用的最大k值
    :param method_k: 某些降维方法所使用的k值
    :param MAX_EIGEN_COUNT: 所能使用的最多的特征向量个数
    :param method_name: 降维方法的名称
    :param yita: 扰动率
    :param save_path: 储存中间结果的路径
    :return:
    """

    """目前该方法仅支持MDS方法  时七年四月二十二日"""
    data_shape = data.shape
    n = data_shape[0]
    dim = data_shape[1]

    # 检查method_k的值是否合理
    if method_k <= dim:
        method_k = dim + 1
    elif method_k > n:
        method_k = n

    save_path = save_path + "【weighted】"
    knn = Preprocess.knn(data, MAX_K)
    np.savetxt(save_path+"knn.csv", knn, fmt="%d", delimiter=",")

    y_list_add = []  # 储存的元素是矩阵
    y_list_sub = []

    eigen_vectors_list = []  # 储存的元素是特征向量矩阵，第i个元素里面存放的是每个点的第i个特征向量
    eigen_values = np.zeros((n, dim))  # 存储每个点的localPCA所得的特征值

    eigen_weights = np.zeros((n, dim))  # 计算每个特征值占所有特征值的比重

    for i in range(0, MAX_EIGEN_COUNT):
        eigen_vectors_list.append(np.zeros((n, dim)))

    for i in range(0, n):
        local_data = np.zeros((int(prefect_k[i]), dim))
        for j in range(0, int(prefect_k[i])):
            local_data[j, :] = data[knn[i, j], :]
        temp_vectors, eigen_values[i, :] = LocalPCA.local_pca_dn(local_data)

        for j in range(0, MAX_EIGEN_COUNT):
            eigenvectors = eigen_vectors_list[j]
            eigenvectors[i, :] = temp_vectors[j, :]

        temp_eigen_sum = sum(eigen_values[i, :])
        for j in range(0, dim):
            eigen_weights[i, j] = eigen_values[i, j]/temp_eigen_sum

    eigen1_div_2 = pD.eigen1_divide_eigen2(eigen_values)
    np.savetxt(save_path + "eigen1_div_eigen2_original.csv", eigen1_div_2, fmt="%f", delimiter=",")

    np.savetxt(save_path + "eigenvalues.csv", eigen_values, fmt="%f", delimiter=",")
    np.savetxt(save_path + "eigenweights.csv", eigen_weights, fmt="%f", delimiter=",")

    # 计算高维中第一特征向量与第二特征向量的角度
    eigen_vectors1 = eigen_vectors_list[0]
    eigen_vectors2 = eigen_vectors_list[1]

    angles_v1_v2 = pD.angle_v1_v2(eigen_vectors1, eigen_vectors2)
    np.savetxt(save_path + "angles_v1_v2_original.csv", angles_v1_v2, fmt="%f", delimiter=",")

    # 准备降维数据
    x_merge = np.zeros((n*(2*MAX_EIGEN_COUNT+1), dim))  # 合并版的高维数据集，包含了所有的扰动情况
    x_merge[0:n, :] = data[0:n, :]

    for loop_index in range(0, MAX_EIGEN_COUNT):
        eigenvectors = eigen_vectors_list[loop_index]
        np.savetxt(save_path+"eigenvectors"+str(loop_index)+".csv", eigenvectors, fmt="%f", delimiter=",")
        x_add_v = np.zeros((n, dim))
        x_sub_v = np.zeros((n, dim))

        for i in range(0, n):
            if eigen_numbers[i] > loop_index:
                x_add_v[i, :] = data[i, :] + yita*eigen_weights[i, loop_index]*eigenvectors[i, :]
                x_sub_v[i, :] = data[i, :] - yita*eigen_weights[i, loop_index]*eigenvectors[i, :]
            else:
                x_add_v[i, :] = data[i, :]  # 控制使用特征向量的个数，修改一下可以改成都使用相同数目的特征向量
                x_sub_v[i, :] = data[i, :]
        np.savetxt(save_path+"x_add_v"+str(loop_index)+".csv", x_add_v, fmt="%f", delimiter=",")
        np.savetxt(save_path+"x_sub_v"+str(loop_index)+".csv", x_sub_v, fmt="%f", delimiter=",")

        x_merge[((loop_index+1)*2-1)*n:(loop_index+1)*2*n, :] = x_add_v[0:n, :]
        x_merge[(loop_index+1)*2*n:((loop_index+1)*2+1)*n, :] = x_sub_v[0:n, :]

    np.savetxt(save_path+"x_merge.csv", x_merge, fmt="%f", delimiter=",")

    # 执行降维
    if method == "newMDS":
        print("当前使用合并执行的MDS方法")
        y = DimReduce.dim_reduce(data, method="MDS")
        y_merge = DimReduce.dim_reduce(x_merge, method="MDS")
        np.savetxt(save_path+"y_merge.csv", y_merge, fmt="%f", delimiter=",")
        y_new, y_list_add, y_list_sub = divide_y(y_merge, MAX_EIGEN_COUNT, n)
    else:
        print("没有匹配到合适的降维方法")
        return

    return y_new, y_list_add, y_list_sub


def divide_y(y_merge, MAX_EIGEN_COUNT, n):
    """
    将合并执行所得到的降维结果拆分开来，还原为原来老版本中的数据结构
    时七年四月二十二日
    :param y_merge: 合并执行所得到的降维结果
    :param MAX_EIGEN_COUNT: 最多使用的特征向量个数
    :param n: 原始数据集的样本个数
    :return:
    """
    y_list_add = []
    y_list_sub = []

    y = np.zeros((n, 2))
    y[0:n, :] = y_merge[0:n, :]

    for loop_index in range(0, MAX_EIGEN_COUNT):
        y_add = np.zeros((n, 2))
        y_sub = np.zeros((n, 2))
        y_add[0:n, :] = y_merge[(2*(loop_index+1)-1)*n:2*(loop_index+1)*n, :]
        y_sub[0:n, :] = y_merge[2*(loop_index+1)*n:(2*(loop_index+1)+1)*n, :]

        y_list_add.append(y_add)
        y_list_sub.append(y_sub)
    return y, y_list_add, y_list_sub


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


def run(main_path, data_name, yita=0.1, threshold=0.75, method_k=30, max_eigen_numbers=5, MAX_K=70, method="MDS",
        draw_kind="line", has_line=False, hasLabel=False, adaptive_threshold=0.0, to_normalize=False, do_straight=True,
        weighted=True, MAX_NK=(0.36, 0.5)):
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
                        polygon: 严格地连接各个扰动的投影组成多边形，很有可能不是凸多边形
                        bezier: 使用各个扰动点计算bezier曲线，画bezier曲线
    :param has_line: 在画凸包或星多边形的时候，是否将线也画上
    :param hasLabel: 数据是否是有标签的数据
    :param adaptive_threshold: 适应度值
    :param to_normalize: 是否需要对数据先做normalize，当使用一些比较常用的简单的三维数据时，如果使用normalize会与网上的一些操作不一致
                        因为它们大都没有进行normalize操作
    :param weighted: 在使用特征向量作为扰动的时候是否需要根据其特征值分配权重
    :param MAX_NK: 是一个元组，并且这个元组中的两个数都是0~1之间的值。第一个值控制k值差别大小，第二个值控制超过差别阈值的邻居个数
                    如果一个点的dim+1邻域中有超过(dim+1)*MAX_NK[1]个点与这个点的k值差别超过了(MAX_K-dim-1)*MAX_NK[0]，则我们需要考虑采取措施

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

    save_path = main_path + method + "\\" + data_name + "\\yita(" + str(yita) + ")method_k(" + str(method_k) + ")max_k(" + str(max_k) +")numbers("+str(max_eigen_numbers) + ")proportion(" + str(threshold) +")adapt_threshold("+str(adaptive_threshold)+")"
    save_path = save_path + "_" + draw_kind
    if weighted:
        save_path = save_path + "_weighted"
    else:
        save_path = save_path + "_withoutweight"
    save_path = save_path + "MAX_NK("+str(MAX_NK[0])+"-"+str(MAX_NK[1])+")"
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

    if MAX_K > n:
        MAX_K = n
        print("所输入的最大k值过大")

    # 检查localPCA是否已经计算过
    # has_local_pca = local_pca_calculated(main_path, data_name, MAX_K, max_eigen_numbers, threshold, adapt_threshold)
    local_pca_path = main_path+"localPCA\\"+data_name+"\\"+"max_k("+str(MAX_K)+") number("+str(max_eigen_numbers)+") proportionThreshold("+str(threshold)+") adaptThreshold("+str(adapt_threshold)+")"
    local_pca_path = local_pca_path + "MAX_NK("+str(MAX_NK[0]) + "-" + str(MAX_NK[1])+")"
    local_pca_path = local_pca_path + "\\"

    has_local_pca = local_pca_calculated(local_pca_path)

    local_pca_start = time()
    if has_local_pca:
        print("已经计算过prefect-k，直接从文件中读取")
        prefect_k_reader = np.loadtxt(local_pca_path+"prefect_k.csv", dtype=np.str, delimiter=",")
        prefect_k = prefect_k_reader.astype(np.int)
        eigen_numbers_reader = np.loadtxt(local_pca_path+"eigen_numbers.csv", dtype=np.str, delimiter=",")
        eigen_numbers = eigen_numbers_reader.astype(np.int)
    else:
        print("尚未计算过prefect-k，现在开始计算")
        print("求解最佳k值")
        Preprocess.check_filepath(local_pca_path)
        prefect_k = np.zeros((n, 1))  # 存储每个点的最佳k值
        eigen_numbers = np.zeros((n, 1))  # 存储每个点需要的特征值数量
        enough = np.zeros((n, 1))
        final_eigen_count = np.zeros((n, 1))

        for loop_index in range(1, max_eigen_numbers):
            eigen_count, k_start = LocalPCA.how_many_k(x, MAX_K=MAX_K, eigen_numbers=loop_index+1)
            temp_prefect_k, max_eigens_count = LocalPCA.find_max_k_adaptive(eigen_count, k0=k_start, low_k=k_start, up_k=MAX_K, threshold=adaptive_threshold)

            for i in range(0, n):
                if enough[i] == 1:
                    continue
                if max_eigens_count[i] > threshold:
                    prefect_k[i] = temp_prefect_k[i]
                    eigen_numbers[i] = loop_index + 1
                    enough[i] = 1
                    final_eigen_count[i] = max_eigens_count[i]

            # 检查是否可以结束
            finished = True
            for i in range(0, n):
                if prefect_k[i] == 0:
                    finished = False
                    break
            if finished:
                break

            # 如果将能够允许使用的所有特征值都使用了，还是不能达到要求比重，需要强行终止
            if loop_index == max_eigen_numbers-1:
                for i in range(0, n):
                    if prefect_k[i] == 0:
                        prefect_k[i] = temp_prefect_k[i]
                        eigen_numbers[i] = max_eigen_numbers
                        final_eigen_count[i] = max_eigens_count[i]
        print('最佳k值求解完成，下面开始降维计算')

        # --------------------------------------------------------------------------------------------------------------
        # 这里增加一个使用第二个阈值的方法，时七年五月十三日
        # MAX_NK = 0.36  # 第二个阈值，每个点的k值与它的邻居的k值的差别不要超过整个k值范围大小的MAX_NK倍
        nbr_s = NearestNeighbors(n_neighbors=dim+1, algorithm='ball_tree').fit(data)
        distance, knn = nbr_s.kneighbors(data)
        # prefect_k2 = np.zeros((n, 1))  # 一个新的存储最佳k值的向量
        for i in range(0, n):
            bad_number = 0  # 统计它的邻居当中有多少个的k值与它自己的k值差别超过了阈值
            nbrs_k = []  # 暂时存储它的邻居的k值都是多少
            for j in range(0, dim+1):
                nbrs_k.append(prefect_k[knn[i, j]])
                if np.abs(prefect_k[knn[i, j]] - prefect_k[i]) > (MAX_K-dim-1)*MAX_NK[0]:
                    bad_number = bad_number + 1

            if bad_number > (dim+1)*MAX_NK[1]:  # 如果与超过一半的邻居差别很大
                print(i, 'need to do something')  # 直接在原来的向量上面改？
                prefect_k[i] = math.ceil(np.mean(nbrs_k))
        # 上面增加了一个使用第二个阈值的方法
        # --------------------------------------------------------------------------------------------------------------

        np.savetxt(local_pca_path + "eigens_counts.csv", final_eigen_count, fmt="%f", delimiter=",")
        np.savetxt(local_pca_path + "enough.csv", enough, fmt="%d", delimiter=",")
        np.savetxt(local_pca_path + "eigen_numbers.csv", eigen_numbers, fmt="%d", delimiter=",")
        np.savetxt(local_pca_path + "prefect_k.csv", prefect_k, fmt="%d", delimiter=",")
        plt.hist(prefect_k, color='darkblue', bins=20)  # 高维空间中的距离比
        plt.title("prefect_k")
        plt.savefig(local_pca_path + "prefect_k.png")
        plt.close()

        plt.hist(final_eigen_count, color='darkblue', bins=20)  # 高维空间中的距离比
        plt.title("proportion")
        plt.savefig(local_pca_path + "proportion.png")
        plt.close()

    local_pca_end = time()
    print("计算prefect-k的时间为\t", local_pca_end-local_pca_start)

    perturb_start = time()
    if method == "newMDS":  # 采用合并的降维方式
        y, y_list_add, y_list_sub = perturb_together(x, prefect_k, eigen_numbers, y_random, MAX_K,
                                                          method_k=method_k,
                                                          MAX_EIGEN_COUNT=max_eigen_numbers, method_name=method,
                                                          yita=yita,
                                                          save_path=save_path)
    else:
        y, y_list_add, y_list_sub = perturb_once_weighted(x, prefect_k, eigen_numbers, y_random, MAX_K,
                                                          method_k=method_k,
                                                          MAX_EIGEN_COUNT=max_eigen_numbers, method_name=method,
                                                          yita=yita,
                                                          save_path=save_path, weighted=weighted)
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
            # y_add_v = y_list_add[j]  # 临时修改为了画正反方向 鸿武七年三月十八日
            # y_sub_v = y_list_sub[j]

            for i in range(0, n):
                if eigen_numbers[i] > j:
                    plt.plot([y[i, 0], y_add_v[i, 0]], [y[i, 1], y_add_v[i, 1]], linewidth=0.8, c=colors[j], alpha=0.9)
                    plt.plot([y[i, 0], y_sub_v[i, 0]], [y[i, 1], y_sub_v[i, 1]], linewidth=0.8, c=colors[j], alpha=0.9)

    if draw_kind == "convex_hull" or draw_kind == "b-spline":
        print('使用凸包画法')
        total_list = []  # 存放所有数据的所有扰动投影点
        for i in range(0, n):
            temp_array = np.zeros((int(eigen_numbers[i]*2+1), 2))
            temp_point = y[i, :]
            temp_array[int(eigen_numbers[i]*2), :] = temp_point  # 最后存储的是原始数值降维的结果
            total_list.append(temp_array)

        for j in range(0, max_eigen_numbers):
            y_add_v = y_list_add_adjust[j]
            y_sub_v = y_list_sub_adjust[j]
            for i in range(0, n):
                if eigen_numbers[i] > j:
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
        # temp_path = save_path + "temp\\"
        # Preprocess.check_filepath(temp_path)
        # for i in range(0, n):
        #     temp_array = total_list[i]
        #     np.savetxt(temp_path+str(i)+".csv", temp_array, fmt="%f", delimiter=",")

    if draw_kind == "b-spline":
        print("使用B样条画法")
        total_list = []  # 存放所有数据的所有扰动投影点
        for i in range(0, n):
            temp_array = np.zeros((int(eigen_numbers[i] * 2 + 1), 2))
            temp_point = y[i, :]
            temp_array[int(eigen_numbers[i] * 2), :] = temp_point  # 最后存储的是不添加扰动降维的结果
            total_list.append(temp_array)
        print("total_list 0 ", len(total_list))

        for j in range(0, max_eigen_numbers):
            y_add_v = y_list_add_adjust[j]
            y_sub_v = y_list_sub_adjust[j]
            for i in range(0, n):
                if eigen_numbers[i] > j:
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
            if len(temp_convex) > 3:
                splines = b_spline.bspline(temp_convex, n=100, degree=3, periodic=True)
                spline_x, spline_y = splines.T
            else:
                spline_x = []
                spline_y = []
                for j in range(0, len(temp_convex)):
                    spline_x.append(temp_convex[j, 0])
                    spline_y.append(temp_convex[j, 1])

            if len(temp_convex) < 4:
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
            temp_array = np.zeros((int(eigen_numbers[i]*2+1), 2))
            temp_point = y[i, :]
            temp_array[int(eigen_numbers[i]*2), :] = temp_point
            total_list.append(temp_array)

        for j in range(0, max_eigen_numbers):
            y_add_v = y_list_add[j]
            y_sub_v = y_list_sub[j]
            for i in range(0, n):
                if eigen_numbers[i] > j:
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
    final_star_shape_list = star_polygons(y, y_list_add, y_list_sub, max_eigen_numbers, eigen_numbers)
    SaveData.save_lists(final_star_shape_list, save_path+"final_star_polygon.csv")

    # 计算形变信息，鸿武七年九月十七日
    OutShape.angle1_2(save_path)
    OutShape.angle_p_n_weighted(save_path=save_path)


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


def star_polygons(y, y_list_add, y_list_sub, max_eigen_numbers, eigen_numbers):
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


if __name__ == "__main__":
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
    main_path_without_straighten = "F:\\result2019\\result0425without_straighten\\"
    main_path = "F:\\result2019\\result0917temp\\"

    data_name = "Wine"
    method = "PCA"
    yita = 0.3
    adapt_threshold = 1.0
    max_k = 30
    method_k = max_k
    eigen_numbers = 4
    draw_kind = "b-spline"
    threshold = 1.0
    normalize = True
    straighten = True  # 是否进行校直操作
    weighted = True  # 当使用特征向量作为扰动的时候是否添加权重
    MAX_NK = (1.5, 1.5)  # 用于控制adaptive k-value选择的数值

    # 默认是需要进行normalize的，如果不进行normalize需要更换主文件目录
    if not normalize:
        main_path = main_path_without_normalize

    if not straighten:
        main_path = main_path_without_straighten

    if (not normalize) and (not straighten):
        print("暂不支持，该组合形式")

    run(main_path, data_name, yita=yita, threshold=threshold, method_k=method_k, max_eigen_numbers=eigen_numbers, method=method, MAX_K=max_k,
        draw_kind=draw_kind, has_line=True, hasLabel=True, adaptive_threshold=adapt_threshold, to_normalize=normalize, do_straight=straighten,
        weighted=weighted, MAX_NK=MAX_NK)

    json_start = time()
    group_num = 10
    # main_path2 = main_path + method + "\\" + data_name + "\\"
    polygon_json.merge_json(main_path, data_name, method, yita, method_k, max_k, eigen_numbers, threshold, adapt_threshold, group_num, draw_kind, MAX_EIGEN_NUMBER=eigen_numbers, weighted=weighted, MAX_NK=MAX_NK)
    json_end = time()
    print("合成json文件的时间为\t", json_end-json_start)

    end_time = time()
    print("程序的总运行时间为\t", end_time-start_time)
