# random的扰动方法的实现
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from MyDR import cTSNE
import random


def all_perturbed(perturb_count, n):
    """
    判断是否所有的点都被扰动过
    :param perturb_count:
    :param n:
    :return:
    """
    for i in range(0, n):
        if perturb_count[i] == 0:
            return False
    return True


def random_perturb(X, Y, vectors, method="MDS", perplexity=30.0):
    """
    对每个点都加上一个向量做扰动
    :param X: 原始的高维数据矩阵
    :param Y: 没有扰动的降维结果
    :param vectors: 扰动向量集合
    :param method: 降维方法
    :param perplexity: 困惑度
    :return:
    """
    (n, m) = X.shape
    perturb_count = np.zeros((n, 1))
    Y2 = np.zeros((n, 2))
    indexs = range(0, n)

    while not all_perturbed(perturb_count, n):
        selected_indexs = random.sample(indexs, n//2)
        X2 = X.copy()
        for i in selected_indexs:
            X2[i, :] = X2[i, :] + vectors[i, :]
            perturb_count[i] += 1

        if method == "MDS":
            mds = MDS(n_components=2, n_init=1, max_iter=200, eps=-1.0)
            temp_y = mds.fit_transform(X2, init=Y)
        elif method == "cTSNE":
            tsne = cTSNE.cTSNE(n_component=2, perplexity=perplexity)
            temp_y = tsne.fit_transform(X2, max_iter=50, early_exaggerate=False, y_random=Y, follow_gradient=False)
        else:
            print("不支持该方法")
            return

        for i in selected_indexs:
            Y2[i, :] = Y2[i, :] + temp_y[i, :]

    for i in range(0, n):
        Y2[i, :] = Y2[i, :] / perturb_count[i]

    return Y2


def random_perturb_all(X, Y, method, vector_list, weights, perplexity=30.0):
    eigen_number = len(vector_list)
    y_add_list = []
    y_sub_list = []

    for loop_index in range(0, eigen_number):
        vectors = vector_list[loop_index].copy()
        (n, d) = vectors.shape
        for i in range(0, n):
            vectors[i, :] = vectors[i, :] * weights[i, loop_index]
        y_add = random_perturb(X, Y, vectors, method, perplexity)
        y_sub = Y + Y - y_add
        y_add_list.append(y_add)
        y_sub_list.append(y_sub)

    return y_add_list, y_sub_list
