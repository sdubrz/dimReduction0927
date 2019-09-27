import numpy as np
import numpy.linalg as LA


"""
    计算k近邻的半径大小，也就是计算样本与其所使用的最远的邻居之间的距离
    
    @author: sdubrz
    @date: 2019/02/12
"""


def nbr_radius(x, knn, k):
    """

    :param knn: k近邻矩阵
    :param k: 每个样本所使用的k值
    :return:
    """
    data_shape = x.shape
    n = data_shape[0]

    radius = np.zeros((n, 1))
    for i in range(0, n):
        radius[i] = LA.norm(x[i, :] - x[knn[i, k[i]], :])

    return radius


def run(main_path):
    x_reader = np.loadtxt(main_path+"x.csv", dtype=np.str, delimiter=",")
    x = x_reader[:, :].astype(np.float)
    knn_reader = np.loadtxt(main_path+"【weighted】knn.csv", dtype=np.str, delimiter=",")
    knn = knn_reader[:, :].astype(np.int)
    perfect_k_reader = np.loadtxt(main_path+"prefect_k.csv", dtype=np.str, delimiter=",")
    perfect_k = perfect_k_reader.astype(np.int)

    radius = nbr_radius(x, knn, perfect_k)
    np.savetxt(main_path+"knn_radius.csv", radius, fmt="%f", delimiter=",")


if __name__ == "__main__":
    path = "F:\\result2019\\0130\\PCA\\Wine\\yita(0.5)method_k(50)max_k(50)numbers(4)k_threshold0.05_convex_hull\\"
    run(path)
