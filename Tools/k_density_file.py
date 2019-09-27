import numpy as np


"""从文件中的结果来计算每个点被多少个样本的邻域所包含
    姑且称之为 k_density
    @author: sdubrz
    @date: 2019/02/11

"""


def k_density(read_path):
    """

    :param read_path:
    :return:
    """
    knn_reader = np.loadtxt(read_path+"【weighted】knn.csv", dtype=np.str, delimiter=",")
    knn = knn_reader[:, :].astype(np.int)
    prefect_k_reader = np.loadtxt(read_path+"prefect_k.csv", dtype=np.str, delimiter=",")
    prefect_k = prefect_k_reader.astype(np.int)

    data_shape = knn.shape
    n = data_shape[0]
    density = np.zeros((n, 1))

    for i in range(0, n):
        for j in range(0, prefect_k[i]):
            density[knn[i, j]] = density[knn[i, j]] + 1

    np.savetxt(read_path+"k_density.csv", density, fmt="%d", delimiter=",")


if __name__ == "__main__":
    path = "F:\\result2019\\0130\\PCA\\Wine\\yita(0.5)method_k(50)max_k(50)numbers(4)k_threshold0.05_convex_hull\\"
    k_density(path)
