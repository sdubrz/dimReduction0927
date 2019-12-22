# 研究扰动量与导致的降维结果之间的变化情况
import numpy as np
import matplotlib.pyplot as plt
from Main import Preprocess
from MyDR import cTSNE


def preturb_scatter():
    """
    每个点的输入的改变量与降维结果的变化量之间的关系图
    :return:
    """
    # path = "E:\\Project\\result2019\\result1026without_straighten\\datasets\\Iris\\"
    path = "E:\\Project\\result2019\\result1026without_straighten\\cTSNE\\coil20obj_16_3class\\yita(0.01)nbrs_k(20)method_k(20)numbers(4)_b-spline_weighted\\"
    X = np.loadtxt(path+"x.csv", dtype=np.float, delimiter=",")
    (n, m) = X.shape
    X1 = np.loadtxt(path+"【weighted】x_add_v0.csv", dtype=np.float, delimiter=",")
    Y = np.loadtxt(path+"y.csv", dtype=np.float, delimiter=",")
    Y1 = np.loadtxt(path+"y_add_1.csv", dtype=np.float, delimiter=",")

    dx = np.zeros((n, 1))
    dy = np.zeros((n, 1))
    for i in range(0, n):
        dx[i] = np.linalg.norm(X1[i, :] - X[i, :])
        dy[i] = np.linalg.norm(Y1[i, :] - Y[i, :])

    plt.scatter(dx, dy)
    plt.xlabel("dx")
    plt.ylabel("dy")
    plt.show()

    plt.hist(dy)
    plt.xlabel("dy")
    plt.show()


if __name__ == '__main__':
    preturb_scatter()





