# 散点图颜色测试
import numpy as np
import matplotlib.pyplot as plt


def draw_scatter():
    path = "E:\\Project\\result2019\\result1026without_straighten\\PCA\\Wine\\yita(0.05)nbrs_k(40)method_k(40)numbers(4)_b-spline_weighted\\"
    Y = np.loadtxt(path+"y.csv", dtype=np.float, delimiter=",")
    X = np.loadtxt(path+"x.csv", dtype=np.float, delimiter=",")
    (n, m) = X.shape

    plt.scatter(Y[:, 0], Y[:, 1], c=X[:, 0])
    plt.colorbar()
    ax = plt.gca()
    ax.set_aspect(1)
    plt.show()


if __name__ == '__main__':
    draw_scatter()
