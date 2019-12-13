# 扰动测试
import numpy as np
import matplotlib.pyplot as plt
from Main import DimReduce


def preturb():
    path = "E:\\Project\\result2019\\result1026without_straighten\\tsne\\Iris\\yita(0.0)nbrs_k(20)method_k(20)numbers(4)_b-spline_weighted\\"
    # path = "E:\\Project\\result2019\\result1026without_straighten\\tsne\\MNIST50mclass1_985\\yita(0.001)nbrs_k(90)method_k(90)numbers(4)_b-spline_weighted\\"
    X = np.loadtxt(path+"x.csv", dtype=np.float, delimiter=",")
    vectors = np.loadtxt(path+"【weighted】eigenvectors0.csv", dtype=np.float, delimiter=",")

    yita = 0.0
    k = 90
    X2 = X + yita*vectors
    Y = DimReduce.dim_reduce(X, method='tsne', method_k=k, n_iters=5000)
    Y2 = DimReduce.dim_reduce(X, method='tsne', method_k=k, y_random=Y, n_iters=5000)
    Y3 = DimReduce.dim_reduce(X, method='tsne', method_k=k, y_random=Y, n_iters=5010)

    plt.scatter(Y[:, 0], Y[:, 1], c='r')
    plt.scatter(Y2[:, 0], Y2[:, 1], c='g')
    plt.scatter(Y3[:, 0], Y3[:, 1], c='b')

    plt.show()


if __name__ == '__main__':
    preturb()


