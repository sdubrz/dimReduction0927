# 在某一个单独的属性上做扰动
import numpy as np
import matplotlib.pyplot as plt


def perturb_single_attr(path, attr=0, eta=0.02, method='PCA'):
    Y = np.loadtxt(path+"y.csv", dtype=np.float, delimiter=",")
    X = np.loadtxt(path+"x.csv", dtype=np.float, delimiter=",")
    label = np.loadtxt(path+"label.csv", dtype=np.int, delimiter=",")
    (n, m) = X.shape

    p_path = path
    if method == 'PCA':
        p_path = path + "【weighted】P.csv"
    elif method == "MDS":
        p_path = path + "MDS_Pxy.csv"
    elif method == "cTSNE":
        p_path = path + "cTSNE_Pxy.csv"

    P = np.loadtxt(p_path, dtype=np.float, delimiter=",")

    colors = ['r', 'g', 'b', 'c', 'm', 'y']
    for i in range(0, n):
        color_i = colors[(label[i]-1)%len(colors)]
        Pi = np.zeros((2, m))
        if method == "PCA":
            Pi = P.T
        else:
            Pi = P[i*2:i*2+2, m*i:m*i+m]
        y11 = Y[i, 0] + eta * Pi[0, attr]
        y12 = Y[i, 1] + eta * Pi[1, attr]

        plt.scatter(Y[i, 0], Y[i, 1], c=color_i, alpha=0.7)
        plt.plot([Y[i, 0], y11], [Y[i, 1], y12], c=color_i, alpha=0.8)

    plt.title(method+"attr "+str(attr))
    ax = plt.gca()
    ax.set_aspect(1)
    plt.show()


if __name__ == '__main__':
    path = "E:\\文件\\IRC\\特征向量散点图项目\\result2020\\result0119\\cTSNE\\Wine\\yita(0.202003062)nbrs_k(40)method_k(90)numbers(4)_b-spline_weighted\\"
    perturb_single_attr(path, attr=0, eta=0.1, method='cTSNE')

