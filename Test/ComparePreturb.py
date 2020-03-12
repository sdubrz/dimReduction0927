# 比较不同扰动方法的区别
import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import MDS
from Derivatives import MDS_Derivative
from Perturb2020 import MDS_Perturb
from Derivatives import TSNE_Derivative
from MyDR import cTSNE
import random
import time


class Compare:
    def __init__(self, data, vectors, eta=0.1, method='MDS'):
        """
        :param data: 高维数据
        :param vectors: 扰动向量
        :param eta: 扰动率
        """
        self.X = data
        self.vectors = vectors
        self.Y = None  # 不扰动时的降维结果
        self.Y1 = None  # 一个点一个点的扰动
        self.Y2 = None  # 随机选一半扰动
        self.Y3 = None  # 用求导的方法扰动
        self.method = method  # 降维方法
        self.time1 = 0  # 一次扰动一个点所花的时间
        self.time2 = 0  # 随机选点扰动所花的时间
        self.time3 = 0  # 使用导数扰动所花的时间
        self.time_d = 0  # 求导用的时间
        self.dr_count = 0  # 随机扰动的方法，执行降维的次数
        self.perplexity = 30.0  # t-SNE的困惑度
        self.eta = eta
        self.tsne_P = None
        self.tsne_Q = None
        self.tsne_P0 = None
        self.beta = None
        self.init_y()

    def init_y(self):
        # 计算初始的降维结果
        if self.method == "MDS":
            mds = MDS(n_components=2, max_iter=10000, eps=-1.0)
            self.Y = mds.fit_transform(self.X)
        elif self.method == "cTSNE":
            tsne = cTSNE.cTSNE(n_component=2, perplexity=self.perplexity)
            self.Y = tsne.fit_transform(self.X, max_iter=30000)
            self.tsne_P = tsne.P
            self.tsne_Q = tsne.Q
            self.tsne_P0 = tsne.P0
            self.beta = tsne.beta
        print("init finished")

    def preturb_1by1(self):
        # 一次降维扰动一个点
        t1 = time.time()
        (n, m) = self.X.shape
        self.Y1 = np.zeros((n, 2))
        for i in range(0, n):
            X = self.X.copy()
            X[i, :] = X[i, :] + self.eta * self.vectors[i, :]
            if self.method == "MDS":
                mds = MDS(n_components=2, n_init=1, max_iter=200, eps=-1.0)
                temp_y = mds.fit_transform(X, init=self.Y)
            else:
                tsne = cTSNE.cTSNE(n_component=2, perplexity=self.perplexity)
                temp_y = tsne.fit_transform(X, max_iter=20, early_exaggerate=False, y_random=self.Y, follow_gradient=False)

            self.Y1[i, :] = temp_y[i, :]
        t2 = time.time()
        self.time1 = t2 - t1
        print("1 by 1 finished")
        return self.Y1

    def all_preturbed(self, preturb_count, n):
        """
        判断是否所有的点都被扰动过了
        :param preturb_count:
        :param n:
        :return:
        """
        for i in range(0, n):
            if preturb_count[i] == 0:
                return False
        return True

    def preturb_random(self):
        # 每次降维随机选一半扰动
        t1 = time.time()
        (n, m) = self.X.shape
        # self.Y2 = np.zeros((n, 2))
        preturb_count = np.zeros((n, 1))
        indexs = range(0, n)
        Y2 = np.zeros((n, 2))

        while not self.all_preturbed(preturb_count, n):
            self.dr_count += 1
            selected_indexs = random.sample(indexs, n//2)
            X = self.X.copy()
            for i in selected_indexs:
                X[i, :] = X[i, :] + self.eta * self.vectors[i, :]
                preturb_count[i] += 1
            if self.method == "MDS":
                mds = MDS(n_components=2, n_init=1, max_iter=200, eps=-1.0)
                temp_y = mds.fit_transform(X, init=self.Y)
            else:
                tsne = cTSNE.cTSNE(n_component=2, perplexity=self.perplexity)
                temp_y = tsne.fit_transform(X, max_iter=20, early_exaggerate=False, y_random=self.Y, follow_gradient=False)

            for i in selected_indexs:
                Y2[i, :] = Y2[i, :] + temp_y[i, :]

        for i in range(0, n):
            Y2[i, :] = Y2[i, :] / preturb_count[i]
        self.Y2 = Y2
        t2 = time.time()
        self.time2 = t2 - t1
        print("random finished")
        return self.Y2

    def our_preturb(self):
        # 我们的扰动方法，先求导，再扰动
        t1 = time.time()
        (n, m) = self.X.shape
        if self.method == "MDS":
            derivative = MDS_Derivative.MDS_Derivative()
            P = derivative.getP(self.X, self.Y)
        else:
            derivative = TSNE_Derivative.TSNE_Derivative()
            P = derivative.getP(self.X, self.Y, self.tsne_P, self.tsne_Q, self.tsne_P0, self.beta)
        t2 = time.time()

        dY = np.zeros((n, 2))
        for i in range(0, n):
            dY[i, :] = np.matmul(self.vectors[i, :], P[2*i:2*i+2, m*i:m*i+m].T)
        self.Y3 = self.Y + dY * self.eta
        t3 = time.time()
        self.time3 = t3 - t1
        self.time_d = t2 - t1
        print("our method finished")
        return self.Y3

    def preturb_compare(self):
        # 具体进行扰动比较的入口
        self.preturb_1by1()
        self.preturb_random()
        self.our_preturb()


def run():
    from Main import Preprocess
    path = "E:\\文件\\IRC\\特征向量散点图项目\\小实验\\不同扰动方法\\Iris3\\"
    method = "cTSNE"
    path = path + method + "\\"

    data = np.loadtxt(path+"data.csv", dtype=np.float, delimiter=",")
    (n, m) = data.shape
    X = Preprocess.normalize(data, -1, 1)
    label = np.loadtxt(path+"label.csv", dtype=np.int, delimiter=",")
    # vectors = np.loadtxt(path+"vectors.csv", dtype=np.float, delimiter=",")
    vectors = np.zeros((n, m))
    vectors[:, 0] = 1.0

    compare = Compare(X, vectors, eta=0.05, method=method)
    Y = compare.Y
    Y1 = compare.preturb_1by1()
    Y2 = compare.preturb_random()
    Y3 = compare.our_preturb()

    print("一次降维扰动一个点的方法耗时 ", compare.time1)
    print("随机选取点扰动耗时 ", compare.time2)
    print("随机扰动方法执行的降维次数是 ", compare.dr_count)
    print("我们的方法耗时 ", compare.time3)
    print("其中，求导耗时 ", compare.time_d)

    np.savetxt(path + "Y.csv", Y, fmt='%.18e', delimiter=",")
    np.savetxt(path + "Y1.csv", Y1, fmt='%.18e', delimiter=",")
    np.savetxt(path + "Y2.csv", Y2, fmt='%.18e', delimiter=",")
    np.savetxt(path + "Y3.csv", Y3, fmt='%.18e', delimiter=",")

    dY1 = Y1 - Y
    dY2 = Y2 - Y
    dY3 = Y3 - Y
    np.savetxt(path + "dY1.csv", dY1, fmt='%.18e', delimiter=",")
    np.savetxt(path + "dY2.csv", dY2, fmt='%.18e', delimiter=",")
    np.savetxt(path + "dY3.csv", dY3, fmt='%.18e', delimiter=",")

    plt.scatter(Y[:, 0], Y[:, 1], c='k', alpha=0.8)
    plt.scatter(Y1[:, 0], Y1[:, 1], c='r', alpha=0.8)
    plt.scatter(Y2[:, 0], Y2[:, 1], c='g', alpha=0.8)
    plt.scatter(Y3[:, 0], Y3[:, 1], c='b', alpha=0.8)
    for i in range(0, n):
        plt.plot([Y[i, 0], Y1[i, 0]], [Y[i, 1], Y1[i, 1]], linewidth=0.7, c='r', alpha=0.8)
        plt.plot([Y[i, 0], Y2[i, 0]], [Y[i, 1], Y2[i, 1]], linewidth=0.7, c='g', alpha=0.8)
        plt.plot([Y[i, 0], Y3[i, 0]], [Y[i, 1], Y3[i, 1]], linewidth=0.7, c='b', alpha=0.8)

    ax = plt.gca()
    ax.set_aspect(1)
    plt.show()


def errors_statistics():
    """
    统计不同扰动方法的误差
    :return:
    """
    path = "E:\\文件\\IRC\\特征向量散点图项目\\小实验\\不同扰动方法\\灵越运行结果\\datasets\\"
    name = "digits40m1"
    method = "cTSNE"

    path = path + name + "\\" + method + "\\"
    dY1 = np.loadtxt(path+"dY1.csv", dtype=np.float, delimiter=",")
    dY2 = np.loadtxt(path + "dY2.csv", dtype=np.float, delimiter=",")
    dY3 = np.loadtxt(path + "dY3.csv", dtype=np.float, delimiter=",")
    (n, m) = dY1.shape

    dY21 = dY2 - dY1
    dY31 = dY3 - dY1
    norm2 = np.zeros((n, 1))
    norm3 = np.zeros((n, 1))
    for i in range(0, n):
        norm2[i] = np.linalg.norm(dY21[i, :]) / np.linalg.norm(dY1[i, :])
        norm3[i] = np.linalg.norm(dY31[i, :]) / np.linalg.norm(dY1[i, :])
    print("random 的误差 ", np.mean(norm2))
    print("我们方法的误差 ", np.mean(norm3))


def test():
    list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    slice = random.sample(list, 5)  # 从list中随机获取5个元素，作为一个片断返回
    print(slice)
    print(list)
    a = []
    a = range(0, 8)
    b = random.sample(a, 3)
    print(b)
    print(a)

    A = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])
    B = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12]])
    C = np.matmul(A[1, :], B[0:2, 0:3].T)
    print(C)

    print('------------------------------------')
    A = A + np.ones((3, 3))*0.1
    D = np.zeros((3, 1))
    D[1] += 2
    print(A)
    print(D)
    A[0, :] = A[0, :] / D[1]
    print(A)


if __name__ == '__main__':
    # test()
    # run()
    errors_statistics()
