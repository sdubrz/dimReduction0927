# 分批降维，主要应用于较大数据，一天无法完成迭代。每次迭代一定次数，下次接着上次迭代
import numpy as np
from sklearn.manifold import MDS
from sklearn.manifold import TSNE
from MyDR import cTSNE
import matplotlib.pyplot as plt


def dr_steps(X, method="MDS", unit_loop=1000, n_steps=10, path="", perplexity=30.0, label=None):
    """
    分步降维，用于一次计算不完的较大数据，可以分步骤多次开始
    :param X: 预处理好的数据矩阵
    :param method: 降维方法名，目前支持 "MDS" 与 "cTSNE"
    :param unit_loop: 单步迭代的次数
    :param n_steps: 需要循环的步数，总共迭代的次数为 unit_loop * n_steps
    :param path: 存储中间结果的文件路径
    :param perplexity: t-SNE 方法的困惑度
    :param label: 数据标签
    :return:
    """
    (n, m) = X.shape

    if method == "MDS":
        mds = MDS(n_components=2, max_iter=unit_loop, eps=-1)
        Y0 = mds.fit_transform(X)
    elif method == "cTSNE":  # 第一次运行直接用sklearn中的方法加速计算
        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=unit_loop)
        Y0 = tsne.fit_transform(X)
    else:
        print("暂不支持该方法： ", method)
        return
    np.savetxt(path+method+"Y0.csv", Y0, fmt='%.18e', delimiter=",")
    plt.figure(figsize=(16, 16))
    if label is None:
        plt.scatter(Y0[:, 0], Y0[:, 1])
    else:
        plt.scatter(Y0[:, 0], Y0[:, 1], c=label)
    plt.title(method+"Y0")
    ax = plt.gca()
    ax.set_aspect(1)
    plt.savefig(path+method+"Y0.png")
    plt.close()

    for loop in range(1, n_steps):
        print(loop)
        Y0 = np.loadtxt(path+method+"Y"+str(loop-1)+".csv", dtype=np.float, delimiter=",")
        if method == "MDS":
            mds = MDS(n_components=2, max_iter=unit_loop, eps=-1, n_init=1)
            Y = mds.fit_transform(X, init=Y0)
        elif method == "cTSNE":
            tsne = cTSNE.cTSNE(n_component=2, perplexity=perplexity)
            Y = tsne.fit_transform(X, max_iter=unit_loop, early_exaggerate=False, y_random=Y0)
        else:
            print("暂不支持该方法")
            return
        np.savetxt(path+method+"Y"+str(loop)+".csv", Y, fmt='%.18e', delimiter=",")
        plt.figure(figsize=(16, 16))
        if label is None:
            plt.scatter(Y0[:, 0], Y0[:, 1])
        else:
            plt.scatter(Y0[:, 0], Y0[:, 1], c=label)
        plt.title(method+"Y"+str(loop))
        ax = plt.gca()
        ax.set_aspect(1)
        plt.savefig(path+method+"Y"+str(loop)+".png")
        plt.close()


if __name__ == '__main__':
    path = "E:\\文件\\IRC\\特征向量散点图项目\\result2020\\result0119\\datasets\\Iris3\\"
    X = np.loadtxt(path+"data.csv", dtype=np.float, delimiter=",")
    from Main import Preprocess
    X = Preprocess.normalize(X, -1, 1)
    label = np.loadtxt(path+"label.csv", dtype=np.int, delimiter=",")

    dr_steps(X, method="MDS", unit_loop=1000, n_steps=10, path=path, perplexity=30.0, label=label)




