# 处理TicTacToe数据集
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.manifold import TSNE


def pre_process():
    """
    预处理数据
    :return:
    """
    path = "E:\\文件\\IRC\\特征向量散点图项目\\DataLab\\TicTacToe\\"
    data = np.loadtxt(path+"tic-tac-toe.data", dtype=np.str, delimiter=",")
    (n, m) = data.shape

    X = np.zeros((n, m-1))
    label = np.ones((n, 1))
    for i in range(0, n):
        for j in range(0, m-1):
            if data[i, j] == 'x':
                X[i, j] = 1
            elif data[i, j] == 'o':
                X[i, j] = -1
        if data[i, m-1] == 'negative':
            label[i] = 2

    np.savetxt(path+"data.csv", X, fmt='%d', delimiter=",")
    np.savetxt(path+"label.csv", label, fmt='%d', delimiter=",")

    pca = PCA(n_components=2)
    Y1 = pca.fit_transform(X)
    plt.scatter(Y1[0:626, 0], Y1[0:626, 1], c='r')
    plt.scatter(Y1[626:n, 0], Y1[626:n, 1], c='b')
    plt.title('PCA')
    plt.show()

    mds = MDS(n_components=2)
    Y2 = mds.fit_transform(X)
    plt.scatter(Y2[0:626, 0], Y2[0:626, 1], c='r')
    plt.scatter(Y2[626:n, 0], Y2[626:n, 1], c='b')
    plt.title('MDS')
    plt.show()

    tsne = TSNE(n_components=2, perplexity=30)
    Y3 = tsne.fit_transform(X)
    plt.scatter(Y3[0:626, 0], Y3[0:626, 1], c='r')
    plt.scatter(Y3[626:n, 0], Y3[626:n, 1], c='b')
    plt.title('t-SNE')
    plt.show()


if __name__ == '__main__':
    pre_process()

