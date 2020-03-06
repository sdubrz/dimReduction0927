# 处理yeast数据集
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.manifold import TSNE


def pre_process():
    path = "E:\\文件\\IRC\\特征向量散点图项目\\DataLab\\Yeast\\"
    data = np.loadtxt(path+"yeast.data", dtype=np.str)
    print(data.shape)
    (n, m) = data.shape
    label1 = data[:, 0]
    label2 = data[:, m-1]

    X0 = data[:, 1:m-1]
    X = X0.astype(np.float)
    np.savetxt(path+"data.csv", X0, delimiter=",", fmt='%s')
    np.savetxt(path+"label1.csv", label1, fmt='%s')
    np.savetxt(path+"label2.csv", label2, fmt='%s')

    pca = PCA(n_components=2)
    Y1 = pca.fit_transform(X)
    plt.scatter(Y1[:, 0], Y1[:, 1])
    plt.title('PCA')
    plt.show()

    mds = MDS(n_components=2)
    Y2 = mds.fit_transform(X)
    plt.scatter(Y2[:, 0], Y2[:, 1])
    plt.title('MDS')
    plt.show()

    tsne = TSNE(n_components=2, perplexity=30)
    Y3 = tsne.fit_transform(X)
    plt.scatter(Y3[:, 0], Y3[:, 1])
    plt.title('t-SNE')
    plt.show()



if __name__ == '__main__':
    pre_process()
