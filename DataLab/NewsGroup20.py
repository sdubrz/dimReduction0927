# 处理20newsGroup数据
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from gensim.models import word2vec


def test():
    path = "E:\\Project\\DataLab\\20news-18828\\"
    data = np.loadtxt(path+"vector.csv", dtype=np.float, delimiter=",")
    (n, m) = data.shape

    pca = PCA(n_components=2)
    Y = pca.fit_transform(data)
    plt.scatter(Y[:, 0], Y[:, 1])
    plt.show()


if __name__ == '__main__':
    test()
