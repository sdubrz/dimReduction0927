# 旅游数据
# https://archive.ics.uci.edu/ml/datasets/Travel+Reviews
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.manifold import TSNE


def clean_data():
    path = "E:\\文件\\IRC\\特征向量散点图项目\\DataLab\\TravelReviews\\"
    matrix = np.loadtxt(path + "tripadvisor_review.csv", dtype=np.str, delimiter=",")
    (n, m) = matrix.shape

    data = matrix[1:n, 1:m]
    np.savetxt(path+"data.csv", data, fmt='%s', delimiter=",")
    np.savetxt(path+"label.csv", np.ones((n, 1)), fmt='%d', delimiter=",")
    X = data[:, :].astype(np.float)

    pca = PCA(n_components=2)
    Y = pca.fit_transform(X)
    plt.scatter(Y[:, 0], Y[:, 1])
    plt.show()


if __name__ == '__main__':
    clean_data()


