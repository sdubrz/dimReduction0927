# sklearn库中的Isomap是使用 kernel PCA实现的
# 我们需要实现一个不是以特征值分解的方式实现的版本，否则结果会不太稳定
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import check_array
from sklearn.utils.graph import graph_shortest_path
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.manifold import MDS
from sklearn.manifold import Isomap as Isomap0


class Isomap:
    def __init__(self, n_neighbors=5, n_components=2, max_iter=300, init=None):
        """
        Isomap
        :param n_neighbors: number of neighbors consider for each point.
        :param n_components: number of coordinates for the manifold.
        :param max_iter: Maximum number of iterations for the MDS.
        """
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.max_iter = max_iter
        self.nbrs_ = None
        self.dist_matrix_ = None
        self.init = init
        self.embedding_ = None  # 低维结果

    def fit_transform(self, X):
        X = check_array(X, accept_sparse='csr')
        self.nbrs_ = NearestNeighbors(n_neighbors=self.n_neighbors)
        self.nbrs_.fit(X)
        kng = kneighbors_graph(self.nbrs_, self.n_neighbors, mode='distance')
        self.dist_matrix_ = graph_shortest_path(kng, method='FW', directed=False)

        mds = MDS(n_components=self.n_components, max_iter=self.max_iter, dissimilarity="precomputed")
        self.embedding_ = mds.fit_transform(self.dist_matrix_, init=self.init)

        return self.embedding_


def test():
    path = "E:\\Project\\result2019\\result0927\\Isomap\\test\\swiss2roll\\"
    X = np.loadtxt(path+"x.csv", dtype=np.float, delimiter=',')
    label = np.loadtxt(path+"label.csv", dtype=np.int, delimiter=",")
    (n, m) = X.shape
    print((n, m))
    k = 5

    # 用 sklearn 库中的 Isomap 做实验
    iso_map0 = Isomap0(n_neighbors=k, n_components=2, eigen_solver='dense')
    y0 = iso_map0.fit_transform(X)

    # 使用自己写的 Isomap 做实验
    iso_map = Isomap(n_neighbors=k)
    y = iso_map.fit_transform(X)

    colors = ['r', 'g', 'b', 'm', 'c', 'y']
    plt.subplot(121)
    plt.scatter(y[:, 0], y[:, 1], marker='o', c=label)
    plt.title("My Isomap ")
    plt.subplot(122)
    plt.scatter(y0[:, 0], y0[:, 1], marker='o', c=label)
    plt.title("sklearn Isomap")
    plt.show()


if __name__ == '__main__':
    test()
