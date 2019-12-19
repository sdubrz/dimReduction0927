from SMMC_.SMMC import SMMC
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


"""
 clustering using SMMC
"""


def run_clustering(X, d_latent, n_pca, n_clusters, k_knn, o=8, max_iter=100):
    """
    clustering using SMMC
    :param X: data matrix
    :param d_latent: dim of latent space
    :param n_pca:
    :param n_clusters:
    :param o:
    :param k_knn:
    :param max_iter:
    :return:
    """
    (n, m) = X.shape
    test = SMMC(X)
    test.train_mppca(d=d_latent, M=n_pca, max_iter=max_iter, tol=1e-4, kmeans_init=False)
    locs = test.run_cluster(o, k_knn, n_clusters)

    labels = []
    locs_shape = locs.shape
    for i in range(0, n):
        for j in range(0, locs_shape[1]):
            if locs[i, j]:
                labels.append(j)
                continue

    return labels


def run_clustering_path(path, d_latent, n_pca, n_clusters, k_knn, o=8, max_iter=100):
    X = np.loadtxt(path+"x.csv", dtype=np.float, delimiter=",")
    labels = run_clustering(X, d_latent, n_pca, n_clusters, k_knn, o=o, max_iter=max_iter)

    return labels


def run():
    # path = "E:\\Project\\result2019\\result1026without_straighten\\PCA\\MNIST50mclass1_985\\yita(0.1)nbrs_k(54)method_k(30)numbers(4)_b-spline_weighted\\"
    # path = "E:\\Project\\result2019\\result1026without_straighten\\PCA\\coil20obj_16_3class\\yita(0.03)nbrs_k(20)method_k(20)numbers(4)_b-spline_weighted\\"
    path = "E:\\Project\\result2019\\result1026without_straighten\\PCA\\2splane_60degree\\yita(0.05)nbrs_k(20)method_k(20)numbers(3)_b-spline_weighted\\"

    labels = run_clustering_path(path, d_latent=3, n_pca=20, n_clusters=2, k_knn=20, o=8, max_iter=100)

    Y = np.loadtxt(path+"y.csv", dtype=np.float, delimiter=",")
    np.savetxt(path+"smmc.csv", labels, fmt='%d', delimiter=",")

    plt.scatter(Y[:, 0], Y[:, 1], c=labels)
    ax = plt.gca()
    ax.set_aspect(1)
    plt.show()


if __name__ == '__main__':
    run()
