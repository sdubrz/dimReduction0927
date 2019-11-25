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

    labels = np.zeros((n, 1))
    locs_shape = locs.shape
    for i in range(0, n):
        for j in range(0, locs_shape[1]):
            if locs[i, j]:
                labels[i] = j
                continue

    return labels


def run_clustering_path(path, d_latent, n_pca, n_clusters, k_knn, o=8, max_iter=100):
    X = np.loadtxt(path+"data.csv", dtype=np.float, delimiter=",")
    labels = run_clustering(X, d_latent, n_pca, n_clusters, k_knn, o=o, max_iter=max_iter)

    return labels
