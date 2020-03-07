# Locality Preserving Projections(LPP)降维方法实现
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import euclidean_distances
from sklearn.neighbors import NearestNeighbors


class LPP:
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.Y = None
        self.vectors = None
        self.values = None

    def fit_transform(self, x, n_nbrs=15):
        """

        :param x:
        :param n_nbrs:
        :return:
        """
        (n, m) = x.shape

