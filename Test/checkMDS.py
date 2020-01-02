from sklearn.metrics import euclidean_distances
from sklearn.manifold import MDS
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as data


def run1():
    # X = np.array([[1, 2, 1],
    #               [3, 4, 5],
    #               [4, 2, 6],
    #               [5, 2, 3],
    #               [3, 3, 2]])
    iris = data.load_iris()
    X = iris.data
    (n, m) = X.shape
    D = euclidean_distances(X)

    mds = MDS(n_components=2)
    Y = mds.fit_transform(X)
    print("stress = ", mds.stress_)

    D2 = euclidean_distances(Y)

    s = 0
    dD = D2 - D
    for i in range(0, n):
        for j in range(0, n):
            s = s + dD[i, j] * dD[i, j]

    print(s/2)
    print(s/n)


if __name__ == '__main__':
    run1()
