# 三维流形的测试
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from Main import Preprocess


def pca_test():
    path = "E:\\Project\\result2019\\result1026without_straighten\\datasets\\Wine\\"
    X = np.loadtxt(path+"data.csv", dtype=np.float, delimiter=",")
    (n, m) = X.shape
    X = Preprocess.normalize(X, -1, 1)

    pca = PCA(n_components=3)
    Y = pca.fit_transform(X)
    np.savetxt(path+"pca3d.csv", Y, fmt="%f", delimiter=",")


if __name__ == '__main__':
    pca_test()


