import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def see_data():
    path = "E:\\Project\\DataLab\\TIMIT\\"
    data = np.loadtxt(path+"data.csv", dtype=np.float, delimiter=",")
    (n, m) = data.shape

    pca = PCA(n_components=2)
    t_sne = TSNE(n_components=2)
    Y = pca.fit_transform(data)
    plt.scatter(Y[:, 0], Y[:, 1])
    ax = plt.gca()
    ax.set_aspect(1)
    plt.show()


if __name__ == '__main__':
    see_data()

