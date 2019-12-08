import numpy as np
from Main import DimReduce
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os
from Tools import ImageScatter
from Main import Preprocess
from Test import cleanData
from PIL import Image


def create_images():
    path = "E:\\Project\\DataLab\\FreyFace\\"
    X = np.loadtxt(path + "origin.csv", dtype=np.int, delimiter=",")
    (n, m) = X.shape
    # X = 255 * np.ones(X.shape) - X

    picture_path = path + "pictures\\"
    if not os.path.exists(picture_path):
        os.makedirs(picture_path)

    for i in range(0, n):
        new_data = np.reshape(X[i, :], (28, 20))
        im = Image.fromarray(new_data.astype(np.uint8))
        im.save(picture_path + str(i) + ".png")
        if i % 1000 == 0:
            print(i)

    print("finished")


def see_data():
    path = "E:\\Project\\DataLab\\FreyFace\\"
    X = np.loadtxt(path + "origin.csv", dtype=np.int, delimiter=",")
    (n, m) = X.shape

    pca = PCA(n_components=2)
    Y = pca.fit_transform(X)
    np.savetxt(path+"pca.csv", Y, fmt='%f', delimiter=",")
    plt.scatter(Y[:, 0], Y[:, 1])
    ax = plt.gca()
    ax.set_aspect(1)
    plt.show()


def pca_art_scatter():
    """
    对每个类画艺术散点图
    :return:
    """
    option = 2
    path = "E:\\Project\\DataLab\\FreyFace\\"
    Y = np.loadtxt(path+"pca.csv", dtype=np.float, delimiter=",")

    if option == 1:
        plt.scatter(Y[:, 0], Y[:, 1])
        ax = plt.gca()
        ax.set_aspect(1)
        plt.show()
    else:
        ImageScatter.mnist_images(path, eta=0.8, image_shape=(28, 20))


def sampling():
    path = "E:\\Project\\DataLab\\FreyFace\\"
    X = np.loadtxt(path + "origin.csv", dtype=np.int, delimiter=",")
    (n, m) = X.shape

    data = X.tolist()
    small_data = []
    for i in range(0, n):
        if i%3 == 0:
            small_data.append(data[i])
    np.savetxt(path+"small.csv", np.array(small_data), fmt='%d', delimiter=",")


if __name__ == '__main__':
    # create_images()
    # see_data()
    pca_art_scatter()
