import numpy as np
from Main import DimReduce
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
from Tools import ImageScatter
from Main import Preprocess
from PIL import Image


def see_data():
    path = "E:\\Project\\DataLab\\OlivettiFaces\\"
    data = np.load(path+"olivetti_faces.npy")
    print(data.shape)
    (n, m, k) = data.shape

    X = np.zeros((n, m*k))
    for i in range(0, n):
        a = data[i]
        X[i, :] = a.reshape(1, m*k)

    np.savetxt(path+"data.csv", X, fmt='%f', delimiter=",")


def faces_pictures(path=""):
    """生成MNIST的图片"""

    path = "E:\\Project\\DataLab\\OlivettiFaces\\"
    data = np.load(path + "olivetti_faces.npy")
    print(data.shape)
    (n, m, k) = data.shape

    picture_path = path + "pictures\\"
    if not os.path.exists(picture_path):
        os.makedirs(picture_path)

    for i in range(0, n):
        new_data = data[i] * 255
        im = Image.fromarray(new_data.astype(np.uint8))
        im.save(picture_path+str(i)+".png")

        if i % 1000 == 0:
            print(i)

    print("finished")


def sub_sampling():
    path = "E:\\Project\\DataLab\\OlivettiFaces\\"
    data = np.load(path + "olivetti_faces.npy")
    print(data.shape)
    (n, m, k) = data.shape

    X = np.zeros((n, 256))
    for i in range(0, n):
        face = data[i]
        sub_face = np.zeros((16, 16))
        for row in range(0, 16):
            for column in range(0, 16):
                s = 0
                for j1 in range(row*4-4, row*4):
                    for j2 in range(column*4-4, column*4):
                        s = s + face[j1, j2]
                sub_face[row, column] = s / 16
        X[i, :] = sub_face.reshape(1, 256)
    np.savetxt(path+"16维\\data.csv", X, fmt='%f', delimiter=",")


def small_face_pictures():
    """
    生成亚采样之后的图片
    :return:
    """
    path = "E:\\Project\\DataLab\\OlivettiFaces\\16维\\"
    data = np.loadtxt(path+"data.csv", dtype=np.float, delimiter=",")
    (n, m) = data.shape

    picture_path = path + "pictures\\"
    if not os.path.exists(picture_path):
        os.makedirs(picture_path)

    for i in range(0, n):
        x = data[i].reshape(16, 16)
        new_data = x * 255
        im = Image.fromarray(new_data.astype(np.uint8))
        im.save(picture_path + str(i) + ".png")

        if i % 1000 == 0:
            print(i)


def pca_research():
    """
    进行PCA变换，观察需要用多少个特征向量来表示原来的数据
    :return:
    """
    path = "E:\\Project\\DataLab\\OlivettiFaces\\16维\\"
    data = np.loadtxt(path + "data.csv", dtype=np.float, delimiter=",")
    (n, m) = data.shape
    # label = np.loadtxt(path + "label.csv", dtype=np.int, delimiter=",")

    pca = PCA(n_components=m)
    pca.fit(data)
    vectors = pca.components_  # 所有的特征向量
    values = pca.explained_variance_  # 所有的特征值
    np.savetxt(path+"eigenvectors.csv", vectors, fmt='%f', delimiter=",")
    np.savetxt(path+"eigenvalues.csv", values, fmt='%f', delimiter=",")
    print("finished")


def pre_dr():
    """
    使用PCA预降维
    :return:
    """
    path = "E:\\Project\\DataLab\\OlivettiFaces\\16维\\"
    data = np.loadtxt(path + "data.csv", dtype=np.float, delimiter=",")
    (n, m) = data.shape

    label = np.zeros((n, 1))
    for i in range(0, 40):
        for j in range(0, 10):
            label[i*10-10+j] = i

    np.savetxt(path+"label.csv", label, fmt='%d', delimiter=",")

    pca1 = PCA(n_components=9)
    pca2 = PCA(n_components=30)
    y1 = pca1.fit_transform(data)
    y2 = pca2.fit_transform(data)

    np.savetxt(path+"data9m.csv", y1, fmt='%f', delimiter=",")
    np.savetxt(path+"data30m.csv", y2, fmt='%f', delimiter=",")


if __name__ == '__main__':
    # see_data()
    # faces_pictures()
    # sub_sampling()
    # small_face_pictures()
    # pca_research()
    pre_dr()
