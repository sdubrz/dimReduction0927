# 在散点图中插入图片
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.decomposition import PCA
from sklearn import preprocessing
from PIL import Image
import os


def small_image(eta=0.5, in_path="", out_path=""):
    """
    将图片缩小
    :param eta: 缩小的倍数
    :param in_path: 读取图片路径
    :param out_path: 保存缩小后的图片的路径
    :return:
    """
    # path1 = "E:\\Project\\DataLab\\duck\\images\\"
    # path2 = "E:\\Project\\DataLab\\duck\\smallimages\\"
    for root, middle, files in os.walk(in_path):
        for file_name in files:
            im = Image.open(in_path+file_name)
            (x, y) = im.size
            s_img = im.resize((int(x*eta), int(y*eta)), Image.ANTIALIAS)
            s_img.save(out_path+file_name)


def get_image(path):
    return OffsetImage(plt.imread(path))


def image_scatter():
    path = "E:\\Project\\DataLab\\duck\\"
    path1 = "E:\\Project\\DataLab\\duck\\images\\"
    path2 = "E:\\Project\\DataLab\\duck\\smallimages\\"
    small_image(eta=0.3, in_path=path1, out_path=path2)
    image_path = path + "smallimages\\obj1__"
    X = np.loadtxt(path+"data.csv", dtype=np.float, delimiter=",")
    X = preprocessing.minmax_scale(X)
    (n, m) = X.shape
    print((n, m))

    pca = PCA(n_components=2)
    Y = pca.fit_transform(X)

    fig, ax = plt.subplots()
    ax.scatter(Y[:, 0], Y[:, 1])
    for i in range(0, n):
        ab = AnnotationBbox(get_image(image_path+str(i)+".png"), (Y[i, 0], Y[i, 1]), frameon=False)
        ax.add_artist(ab)
    ax = plt.gca()
    ax.set_aspect(1)
    plt.show()


def coil_image_scatter():
    """在coil数据的降维结果中展示图片"""
    path = "E:\\Project\\DataLab\\imageScatter\\"
    small_image(eta=0.15, in_path=path+"images\\", out_path=path+"smallImages\\")
    Y = np.loadtxt(path+"pca.csv", dtype=np.float, delimiter=",")
    (n, m) = Y.shape
    fig, ax = plt.subplots()
    ax.scatter(Y[:, 0], Y[:, 1])
    for i in range(0, n):
        obj = i // 72 + 1
        img_index = i % 72
        ab = AnnotationBbox(get_image(path+"smallImages\\obj"+str(obj)+"__"+str(img_index)+".png"), (Y[i, 0], Y[i, 1]), frameon=False)
        ax.add_artist(ab)
    ax = plt.gca()
    ax.set_aspect(1)
    plt.show()


if __name__ == '__main__':
    # image_scatter()
    # path1 = "E:\\Project\\DataLab\\duck\\images\\"
    # for i, j, k in os.walk(path1):
    #     for file in k:
    #         print(file)
    coil_image_scatter()
