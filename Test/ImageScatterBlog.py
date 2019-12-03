# 用于发表在CSDN的代码
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import os
from PIL import Image
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def mnist_pictures(path=""):
    """生成MNIST的图片"""
    X = np.loadtxt(path+"data.csv", dtype=np.int, delimiter=",")
    (n, m) = X.shape
    X = 255*np.ones(X.shape) - X  # 黑白颠倒

    picture_path = path + "pictures\\"
    if not os.path.exists(picture_path):
        os.makedirs(picture_path)

    for i in range(0, n):
        new_data = np.reshape(X[i, :], (28, 28))
        im = Image.fromarray(new_data.astype(np.uint8))
        im.save(picture_path+str(i)+".png")
        if i % 1000 == 0:
            print(i)


def small_image(eta=0.5, in_path="", out_path=""):
    """
    将图片进行缩放
    :param eta: 缩小的倍数
    :param in_path: 读取图片路径
    :param out_path: 保存缩小后的图片的路径
    :return:
    """
    for root, middle, files in os.walk(in_path):
        for file_name in files:
            im = Image.open(in_path+file_name)
            (x, y) = im.size
            s_img = im.resize((int(x*eta), int(y*eta)), Image.ANTIALIAS)
            s_img.save(out_path+file_name)


def get_image(path):
    return OffsetImage(plt.imread(path))


def mnist_images(path=None, eta=0.4):
    """
    用MNIST数据画艺术散点图
    :return:
    """
    mnist_pictures(path)  # 先生成图片

    small_path = path + "smallImages\\"
    if not os.path.exists(small_path):
        os.makedirs(small_path)
    small_image(eta=eta, in_path=path+"pictures\\", out_path=small_path)
    Y = np.loadtxt(path + "y.csv", dtype=np.float, delimiter=",")
    (n, m) = Y.shape
    fig, ax = plt.subplots()

    ax.scatter(Y[:, 0], Y[:, 1])
    plt.set_cmap(cm.gray)  # 修改颜色映射
    for i in range(0, n):
        ab = AnnotationBbox(get_image(small_path + str(i)+".png"), (Y[i, 0], Y[i, 1]), frameon=False)
        ax.add_artist(ab)
    ax = plt.gca()
    ax.set_aspect(1)
    plt.show()


if __name__ == '__main__':
    path = "E:\\Project\\DataLab\\MNISTdigit\\"
    # X = np.loadtxt(path+"data.csv", dtype=np.int, delimiter=",")
    # (n, m) = X.shape
    # pca = PCA(n_components=50)
    # pca_y = pca.fit_transform(X)  # 先用PCA初步降维
    # t_sne = TSNE(n_components=2, perplexity=30.0)
    # Y = t_sne.fit_transform(pca_y)  # 用t-SNE得到最后的二维坐标
    # np.savetxt(path+"y.csv", Y, fmt="%f", delimiter=",")
    mnist_images(path=path, eta=0.5)
