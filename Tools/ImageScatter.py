# 在散点图中插入图片
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.decomposition import PCA
from sklearn import preprocessing
from PIL import Image


def small_image(eta=0.5):
    """
    将图片缩小
    :param eta: 缩小的倍数
    :return:
    """
    path1 = "E:\\Project\\DataLab\\duck\\images\\"
    path2 = "E:\\Project\\DataLab\\duck\\smallimages\\"

    for i in range(0, 72):
        im = Image.open(path1+"obj1__"+str(i)+".png")
        (x, y) = im.size
        s_img = im.resize((int(x*eta), int(y*eta)), Image.ANTIALIAS)
        s_img.save(path2+"obj1__"+str(i)+".png")


def get_image(path):
    return OffsetImage(plt.imread(path))


def image_scatter():
    path = "E:\\Project\\DataLab\\duck\\"
    small_image(eta=0.3)
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


if __name__ == '__main__':
    image_scatter()

