# 在散点图中插入图片
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.decomposition import PCA
from sklearn import preprocessing
from PIL import Image
import os
from Test import cleanData


def create_pictures(path="", image_shape=(28, 28)):
    """生成MNIST的图片"""
    # path = "E:\\Project\\DataLab\\MNIST\\"
    # path = "E:\\Project\\result2019\\result1026without_straighten\\datasets\\MNIST50mclass2_874\\"
    X = np.loadtxt(path+"origin.csv", dtype=np.int, delimiter=",")
    (n, m) = X.shape
    # X = 255*np.ones(X.shape) - X  # 反转
    # X = np.maximum(X, 1)
    # label = np.loadtxt(path+"label.csv", dtype=np.int, delimiter=",")
    # count = np.zeros((10, 1))
    picture_path = path + "pictures\\"
    if not os.path.exists(picture_path):
        os.makedirs(picture_path)

    for i in range(0, n):
        new_data = np.reshape(X[i, :], image_shape)
        im = Image.fromarray(new_data.astype(np.uint8))
        # plt.imshow(new_data, cmap=plt.cm.gray, interpolation='nearest')
        # im.show()
        # im.save(path+"pictures\\"+str(label[i])+"\\"+str(int(count[label[i]]))+".png")
        im.save(picture_path+str(i)+".png")
        # count[label[i]] += 1
        if i % 1000 == 0:
            print(i)

    print("finished")


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
    Y = np.loadtxt(path+"mds3classk20yita01.csv", dtype=np.float, delimiter=",")
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


def mnist_images(path=None, eta=0.4, y_name="PCA.csv", label=None, image_shape=(28, 28), colormap='gray'):
    """
    用MNIST数据画艺术散点图
    :return:
    """
    if path is None:
        path = "E:\\Project\\result2019\\result1026without_straighten\\datasets\\MNIST50mclass019\\"
    # path = "E:\\Project\\result2019\\result1026without_straighten\\datasets\\winequality1000\\"

    # 如果事前没有生成图片，则需要先生成图片
    create_pictures(path, image_shape=image_shape)

    small_path = path + "smallImages\\"
    if not os.path.exists(small_path):
        os.makedirs(small_path)
    small_image(eta=eta, in_path=path+"pictures\\", out_path=small_path)
    Y = np.loadtxt(path + y_name, dtype=np.float, delimiter=",")
    (n, m) = Y.shape
    fig, ax = plt.subplots()
    # plt.colormaps()
    ax.scatter(Y[:, 0], Y[:, 1])
    if colormap == 'gray':
        plt.set_cmap(cm.gray)
    for i in range(0, n):
        # if label[i] != 5:
        #     continue
        ab = AnnotationBbox(get_image(small_path + str(i)+".png"), (Y[i, 0], Y[i, 1]), frameon=False)
        ax.add_artist(ab)
    ax = plt.gca()
    ax.set_aspect(1)
    plt.show()


def mnist_scatter():
    option = 2
    # path = "E:\\Project\\result2019\\result1026without_straighten\\datasets\\coil20obj_16_3class\\"
    # path = "E:\\Project\\result2019\\result1112without_normalize\\datasets\\fashion50mclass568\\"
    # path = "E:\\Project\\result2019\\result1224\\datasets\\MNIST50mclass1_985\\"
    # path = "E:\\Project\\result2020\\result0103\\datasets\\MNIST50mclass1_985\\"  # 华硕
    # path = "E:\\Project\\result2020\\result0104without_normalize\\datasets\\fashion50mclass568\\"  # 华硕
    path = "E:\\文件\\IRC\\特征向量散点图项目\\result2020\\result0119\\datasets\\coil20obj_10_3class\\"  # XPS
    # path = "E:\\文件\\IRC\\特征向量散点图项目\\result2020\\result0119_withoutnormalize\\datasets\\MNIST50mclass1_985\\"  # XPS
    # path = "E:\\文件\\IRC\\特征向量散点图项目\\DataLab\\optdigits\\optdigitClass9_562\\"
    if option == 1:  # 直接画散点图
        Y = np.loadtxt(path + "PCA.csv", dtype=np.float, delimiter=",")
        (n, m) = Y.shape
        label = np.loadtxt(path+"label.csv", dtype=np.int, delimiter=",")
        # plt.scatter(Y[:, 0], Y[:, 1], c=label)
        colors = ['r', 'g', 'b']
        for i in range(0, n):
            c = 'r'
            if label[i] == 5:
                c = 'r'
            elif label[i] == 6:
                c = 'g'
            else:
                c = 'b'
            plt.scatter(Y[i, 0], Y[i, 1], c=c, alpha=0.7)

        ax = plt.gca()
        ax.set_aspect(1)
        # plt.colorbar()
        plt.show()
    else:  # 画艺术散点图
        mnist_images(path, eta=0.15, y_name="cTSNE.csv", image_shape=(128, 128), colormap='gray')  # 搜 反转


if __name__ == '__main__':
    mnist_scatter()
