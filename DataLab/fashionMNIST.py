import numpy as np
from Main import DimReduce
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
from Tools import ImageScatter
from Main import Preprocess


def sampling():
    """
    从测试数据集中采样出少量数据
    :return:
    """
    path = "E:\\Project\\DataLab\\fashionMnist\\"
    data = np.loadtxt(path+"fashion-mnist_test.csv", dtype=np.int, delimiter=",")
    (n, m) = data.shape

    test_data = data[:, 1:m]
    test_label = data[:, 0]
    test_data_list = test_data.tolist()

    label = []
    X = []
    for i in range(0, n):
        if i % 4 == 0:
            X.append(test_data_list[i])
            label.append(test_label[i])
    np.savetxt(path+"data.csv", np.array(X), fmt='%d', delimiter=",")
    np.savetxt(path+"label.csv", np.transpose(np.array(label)), fmt='%d', delimiter=",")
    print("finished")


def see_data():
    """
    用部分降维算法，查看数据大致面貌
    :return:
    """
    path = "E:\\Project\\DataLab\\fashionMnist\\"
    data = np.loadtxt(path + "data.csv", dtype=np.int, delimiter=",")
    (n, m) = data.shape
    label = np.loadtxt(path + "label.csv", dtype=np.int, delimiter=",")

    Y = DimReduce.dim_reduce(data, method="MDS", method_k=90)
    plt.scatter(Y[:, 0], Y[:, 1], c=label)
    plt.colorbar()
    ax = plt.gca()
    ax.set_aspect(1)
    plt.show()


def pca_research():
    """
    进行PCA变换，观察需要用多少个特征向量来表示原来的数据
    :return:
    """
    path = "E:\\Project\\DataLab\\fashionMnist\\"
    data = np.loadtxt(path + "data.csv", dtype=np.float, delimiter=",")
    (n, m) = data.shape
    label = np.loadtxt(path + "label.csv", dtype=np.int, delimiter=",")

    pca = PCA(n_components=m)
    pca.fit(data)
    vectors = pca.components_  # 所有的特征向量
    values = pca.explained_variance_  # 所有的特征值
    np.savetxt(path+"eigenvectors.csv", vectors, fmt='%f', delimiter=",")
    np.savetxt(path+"eigenvalues.csv", values, fmt='%f', delimiter=",")
    print("finished")


def classify():
    """
    划分成每个类的单独数据集
    :return:
    """
    path = "E:\\Project\\DataLab\\fashionMnist\\"
    origin_data = np.loadtxt(path + "data.csv", dtype=np.int, delimiter=",")
    label = np.loadtxt(path + "label.csv", dtype=np.int, delimiter=",")
    (n, m) = origin_data.shape
    print((n, m))

    pca = PCA(n_components=50)
    data_50m = pca.fit_transform(origin_data)

    np.savetxt(path+"data50m.csv", data_50m, fmt='%f', delimiter=",")
    data_50_list = data_50m.tolist()
    origin_data_list = origin_data.tolist()

    data_list = []
    origin_list = []
    for i in range(0, 10):
        data_list.append([])
        origin_list.append([])

    for i in range(0, n):
        # print(i)
        obj = label[i]
        temp_list1 = data_list[obj]
        temp_list2 = origin_list[obj]
        temp_list1.append(data_50_list[i])
        temp_list2.append(origin_data_list[i])

    for i in range(0, 10):
        obj_path = path + "fashion50mclass" + str(i) + "_" + str(len(data_list[i])) + "\\"
        if not os.path.exists(obj_path):
            os.makedirs(obj_path)

        temp_data = np.array(data_list[i])
        (temp_n, temp_m) = temp_data.shape
        np.savetxt(obj_path+"data.csv", temp_data, fmt='%f', delimiter=",")
        np.savetxt(obj_path + "origin.csv", np.array(origin_list[i]), fmt='%d', delimiter=",")
        np.savetxt(obj_path+"label.csv", i*np.ones((temp_n, 1)), fmt='%d', delimiter=",")
        print(i)


def pca_art_scatter():
    """
    对每个类画艺术散点图
    :return:
    """
    obj_index = 6
    option = 2
    path = "E:\\Project\\DataLab\\fashionMnist\\"
    fashion_count = [238, 254, 251, 262, 264, 251, 251, 244, 227, 258]
    obj_path = path + "fashion50mclass" + str(obj_index) + "_" + str(fashion_count[obj_index]) + "\\"
    data = np.loadtxt(obj_path+"data.csv", dtype=np.float, delimiter=",")
    # data = Preprocess.normalize(data)
    pca = PCA(n_components=2)
    Y = pca.fit_transform(data)
    np.savetxt(obj_path+"PCA.csv", Y, fmt='%f', delimiter=",")

    if option == 1:
        plt.scatter(Y[:, 0], Y[:, 1])
        ax = plt.gca()
        ax.set_aspect(1)
        # plt.colorbar()
        plt.show()
    else:
        ImageScatter.mnist_images(obj_path, eta=0.8)


if __name__ == '__main__':
    # sampling()
    # see_data()
    # pca_research()
    # classify()
    pca_art_scatter()
