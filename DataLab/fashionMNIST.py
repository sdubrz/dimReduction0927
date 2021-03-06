import numpy as np
from Main import DimReduce
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os
from Tools import ImageScatter
from Main import Preprocess
from MyDR import cTSNE


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
    obj_index = 9
    option = 2
    path = "E:\\Project\\DataLab\\fashionMnist\\"
    fashion_count = [238, 254, 251, 262, 264, 251, 251, 244, 227, 258]
    obj_path = path + "fashion50mclass" + str(obj_index) + "_" + str(fashion_count[obj_index]) + "\\"
    obj_path = "E:\\Project\\result2019\\result1112without_normalize\\datasets\\fashion50mclass7_244\\"
    data = np.loadtxt(obj_path+"data.csv", dtype=np.float, delimiter=",")
    # data = Preprocess.normalize(data)
    # pca = PCA(n_components=2)
    # Y = pca.fit_transform(data)
    # np.savetxt(obj_path + "PCA.csv", Y, fmt='%f', delimiter=",")
    Y = np.loadtxt(obj_path+"PCA.csv", dtype=np.float, delimiter=",")
    label = np.loadtxt(obj_path+"label.csv", dtype=np.int, delimiter=",")

    if option == 1:
        plt.scatter(Y[:, 0], Y[:, 1], c=label)
        plt.colorbar()
        ax = plt.gca()
        ax.set_aspect(1)
        # plt.colorbar()
        plt.show()
    else:
        ImageScatter.mnist_images(obj_path, eta=0.8, label=label)


def combination():
    """
    对数据进行组合，每三个类组合成一个数据
    :return:
    """
    path = "E:\\Project\\DataLab\\fashionMnist\\"
    fashion_count = [238, 254, 251, 262, 264, 251, 251, 244, 227, 258]

    for i in range(0, 8):
        X1 = np.loadtxt(path + "fashion50mclass" + str(i) + "_" + str(fashion_count[i]) + "\\data.csv", dtype=np.float,
                        delimiter=",")
        X1_origin = np.loadtxt(path + "fashion50mclass" + str(i) + "_" + str(fashion_count[i]) + "\\origin.csv",
                               dtype=np.float, delimiter=",")
        (n1, m1) = X1.shape
        for j in range(i + 1, 9):
            X2 = np.loadtxt(path + "fashion50mclass" + str(j) + "_" + str(fashion_count[j]) + "\\data.csv", dtype=np.float,
                            delimiter=",")
            X2_origin = np.loadtxt(path + "fashion50mclass" + str(j) + "_" + str(fashion_count[j]) + "\\origin.csv",
                                   dtype=np.float, delimiter=",")
            (n2, m2) = X2.shape
            for k in range(j + 1, 10):
                X3 = np.loadtxt(path + "fashion50mclass" + str(k) + "_" + str(fashion_count[k]) + "\\data.csv",
                                dtype=np.float, delimiter=",")
                X3_origin = np.loadtxt(path + "fashion50mclass" + str(k) + "_" + str(fashion_count[k]) + "\\origin.csv",
                                       dtype=np.float, delimiter=",")
                (n3, m3) = X3.shape
                X = np.zeros((n1 + n2 + n3, m1))
                X_origin = np.zeros((n1 + n2 + n3, 784))
                X[0:n1, :] = X1[:, :]
                X[n1:n1 + n2, :] = X2[:, :]
                X[n1 + n2:n1 + n2 + n3, :] = X3[:, :]
                X_origin[0:n1, :] = X1_origin[:, :]
                X_origin[n1:n1 + n2, :] = X2_origin[:, :]
                X_origin[n1 + n2:n1 + n2 + n3, :] = X3_origin[:, :]

                label = []
                for index in range(0, n1):
                    label.append(i)
                for index in range(0, n2):
                    label.append(j)
                for index in range(0, n3):
                    label.append(k)

                temp_path = path + "combination\\" + "fashion50mclass" + str(i) + str(j) + str(k) + "\\"
                if not os.path.exists(temp_path):
                    os.makedirs(temp_path)
                np.savetxt(temp_path + "data.csv", X, fmt="%f", delimiter=",")
                np.savetxt(temp_path + "label.csv", np.array(label).T, fmt='%d', delimiter=",")
                np.savetxt(temp_path + "origin.csv", X_origin, fmt='%d', delimiter=",")
                print(i, j, k)


def combination4():
    """
    四种物体组合, 6 和 8 是比价好的，保持6和8
    :return:
    """
    path = "E:\\文件\\IRC\\特征向量散点图项目\\DataLab\\fashion\\"
    fashion_count = [238, 254, 251, 262, 264, 251, 251, 244, 227, 258]
    in_path = path + "single\\fashion50mclass"
    data6 = np.loadtxt(in_path+"6_251\\data.csv", dtype=np.float, delimiter=",")
    label6 = np.loadtxt(in_path+"6_251\\label.csv", dtype=np.int, delimiter=",")
    origin6 = np.loadtxt(in_path+"6_251\\origin.csv", dtype=np.int, delimiter=",")
    data8 = np.loadtxt(in_path + "8_227\\data.csv", dtype=np.float, delimiter=",")
    label8 = np.loadtxt(in_path + "8_227\\label.csv", dtype=np.int, delimiter=",")
    origin8 = np.loadtxt(in_path + "8_227\\origin.csv", dtype=np.int, delimiter=",")

    for i in range(0, 9):
        if i == 6 or i == 8:
            continue
        in_path_i = in_path+str(i)+"_"+str(fashion_count[i])+"\\"
        Xi_data = np.loadtxt(in_path_i+"data.csv", dtype=np.float, delimiter=",")
        Xi_label = np.loadtxt(in_path_i+"label.csv", dtype=np.int, delimiter=",")
        Xi_origin = np.loadtxt(in_path_i+"origin.csv", dtype=np.int, delimiter=",")
        for j in range(i+1, 10):
            if j == 6 or j == 8:
                continue
            in_path_j = in_path + str(j) + "_" + str(fashion_count[j]) + "\\"
            Xj_data = np.loadtxt(in_path_j + "data.csv", dtype=np.float, delimiter=",")
            Xj_label = np.loadtxt(in_path_j + "label.csv", dtype=np.int, delimiter=",")
            Xj_origin = np.loadtxt(in_path_j + "origin.csv", dtype=np.int, delimiter=",")

            out_path = path + "combination4\\fashion50mClass68"+str(i)+str(j)+"\\"
            image_path = path + "combination4PCA\\"
            if os.path.exists(out_path):
                pass
            else:
                os.makedirs(out_path)
            n = fashion_count[6] + fashion_count[8] + fashion_count[i] + fashion_count[j]
            data = np.zeros((n, 50))
            label = np.zeros((n, 1))
            origin = np.zeros((n, 784))

            n1 = fashion_count[6]
            n2 = n1 + fashion_count[8]
            n3 = n2 + fashion_count[i]

            data[0:n1, :] = data6[:, :]
            origin[0:n1, :] = origin6[:, :]
            data[n1:n2, :] = data8[:, :]
            origin[n1:n2, :] = origin8[:, :]
            data[n2:n3, :] = Xi_data[:, :]
            origin[n2:n3, :] = Xi_origin[:, :]
            data[n3:n, :] = Xj_data[:, :]
            origin[n3:n, :] = Xj_origin[:, :]

            label[0:n1] = 6
            label[n1:n2] = 8
            label[n2:n3] = i
            label[n3:n] = j

            np.savetxt(out_path+"data.csv", data, fmt='%f', delimiter=",")
            np.savetxt(out_path+"label.csv", label, fmt='%d', delimiter=",")
            np.savetxt(out_path+"origin.csv", origin, fmt='%d', delimiter=",")
            label2 = np.loadtxt(out_path+"label.csv", dtype=np.int, delimiter=",")

            pca1 = PCA(n_components=2)
            Y1 = pca1.fit_transform(data)
            pca2 = PCA(n_components=50)
            data2 = pca2.fit_transform(origin)
            pca3 = PCA(n_components=2)
            Y2 = pca3.fit_transform(origin)

            np.savetxt(out_path+"data2.csv", data2, fmt='%f', delimiter=",")
            plt.scatter(Y1[:, 0], Y1[:, 1], c=label2)
            plt.title("68"+str(i)+str(j)+"A")
            plt.colorbar()
            plt.savefig(image_path+"68"+str(i)+str(j)+"A.png")
            plt.close()
            plt.scatter(Y2[:, 0], Y2[:, 1], c=label2)
            plt.title("68" + str(i) + str(j) + "B")
            plt.colorbar()
            plt.savefig(image_path + "68" + str(i) + str(j) + "B.png")
            plt.close()

            print("68" + str(i) + str(j))


def fashion568_test():
    """
    探究fashion568在t-SNE中不正常的原因
    :return:
    """
    path = "E:\\文件\\IRC\\特征向量散点图项目\\result2020\\result0119_withoutnormalize\\datasets\\fashion50mclass568\\"
    data = np.loadtxt(path+"data.csv", dtype=np.float, delimiter=",")
    label = np.loadtxt(path+"label.csv", dtype=np.int, delimiter=",")

    # t_sne = TSNE(n_components=2, perplexity=30.0)  # sklearn 中的t-SNE
    t_sne = cTSNE.cTSNE(n_component=2, perplexity=30.0)
    Y = t_sne.fit_transform(data)
    # np.savetxt(path+"sTSNE.csv", Y, fmt='%f', delimiter=",")

    plt.scatter(Y[:, 0], Y[:, 1], c=label)
    plt.show()


if __name__ == '__main__':
    # sampling()
    # see_data()
    # pca_research()
    # classify()
    # pca_art_scatter()
    # combination()
    # fashion568_test()
    combination4()
