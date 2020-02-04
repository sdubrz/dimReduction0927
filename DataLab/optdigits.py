# 处理optdigits
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.manifold import TSNE


def classify():
    """
    划分成每个类的单独数据集
    :return:
    """
    path = "E:\\文件\\IRC\\特征向量散点图项目\\DataLab\\optdigits\\"
    origin_data = np.loadtxt(path + "data.csv", dtype=np.int, delimiter=",")
    label = np.loadtxt(path + "label.csv", dtype=np.int, delimiter=",")
    (n, m) = origin_data.shape
    print((n, m))

    origin_data_list = origin_data.tolist()

    origin_list = []
    for i in range(0, 10):
        origin_list.append([])

    for i in range(0, n):
        # print(i)
        obj = label[i]
        temp_list2 = origin_list[obj]
        temp_list2.append(origin_data_list[i])

    for i in range(0, 10):
        obj_path = path + "optdigitClass" + str(i) + "_" + str(len(origin_list[i])) + "\\"
        if not os.path.exists(obj_path):
            os.makedirs(obj_path)

        np.savetxt(obj_path + "data.csv", np.array(origin_list[i]), fmt='%d', delimiter=",")
        np.savetxt(obj_path+"label.csv", i*np.ones((len(origin_list[i]), 1)), fmt='%d', delimiter=",")
        print(i)


def single_dr():
    """
    查看单独一类的降维结果
    :return:
    """
    main_path = "E:\\文件\\IRC\\特征向量散点图项目\\DataLab\\optdigits\\"
    digit_count = [554, 571, 557, 572, 568, 558, 558, 566, 554, 562]
    index = 9

    path = main_path + "optdigitClass" + str(index) + "_" + str(digit_count[index])+"\\"
    data = np.loadtxt(path+"data.csv", dtype=np.float, delimiter=",")
    label = np.loadtxt(path+"label.csv", dtype=np.int, delimiter=",")

    pca = PCA(n_components=2)
    Y = pca.fit_transform(data)

    np.savetxt(path+"PCA.csv", Y, fmt='%f', delimiter=",")

    plt.scatter(Y[:, 0], Y[:, 1])
    ax = plt.gca()
    ax.set_aspect(1)
    plt.title(str(index))
    plt.show()


def combination3():
    """
    对数据进行组合，每三个类组合成一个数据
    :return:
    """
    path = "E:\\文件\\IRC\\特征向量散点图项目\\DataLab\\optdigits\\"
    fashion_count = [554, 571, 557, 572, 568, 558, 558, 566, 554, 562]

    for i in range(0, 8):
        X1 = np.loadtxt(path + "optdigitClass" + str(i) + "_" + str(fashion_count[i]) + "\\data.csv", dtype=np.float,
                        delimiter=",")
        X1_origin = np.loadtxt(path + "optdigitClass" + str(i) + "_" + str(fashion_count[i]) + "\\origin.csv",
                               dtype=np.float, delimiter=",")
        (n1, m1) = X1.shape
        for j in range(i + 1, 9):
            X2 = np.loadtxt(path + "optdigitClass" + str(j) + "_" + str(fashion_count[j]) + "\\data.csv", dtype=np.float,
                            delimiter=",")
            X2_origin = np.loadtxt(path + "optdigitClass" + str(j) + "_" + str(fashion_count[j]) + "\\origin.csv",
                                   dtype=np.float, delimiter=",")
            (n2, m2) = X2.shape
            for k in range(j + 1, 10):
                X3 = np.loadtxt(path + "optdigitClass" + str(k) + "_" + str(fashion_count[k]) + "\\data.csv",
                                dtype=np.float, delimiter=",")
                X3_origin = np.loadtxt(path + "optdigitClass" + str(k) + "_" + str(fashion_count[k]) + "\\origin.csv",
                                       dtype=np.float, delimiter=",")
                (n3, m3) = X3.shape
                X = np.zeros((n1 + n2 + n3, m1))
                X_origin = np.zeros((n1 + n2 + n3, 64))
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

                temp_path = path + "combination\\" + "optdigitClass" + str(i) + str(j) + str(k) + "\\"
                if not os.path.exists(temp_path):
                    os.makedirs(temp_path)
                np.savetxt(temp_path + "data.csv", X, fmt="%f", delimiter=",")
                np.savetxt(temp_path + "label.csv", np.array(label).T, fmt='%d', delimiter=",")
                np.savetxt(temp_path + "origin.csv", X_origin, fmt='%d', delimiter=",")
                print(i, j, k)

                pca = PCA(n_components=2)
                Y1 = pca.fit_transform(X)
                plt.scatter(Y1[:, 0], Y1[:, 1], c=label)
                plt.title(str(i) + str(j) + str(k)+" PCA")
                plt.savefig(temp_path+"PCA.png")
                plt.close()

                # mds = MDS(n_components=2)
                # Y2 = mds.fit_transform(X)
                # plt.scatter(Y2[:, 0], Y2[:, 1], c=label)
                # plt.title(str(i) + str(j) + str(k) + " MDS")
                # plt.savefig(temp_path + "MDS.png")
                # plt.close()
                #
                # t_sne = TSNE(n_components=2, perplexity=30.0)
                # Y3 = t_sne.fit_transform(X)
                # plt.scatter(Y3[:, 0], Y3[:, 1], c=label)
                # plt.title(str(i) + str(j) + str(k) + " t-SNE")
                # plt.savefig(temp_path + "t-SNE.png")
                # plt.close()


def combination3_dr():
    """
    对三三组合进行降维
    :return:
    """
    path = "E:\\文件\\IRC\\特征向量散点图项目\\DataLab\\optdigits\\"
    out_path = path + "combination3MDS\\"
    for i in range(0, 8):
        for j in range(i+1, 9):
            for k in range(j+1, 10):
                case_name = str(i) + str(j) + str(k)
                in_path = path + "combination3\\optdigitClass" + case_name + "\\"
                data = np.loadtxt(in_path+"data.csv", dtype=np.float, delimiter=",")
                label = np.loadtxt(in_path+"label.csv", dtype=np.int, delimiter=",")
                pca = MDS(n_components=2)
                Y = pca.fit_transform(data)
                plt.scatter(Y[:, 0], Y[:, 1], c=label)
                plt.title(case_name)
                plt.colorbar()
                plt.savefig(out_path+case_name+".png")
                plt.close()
                print(case_name)


if __name__ == '__main__':
    # classify()
    # single_dr()
    # combination3()
    combination3_dr()
