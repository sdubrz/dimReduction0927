# 将数据的KNN信息在降维之后的散点图中显示出来
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.neighbors import NearestNeighbors


def draw_knn(path=""):
    """
    绘制 KNN 关系信息
    :param path: json文件存放目录
    :return:
    """
    save_path = path + "knn\\"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    KNN = np.loadtxt(path+"【weighted】knn.csv", dtype=np.int, delimiter=",")
    Y = np.loadtxt(path+"y.csv", dtype=np.float, delimiter=",")
    labels = np.loadtxt(path+"label.csv", dtype=np.int, delimiter=",")
    (n, k) = KNN.shape

    plt.scatter(Y[:, 0], Y[:, 1], c=labels)
    for i in range(0, n):
        plt.text(Y[i, 0], Y[i, 1], str(i))
    ax = plt.gca()
    ax.set_aspect(1)
    plt.savefig(save_path+"index.png")
    plt.close()

    for i in range(0, n):
        temp_label = 0*labels
        for j in range(0, k):
            temp_label[KNN[i, j]] = 1
        temp_label[i] = 2
        plt.scatter(Y[:, 0], Y[:, 1], c=temp_label)
        for j in range(0, k):
            temp_index = KNN[i, j]
            # plt.text(Y[temp_index, 0], Y[temp_index, 1], str(temp_index))
        ax2 = plt.gca()
        ax2.set_aspect(1)
        plt.savefig(save_path+str(i)+".png")
        plt.close()

    print("KNN关系图保存完毕")


def KNN_similar(path=""):
    """
    计算每个点之间的KNN的相似性
    由于我们在计算local PCA时，并没有各个点分配权重，因为在计算相似度时，KNN中的点不分先后。
    :param path: 计算结果的存放路径
    :return:
    """
    KNN = np.loadtxt(path + "【weighted】knn.csv", dtype=np.int, delimiter=",")
    (n, k) = KNN.shape

    similar = np.zeros((n, n))
    KNN_list = KNN.tolist()
    for i in range(0, n-1):
        i_list = KNN_list[i]
        for j in range(i+1, n):
            j_list = KNN_list[j]
            count = 0
            for index in i_list:
                if index in j_list:
                    count += 1
            similar[i, j] = count / k
            similar[j, i] = count / k
        similar[i, i] = 1

    np.savetxt(path+"knn_similar.csv", similar, fmt='%f', delimiter=",")
    print("KNN相似性计算完毕")


def knn_keep(path=""):
    """
    计算降维前后 KNN 的保持程度
    :param path: 中间结果的存放路径
    :return:
    """
    Y = np.loadtxt(path+"y.csv", dtype=np.float, delimiter=",")
    high_knn = np.loadtxt(path+"【weighted】knn.csv", dtype=np.int, delimiter=",")
    (n, k) = high_knn.shape

    nbr_s = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(Y)
    distance, d2_knn = nbr_s.kneighbors(Y)

    high_knn_list = high_knn.tolist()
    low_knn_list = d2_knn.tolist()
    knn_score = np.zeros((n, 1))

    for i in range(0, n):
        list_h = high_knn_list[i]
        list_l = low_knn_list[i]
        count = 0
        for index in list_h:
            if index in list_l:
                count += 1
        knn_score[i, 0] = count / k

    np.savetxt(path+"knn_keep.csv", knn_score, fmt='%f', delimiter=",")
    return knn_score


def draw_index(path=""):
    """
    画散点的索引图
    :param path:
    :return:
    """
    Y = np.loadtxt(path + "y.csv", dtype=np.float, delimiter=",")
    labels = np.loadtxt(path + "label.csv", dtype=np.int, delimiter=",")
    (n, m) = Y.shape

    plt.scatter(Y[:, 0], Y[:, 1], c=labels)
    for i in range(0, n):
        plt.text(Y[i, 0], Y[i, 1], str(i))
    ax = plt.gca()
    ax.set_aspect(1)
    plt.show()


def run_test():
    path = "E:\\Project\\result2019\\result1026without_straighten\\PCA\\coil20obj_16_5class\\yita(0.1)nbrs_k(20)method_k(20)numbers(4)_b-spline_weighted\\"
    # draw_knn(path)
    # knn_keep(path)
    draw_index(path)


def run_test2():
    list1 = [1, 2, 3, 4]
    list2 = [6, 7, 8, 9]
    for (a, b) in (list1, list2):
        print("a="+str(a)+", b="+str(b))


if __name__ == '__main__':
    run_test()
