# 将数据的KNN信息在降维之后的散点图中显示出来
import numpy as np
import matplotlib.pyplot as plt
import os


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


def run_test():
    path = "E:\\Project\\result2019\\result1026without_straighten\\PCA\\Wine\\yita(0.1)nbrs_k(45)method_k(20)numbers(4)_b-spline_weighted\\"
    draw_knn(path)


if __name__ == '__main__':
    run_test()
