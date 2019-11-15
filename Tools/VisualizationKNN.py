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


def run_test():
    path = "E:\\Project\\result2019\\result1026without_straighten\\PCA\\Wine\\yita(0.1)nbrs_k(45)method_k(20)numbers(4)_b-spline_weighted\\"
    draw_knn(path)


if __name__ == '__main__':
    run_test()
