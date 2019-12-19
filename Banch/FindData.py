# 观察一些数据的降维散点图效果
# 用于寻找数据
import numpy as np
import matplotlib.pyplot as plt
import os

from Main import DimReduce
from Main import Preprocess


def look_data(path, iso_k=30):
    """
    查看数据集使用基本的降维方法的效果
    :param path: 数据的存放目录
    :param iso_k: 计算测地线距离的k值
    :return:
    """
    data_reader = np.loadtxt(path+"data.csv", dtype=np.str, delimiter=',')
    data = data_reader[:, :].astype(np.float)
    label_reader = np.loadtxt(path + "label.csv", dtype=np.str, delimiter=',')
    label = label_reader.astype(np.int)
    (n, dim) = data.shape
    print(data.shape)

    # 如果目前还没有初始的随机结果，现在就生成并保存
    if not os.path.exists(path+"y_random.csv"):
        y_random = np.random.random((n, 2))
        np.savetxt(path+"y_random.csv", y_random, fmt='%f', delimiter=',')

    X = Preprocess.normalize(data)
    pca_y = DimReduce.dim_reduce(X, method='PCA')
    mds_y = DimReduce.dim_reduce(X, method='MDS')
    tsne_y = DimReduce.dim_reduce(X, method='t-SNE', method_k=90)
    iso_y = DimReduce.dim_reduce(X, method='Isomap', method_k=iso_k)
    print('降维结束')

    plt.subplot(221)
    plt.scatter(pca_y[:, 0], pca_y[:, 1], c=label, marker='o')
    plt.title('PCA')
    plt.subplot(222)
    plt.scatter(mds_y[:, 0], mds_y[:, 1], c=label, marker='o')
    plt.title('MDS')
    plt.subplot(223)
    plt.scatter(tsne_y[:, 0], tsne_y[:, 1], c=label, marker='o')
    plt.title('t-SNE')
    plt.subplot(224)
    plt.scatter(iso_y[:, 0], iso_y[:, 1], c=label, marker='o')
    plt.title('Isomap')

    plt.show()


def fengkang_data(path):
    """如果数据集来自于冯康的数据集，首先改成我们的格式"""
    old_data = np.loadtxt(path+"data.data", dtype=np.float, delimiter=',')
    (n, m) = old_data.shape
    np.savetxt(path+"data.csv", old_data[:, 1:m], fmt='%f', delimiter=',')
    np.savetxt(path+"label.csv", old_data[:, 0], fmt='%d', delimiter=',')
    print('数据转换完成')


def run():
    main_path = "E:\\Project\\result2019\\result0927\\"
    data_name = 'ecoli'
    path = main_path + "datasets\\" + data_name + "\\"
    # fengkang_data(path)
    look_data(path, iso_k=15)


if __name__ == '__main__':
    run()
