# 观察一些数据的降维散点图效果
# 用于寻找数据
import numpy as np
import matplotlib.pyplot as plt

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


def run():
    main_path = "E:\\Project\\result2019\\result0927\\"
    data_name = 'swissroll2rolls'
    path = main_path + "datasets\\" + data_name + "\\"
    look_data(path, iso_k=15)


if __name__ == '__main__':
    run()
