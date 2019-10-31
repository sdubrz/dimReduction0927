# 清洗数据
import numpy as np
from Main import DimReduce
from Main import Preprocess
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def wine_quality_red():
    """红酒质量数据"""
    in_path = "E:\\Project\\result2019\\result0927\\datasets\\winequalityred\\"
    data0 = np.loadtxt(in_path+"winequality-red.csv", dtype=np.float, delimiter=";")

    (n, m) = data0.shape
    print((n, m))
    np.savetxt(in_path+"data.csv", data0[:, 0:m-1], fmt='%f', delimiter=',')
    np.savetxt(in_path+"label.csv", data0[:, m-1], fmt='%d', delimiter=',')

    print('数据清洗完毕')


def bostonHouse6912():
    path = "E:\\Project\\result2019\\result1026without_straighten\\datasets\\bostonHouse\\"
    data = np.loadtxt(path+"data.csv", dtype=np.float, delimiter=",")
    (n, m) = data.shape

    data2 = np.zeros((n, 3))
    data2[:, 0] = data[:, 5]
    data2[:, 1] = data[:, 8]
    data2[:, 2] = data[:, 13]

    np.savetxt(path+"data2.csv", data2, fmt="%f", delimiter=",")
    print("数据处理完毕")


def coil_20():
    path = "E:\\Project\\result2019\\result1026without_straighten\\datasets\\coil20obj\\"
    data = np.loadtxt(path+"data.csv", dtype=np.float, delimiter=",")
    label = np.loadtxt(path+"label.csv", dtype=np.int, delimiter=",")
    (n, m) = data.shape
    X = Preprocess.normalize(data, -1, 1)

    # dr_method = "PCA"
    # Y = DimReduce.dim_reduce(X[0:5*72, :], method=dr_method, method_k=15)
    # plt.scatter(Y[:, 0], Y[:, 1], marker='o', c=label[0:5*72])

    # Y = DimReduce.dim_reduce(X, method=dr_method, method_k=15)
    # plt.scatter(Y[:, 0], Y[:, 1], marker='o', c=label)
    # plt.title(dr_method)
    # ax = plt.gca()
    # ax.set_aspect(1)
    # plt.show()

    # 用PCA将数据压缩到 16 维
    pca = PCA(n_components=16)
    Y = pca.fit_transform(X)
    np.savetxt(path+"Y16.csv", Y, fmt='%f', delimiter=",")
    np.savetxt(path+"Y16_5class.csv", Y[0:5*72, :], fmt='%f', delimiter=",")
    print("数据处理完毕")


if __name__ == '__main__':
    # wine_quality_red()
    # bostonHouse6912()
    coil_20()
