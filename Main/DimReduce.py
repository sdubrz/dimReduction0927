import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
# from sklearn.manifold import Isomap
from Main.MyIsomap import Isomap
from Main.LDA import LDA
from sklearn.manifold import LocallyLinearEmbedding
from Main import Preprocess
import matplotlib.pyplot as plt


"""
降维的主要实现函数
"""


def dim_reduce(data, method="MDS", method_k=30, y_random=None, label=None):
    """
    对数据进行降维，返回二维的投影结果
    :param data: 原始的高维数据矩阵，每一行是一条高维数据记录
    :param method: 降维方法名称，目前已经支持的降维方法有
                    MDS ， tsne , LLE , Hessien_eigenmap ， Isomap
                    默认使用的方法是 MDS
    :param method_k: 某些非线性降维方法所需的k值
    :param y_random: 某些降维方法所需要的初始随机结果矩阵，如果为None则调用numpy中的相关函数生成一个随机矩阵
    :param label: 数据的分类标签， LDA方法会用到
    :return: 降维之后的二维结果矩阵
    """
    data_shape = data.shape
    n = data_shape[0]
    dim = data_shape[1]

    if method_k > n-1:
        print("[DimReduce]\t警告：输入的method_k值过大")
        method_k = n-1

    y = np.zeros((n, 2))

    if method == 'tsne' or method == 't-SNE':
        print("[DimReduce]\t当前使用 t-SNE 降维方法")
        if y_random is None:
            y_random = np.random.random((n, 2))
        tsne = TSNE(n_components=2, n_iter=5000, perplexity=method_k / 3, init=y_random)
        y = tsne.fit_transform(data)

    elif method == 'MDS' or method == 'mds':
        print("[DimReduce]\t当前使用 MDS 降维方法")
        if y_random is None:
            mds = MDS(n_components=2, max_iter=3000, n_init=10)
            y = mds.fit_transform(data)
        else:
            mds = MDS(n_components=2, max_iter=3000)
            y = mds.fit_transform(data, init=y_random)

    elif method == 'isomap' or method == 'Isomap':
        print("[DimReduce]\t当前使用 Isomap 降维方法")
        iso_map = Isomap(n_neighbors=method_k, n_components=2, init=y_random)
        y = iso_map.fit_transform(data)

    elif method == 'LLE' or method == 'lle':
        print("[DimReduce]\t当前使用 LLE 降维方法")
        lle = LocallyLinearEmbedding(n_neighbors=method_k, n_components=2, n_jobs=1)  # eigen_solver='dense'
        y = lle.fit_transform(data)

    elif method == 'Hessien_eigenmap' or method == 'hessien_eigenmap' or method == 'Hessien' or method == 'hessien':
        print("[DimReduce]\t当前使用 Hessien_eigenmap 降维方法")
        hessien = LocallyLinearEmbedding(n_neighbors=method_k, n_components=2, method='hessian', eigen_solver='dense')
        y = hessien.fit_transform(data)

    elif method == 'PCA' or method == 'pca':
        print("[DimReduce]\t当前使用 PCA 降维方法")
        pca = PCA(n_components=2)
        y = pca.fit_transform(data)
    elif method == 'LDA' or method == 'lda':
        print("[DimReduce]\t当前使用 LDA 降维方法")
        lda = LDA(n_component=2)
        y = lda.fit_transform(data, label)
    else:
        print("[DimReduce]\t未能匹配到合适的降维方法")

    return y


def run_test():
    # path = "E:\\Project\\result2019\\result1026without_straighten\\datasets\\coil20obj_16_5class\\"
    path = "E:\\Project\\DataLab\\wineQuality\\"
    X = np.loadtxt(path+"data.csv", dtype=np.float, delimiter=",")
    label = np.loadtxt(path+"label.csv", dtype=np.int, delimiter=",")
    (n, m) = X.shape
    X = Preprocess.normalize(X, -1, 1)
    Y = dim_reduce(X, method="tsne", method_k=90)

    plt.scatter(Y[:, 0], Y[:, 1], c=label)
    plt.colorbar()
    ax = plt.gca()
    ax.set_aspect(1)
    plt.show()


if __name__ == '__main__':
    run_test()

