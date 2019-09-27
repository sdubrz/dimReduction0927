import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.manifold import Isomap
from sklearn.manifold import LocallyLinearEmbedding


"""
降维的主要实现函数
"""


def dim_reduce(data, method="MDS", method_k=30, y_random=None):
    """
    对数据进行降维，返回二维的投影结果
    :param data: 原始的高维数据矩阵，每一行是一条高维数据记录
    :param method: 降维方法名称，目前已经支持的降维方法有
                    MDS ， tsne , LLE , Hessien_eigenmap ， Isomap
                    默认使用的方法是 MDS
    :param method_k: 某些非线性降维方法所需的k值
    :param y_random: 某些降维方法所需要的初始随机结果矩阵，如果为None则调用numpy中的相关函数生成一个随机矩阵
    :return: 降维之后的二维结果矩阵
    """
    data_shape = data.shape
    n = data_shape[0]
    dim = data_shape[1]

    if method_k > n-1:
        print("[DimReduce]\t警告：输入的method_k值过大")
        method_k = n-1

    if y_random is None:
        y_random = np.random.random((n, 2))

    y = np.zeros((n, 2))

    if method == 'tsne' or method == 't-SNE':
        print("[DimReduce]\t当前使用 t-SNE 降维方法")
        tsne = TSNE(n_components=2, n_iter=5000, perplexity=method_k / 3, init=y_random)
        y = tsne.fit_transform(data)

    elif method == 'MDS' or method == 'mds':
        print("[DimReduce]\t当前使用 MDS 降维方法")
        mds = MDS(n_components=2)
        y = mds.fit_transform(data, init=y_random)

    elif method == 'isomap' or method == 'Isomap':
        print("[DimReduce]\t当前使用 Isomap 降维方法")
        iso_map = Isomap(n_neighbors=method_k, n_components=2, eigen_solver='dense')
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
    else:
        print("[DimReduce]\t未能匹配到合适的降维方法")

    return y

