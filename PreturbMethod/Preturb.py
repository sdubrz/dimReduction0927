# 观察各种不同的扰动方式造成的结果
# 好的扰动方法应该是没有添加扰动的样本不会发生太大变化
import numpy as np
import matplotlib.pyplot as plt


def load_data(path):
    """加载数据"""
    X = np.loadtxt(path+"x.csv", dtype=np.float, delimiter=",")
    vectors = np.loadtxt(path+"【weighted】eigenvectors0.csv", dtype=np.float, delimiter=',')

    return X, vectors


def preturb_whole(X, vectors):
    """
    一次性对所有的样本添加扰动。
    为了便于观察和评估这种扰动方法的效果，会随机选择几个点不填加扰动.
    降维方法采用MDS方法
    :param X: 数据矩阵
    :param vectors: 扰动向量矩阵
    :return:
    """
    (n, m) = X.shape
    y_random = np.random.random((n, 2))
    print(y_random)


def run():
    path = "E:\\Project\\result2019\\PreturbTest\\Wine\\"
    X, vecotrs = load_data(path)
    preturb_whole(X, vecotrs)


if __name__ == '__main__':
    run()

