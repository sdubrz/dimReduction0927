# 观察各种不同的扰动方式造成的结果
# 好的扰动方法应该是没有添加扰动的样本不会发生太大变化
import numpy as np
import matplotlib.pyplot as plt
from Main import DimReduce


def load_data(path):
    """加载数据"""
    X = np.loadtxt(path+"x.csv", dtype=np.float, delimiter=",")
    vectors = np.loadtxt(path+"【weighted】eigenvectors0.csv", dtype=np.float, delimiter=',')

    return X, vectors


def random_index(low, up, n):
    """
    随机生成 [low, up) 之间的 n 个索引值
    :param low: 下界
    :param up: 上界
    :param n: 个数
    :return:
    """
    index_list = []
    if up - low < n:
        n = up - low

    while len(index_list) < n:
        temp = np.random.randint(low, up)
        if not(temp in index_list):
            index_list.append(temp)

    return index_list


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

    eta = 0.1
    method = 'MDS'

    Y = DimReduce.dim_reduce(X, method='MDS', y_random=y_random)
    # 随机地挑选10个点不动
    stable = 177
    stable_list = random_index(0, n, stable)
    X2 = X + eta*vectors
    for index in stable_list:
        X2[index, :] = X[index, :]

    Y2 = DimReduce.dim_reduce(X2, method='MDS', y_random=Y)

    plt.scatter(Y[:, 0], Y[:, 1], marker='o', c='y')
    plt.scatter(Y2[:, 0], Y2[:, 1], marker='o', c='deepskyblue')

    for i in range(0, n):
        line_color = 'deepskyblue'
        if i in stable_list:
            line_color = 'r'
        plt.plot([Y[i, 0], Y2[i, 0]], [Y[i, 1], Y2[i, 1]], c=line_color, linewidth=0.6)

    plt.title('eta = ' + str(eta) + ", stable = " + str(stable))
    plt.show()


def run():
    path = "E:\\Project\\result2019\\PreturbTest\\Wine\\"
    X, vecotrs = load_data(path)
    preturb_whole(X, vecotrs)


if __name__ == '__main__':
    run()

