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


def change_norm(Y0, Y2):
    """
    计算两个降维结果的改变量
    :param Y0: 第一次降维结果
    :param Y2: 第二次降维结果
    :return:
    """
    (n, m) = Y0.shape
    change = np.zeros((n, 1))
    for i in range(0, n):
        change[i] = np.linalg.norm(Y0[i, :] - Y2[i, :])

    return change


def draw_change_hist(Y0, Y2, stable_list):
    """
    分别画出扰动的点和未扰动的点，降维结果改变量的直方图
    :param Y0: 第一次降维结果
    :param Y2: 第二次降维结果
    :param stable_list: 没有进行扰动的点的索引集合
    :return:
    """
    (n, m) = Y0.shape
    change = change_norm(Y0, Y2)
    preturb_change = []
    stable_change = []
    for i in range(0, n):
        if i in stable_list:
            stable_change.append(change[i, 0])
        else:
            preturb_change.append(change[i, 0])

    plt.subplot(121)
    plt.hist(preturb_change, bins=20)
    plt.title('preturb points\' change')
    plt.subplot(122)
    plt.hist(stable_change, bins=20)
    plt.title('stable points\' change')
    plt.show()


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
    # method = 'MDS'

    Y = DimReduce.dim_reduce(X, method='MDS')
    # 随机地挑选10个点不动
    stable = 10
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
    ax = plt.gca()
    ax.set_aspect(1)
    plt.show()

    draw_change_hist(Y, Y2, stable_list)


def run():
    path = "E:\\Project\\result2019\\PreturbTest\\Wine\\"
    X, vecotrs = load_data(path)
    preturb_whole(X, vecotrs)


if __name__ == '__main__':
    run()

