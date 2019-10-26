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
    preturb_sum = 0
    stable_sum = 0
    for i in range(0, n):
        if i in stable_list:
            stable_change.append(change[i, 0])
            stable_sum += change[i, 0]
        else:
            preturb_change.append(change[i, 0])
            preturb_sum += change[i, 0]

    preturb_average = preturb_sum / len(preturb_change)
    stable_average = stable_sum / len(stable_change)

    plt.subplot(121)
    plt.hist(preturb_change, bins=100)
    plt.title('preturb points\' change, average='+str(preturb_average))
    plt.subplot(122)
    plt.hist(stable_change, bins=100)
    plt.title('stable points\' change, average='+str(stable_average))
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
    # print(y_random)
    print((n, m))

    eta = 0.1
    # method = 'MDS'

    Y = DimReduce.dim_reduce(X, method='MDS')
    # 随机地挑选10个点不动
    stable = 100
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


def preturb_1_by_1(X, vectors, eta, y_random):
    """
    对一个数据集一次只对一个点进行扰动
    :param X:
    :param vectors:
    :param eta:
    :param y_random:
    :return:
    """
    (n, m) = X.shape
    Y = np.zeros((n, 2))

    for i in range(0, n):
        X2 = X.copy()
        X2[i, :] = X2[i, :] + eta * vectors[i, :]
        temp_y = DimReduce.dim_reduce(X2, method='MDS', y_random=y_random)
        Y[i, :] = temp_y[i, :]

        if i % 10 == 0:
            print(i, ' of ', n)

    return Y


def compare_2_method(X, vectors):
    """
    比较每次只扰动一个点和扰动所有的点之间的区别
    :param X: 数据矩阵
    :param vectors: 扰动向量
    :return:
    """
    eta = 0.1  # 扰动步长
    (n, m) = X.shape
    Y = DimReduce.dim_reduce(X, method='MDS')

    # 一次性对所有点扰动
    print('一次性对所有点扰动......')
    X1 = X + eta * vectors
    Y1 = DimReduce.dim_reduce(X1, method='MDS', y_random=Y)

    # 扰动 one by one
    print('逐个点扰动......')
    Y2 = preturb_1_by_1(X, vectors, eta, Y)

    plt.scatter(Y[:, 0], Y[:, 1], c='k', marker='o')
    plt.scatter(Y1[:, 0], Y1[:, 1], c='g', marker='^')
    plt.scatter(Y2[:, 0], Y2[:, 1], c='m', marker='p')

    for i in range(0, n):
        plt.plot([Y[i, 0], Y2[i, 0]], [Y[i, 1], Y2[i, 1]], c='deepskyblue', linewidth=0.6)
        plt.plot([Y[i, 0], Y1[i, 0]], [Y[i, 1], Y1[i, 1]], c='deepskyblue', linewidth=0.6)

    ax = plt.gca()
    ax.set_aspect(1)
    plt.show()

    return Y, Y1, Y2


def evaluate(Y, Y1, Y2):
    """
    用数据说明两种扰动方法之间的差别
    :param Y: 没有扰动的降维结果
    :param Y1: 一次性扰动的降维结果
    :param Y2: 逐个扰动的降维结果
    :return:
    """
    (n, m) = Y.shape
    change01 = np.zeros((n, 1))
    change02 = np.zeros((n, 1))
    change12 = np.zeros((n, 1))
    relative_change = np.zeros((n, 1))

    for i in range(0, n):
        change01[i, 0] = np.linalg.norm(Y1[i, :] - Y[i, :])
        change02[i, 0] = np.linalg.norm(Y2[i, :] - Y[i, :])
        change12[i, 0] = np.linalg.norm(Y1[i, :] - Y2[i, :])

        if change02[i, 0] != 0:
            relative_change[i, 0] = change12[i, 0] / change02[i, 0]

    cos_list = np.zeros((n, 1))
    for i in range(0, n):
        v1 = Y1[i, :] - Y[i, :]
        v2 = Y2[i, :] - Y[i, :]
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 * norm2 != 0:
            cos_list[i, 0] = np.dot(v1, v2) / (norm1 * norm2)

    plt.subplot(321)
    plt.hist(change01, bins=50)
    plt.title('change between origin and whole')

    plt.subplot(322)
    plt.hist(change02, bins=50)
    plt.title('change between origin and 1 by 1')

    plt.subplot(323)
    plt.hist(change12, bins=50)
    plt.title('difference between two methods')

    plt.subplot(324)
    plt.hist(relative_change, bins=50)
    plt.title('relative change')

    plt.subplot(325)
    plt.hist(cos_list, bins=50)
    plt.title('cos similar')

    # plt.subplot(326)
    # plt.plot(cos_list)

    plt.show()




def run():
    path = "E:\\Project\\result2019\\PreturbTest\\Wine\\"
    X, vecotrs = load_data(path)
    # preturb_whole(X, vecotrs)
    Y, Y1, Y2 = compare_2_method(X, vecotrs)
    np.savetxt(path+"Y.csv", Y, fmt='%f', delimiter=',')
    np.savetxt(path + "Y1.csv", Y1, fmt='%f', delimiter=',')
    np.savetxt(path + "Y2.csv", Y2, fmt='%f', delimiter=',')
    evaluate(Y, Y1, Y2)


if __name__ == '__main__':
    run()

