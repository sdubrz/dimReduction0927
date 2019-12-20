import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.manifold import Isomap
# from Main.MyIsomap import Isomap  # 这不是经典的Isomap，Isomap应该使用cMDS
from Main.LDA import LDA
from sklearn.manifold import LocallyLinearEmbedding
from Main import Preprocess
import matplotlib.pyplot as plt
import os
from MyDR import cTSNE


"""
降维的主要实现函数
"""


def dim_reduce(data, method="MDS", method_k=30, y_random=None, label=None, n_iters=5000, early_exaggeration=12.0, c_early_exage=True):
    """
    对数据进行降维，返回二维的投影结果
    :param data: 原始的高维数据矩阵，每一行是一条高维数据记录
    :param method: 降维方法名称，目前已经支持的降维方法有
                    MDS ， tsne , LLE , Hessien_eigenmap ， Isomap
                    默认使用的方法是 MDS
    :param method_k: 某些非线性降维方法所需的k值
    :param y_random: 某些降维方法所需要的初始随机结果矩阵，如果为None则调用numpy中的相关函数生成一个随机矩阵
    :param label: 数据的分类标签， LDA方法会用到
    :param n_iters: 某些降维方法所使用的迭代次数
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
        tsne = TSNE(n_components=2, n_iter=n_iters, perplexity=method_k / 3, init=y_random, early_exaggeration=early_exaggeration)
        y = tsne.fit_transform(data)

    if method == 'ctsne' or method == 'cTSNE':
        print("[DimReduce]\t当前使用 c-t-SNE 降维方法")
        t_sne = cTSNE.cTSNE(n_component=2, perplexity=method_k/3)
        y = t_sne.fit_transform(data, max_iter=n_iters, y_random=y_random, early_exaggerate=c_early_exage, show_progress=False)

    elif method == 'MDS' or method == 'mds':
        print("[DimReduce]\t当前使用 MDS 降维方法")
        if y_random is None:
            mds = MDS(n_components=2, max_iter=3000)
            y = mds.fit_transform(data)
        else:
            mds = MDS(n_components=2, max_iter=3000)
            y = mds.fit_transform(data, init=y_random)

    elif method == 'isomap' or method == 'Isomap':
        print("[DimReduce]\t当前使用 Isomap 降维方法")
        iso_map = Isomap(n_neighbors=method_k, n_components=2)
        y = iso_map.fit_transform(data)

    elif method == 'LLE' or method == 'lle':
        print("[DimReduce]\t当前使用 LLE 降维方法")
        lle = LocallyLinearEmbedding(n_neighbors=method_k, n_components=2, n_jobs=1)  # eigen_solver='dense'
        y = lle.fit_transform(data)

    elif method == 'Hessien_eigenmap' or method == 'hessien_eigenmap' or method == 'Hessien' or method == 'hessien':
        print("[DimReduce]\t当前使用 Hessien_eigenmap 降维方法")
        hessien = LocallyLinearEmbedding(n_neighbors=method_k, n_components=2, method='hessian', eigen_solver='dense')
        y = hessien.fit_transform(data)

    elif method == 'LTSA' or method == 'ltsa':
        ltsa = LocallyLinearEmbedding(n_neighbors=method_k, n_components=2, method='ltsa', n_jobs=1)
        y = ltsa.fit_transform(data)

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


def dim_reduce_convergence(data, method="cTSNE", method_k=30, n_iter_init=10000, y_random=None):
    """
    降维，直到相对于屏幕看起来收敛
    :param data: 数据矩阵
    :param method: 降维方法名称
    :param method_k: 某些降维方法需要的K值
    :param n_iter_init: 默认的迭代次数
    :param y_random: 随机的初始矩阵
    :return:
    """
    Y0 = dim_reduce(data, method=method, method_k=method_k, n_iters=n_iter_init, y_random=y_random)
    Y1 = dim_reduce(data, method=method, method_k=method_k, n_iters=1000, y_random=Y0, early_exaggeration=False)

    total_count = n_iter_init
    MAX_LOOP_COUNT = 1000000
    while not convergence_screen(Y0, Y1) and total_count < MAX_LOOP_COUNT:
        print("\t当前已经迭代了 %d 次" % total_count)
        Y0 = dim_reduce(data, method=method, method_k=method_k, n_iters=n_iter_init, y_random=Y0, early_exaggeration=False)
        Y1 = dim_reduce(data, method=method, method_k=method_k, n_iters=1000, y_random=Y0, early_exaggeration=False)
        total_count += n_iter_init

    if total_count >= MAX_LOOP_COUNT:
        print('[DimReduce warning]: 仍然没有达到相对屏幕收敛的要求')
    print('[DimReduce log]: 最终迭代的次数是 ', total_count)

    return Y0


def convergence_screen(Y0, Y1):
    """
    判断降维结果是否相对于屏幕收敛， 如果收敛则返回True
    :param Y0: (n×2) 第一次降维结果矩阵
    :param Y1: (n×2) 第二次降维结果矩阵
    :return:
    """
    (n, m) = Y0.shape
    dx = np.max(Y0[:, 0]) - np.min(Y0[:, 0])
    dy = np.max(Y0[:, 1]) - np.min(Y0[:, 1])
    d_screen = max(dx, dy)

    d_norm = np.zeros((n, 1))
    for i in range(0, n):
        d_norm[i] = np.linalg.norm(Y0[i, :] - Y1[i, :])

    plt.scatter(Y0[:, 0], Y0[:, 1], c='r')
    plt.scatter(Y1[:, 0], Y1[:, 1], c='b')
    plt.show()

    d_mean = np.mean(d_norm)
    print(dx, dy, d_mean)
    if dx >= d_mean * 1000 or dy >= d_mean * 1000:
        return True
    else:
        return False


def run_test():
    path = "E:\\Project\\result2019\\result1026without_straighten\\datasets\\MNIST50mclass1_985\\"
    # path = "E:\\Project\\DataLab\\MoCap\\cleanData\\"
    # path = "E:\\Project\\DataLab\\MNIST50m\\"
    # index = 9
    X = np.loadtxt(path+"data.csv", dtype=np.float, delimiter=",", encoding='UTF-8-sig')
    # print(X)
    # label = np.loadtxt(path+"label.csv", dtype=np.int, delimiter=",", encoding='UTF-8')
    # label = np.loadtxt(path + "quality.csv", dtype=np.int, delimiter=",")
    (n, m) = X.shape
    X2 = Preprocess.normalize(X, -1, 1)
    Y = dim_reduce(X2, method="PCA", method_k=90)

    # np.savetxt(path+"y.csv", Y, fmt='%f', delimiter=",")

    # plt.scatter(Y[:, 0], Y[:, 1], c=label)
    # plt.colorbar()
    plt.scatter(Y[:, 0], Y[:, 1])
    ax = plt.gca()
    ax.set_aspect(1)
    # plt.title(str(index))
    plt.show()


def mnist_combination():
    """
    MNIST 数据组合查看降维效果
    :return:
    """
    path = "E:\\Project\\result2019\\result1026without_straighten\\datasets\\"
    save_path = "E:\\Project\\DataLab\\MNIST50m\\combination3\\normalize\\"
    data_path = "E:\\Project\\DataLab\\MNIST50m\\combination3\\data\\"
    # digits_count = [863, 985, 874, 893, 853, 790, 860, 912, 854, 870]
    # digits_count = [461, 526, 466, 477, 455, 421, 459, 487, 455, 464]
    digits_count = [231, 263, 233, 239, 228, 211, 230, 244, 228, 232]

    for i in range(0, 8):
        X1 = np.loadtxt(path+"MNIST50mclass"+str(i)+"_"+str(digits_count[i])+"\\data.csv", dtype=np.float, delimiter=",")
        X1_origin = np.loadtxt(path+"MNIST50mclass"+str(i)+"_"+str(digits_count[i])+"\\origin.csv", dtype=np.float, delimiter=",")
        (n1, m1) = X1.shape
        for j in range(i+1, 9):
            X2 = np.loadtxt(path + "MNIST50mclass" + str(j) + "_" + str(digits_count[j]) + "\\data.csv", dtype=np.float, delimiter=",")
            X2_origin = np.loadtxt(path + "MNIST50mclass" + str(j) + "_" + str(digits_count[j]) + "\\origin.csv", dtype=np.float, delimiter=",")
            (n2, m2) = X2.shape
            for k in range(j+1, 10):
                X3 = np.loadtxt(path + "MNIST50mclass" + str(k) + "_" + str(digits_count[k]) + "\\data.csv", dtype=np.float, delimiter=",")
                X3_origin = np.loadtxt(path + "MNIST50mclass" + str(k) + "_" + str(digits_count[k]) + "\\origin.csv", dtype=np.float, delimiter=",")
                (n3, m3) = X3.shape
                X = np.zeros((n1+n2+n3, m1))
                X_origin = np.zeros((n1+n2+n3, 784))
                X[0:n1, :] = X1[:, :]
                X[n1:n1+n2, :] = X2[:, :]
                X[n1+n2:n1+n2+n3, :] = X3[:, :]
                X_origin[0:n1, :] = X1_origin[:, :]
                X_origin[n1:n1 + n2, :] = X2_origin[:, :]
                X_origin[n1 + n2:n1 + n2 + n3, :] = X3_origin[:, :]
                # label = np.zeros((n1+n2+n3, 1))
                # label[0:n1] = i
                # label[n1:n1+n2] = j
                # label[n1+n2:n1+n2+n3] = k
                # for i in range
                label = []
                for index in range(0, n1):
                    label.append(i)
                for index in range(0, n2):
                    label.append(j)
                for index in range(0, n3):
                    label.append(k)

                temp_path = data_path+"mnist50mminiclass"+str(i)+str(j)+str(k)+"\\"
                if not os.path.exists(temp_path):
                    os.makedirs(temp_path)
                np.savetxt(temp_path+"data.csv", X, fmt="%f", delimiter=",")
                np.savetxt(temp_path+"label.csv", np.array(label).T, fmt='%d', delimiter=",")
                np.savetxt(temp_path+"origin.csv", X_origin, fmt='%d', delimiter=",")

                X = Preprocess.normalize(X, -1, 1)

                Y = dim_reduce(X, method="PCA")
                plt.scatter(Y[:, 0], Y[:, 1], c=label)
                plt.colorbar()
                ax = plt.gca()
                ax.set_aspect(1)
                plt.savefig(save_path+str(i)+str(j)+str(k)+".png")
                plt.close()

                print(i, j, k)


def mnist_50m_small():
    """
    对每个类大约有500个点的数据进行处理，预览降维效果
    :return:
    """
    digit_count = [461, 526, 466, 477, 455, 421, 459, 487, 455, 464]
    path = "E:\\Project\\DataLab\\MNIST50m\\"
    method = "pca"
    for i in range(0, 10):
        this_path = path + "MNIST50mclass" + str(i) + "_" + str(digit_count[i]) + "\\"
        X = np.loadtxt(this_path+"data.csv", dtype=np.float, delimiter=",")
        X = Preprocess.normalize(X, -1, 1)
        Y = dim_reduce(X, method=method)
        np.savetxt(this_path+method+".csv", Y, fmt="%f", delimiter=",")
        plt.scatter(Y[:, 0], Y[:, 1])
        plt.title(str(i)+"-"+method)
        ax = plt.gca()
        ax.set_aspect(1)
        plt.savefig(this_path+str(i)+"-"+method+".png")
        plt.close()
        print(i)


if __name__ == '__main__':
    # run_test()
    mnist_combination()
    # mnist_50m_small()

