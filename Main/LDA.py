# LDA 降维方法
import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt


class LDA:
    # 当然，正宗的 LDA 其实不是这样做的
    """
    假设数据是已经经过必要的预处理的，比如 normalize……
    """
    def __init__(self, n_component=2):
        self.n_component = n_component  # 降维后的数据维数
        # self.X = None  # 高维数据矩阵
        self.P = None  # 投影矩阵
        self.Y = None

    def fit_transform(self, X, label):
        (n, m) = X.shape
        mean_x = np.mean(X, axis=0)

        # 对数据按照类别分类
        cluster_list = []
        label_list = []
        for i in range(0, n):
            if not (label[i] in label_list):
                a_list = []
                a_list.append(X[i, :])
                cluster_list.append(a_list)
                label_list.append(label[i])
            else:
                index = label_list.append(label[i])
                a_list = cluster_list[index]
                a_list.append(X[i, :])

        no_labels = label_list.__len__()  # 类的数量
        array_list = []
        mean_matrix = np.zeros((no_labels, m))

        temp_index = 0
        for cluster in cluster_list:
            this_cluster = np.array(cluster)
            array_list.append(this_cluster)
            mean_matrix[temp_index, :] = np.mean(this_cluster, axis=0)
            temp_index += 1

        Sb = np.zeros((dim, dim))  # 类间差距
        Sw = np.zeros((dim, dim))  # 类内差距

        for i in range(0, no_labels):
            Sb = Sb + len(this_cluster[i])
