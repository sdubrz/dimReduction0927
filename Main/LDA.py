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

    def max_indexs(self, a_list0, num_head=2):
        """获得前几个大的数的索引号"""
        k_list = []
        a_list = []
        for i in a_list0:
            a_list.append(i.real)

        n = len(a_list)
        for i in range(0, n):
            if len(k_list) < num_head:
                k_list.append(i)
                index = len(k_list) - 1
                while index > 0:
                    if a_list[k_list[index]] > a_list[k_list[index - 1]]:
                        temp = k_list[index]
                        k_list[index] = k_list[index - 1]
                        k_list[index - 1] = temp
                        index = index - 1
                    else:
                        break
            else:
                if a_list[k_list[num_head - 1]] < a_list[i]:
                    k_list[num_head - 1] = i
                    index = len(k_list) - 1
                    while index > 0:
                        if a_list[k_list[index]] > a_list[k_list[index - 1]]:
                            temp = k_list[index]
                            k_list[index] = k_list[index - 1]
                            k_list[index - 1] = temp
                            index = index - 1
                        else:
                            break
        return k_list

    def fit_transform(self, X, label, P=None):
        if not (P is None):
            self.Y = np.matmul(X, P)
            return self.Y

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
                index = label_list.index(label[i])
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

        Sb = np.zeros((m, m))  # 类间差距
        Sw = np.zeros((m, m))  # 类内差距

        for i in range(0, no_labels):
            Sb = Sb + len(cluster_list[i]) * np.outer(np.transpose(mean_matrix[i, :]-mean_x), mean_matrix[i, :]-mean_x)
        for i in range(0, n):
            label_index = label_list.index(label[i])
            Sw = Sw + np.outer(np.transpose(X[i, :]-mean_matrix[label_index, :]), X[i, :]-mean_matrix[label_index, :])

        Q = np.dot(LA.inv(Sw), Sb)
        values, vectors = LA.eig(Q)
        selected_index = self.max_indexs(values, num_head=self.n_component)

        P = np.zeros((m, self.n_component))
        for i in range(0, self.n_component):
            P[:, i] = vectors[:, selected_index[i]]

        self.P = P
        self.Y = np.dot(X, P)

        return self.Y


def run_test():
    path = "E:\\Project\\result2019\\result1026without_straighten\\PCA\\Wine\\yita(0.1)nbrs_k(20)method_k(20)numbers(4)_b-spline_weighted\\"
    X = np.loadtxt(path+"x.csv", dtype=np.float, delimiter=",")
    label = np.loadtxt(path+"label.csv", dtype=np.int, delimiter=",")

    (n, m) = X.shape
    lda = LDA(n_component=2)
    Y = lda.fit_transform(X, label)
    P = lda.P
    print(P)
    plt.scatter(Y[:, 0], Y[:, 1], marker='o', c=label)
    plt.show()


if __name__ == '__main__':
    run_test()
