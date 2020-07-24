# 测试每个点的 Trustworthniess 值作为测试属性值
import numpy as np
import json
from Main import Preprocess
import matplotlib.pyplot as plt


def trust_worth_test(path, k):
    """
    计算每个点的 Trustworth 值
    :param path: 文件存储路径
    :param k: K 近邻的k
    :return:
    """
    X = np.loadtxt(path+"x.csv", dtype=np.float, delimiter=",")
    Y = np.loadtxt(path+"y.csv", dtype=np.float, delimiter=",")
    (n, m) = X.shape

    Knn_x = Preprocess.knn(X, k)
    Knn_y = Preprocess.knn(Y, k)

    trust = np.zeros((n, 1))
    for i in range(0, n):
        rank = 0
        for j in range(0, k):
            if not (Knn_y[i, j] in Knn_x[i, :]):
                rank = rank + j - k
                trust[i] = rank
    trust = np.ones((n, 1)) - trust * 2 / (k*(2*m-2*k-1))
    np.savetxt(path+"trust.csv", trust, fmt='%f', delimiter=",")
    return trust


def Trusworth_test_json(path, k):
    """
    生成 test attr为 Trustworthniess 的json文件
    :param path: 文件存储路径
    :param k: 计算Trustworthniess所用的K近邻数
    :return:
    """
    old_file = open(path + "removeStress.json", encoding='utf-8')
    old_data = json.load(old_file)
    n = len(old_data)
    trust = trust_worth_test(path, k)

    error_file = open(path + "Trustworthniess"+str(k)+".json", 'w', encoding='utf-8')
    error_file.write("[")
    index = 0
    for item in old_data:
        item['test_attr'] = trust[index, 0]
        error_file.write(str(item).replace('\'', '\"'))
        if index < n - 1:
            error_file.write(",\n")
        index += 1
    error_file.write("]")
    error_file.close()

    print("Trustworthniess作为测试属性的JSon文件成功")


def test():
    # path = "E:\\文件\\IRC\\特征向量散点图项目\\result2020\\result0119_withoutnormalize\\MDS\\IsomapFace\\yita(0.20200219)nbrs_k(65)method_k(90)numbers(4)_b-spline_weighted\\"
    # path = "E:\\文件\\IRC\\特征向量散点图项目\\result2020\\result0119\\MDS\\Iris3\\yita(0.20200306222)nbrs_k(20)method_k(60)numbers(4)_b-spline_weighted\\"
    path = "E:\\文件\\IRC\\特征向量散点图项目\\result2020\\result0119\\MDS\\Iris3\\yita(0.20200306222)nbrs_k(20)method_k(60)numbers(4)_b-spline_weighted\\"
    # trust_worth_test(path, 20)
    Trusworth_test_json(path, 20)


if __name__ == '__main__':
    test()

