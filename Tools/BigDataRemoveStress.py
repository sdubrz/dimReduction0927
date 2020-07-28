# 对过大的数据集删掉stress属性，以防系统无法加载
import numpy as np
import json


def remove_stress(path):
    """
    用空代替stress
    :param path:
    :return:
    """

    old_file = open(path + "error.json", encoding='utf-8')
    old_data = json.load(old_file)
    n = len(old_data)

    error_file = open(path + "removeStress.json", 'w', encoding='utf-8')
    error_file.write("[")
    index = 0
    for item in old_data:
        item["stress"] = []
        item["test_attr"] = item["test_attr"][0]
        error_file.write(str(item).replace('\'', '\"'))
        if index < n - 1:
            error_file.write(",\n")
        index += 1
    error_file.write("]")
    error_file.close()

    print("finished 删除掉stress属性")


def test():
    # path = "E:\\文件\\IRC\\特征向量散点图项目\\result2020\\result0119_withoutnormalize\\MDS\\IsomapFace\\yita(0.20200219)nbrs_k(65)method_k(90)numbers(4)_b-spline_weighted\\"
    # path = "E:\\文件\\IRC\\特征向量散点图项目\\result2020\\result0119\\PCA\\Iris3\\yita(0.20200303)nbrs_k(20)method_k(90)numbers(4)_b-spline_weighted\\"
    # path = "E:\\文件\\IRC\\特征向量散点图项目\\result2020\\result0119\\MDS\\Iris3\\yita(0.0715)nbrs_k(20)method_k(70)numbers(3)_b-spline_weighted\\"
    # path = "E:\\文件\\IRC\\特征向量散点图项目\\result2020\\result0119_withoutnormalize\\PCA\\IsomapFace\\yita(0.20200219)nbrs_k(65)method_k(90)numbers(4)_b-spline_weighted\\"
    path = "E:\\文件\\IRC\\特征向量散点图项目\\result2020\\result0119_withoutnormalize\\cTSNE\\IsomapFace\\yita(0.20200225)nbrs_k(65)method_k(90)numbers(4)_b-spline_weighted\\"
    remove_stress(path)


if __name__ == '__main__':
    test()
