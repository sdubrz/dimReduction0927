# 生成带有每个点的error的json文件
import json
import numpy as np


def create_json(path=""):
    """
    生成json文件
    :param path: 文件的存放目录
    :return:
    """
    error = np.loadtxt(path+"error.csv", dtype=np.float, delimiter=",")
    (n, ) = error.shape

    old_file = open(path + "stress.json", encoding='utf-8')
    old_data = json.load(old_file)

    error_file = open(path+"error.json", 'w', encoding='utf-8')
    error_file.write("[")
    index = 0
    for item in old_data:
        item["error"] = error[index]
        error_file.write(str(item).replace('\'', '\"'))
        if index < n-1:
            error_file.write(",\n")
        index += 1
    error_file.write("]")
    error_file.close()

    print("已成功生成带有 error 的json文件")


if __name__ == '__main__':
    path = "E:\\文件\\IRC\\特征向量散点图项目\\result2020\\result0119\\cTSNE\\Iris3\\yita(0.08333)nbrs_k(30)method_k(90)numbers(3)_b-spline_weighted\\"
    create_json(path)