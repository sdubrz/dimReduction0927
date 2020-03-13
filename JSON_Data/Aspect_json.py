# 用图形的长宽比代替投影之后的linearity
import numpy as np
import json
from JSON_Data import polygon_json190927


def linearity_aspect(path):
    old_file = open(path + "error.json", encoding='utf-8')
    old_data = json.load(old_file)
    n = len(old_data)

    aspect = np.loadtxt(path+"spline_radios.csv", dtype=np.float, delimiter=",")
    linearity = np.loadtxt(path+"【weighted】eigen1_div_eigen2_original.csv", dtype=np.float, delimiter=",")
    linearityChange = polygon_json190927.linearity_change(aspect, linearity)

    error_file = open(path + "splineAspect.json", 'w', encoding='utf-8')
    error_file.write("[")
    index = 0
    for item in old_data:
        item["lineaProject"] = aspect[index]
        item["linearChange"] = linearityChange[index, 0]
        error_file.write(str(item).replace('\'', '\"'))
        if index < n - 1:
            error_file.write(",\n")
        index += 1
    error_file.write("]")
    error_file.close()

    # print("finished 删除掉stress属性")


if __name__ == '__main__':
    path = "E:\\文件\\IRC\\特征向量散点图项目\\result2020\\result0119\\PCA\\Irismini\\yita(0.102003062)nbrs_k(5)method_k(90)numbers(4)_b-spline_weighted\\"
    linearity_aspect(path)
