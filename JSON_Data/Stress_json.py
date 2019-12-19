# 带有MDS的stress信息的json文件
# 临时独立，后面要合并到默认的json生成程序中去
# 2019.11.19
import numpy as np
import json
import os
from Main import processData


def create_json(path=""):
    """
    生成json文件
    :param path: 文件的存放目录
    :return:
    """
    dD = processData.mds_stress(path)
    (n, n2) = dD.shape

    old_file = open(path + "temp_total.json", encoding='utf-8')
    old_data = json.load(old_file)

    stress_file = open(path+"stress.json", 'w', encoding='utf-8')
    stress_file.write("[")
    index = 0
    for item in old_data:
        item["stress"] = dD[index, :].tolist()
        stress_file.write(str(item).replace('\'', '\"'))
        if index < n-1:
            stress_file.write(",\n")
        index += 1
    stress_file.write("]")
    stress_file.close()

    print("已成功生成带有 MDS stress 的json文件")


if __name__ == '__main__':
    path = "E:\\Project\\result2019\\result1026without_straighten\\MDS\\Wine\\yita(0.1)nbrs_k(45)method_k(20)numbers(4)_b-spline_weighted\\"
    create_json(path)
