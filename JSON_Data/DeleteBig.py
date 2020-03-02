# 删除图形过大的点
import numpy as np
import json
import os


def delete_big_length_width(path="", rate1=0.01, rate2=0.01):
    """
    删除特别大的点，根据长和宽来判断
    :param path: 文件存储的路径
    :param rate1: 删除最长的点的比例
    :param rate2: 删除最短的点的比例
    :return:
    """
    f = open(path + "error.json", encoding='utf-8')
    data = json.load(f)
    n = len(data)


def delete_big_area(path="", rate=0.02):
    """
    删除特别大的点，根据面积进行判断
    :param path: 存储文件的目录
    :param rate: 要删除的点的比例
    :return:
    """
    f = open(path + "error.json", encoding='utf-8')
    data = json.load(f)
    n = len(data)

    size_list = []
    for item in data:
        i_size = item['polygonSize']
        size_list.append(i_size)

    size_list.sort()
    m = int(n*(1-rate))
    grade = size_list[m]

    data2 = []
    for item in data:
        if item['polygonSize'] < grade:
            data2.append(item)

    out_file = open(path + "delete_big_area.json", "w")
    out_file.write('[')
    index = 0
    for item in data2:
        line1 = str(item)
        line2 = ''
        for a_c in line1:
            if a_c == '\'':
                line2 = line2 + '\"'
            else:
                line2 = line2 + a_c
        out_file.write(line2)
        index += 1
        if index == len(data2):
            out_file.write(']')
        else:
            out_file.write(',\n')
    out_file.close()
    print("删除面积过大的点的个数为", n-m)


def test():
    # path = "E:\\文件\\IRC\\特征向量散点图项目\\result2020\\result0119_withoutnormalize\\cTSNE\\IsomapFace\\yita(0.20200225)nbrs_k(65)method_k(90)numbers(4)_b-spline_weighted\\"
    path = "E:\\文件\\IRC\\特征向量散点图项目\\result2020\\result0119_withoutnormalize\\cTSNE\\fashion50mclass568\\yita(50.202002172)nbrs_k(51)method_k(90)numbers(4)_b-spline_weighted\\"
    delete_big_area(path, rate=0.03)


if __name__ == '__main__':
    test()


