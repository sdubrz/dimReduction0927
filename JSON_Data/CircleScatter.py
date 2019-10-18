# 现有系统中的散点图工具不太好用，有些时候点与点之间的区分度不够强
# 根据点的坐标生成全是同样大小的圆圈
import numpy as np
import matplotlib.pyplot as plt
from Tools import Circle

import json
import os


def circle_json(path, r=0.1, n_points=100):
    """
    将数据的图形变成完全相同的圆形，生成的数据作为散点图使用
    :param path: 进行操作的文件目录
    :param r: 圆的半径
    :param n_points: 点数
    :return:
    """
    f = open(path+"temp_total.json", encoding='utf-8')
    data = json.load(f)
    n = len(data)

    for item in data:
        c = Circle.circle(item['x'], item['y'], r, n_points=n_points)
        item['pointsNum'] = n_points
        item['polygon'] = c.tolist()

    out_file = open(path+"scatter.json", "w")
    out_file.write('[')
    index = 0
    for item in data:
        line1 = str(item)
        line2 = ''
        for a_c in line1:
            if a_c == '\'':
                line2 = line2 + '\"'
            else:
                line2 = line2 + a_c
        out_file.write(line2)
        index += 1
        if index == n:
            out_file.write(']')
        else:
            out_file.write(',\n')
    out_file.close()


def test2():
    s = "abandon"
    print(s)
    s2 = ''
    for i in range(0, len(s)):
        if s[i] == 'a':
            s2 = s2 + 'b'
        else:
            s2 = s2 + s[i]
    print(s2)


if __name__ == '__main__':
    circle_json()
