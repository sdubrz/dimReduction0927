# 生成趋势椭圆的 json文件
import numpy as np
import matplotlib.pyplot as plt
import json
from Tools import Eclipse


def trend_eclipse(path='', n_points=100):
    """

    :param path: 文件操作目录
    :param n_points: 每个椭圆的点数
    :return:
    """
    y = np.loadtxt(path+"y.csv", dtype=np.float, delimiter=",")
    eigenvalues = np.loadtxt(path+"【weighted】eigenvalues.csv", dtype=np.float, delimiter=",")
    (n, m) = eigenvalues.shape
    y1 = np.loadtxt(path+"y_add_1.csv", dtype=np.float, delimiter=",")

    eclipse_list = []
    for i in range(0, n):
        a = np.linalg.norm(y1[i, :] - y[i, :])
        b = eigenvalues[i, 1] / eigenvalues[i, 0] * a
        alpha = 0
        if y1[i, 0] == y[i, 0] or a == 0:
            alpha = np.pi / 2
        else:
            # 这个角度算的不对
            alpha = np.arcsin((y1[i, 1] - y[i, 1]) / a)
            if y1[i, 0] < y[i, 0]:
                alpha = np.pi - alpha

        i_eclipse = Eclipse.eclipse(a, b, alpha=alpha, x0=y[i, 0], y0=y[i, 1], n_points=n_points)
        eclipse_list.append(i_eclipse)

    return eclipse_list


def trend_json(path='', draw=False):
    """

    :param path: 存储目录
    :param draw: 是否画出结果，暂时没有实现
    :return:
    """
    eclipse_list = trend_eclipse(path)
    f = open(path + "temp_total.json", encoding='utf-8')
    data = json.load(f)

    index = 0
    for item in data:
        eclipse = eclipse_list[index]
        item['pointsNum'] = eclipse.shape[0]
        item['polygon'] = eclipse.tolist()
        index += 1

    n = index  # 数据的总个数

    out_file = open(path + "trend_oval.json", "w")
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
    print("已成功生成表示高维趋势的json文件")


def run_test():
    for i in range(0, 20):
        angle = np.arccos(i/10-1) / np.pi * 180
        print(angle)


if __name__ == '__main__':
    run_test()
