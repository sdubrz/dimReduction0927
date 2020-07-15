# 根据前两个特征向量生成椭圆
import numpy as np
import json
from Tools import Eclipse


def ellipse_json(path):
    Y0 = np.loadtxt(path+"y.csv", dtype=np.float, delimiter=",")
    Y1 = np.loadtxt(path+"y1+.csv", dtype=np.float, delimiter=",")
    Y2 = np.loadtxt(path+"y2+.csv", dtype=np.float, delimiter=",")

    old_file = open(path + "removeStress.json", encoding='utf-8')
    old_data = json.load(old_file)
    n = len(old_data)

    error_file = open(path + "ellipse.json", 'w', encoding='utf-8')
    error_file.write("[")
    index = 0
    for item in old_data:
        n_points = 50
        glyph = Eclipse.eclipse_glyph(Y0[index, :], Y1[index, :], Y2[index, :], n_points=n_points)
        item['polygon'] = glyph.tolist()
        item['pointsNum'] = n_points
        error_file.write(str(item).replace('\'', '\"'))
        if index < n - 1:
            error_file.write(",\n")
        index += 1
    error_file.write("]")
    error_file.close()

    print("生成椭圆图形成功")


def test():
    path = "E:\\文件\\IRC\\特征向量散点图项目\\result2020\\result0119_withoutnormalize\\MDS\\IsomapFace\\yita(0.20200219)nbrs_k(65)method_k(90)numbers(4)_b-spline_weighted\\"
    ellipse_json(path)


if __name__ == '__main__':
    test()

