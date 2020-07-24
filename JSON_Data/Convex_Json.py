# 凸包的画法
import numpy as np
import json


def convex_json(path):
    # 读取凸包数据
    file = open(path + "real_convex_hull_list.csv")
    glyphs = []
    line = file.readline()
    while line:
        # print(line)
        line = line[0:len(line) - 1]
        items = line.split(',')
        glyph = []
        for i in range(0, len(items) // 2):
            point = [float(items[i * 2]), float(items[i * 2 + 1])]
            glyph.append(point)
        # print(glyph)
        glyphs.append(glyph)
        line = file.readline()
    file.close()

    old_file = open(path + "removeStress.json", encoding='utf-8')
    old_data = json.load(old_file)
    n = len(old_data)

    error_file = open(path + "convex_hull.json", 'w', encoding='utf-8')
    error_file.write("[")
    index = 0
    for item in old_data:
        item['polygon'] = glyphs[index]
        item['pointsNum'] = len(glyphs[index])
        error_file.write(str(item).replace('\'', '\"'))
        if index < n - 1:
            error_file.write(",\n")
        index += 1
    error_file.write("]")
    error_file.close()

    print("生成凸包画法成功")


def test():
    path = "E:\\文件\\IRC\\特征向量散点图项目\\result2020\\result0119\\MDS\\Iris3\\yita(0.20200306222)nbrs_k(21)method_k(60)numbers(4)_b-spline_weighted\\"
    # X = np.loadtxt(path+"real_convex_hull_list.csv", dtype=np.float, delimiter=",")
    # for i in range(0, 100):
    #     print(X[i])

    file = open(path+"real_convex_hull_list.csv")
    glyphs = []
    line = file.readline()
    while line:
        # print(line)
        line = line[0:len(line)-1]
        items = line.split(',')
        glyph = []
        for i in range(0, len(items)//2):
            point = [float(items[i*2]), float(items[i*2+1])]
            glyph.append(point)
        print(glyph)
        glyphs.append(glyph)
        line = file.readline()
    file.close()


def run():
    # path = "E:\\文件\\IRC\\特征向量散点图项目\\result2020\\result0119_withoutnormalize\\MDS\\IsomapFace\\yita(0.20200219)nbrs_k(65)method_k(90)numbers(4)_b-spline_weighted\\"
    # path = "E:\\文件\\IRC\\特征向量散点图项目\\result2020\\result0119\\MDS\\Iris3\\yita(0.20200306222)nbrs_k(20)method_k(60)numbers(4)_b-spline_weighted\\"
    # path = "E:\\文件\\IRC\\特征向量散点图项目\\result2020\\result0119\\PCA\\Wine\\yita(0.202003062)nbrs_k(40)method_k(90)numbers(4)_b-spline_weighted\\"
    # path = "E:\\文件\\IRC\\特征向量散点图项目\\result2020\\result0119\\PCA\\seeds\\yita(0.202003062)nbrs_k(25)method_k(90)numbers(4)_b-spline_weighted\\"
    path = "E:\\文件\\IRC\\特征向量散点图项目\\result2020\\result0119\\MDS\\seeds\\yita(0.202003062)nbrs_k(25)method_k(90)numbers(4)_b-spline_weighted\\"
    print(path)
    convex_json(path)


if __name__ == '__main__':
    run()

