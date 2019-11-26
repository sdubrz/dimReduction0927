# 可以把高维数据转化为一个矩阵，进而转化成一幅灰度图片
import numpy as np
import math
import matplotlib.pyplot as plt
from PIL import Image
import os
from Main import Preprocess


def point_image(path=""):
    """
    将高维数据转化为图片
    每一个点转化为一幅二维的灰度图片
    :param X: 高维数据矩阵
    :param save_path: 图片的保存路径
    :return:
    """
    X = np.loadtxt(path+"data.csv", dtype=np.float, delimiter=",")
    picture_path = path + "pictures\\"
    if not os.path.exists(picture_path):
        os.makedirs(picture_path)

    (n, m) = X.shape
    X = Preprocess.normalize(X, 0, 255)
    im_size = int(math.sqrt(m))
    if im_size * im_size < m:
        im_size += 1

    for i in range(0, n):
        im_matrix = np.zeros((im_size, im_size))
        dim_iter = 0
        for row in range(0, im_size):
            if dim_iter >= m:
                break
            for column in range(0, im_size):
                im_matrix[row, column] = X[i, dim_iter]
                dim_iter += 1
                if dim_iter >= m:
                    break
        im_matrix = 255 - im_matrix
        im = Image.fromarray(im_matrix.astype(np.uint8))
        im.save(picture_path+str(i)+".png")
        if (i+1) % 1000 == 0:
            print("making images " + str(i) + " of " + str(n))


def run_test():
    path = "E:\\Project\\result2019\\result1026without_straighten\\datasets\\Wine\\"
    point_image(path)


if __name__ == '__main__':
    run_test()


