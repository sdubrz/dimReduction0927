# 清洗数据
import numpy as np


def wine_quality_red():
    """红酒质量数据"""
    in_path = "E:\\Project\\result2019\\result0927\\datasets\\winequalityred\\"
    data0 = np.loadtxt(in_path+"winequality-red.csv", dtype=np.float, delimiter=";")

    (n, m) = data0.shape
    print((n, m))
    np.savetxt(in_path+"data.csv", data0[:, 0:m-1], fmt='%f', delimiter=',')
    np.savetxt(in_path+"label.csv", data0[:, m-1], fmt='%d', delimiter=',')

    print('数据清洗完毕')


if __name__ == '__main__':
    wine_quality_red()
