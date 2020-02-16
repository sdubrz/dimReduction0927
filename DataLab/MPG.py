# 对MPG数据进行处理
import numpy as np


def pre_process():
    """

    :return:
    """
    path = "E:\\文件\\IRC\\特征向量散点图项目\\DataLab\\MPG\\"
    data = np.loadtxt(path+"auto-mpg2.data", dtype=np.str, delimiter="\t")
    print(data.shape)
    print(data[0, 0])
    np.savetxt(path+"number.data", data[:, 0], fmt='%s')



if __name__ == '__main__':
    pre_process()

