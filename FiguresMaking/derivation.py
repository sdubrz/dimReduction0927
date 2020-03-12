# 画导数的形式
# 造一个小一点的数据
import numpy as np
import matplotlib.pyplot as plt


def small_data():
    """
    造一个小的数据
    :return:
    """
    path = "E:\\文件\\IRC\\特征向量散点图项目\\DataLab\\Iris3\\"
    data = np.loadtxt(path+"data.csv", dtype=np.float, delimiter=",")
    label = np.loadtxt(path+"label.csv", dtype=np.int, delimiter=",")
    (n, m) = data.shape

    mini = []
    label_mini = []
    for i in range(0, n):
        if i % 10 == 0:
            mini.append(data[i, :].tolist())
            label_mini.append(label[i])

    np.savetxt(path+"datamini.csv", np.array(mini), fmt="%f", delimiter=",")
    np.savetxt(path+"labelmini.csv", label_mini, fmt='%d', delimiter=",")


def heat_map():
    """
    画导数的热图
    :return:
    """

    """
    hot 从黑平滑过度到红、橙色和黄色的背景色，然后到白色。
    cool 包含青绿色和品红色的阴影色。从青绿色平滑变化到品红色。
    gray 返回线性灰度色图。
    bone 具有较高的蓝色成分的灰度色图。该色图用于对灰度图添加电子的视图。
    white 全白的单色色图。 
    spring 包含品红和黄的阴影颜色。 
    summer 包含绿和黄的阴影颜色。
    autumn 从红色平滑变化到橙色，然后到黄色。 
    winter 包含蓝和绿的阴影色。
    原文链接：https://blog.csdn.net/coder_Gray/article/details/81867639
    """
    path = "E:\\文件\\IRC\\特征向量散点图项目\\做图\\VisFigures\\导数\\Irismini\\"
    P = np.loadtxt(path+"MDS_Pxy.csv", dtype=np.float, delimiter=",")
    plt.imshow(P, cmap=plt.cm.cool)
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    # small_data()
    heat_map()
