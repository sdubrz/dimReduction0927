import numpy as np
import matplotlib.pyplot as plt
from Main import Preprocess


def parallel_coordinate(data, label):
    """
    绘制平行坐标系，假设数据已经normalize到(0, 1)
    :param data:
    :param label:
    :return:
    """

    (n, m) = data.shape
    colors = ['r', 'g', 'b', 'orange', 'm', 'k', 'c', 'yellow']

    for i in range(0, n):
        c = colors[label[i] % len(colors)]
        for j in range(0, m-1):
            plt.plot([j, j+1], [data[i, j], data[i, j+1]], c=c, alpha=0.4, linewidth=0.7)

    for i in range(0, m):
        plt.plot([i, i], [0, 1], c='k')

    plt.show()


if __name__ == '__main__':
    path = "E:\\Project\\result2019\\result1026without_straighten\\datasets\\Wine\\"
    data = np.loadtxt(path+"data.csv", dtype=np.float, delimiter=",")
    data = Preprocess.normalize(data, 0, 1)
    label = np.loadtxt(path+"label.csv", dtype=np.int, delimiter=",")

    parallel_coordinate(data[60:131, :], label[60:131])
