import numpy as np
import matplotlib.pyplot as plt


"""
    直方图均衡化
    2019/01/30
"""


def hist_equalization(data, bins_num=100):
    """
        一维的直方图均衡化
    :param data: 数据向量，每一个的具体的值
    :param bins_num: 直方图的bins数目，默认是100
    :return:
    """
    data_shape = data.shape
    n = data_shape[0]

    data_count = np.zeros((bins_num, 1))
    data_distribution = np.zeros((bins_num, 1))
    data_label = np.zeros((n, 1))

    min_data = min(data)
    max_data = max(data)
    bins_length = (max_data-min_data)/bins_num

    if bins_length == 0:
        return data_label

    for i in range(0, n):
        data_label[i] = int((data[i]-min_data) / bins_length)
        if data_label[i] == bins_num:
            data_label[i] = bins_num - 1

        data_count[int(data_label[i])] = data_count[int(data_label[i])] + 1

    data_distribution[0] = data_count[0]
    for i in range(1, bins_num):
        data_distribution[i] = data_distribution[i-1] + data_count[i]

    label_equalized = np.zeros((bins_num, 1))
    for i in range(0, bins_num):
        label_equalized[i] = (data_distribution[i]-data_count[0])/(n-data_count[0]) * (bins_num-1)

    data_equalized = np.zeros((n, 1))
    for i in range(0, n):
        data_equalized[i] = label_equalized[int(data_label[i])]

    return data_equalized


if __name__ == "__main__":
    read_path = "F:\\result2019\\result0218\\MDS\\Wine\\yita(0.3)method_k(30)max_k(30)numbers(4)proportion(0.75)adapt_threshold(0.05)_b-spline\\"
    linearity_reader = np.loadtxt(read_path + "linearityChange.csv", dtype=np.str, delimiter=",")
    linearity = linearity_reader.astype(np.float)
    linearity_e = hist_equalization(linearity)

    np.savetxt(read_path + "linearityChangeEqualize.csv", linearity_e, fmt="%f", delimiter=",")

    plt.hist(linearity_e)
    plt.show()



