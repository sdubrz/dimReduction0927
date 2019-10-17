# 对一些数据进行清洗，以便于使用
import numpy as np
import matplotlib.pyplot as plt


def computer_clean():
    """
    清洗 Computer Hardware Data set
    https://archive.ics.uci.edu/ml/datasets/Computer+Hardware
    这类数据的清洗特点是删除其中的几个列
    :return:
    """
    path = "E:\\Project\\result2019\\result0927\\datasets\\ComputerHardware\\"
    in_file = open(path+"machine.data")
    line = in_file.readline()
    while line:
        print(line)
        line = in_file.readline()
    in_file.close()


if __name__ == '__main__':
    computer_clean()
