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
    path = "E:\\Project\\result2019\\result0927\\datasets\\ecoli\\"
    in_file = open(path+"ecoli.data")

    out_file = open(path+"data.csv", 'w')
    line = in_file.readline()
    n = 0
    while line:
        n += 1
        items = line.split('  ')
        write_line = items[1]
        for i in range(3, len(items)-1):
            write_line = write_line + "," + items[i]
        write_line = write_line + "\n"
        out_file.writelines(write_line)

        line = in_file.readline()
    out_file.close()
    in_file.close()

    temp_label = np.zeros((n, 1))
    np.savetxt(path+"label.csv", temp_label, fmt='%d', delimiter=',')

    print('清洗数据完成')


if __name__ == '__main__':
    computer_clean()
