# 将不符合要求的数据删除，产生更加“干净”的数据集
import numpy as np
from Main import Preprocess


def clean_by_index(data_name, main_path, remove_index=[]):
    """
    根据索引号删除不好的数据
    :param data_name: 原数据集名称
    :param main_path: 主要读写文件目录
    :param remove_index: 要删除的索引号列表
    :return:
    """
    print('通过噪声索引清洗数据')
    read_path = main_path + "datasets\\" + data_name + "\\"
    save_path = main_path + "datasets\\" + data_name + "Clean" + "\\"
    Preprocess.check_filepath(save_path)

    oriangl_data_reader = np.loadtxt(read_path+"data.csv", dtype=np.str, delimiter=',')
    oriangl_label_reader = np.loadtxt(read_path+"label.csv", dtype=np.str, delimiter=',')
    oriangl_data = oriangl_data_reader[:, :].astype(np.float)
    oriangl_label = oriangl_label_reader.astype(np.int)

    (n, dim) = oriangl_data.shape
    n_remove = len(remove_index)
    print('总共需要删除的数据个数是：\t', n_remove)

    clean_data = np.zeros((n-n_remove, dim))
    clean_label = np.zeros((n-n_remove, 1))

    count = 0
    for i in range(0, n):
        if not (i in remove_index):
            clean_data[count, :] = oriangl_data[i, :]
            clean_label[count] = oriangl_label[i]
            count += 1

    y_random = np.random.random((n-n_remove, 2))

    np.savetxt(save_path+"data.csv", clean_data, fmt='%f', delimiter=',')
    np.savetxt(save_path+"label.csv", clean_label, fmt='%d', delimiter=',')
    np.savetxt(save_path+'y_random.csv', y_random, fmt='%f', delimiter=',')
    print('数据清洗完成')


def clean_small_value(data_name, main_path, last_path, attri_file, threshold):
    """
    删除某个属性值过小的数据
    :param data_name: 原数据集的名字
    :param main_path: 主文件目录
    :param last_path: 上次执行降维存放结果的目录
    :param attri_file: 对应属性的文件名
    :param threshold: 阈值，小于该值的数据要删掉
    :return:
    """

    remove_index = []
    values_reader = np.loadtxt(last_path+attri_file, dtype=np.str, delimiter=',')
    values = values_reader.astype(np.float)

    n = len(values)
    for i in range(0, n):
        if values[i] < threshold:
            remove_index.append(i)

    clean_by_index(data_name, main_path, remove_index)


def clean_big_value(data_name, main_path, last_path, attri_file, threshold):
    """
    删除某个属性值过大的数据
    :param data_name: 原数据集的名字
    :param main_path: 主文件目录
    :param last_path: 上次执行降维存放结果的目录
    :param attri_file: 对应属性的文件名
    :param threshold: 阈值，大于该值的数据要删掉
    :return:
    """

    remove_index = []
    values_reader = np.loadtxt(last_path + attri_file, dtype=np.str, delimiter=',')
    values = values_reader.astype(np.float)

    n = len(values)
    for i in range(0, n):
        if values[i] > threshold:
            remove_index.append(i)

    clean_by_index(data_name, main_path, remove_index)
