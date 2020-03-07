# 投影local LDA的主程序
import numpy as np
import matplotlib.pyplot as plt
from Main import Preprocess
from Perturb2020 import MDS_Perturb
from Perturb2020 import TSNE_Perturb
from Main import Preturb


def main_run(main_path, data_name, nbrs_k=30, yita=0.1, method_k=30, max_eigen_numbers=5, method="MDS",
        draw_kind="line", has_line=False, hasLabel=True, to_normalize=False, do_straight=False,
        weighted=True, P_matrix=None, show_result=False, min_proportion=0.9, min_good_points=0.9, y_precomputed=False):
    """

    :param main_path:
    :param data_name:
    :param nbrs_k:
    :param yita:
    :param method_k:
    :param max_eigen_numbers:
    :param method:
    :param draw_kind:
    :param has_line:
    :param hasLabel:
    :param to_normalize:
    :param do_straight:
    :param weighted:
    :param P_matrix:
    :param show_result:
    :param min_proportion:
    :param min_good_points:
    :param y_precomputed:
    :return:
    """
    data_path = main_path + "datasets\\" + data_name + "\\data.csv"
    label_path = main_path + "datasets\\" + data_name + "\\label.csv"
    y_random_path = main_path + "datasets\\" + data_name + "\\y_random.csv"

    data = np.loadtxt(data_path, dtype=np.float, delimiter=",")
    data_shape = data.shape
    n = data_shape[0]
    dim = data_shape[1]
    print(data_shape)

    label = np.zeros((n, 1))
    if hasLabel:
        label = np.loadtxt(label_path, dtype=np.int, delimiter=",")

    save_path = main_path + method + "\\" + data_name + "\\yita(" + str(yita) + ")nbrs_k(" + str(nbrs_k)
    save_path = save_path + ")method_k(" + str(method_k) + ")numbers(" + str(max_eigen_numbers) + ")"
    save_path = save_path + "_" + draw_kind
    if weighted:
        save_path = save_path + "_weighted"
    else:
        save_path = save_path + "_withoutweight"

    save_path = save_path + "\\"

    Preprocess.check_filepath(save_path)
    print(save_path)

    if to_normalize:
        print('进行normalize')
        x = Preprocess.normalize(data)
    else:
        print('不进行normalize')
        x = data
    np.savetxt(save_path + "x.csv", x, fmt="%f", delimiter=",")
    np.savetxt(save_path + "label.csv", label, fmt="%d", delimiter=",")

    if max_eigen_numbers > dim:
        max_eigen_numbers = dim
        print("所要求的的特征值数目过多")

    if method == "MDS":
        y, y_list_add, y_list_sub = MDS_Perturb.perturb_mds_lda_one_by_one(x, nbrs_k=nbrs_k, method_k=method_k,
                                                                   MAX_EIGEN_COUNT=max_eigen_numbers,
                                                                   method_name=method,
                                                                   yita=yita, save_path=save_path, weighted=weighted,
                                                                   label=label, y_precomputed=y_precomputed)
    elif method == "cTSNE":
        y, y_list_add, y_list_sub = TSNE_Perturb.perturb_tsne_lda_one_by_one(x, nbrs_k=nbrs_k, method_k=method_k,
                                                                           MAX_EIGEN_COUNT=max_eigen_numbers,
                                                                           method_name=method,
                                                                           yita=yita, save_path=save_path,
                                                                           weighted=weighted,
                                                                           label=label, y_precomputed=y_precomputed)
    elif method == "PCA":
        y, y_list_add, y_list_sub = Preturb.perturb_lda_once_weighted(x, nbrs_k=nbrs_k,
                                                                  method_k=method_k,
                                                                  method_name=method,
                                                                  yita=yita,
                                                                  save_path=save_path, weighted=weighted,
                                                                  P_matrix=P_matrix,
                                                                  label=label, MAX_EIGEN_COUNT=max_eigen_numbers)
    else:
        print("暂不支持该方法")
        return

    np.savetxt(save_path + "y.csv", y, fmt="%f", delimiter=",")
    temp_count = 0
    for dy in y_list_add:
        temp_count += 1
        np.savetxt(save_path+"y_add"+str(temp_count)+".csv", dy, fmt='%f', delimiter=",")
    temp_count = 0
    for dy in y_list_sub:
        temp_count += 1
        np.savetxt(save_path+"y_sub"+str(temp_count)+".csv", dy, fmt='%f', delimiter=",")

    # 画图部分
    colors = ['r', 'g', 'b', 'm', 'yellow', 'k', 'c']
    for i in range(0, n):
        plt.scatter(y[i, 0], y[i, 1], c=colors[int(label[i] % len(colors))], alpha=0.8)

    if draw_kind == "line" or has_line:
        for j in range(0, max_eigen_numbers):
            y_add_v = y_list_add[j]
            y_sub_v = y_list_sub[j]

            for i in range(0, n):
                plt.plot([y[i, 0], y_add_v[i, 0]], [y[i, 1], y_add_v[i, 1]], linewidth=1.0, c=colors[label[i] % len(colors)], alpha=0.8)
                plt.plot([y[i, 0], y_sub_v[i, 0]], [y[i, 1], y_sub_v[i, 1]], linewidth=1.0, c=colors[label[i] % len(colors)], alpha=0.8)

    ax = plt.gca()
    ax.set_aspect(1)
    plt.show()


def run_test():
    """
    用于调节参数的地方
    :return:
    """
    main_path = "E:\\文件\\IRC\\特征向量散点图项目\\result2020\\locallda\\"
    data_name = "Iris3"

    method = "PCA"  # "PCA" "MDS" "P_matrix" "Isomap" "LDA" "LTSA" "cTSNE"  "MDS2nd"
    yita = 0.05200306
    nbrs_k = 20
    method_k = 90  # if cTSNE perplexity=method_k/3
    eigen_numbers = 4  # 无用

    draw_kind = "line"
    normalize = True  # 是否进行normalize

    main_run(main_path, data_name, nbrs_k=nbrs_k, yita=yita, method_k=method_k, max_eigen_numbers=eigen_numbers, method=method, draw_kind=draw_kind, to_normalize=normalize)


if __name__ == '__main__':
    run_test()
