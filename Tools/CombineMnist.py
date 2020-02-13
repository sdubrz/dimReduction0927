# 采样，某些数据样本过多
import numpy as np
import os
from Tools import PCAClean

def sample():
    nbrs = ['3','5','7']
    in_path = "D:\\Exp\\datasets\\coil-20\\"
    out_path = "D:\\Exp\\datasets\\coil-20"

    data0 = np.loadtxt(in_path+"data.csv", dtype=np.int, delimiter=',')
    label0 = np.loadtxt(in_path+"label.csv", dtype=np.str, delimiter=',')
    (n, m) = data0.shape

    data = []
    label = []

    for i in range(0, n):
        if label0[i][3:] in nbrs:
            data.append(data0[i, :])
            label.append([label0[i][3:]])

    path = out_path+'-'+'-'.join(nbrs)+"\\"
    if not os.path.exists(path):
        os.makedirs(path)
    np.savetxt(path+"data.csv", np.array(data), fmt='%d', delimiter=',')
    np.savetxt(path+"label.csv", np.array(label), fmt='%s', delimiter=',')
    #PCAClean.main(path)

    print('采样完毕')


if __name__ == '__main__':
    sample()

