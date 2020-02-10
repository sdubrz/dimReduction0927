# 采样，某些数据样本过多
import numpy as np
import os

def sample():
    nbrs = [0,2,4,5]
    in_path = "C:\\Users\\Hayim\\Desktop\\testrun\\datasets\\MNIST\\"
    out_path = "C:\\Users\\Hayim\\Desktop\\testrun\\datasets\\MNIST"

    data0 = np.loadtxt(in_path+"data.csv", dtype=np.float, delimiter=',')
    label0 = np.loadtxt(in_path+"label.csv", dtype=np.int, delimiter=',')
    (n, m) = data0.shape

    data = []
    label = []

    for i in range(0, n):
        if i % 20 == 0 and (label0[i] in nbrs):
            data.append(data0[i, :])
            label.append([label0[i]])
    path = out_path+"0245\\"
    if not os.path.exists(path):
        os.makedirs(path)
    np.savetxt(out_path+"0245\\"+"data.csv", np.array(data), fmt='%f', delimiter=',')
    np.savetxt(out_path+"0245\\"+"label.csv", np.array(label), fmt='%d', delimiter=',')

    print('采样完毕')


if __name__ == '__main__':
    sample()

