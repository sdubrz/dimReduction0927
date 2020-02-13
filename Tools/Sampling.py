# 采样，某些数据样本过多
import numpy as np


def sample():
    in_path = "C:\\Users\\Hayim\\Desktop\\testrun\\datasets\\MNIST_568\\"
    out_path = "C:\\Users\\Hayim\\Desktop\\testrun\\datasets\\MNIST_568\\"

    data0 = np.loadtxt(in_path+"new.csv", dtype=np.float, delimiter=',')
    label0 = np.loadtxt(in_path+"label.csv", dtype=np.int, delimiter=',')
    (n, m) = data0.shape

    data = []
    label = []

    for i in range(0, n):
        if i % 10 == 0:
            data.append(data0[i, :])
            label.append([label0[i]])

    np.savetxt(out_path+"new2.csv", np.array(data), fmt='%f', delimiter=',')
    np.savetxt(out_path+"label.csv", np.array(label), fmt='%d', delimiter=',')

    print('采样完毕')


if __name__ == '__main__':
    sample()

