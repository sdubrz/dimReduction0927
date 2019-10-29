# 参数插值尝试
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate


def test():
    x = [4.913, 4.913, 4.918, 4.938, 4.955, 4.949, 4.911, 4.848]
    y = [5.2785, 5.2875, 5.291, 5.289, 5.28, 5.26, 5.245, 5.245]

    plt.scatter(x, y, marker='o')

    for s in (0, 1e-4):
        print(s)
        tck, t = interpolate.splprep([x, y], s=s)
        xi, yi = interpolate.splev(np.linspace(t[0], t[-1], 200), tck)
        plt.plot(xi, yi)

    plt.show()


def test2():
    x = [51, 188, 322, 306, 68, 51, 188]
    y = [51, 69, 95, 392, 185, 51, 69]
    for i in range(0, len(y)):
        y[i] = y[i] * -1

    plt.scatter(x, y, marker='o')

    # 计算曲线
    for s in range(0, 5):
        tck, t = interpolate.splprep([x, y], s=1)
        xi, yi = interpolate.splev(np.linspace(t[0], t[-1], 200), tck)
        plt.plot(xi, yi)

    ax = plt.gca()
    ax.set_aspect(1)
    plt.show()


if __name__ == '__main__':
    test2()
