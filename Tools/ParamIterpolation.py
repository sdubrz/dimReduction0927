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


def cubic_spline_test():
    x = np.arange(0, 2 * np.pi + np.pi / 4, 2 * np.pi / 8)
    y = np.sin(x)
    tck = interpolate.splrep(x, y, s=0)
    xnew = np.arange(0, 2 * np.pi, np.pi / 50)
    ynew = interpolate.splev(xnew, tck, der=0)

    plt.figure()
    plt.plot(x, y, 'x', xnew, ynew, xnew, np.sin(xnew), x, y, 'b')
    plt.legend(['Linear', 'Cubic Spline', 'True'])
    plt.axis([-0.05, 6.33, -1.05, 1.05])
    plt.title('Cubic-spline interpolation')
    plt.show()


def test3():
    x = [51, 188, 322, 306, 68]
    y = [51, 69, 95, 392, 185]
    for i in range(0, len(y)):
        y[i] = y[i] * -1

    plt.scatter(x, y, marker='o')


if __name__ == '__main__':
    # test2()
    cubic_spline_test()
