# 计算椭圆
import numpy as np
import matplotlib.pyplot as plt


def eclipse(a, b, alpha=0.0, x0=0.0, y0=0.0, n_points=100):
    """
    计算一个椭圆
    :param a: 轴长（与x有关的轴）
    :param b: 轴长（与y有关的轴）
    :param alpha: 轴沿逆时针方向的偏移角度
    :param x0: 中心的横坐标
    :param y0: 中心的纵坐标
    :param n_points: 点数
    :return:
    """
    points = np.zeros((n_points, 2))
    for i in range(0, n_points):
        t = i * 2 * np.pi / n_points
        x1 = a * np.cos(t)
        y1 = b * np.sin(t)

        points[i, 0] = x1 * np.cos(alpha) - y1 * np.sin(alpha) + x0
        points[i, 1] = x1 * np.sin(alpha) + y1 * np.cos(alpha) + y0

    return points


def run():
    a = eclipse(4, 2, alpha=np.pi/8, x0=2, y0=2)
    plt.plot(a[:, 0], a[:, 1])
    ax = plt.gca()
    ax.set_aspect(1)
    plt.show()


if __name__ == '__main__':
    run()
