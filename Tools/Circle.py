import numpy as np
import matplotlib.pyplot as plt


def circle(x, y, r, n_points=100):
    """

    :param x:
    :param y:
    :param r:
    :param n_points:
    :return:
    """
    c = np.zeros((n_points, 2))
    for i in range(0, n_points):
        sita = i * 2 * np.pi / n_points
        c[i, 0] = np.cos(sita) * r + x
        c[i, 1] = np.sin(sita) * r + y

    return c


def draw(c):
    """画"""
    (n, m) = c.shape
    plt.plot(c[:, 0], c[:, 1])
    plt.show()


if __name__ == '__main__':
    c = circle(1, 1, 1)
    draw(c)


