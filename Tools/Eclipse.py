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


def eclipse_glyph(center, point1, point2, n_points=50):
    """
    以投影较长的特征向量作为长轴，生成椭圆图形
    if |center-point1| > |center-point2|:
        center--point1为长轴
    else:
        center--point2为长轴
    :param center: 椭圆的中心点
    :param point1: 椭圆上的一个点
    :param point2: 椭圆上的另一个点
    :param n_points: 椭圆点的个数
    :return:
    """
    points = np.zeros((n_points, 2))
    axis1 = (center[0]-point1[0])**2 + (center[1]-point1[1])**2
    axis2 = (center[0]-point2[0])**2 + (center[1]-point2[1])**2

    # 确定长轴
    if axis1 > axis2:
        point_a = point1
        point_b = point2
        a = np.sqrt(axis1)
        a2 = np.sqrt(axis2)
    else:
        point_a = point2
        point_b = point1
        a = np.sqrt(axis2)
        a2 = np.sqrt(axis1)

    # 计算长轴倾斜角度
    sita = np.pi/2
    if point_a[0] != center[0]:  # 似乎应该考虑计量误差
        sita = np.arctan((point_a[1]-center[1])/(point_a[0]-center[0]))
    # print("sita = ", sita/np.pi)

    # 计算 t 的值
    if point_b[1] < center[1]:
        point_b[1] = 2*center[1] - point_b[1]
        point_b[0] = 2*center[0] - point_b[0]
    t = np.pi/2
    if point_b[0] != center[0]:  # 似乎应该考虑计量误差
        t = np.arccos((point_b[0] - center[0]) / a2)
        # t = np.arccos((point_b[0] - center[0]) / np.sqrt(np.min(axis1, axis2)))
    t = t - sita
    # print("t = "+str(t/np.pi)+ " * Pi")  # t应该没错

    # 计算 b 的值
    # if np.sin(sita) == 0:
    #     x1 = point_b[0] - center[0]
    #     y1 = point_b[1] - center[1]
    # elif np.cos(sita) == 0:
    #     x1 = point_b[1] - center[1]
    #     y1 = center[0] - point_b[0]
    # else:
    #     # y1 = (point_b[1]-center[1]-(point_b[0]-center[0])/np.cos(sita)) / (np.tan(sita)+np.cos(sita))
    #     y1 = (point_b[1]-center[1])*np.cos(sita) - (point_b[0]-center[0])*np.sin(sita)
    #     x1 = (point_b[0]-center[0]+y1*np.sin(sita)) / np.cos(sita)
    # b = y1 / np.sin(t)
    #
    # print(point_a)
    # print(point_b)
    # print("a = ", a)
    # print("b = ", b)

    x = point_b[0] - center[0]
    y = point_b[1] - center[1]
    x2 = x*np.cos(sita) + y*np.sin(sita)
    y2 = -1*x*np.sin(sita) + y*np.cos(sita)
    b = np.sqrt(y2*y2 / (1-x2*x2/(a*a)))
    # print("b = ", b)

    for i in range(0, n_points):
        angle = np.pi * 2 / n_points * i
        x1 = a * np.cos(angle)
        y1 = b * np.sin(angle)
        points[i, 0] = x1 * np.cos(sita) - y1 * np.sin(sita) + center[0]
        points[i, 1] = x1 * np.sin(sita) + y1 * np.cos(sita) + center[1]

    return points


def run():
    # a = eclipse(4, 2, alpha=-5*np.pi/8, x0=2, y0=2)
    # plt.plot(a[:, 0], a[:, 1])
    # ax = plt.gca()
    # ax.set_aspect(1)
    # plt.show()
    center = [1.0, 3.0]
    points1 = [3.0 + center[0], 0.0 + center[1]]
    points2 = [0.0 + center[0], -2.5 + center[1]]
    a = eclipse_glyph(center, points1, points2, n_points=500)
    plt.plot(a[:, 0], a[:, 1])

    plt.scatter(center[0], center[1], c='r')
    plt.scatter(points1[0], points1[1], c='g')
    plt.scatter(points2[0], points2[1], c='b')

    plt.plot([center[0], points1[0]], [center[1], points1[1]], c='g')
    plt.plot([center[0], points2[0]], [center[1], points2[1]], c='b')
    plt.plot([center[0], center[0]-(points1[0]-center[0])], [center[1], 2*center[1]-points1[1]], c='g')
    plt.plot([center[0], 2*center[0]-points2[0]], [center[1], 2*center[1]-points2[1]], c='b')

    ax = plt.gca()
    ax.set_aspect(1)
    plt.show()


if __name__ == '__main__':
    run()
