import numpy as np


def run1():
    # n = 5
    # num = np.ones((n, n))
    # num[range(n), range(n)] = 0.
    # print(num)

    Y = np.array([[1, 2],
                  [3, 4],
                  [5, 6],
                  [7, 8]])
    square_Y = np.square(Y)
    print(square_Y)
    sum_Y = np.sum(square_Y, 1)
    print("sum_Y = ", sum_Y)

    num = -2. * np.dot(Y, Y.T)
    print("num = ", num)

    print(np.add(num, sum_Y))

    num = (1. + np.add(np.add(num, sum_Y).T, sum_Y))
    print(num)

    print(np.sum(num))

    print(num / np.sum(num))


if __name__ == '__main__':
    run1()