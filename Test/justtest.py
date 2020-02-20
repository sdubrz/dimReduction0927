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


def run2():
    Y = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])
    X = np.zeros((3, 3))
    X[range(3), range(3)] = Y[0, :]
    print(X)
    C = np.eye(4)
    print(C)
    print(np.outer(Y[0, :], Y[0, :]))
    print(np.power(Y[0, 1], 3))
    y = Y[0, :]*Y[1, :]
    print(y)
    yy = np.dot(Y[0, :], Y[2, :])
    print(yy)


def test3():
    if 0.1+0.1 == 0.2:
        print("one")
    if 0.1+0.2 == 0.3:
        print("two")


def test4():
    n = 10
    PQ = np.random.random((n, n))
    E = np.random.random((n, n))
    a = 1
    S3_0 = (-1)*np.tile(PQ[a, :] * E[a, :], (4, 1)).T  # 利用了P和Q是对称矩阵的性质
    print(S3_0)
    S3_0[:, 1:3] = 0
    S3 = (S3_0.reshape((2*n, 2))).T
    print(S3)


def test5():
    a = np.random.random((5, 5))
    print("a = ", a)
    b = np.tile(a[1, :], (2, 1)).T
    print("b = ", b)
    c = np.random.random((5, 2))
    print("c = ", c)
    print("b*c = ", b*c)


def test6():
    a = np.log(np.exp(1))
    print(a)


if __name__ == '__main__':
    # run2()
    test6()
