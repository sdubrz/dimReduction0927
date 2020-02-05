# 测试所谓的张量的运算
import numpy as np


def run1():
    """
    测试所谓的张量的计算

    :return:
    """
    A = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])
    B = np.ones((3, 3, 3))
    B[0, :, :] = A[:, :]
    print("B = ", B)

    """
    这种情况下，是A分别与B中的元素相乘
    """
    C = np.matmul(A, B)
    print("C = ", C)
    print(B.T)


def run2():
    """

    :return:
    """
    A = np.array([[[1, 2],
                   [3, 4]],
                  [[5, 6],
                   [7, 8]]])
    B = np.array([[1, 2, 3],
                  [4, 5, 6]])
    print("A = ", A)
    print("B = ", B)
    C = np.matmul(A, B)
    print("A*B = ", C)
    np.savetxt("F:\\test.csv", C, fmt='%f', delimiter=",")


def run3():
    a = np.array([[1]
                  [2]])
    b = bp.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])
    c = np.outer(a, b)
    print(c)


def run4():
    a = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])
    b = a[0, :]
    print(b[1])


def run5():
    a = np.ones((2, 2, 2))
    print(a)
    b = np.array([[1, 2], [3, 4]])
    a[:, :, 0] = b[:, :]
    print(a)


if __name__ == '__main__':
    run5()

