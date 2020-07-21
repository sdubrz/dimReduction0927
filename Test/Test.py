import numpy as np


def test():
    n = 7494*16
    m = 7494*2
    A = np.zeros((n, m))
    count = 0
    for i in range(0, n):
        if i % 1000 == 0:
            print(i)
        for j in range(0, m):
            A[i, j] = count / 3.1
            count = count + 1

    print("A finished")
    B = np.zeros((m, m))
    count = 0
    for i in range(0, m):
        for j in range(0, m):
            B[i, j] = count / 2.3
            count = count + 1

    print("B finished")
    C = np.linalg.pinv(B)
    print("pinv finished")

    D = np.zeros((m, 16))
    count = 0
    for i in range(0, m):
        for j in range(0, 16):
            D[i, j] = count / 1.4
            count = count + 1

    print("D finished")

    np.savetxt("E:\\D.csv", D, fmt='%.18e', delimiter=",")
    print("finished")


if __name__ == '__main__':
    A = np.array([1, 2, 3])
    print(np.inner(A, A))

    B = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])
    print(B[:, 0:2])

