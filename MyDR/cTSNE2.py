# 有显式 KNN 计算的经典 t-SNE算法实现

import numpy as np
import matplotlib.pyplot as plt
from Main import Preprocess
from Main import DimReduce
from MyDR import Probabilities_NN


temp_path = "E:\\Project\\result2019\\result1026without_straighten\\datasets\\coil20obj_16_3class\\"


class cTSNE:
    def __init__(self, n_component=2, perplexity=30.0):
        """
        init function
        :param n_component: 降维后的维度数
        :param perplexity: 困惑度
        """
        self.n_component = n_component
        self.perplexity = perplexity

    def Hbeta(self, D=np.array([]), beta=1.0):
        """
            Compute the perplexity and the P-row for a specific value of the
            precision of a Gaussian distribution.
        """

        # Compute P-row and corresponding perplexity
        P = np.exp(-D.copy() * beta)
        sum_p = sum(P)
        H = np.log(sum_p) + beta * np.sum(D * P) / sum_p
        P = P / sum_p
        return H, P

    def x2p2(self, X=np.array([]), tol=1e-5, perplexity=30.0):
        (n, d) = X.shape
        # sum_X = np.sum(np.square(X), 1)
        # D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
        k = min(n - 1, int(3. * self.perplexity + 1))
        distance = np.zeros((n, k))
        knn = Preprocess.knn(X, k)
        for i in range(0, n):
            for j in range(0, k):
                norm = np.linalg.norm(X[i, :]-X[j, :])
                distance[i, j] = norm * norm

        P = Probabilities_NN._joint_probabilities_nn(distance, knn, perplexity, 0)
        # np.savetxt(temp_path+"P.csv", P.A, fmt='%f', delimiter=",")
        # P2 = np.zeros((n, n))
        # for index in P:
        #     print(index[0])

        return P.A

    def x2p(self, X=np.array([]), tol=1e-5, perplexity=30.0):
        """
            Performs a binary search to get P-values in such a way that each
            conditional Gaussian has the same perplexity.
        """

        # Initialize some variables
        # print("\tComputing pairwise distances...")
        (n, d) = X.shape
        sum_X = np.sum(np.square(X), 1)
        D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
        np.savetxt(temp_path+"D.csv", D, fmt='%f', delimiter=",")
        P = np.zeros((n, n))
        beta = np.ones((n, 1))
        logU = np.log(perplexity)

        # Loop over all datapoints
        for i in range(n):

            # Print progress
            # if i % 500 == 0:
            #     print("\tComputing P-values for point %d of %d..." % (i, n))

            # Compute the Gaussian kernel and entropy for the current precision
            betamin = -np.inf
            betamax = np.inf
            Di = D[i, np.concatenate((np.r_[0:i], np.r_[i + 1:n]))]
            (H, thisP) = self.Hbeta(Di, beta[i])

            # Evaluate whether the perplexity is within tolerance
            Hdiff = H - logU
            tries = 0
            while np.abs(Hdiff) > tol and tries < 50:

                # If not, increase or decrease precision
                if Hdiff > 0:
                    betamin = beta[i].copy()
                    if betamax == np.inf or betamax == -np.inf:
                        beta[i] = beta[i] * 2.
                    else:
                        beta[i] = (beta[i] + betamax) / 2.
                else:
                    betamax = beta[i].copy()
                    if betamin == np.inf or betamin == -np.inf:
                        beta[i] = beta[i] / 2.
                    else:
                        beta[i] = (beta[i] + betamin) / 2.

                # Recompute the values
                (H, thisP) = self.Hbeta(Di, beta[i])
                Hdiff = H - logU
                tries += 1

            # Set the final row of P
            P[i, np.concatenate((np.r_[0:i], np.r_[i + 1:n]))] = thisP

        # Return final P-matrix
        # print("\tMean value of sigma: %f" % np.mean(np.sqrt(1 / beta)))
        return P

    def fit_transform(self, X, max_iter=1000, early_exaggerate=True, y_random=None, dY=None, iY=None, gains=None, show_progress=True):
        """
        执行降维
        :param X: 高维数据
        :param max_iter: 最大迭代次数
        :param early_exaggerate: 是否早期放大
        :param y_random: 随机的初始矩阵，如果为None,需要在本函数中随机生成
        :param dY: 用于迭代的一个参数
        :param iY: 用于迭代的一个参数
        :param gains: 用于迭代的一个参数
        :return:
        """
        # print("\tearly-exaggerate: ", early_exaggerate)
        (n, d) = X.shape
        no_dims = self.n_component
        initial_momentum = 0.5
        final_momentum = 0.8
        eta = 500
        min_gain = 0.01

        # Initialize variables
        if y_random is None:
            Y = np.random.randn(n, no_dims)
        else:
            Y = y_random
        dY = np.zeros((n, no_dims))

        if dY is None:
            dY = np.zeros((n, no_dims))
        if iY is None:
            iY = np.zeros((n, no_dims))  # iY 和 gains可能会影响扰动效果，如果扰动效果不好，需要对这两个下手
        if gains is None:
            gains = np.ones((n, no_dims))

        # Compute P-values
        P = self.x2p2(X, 1e-5, self.perplexity)
        P = P + np.transpose(P)
        P = P / np.sum(P)
        if early_exaggerate:
            P = P * 4.  # early exaggeration
        P = np.maximum(P, 1e-12)

        # Run iterations
        for iter in range(max_iter):

            # Compute pairwise affinities
            sum_Y = np.sum(np.square(Y), 1)
            num = -2. * np.dot(Y, Y.T)
            num = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))
            num[range(n), range(n)] = 0.
            Q = num / np.sum(num)
            Q = np.maximum(Q, 1e-12)

            # Compute gradient
            PQ = P - Q
            for i in range(n):
                dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0)  # dY是完全重新计算了，应该不会影响

            # Perform the update
            if iter < 20 and early_exaggerate:
                momentum = initial_momentum
            else:
                momentum = final_momentum
            gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + (gains * 0.8) * ((dY > 0.) == (iY > 0.))
            gains[gains < min_gain] = min_gain
            iY = momentum * iY - eta * (gains * dY)
            Y = Y + iY
            Y = Y - np.tile(np.mean(Y, 0), (n, 1))

            # Compute current value of cost function
            if (iter + 1) % 1000 == 0 and show_progress:
                C = np.sum(P * np.log(P / Q))
                print("\tIteration %d: error is %f" % (iter + 1, C))
                # print("eta = ", eta)

            # Stop lying about P-values
            if iter == 100 and early_exaggerate:
                P = P / 4.

        # Return solution
        return Y


def run():
    data = np.loadtxt(temp_path+"data.csv", dtype=np.float, delimiter=",")
    data = Preprocess.normalize(data)
    (n, m) = data.shape

    distance = np.zeros((n, n))
    for i in range(0, n):
        for j in range(0, n):
            distance[i, j] = np.linalg.norm(data[i, :]-data[j, :])
    np.savetxt(temp_path+"distance.csv", distance, fmt='%f', delimiter=",")

    t_sne = cTSNE(n_component=2, perplexity=10.0)
    Y = t_sne.fit_transform(data, max_iter=10000)
    plt.scatter(Y[:, 0], Y[:, 1])
    plt.show()


if __name__ == '__main__':
    run()

