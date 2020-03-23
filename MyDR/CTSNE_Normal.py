# 经典的t-SNE方法改写，根据 Maaten 的程序改写为面向对象形式
# 改为所有的点都用相同的方差，beta的取值为所有点beta值的平均
import numpy as np
import matplotlib.pyplot as plt
from Main import Preprocess
from Main import DimReduce
from Main import Preprocess


class cTSNE:
    def __init__(self, n_component=2, perplexity=30.0):
        """
        init function
        :param n_component: 降维后的维度数
        :param perplexity: 困惑度
        """
        self.n_component = n_component
        self.perplexity = perplexity
        self.beta = None
        self.kl = []
        self.final_kl = None
        self.final_iter = 0
        self.P = None
        self.P0 = None
        self.Q = None

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

    def x2p_beta(self, X=np.array([]), tol=1e-5, perplexity=30.0, beta=None):
        """
        根据已有的beta计算概率矩阵
        :param X: 数据矩阵
        :param tol:
        :param perplexity:
        :param beta:
        :return:
        """
        (n, d) = X.shape
        sum_X = np.sum(np.square(X), 1)
        D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
        P = np.zeros((n, n))

        for i in range(0, n):
            Di = D[i, np.concatenate((np.r_[0:i], np.r_[i + 1:n]))]
            (H, thisP) = self.Hbeta(Di, beta[i])
            P[i, np.concatenate((np.r_[0:i], np.r_[i + 1:n]))] = thisP

        return P

    def x2p(self, X=np.array([]), tol=1e-12, perplexity=30.0):
        """原来的tol是1e-5
            Performs a binary search to get P-values in such a way that each
            conditional Gaussian has the same perplexity.
        """

        # Initialize some variables
        # print("\tComputing pairwise distances...")
        (n, d) = X.shape
        sum_X = np.sum(np.square(X), 1)
        D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
        P = np.zeros((n, n))
        beta = np.ones((n, 1)) / np.max(D)  # 加上一个除法，有的数据不normalize会报0除错误，系初始的beta设置过大导致
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
            while np.abs(Hdiff) > tol and tries < 1000:  # 原来是50

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
        self.beta = beta
        return P

    def fit_transform(self, X, max_iter=1000, early_exaggerate=True, y_random=None, dY=None, iY=None, gains=None,
                      show_progress=False, min_kl=None, follow_gradient=True):
        """
        执行降维
        :param X: 高维数据
        :param max_iter: 最大迭代次数
        :param early_exaggerate: 是否早期放大
        :param y_random: 随机的初始矩阵，如果为None,需要在本函数中随机生成
        :param dY: 用于迭代的一个参数
        :param iY: 用于迭代的一个参数
        :param gains: 用于迭代的一个参数
        :param show_progress: 是否展示中间结果
        :param min_kl: 当KL散度小于该值时停止迭代
        :param follow_gradient: 是否记录迭代过程中的导数
        :return:
        """
        # print("\tearly-exaggerate: ", early_exaggerate)
        (n, d) = X.shape
        no_dims = self.n_component
        initial_momentum = 0.5
        final_momentum = 0.8
        if not early_exaggerate:  # 20191231
            final_momentum = 0.0
        eta = 500
        min_gain = 0.01  # 原为0.01

        if show_progress:
            self.kl = []

        # Initialize variables
        Y2 = np.random.randn(n, no_dims)
        if y_random is None:
            Y2 = np.random.randn(n, no_dims)
        else:
            Y2 = y_random
        dY = np.zeros((n, no_dims))

        if dY is None:
            dY = np.zeros((n, no_dims))
        if iY is None:
            iY = np.zeros((n, no_dims))  # 上次迭代的改变量
        if gains is None:
            gains = np.ones((n, no_dims))

        # Compute P-values
        P = self.x2p(X, 1e-15, self.perplexity)  # 第二个参数原来是1e-5
        mean_beta = np.mean(self.beta)
        self.beta = mean_beta * np.ones((n, 1))
        P = self.x2p_beta(X, beta=self.beta)

        self.P0 = P.copy()
        P = P + np.transpose(P)
        # P = P / np.sum(P)
        P = P / (2*n)
        P = np.maximum(P, 1e-120)  # 1e-12太大了
        self.P = P.copy()

        if early_exaggerate:
            P = P * 4.  # early exaggeration

        firsts = []  # 临时所加，用于统计一阶导的变化规律

        # Run iterations
        Y = Y2
        for iter in range(max_iter):
            # Y = Y2  # 原版本的t-sne是有这个中心化的操作的
            # Compute pairwise affinities
            sum_Y = np.sum(np.square(Y), 1)  # square是将矩阵中的每个元素计算平方，sum_Y里面存储的是每个点的模的平方
            num = -2. * np.dot(Y, Y.T)
            num = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))
            num[range(n), range(n)] = 0.  # 把对角线设置为0
            Q = num / np.sum(num)
            Q = np.maximum(Q, 1e-120)

            if show_progress:
                c = np.sum(P*np.log(P/Q))
                self.kl.append(c)

            if iter == max_iter - 1:
                self.final_kl = np.sum(P*np.log(P/Q))

            if not (min_kl is None) and iter > 1:
                c = np.sum(P*np.log(P/Q))
                if c <= min_kl:
                    break

            # Compute gradient
            PQ = P - Q
            for i in range(n):
                dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0)  # dY是完全重新计算了，应该不会影响

            if follow_gradient:
                firsts.append(dY[0, :].tolist())

            # Perform the update
            if iter < 20 and early_exaggerate:
                momentum = initial_momentum
            else:
                momentum = final_momentum
            # gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + (gains * 0.8) * ((dY > 0.) == (iY > 0.))  # 感觉它这里代码有错误
            gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + (gains * 0.8) * ((dY > 0.) == (iY > 0.))
            if not early_exaggerate:  # 2020.02.17为了提高收敛精度，将此注释
                gains[gains < min_gain] = min_gain
            iY = momentum * iY - eta * (gains * dY)
            Y = Y + iY  # 原来的式子
            # Y = Y - eta*dY  # 新改的实验方法
            Y2 = Y - np.tile(np.mean(Y, 0), (n, 1))

            # Compute current value of cost function
            # if (iter + 1) % 1000 == 0 and show_progress:
            #     C = np.sum(P * np.log(P / Q))
            #     print("\tIteration %d: error is %f" % (iter + 1, C))
            #     # print("eta = ", eta)

            # Stop lying about P-values
            if iter == 100 and early_exaggerate:
                P = P / 4.

        # 最后更新低维空间中的概率矩阵
        if follow_gradient:
            sum_Y = np.sum(np.square(Y), 1)
            num = -2. * np.dot(Y, Y.T)
            num = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))
            num[range(n), range(n)] = 0.  # 把对角线设置为0
            Q = num / np.sum(num)
            Q = np.maximum(Q, 1e-120)
            self.Q = Q

        self.final_iter = iter
        if show_progress:
            kl = self.kl
            # print(kl)
            # plt.scatter(range(0, max_iter), kl)
            plt.plot(kl)
            plt.title("KL divergence, min="+str(min(kl)))
            plt.show()

        if follow_gradient:
            print("最终迭代的次数是 ", iter)
            # np.savetxt("F:\\first.csv", np.array(firsts), fmt='%.18e', delimiter=",")
            print(firsts[len(firsts)-1])

            plt.plot(np.log10(firsts))
            plt.title("log10 der")
            plt.show()

        return Y


if __name__ == '__main__':
    # dr_3d()
    # perturbation_one_by_one()
    # run()
    # show_kl()
    # knn_change()
    run2()





