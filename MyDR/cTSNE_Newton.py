# 用牛顿法解t-SNE
import numpy as np
import matplotlib.pyplot as plt


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
        eta = 500

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
        self.P0 = P.copy()
        P = P + np.transpose(P)
        # P = P / np.sum(P)
        P = P / (2*n)
        P = np.maximum(P, 1e-120)  # 1e-12太大了
        self.P = P.copy()

        # if early_exaggerate:
        #     P = P * 4.  # early exaggeration

        firsts = []  # 临时所加，用于统计一阶导的变化规律

        # Run iterations
        for iter in range(max_iter):
            if iter % 1000 == 0:
                print(iter)
            Y = Y2
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

            C = np.sum(P * (np.log(P) - np.log(Q)))
            if C != 0:
                gains = (-1) / C * dY
            else:
                break

            if follow_gradient:
                firsts.append(dY[0, :].tolist())

            Y = Y + gains  # 原来的式子
            # Y = Y - eta*dY  # 新改的实验方法
            Y2 = Y - np.tile(np.mean(Y, 0), (n, 1))

            # Compute current value of cost function
            # if (iter + 1) % 1000 == 0 and show_progress:
            #     C = np.sum(P * np.log(P / Q))
            #     print("\tIteration %d: error is %f" % (iter + 1, C))
            #     # print("eta = ", eta)

            # Stop lying about P-values
            # if iter == 100 and early_exaggerate:
            #     P = P / 4.

        # 最后更新低维空间中的概率矩阵
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


def test():
    path = "E:\\文件\\IRC\\特征向量散点图项目\\result2020\\result0119\\datasets\\Iris3\\"
    from Main import Preprocess
    data = np.loadtxt(path+"data.csv", dtype=np.float, delimiter=",")
    label = np.loadtxt(path+"label.csv", dtype=np.int, delimiter=",")

    X = Preprocess.normalize(data)

    tsne = cTSNE(n_component=2, perplexity=30.0)
    Y = tsne.fit_transform(X, max_iter=100000)

    plt.scatter(Y[:, 0], Y[:, 1], c=label)
    plt.show()

    from Derivatives import TSNE_Derivative
    der = TSNE_Derivative.TSNE_Derivative()
    P = der.getP(X, Y, tsne.P, tsne.Q, tsne.P0, tsne.beta)
    np.savetxt("F:\\P.csv", P, fmt='%.18e', delimiter=",")
    print("finished")


if __name__ == '__main__':
    test()
