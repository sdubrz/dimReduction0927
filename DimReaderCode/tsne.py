#
#  tsne.py
#
# Implementation of t-SNE in Python. The implementation was tested on Python 2.7.10, and it requires a working
# installation of NumPy. The implementation comes with an example on the MNIST dataset. In order to plot the
# results of this example, a working installation of matplotlib is required.
#
# The example can be run by executing: `ipython tsne.py`
#
#
#  Created by Laurens van der Maaten on 20-12-08.
#  Copyright (c) 2008 Tilburg University. All rights reserved.

import numpy as np
from DimReaderCode import DualNum


class tSNE:
    def __init__(self,X = np.array([]), no_dims = 2, initial_dims = 50, perplexity = 30.0,initY = None,maxIter = 1000,initBeta = None,betaTries = 50,initIY = None):
        self.X = X
        self.no_dims=no_dims
        self.initial_dims = initial_dims
        self.perplexity = perplexity
        self.initY = initY
        self.maxIter = maxIter
        self.betaTries = betaTries
        self.initBeta = initBeta
        self.initIY = initIY

    def Hbeta(self,D = np.array([]), beta = 1.0):
        """Compute the perplexity and the P-row for a specific value of the precision of a Gaussian distribution."""

        # Compute P-row and corresponding perplexity
        if self.dNum:
            tempD = DualNum.DualNum(D.val,D.dot)
            n=len(tempD.val)

            temp =DualNum.DualNum(-beta*np.identity(n),np.zeros((n,n)))*tempD
            P = temp.exp()
            sumP = np.maximum(sum(P).val, 1e-12)

        else:
            P = np.exp(-D.copy() * beta)
            sumP = np.maximum(np.sum(P), 1e-12)

        PD = D*P

        bPD =(beta * np.sum(PD))[0]
        if self.dNum:
            H = np.log(sumP) + bPD.val  / sumP
            P = P / DualNum.DualNum(sumP, 0)
        else:
            H = np.log(sumP ) + bPD / sumP
            P = P / sumP
        return H, P

    def calcPtDist(self,points):
            dists = []
            for i in range(len(points)):
                dists.append([])
                for j in range(len(points)):
                    diff = points[i] - points[j]
                    norm = 0
                    for k in range(len(diff)):
                        norm += pow(diff[k], 2)
                    norm = pow(norm, .5)
                    dists[i].append(norm)


            return dists

    def x2p(self,X = np.array([]), tol = 1e-5, perplexity = 30.0,initBeta = None,betaTries=50):
        """Performs a binary search to get P-values in such a way that each conditional Gaussian has the same perplexity."""

        # Initialize some variables
        # print( "Computing pairwise distances...")
        if self.dNum:
           temp = pow(X,2)
           tempT = DualNum.DualNum(temp.val.T,temp.dot.T)
           sum_X = sum(tempT)

        else:
            sum_X = np.sum(np.square(X),1)


        if self.dNum:
            (n, d) = X.val.shape

            XT = DualNum.DualNum(X.val.T,X.dot.T )
            xxt = X*XT
        else:
            (n, d) = X.shape

            xxt = np.dot(X, X.T)

            xxt = np.array(xxt)
        temp = np.add(-2 * xxt, sum_X)

        if self.dNum:
            tempT = DualNum.DualNum(temp.val.T,temp.dot.T)
        else:
            tempT = temp.T
        D = np.add(tempT, sum_X)
        if not self.dNum:
            D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X);

        P = []
        if self.dNum:
            P = DualNum.DualNum(np.zeros((n,n)),np.zeros((n,n)))
        else:
            P = np.zeros((n, n))

        if initBeta is None:
            beta = np.ones((n, 1))
        else:
            beta = initBeta
        if perplexity.__class__ is DualNum.DualNum:
            logU = perplexity.log()
        else:
            logU = np.log(perplexity)

        # Loop over all datapoints
        for i in range(n):
            # print(progress
            # if i % 500 == 0:
                # print( "Computing P-values for point ", i, " of ", n, "...")

            # Compute the Gaussian kernel and entropy for the current precision
            betamin = -np.inf
            betamax =  np.inf
            if self.dNum:
                Di = DualNum.DualNum(D.val[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))],D.dot[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))])
            else:
                Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
            (H, thisP) = self.Hbeta(Di, beta[i])

            # Evaluate whether the perplexity is within tolerance
            Hdiff = H - logU
            tries = 0
            while np.abs(Hdiff) > tol and tries < betaTries:

                # If not, increase or decrease precision
                if Hdiff > 0:
                    betamin = beta[i].copy()
                    if betamax == np.inf or betamax == -np.inf:
                        beta[i] = beta[i] * 2
                    else:
                        beta[i] = (beta[i] + betamax) / 2
                else:
                    betamax = beta[i].copy()
                    if betamin == np.inf or betamin == -np.inf:
                        beta[i] = beta[i] / 2
                    else:
                        beta[i] = (beta[i] + betamin) / 2

                # Recompute the values
                (H, thisP) = self.Hbeta(Di, beta[i])
                Hdiff = H - logU
                tries = tries + 1

            if self.dNum:
                for j in range(len(thisP.val)+1):
                    if j !=i:
                        if j>i:
                            P.val[i][j] = thisP.val[j-1]
                            P.dot[i][j] = thisP.dot[j-1]

                        else:
                            P.val[i][j] = thisP.val[j]
                            P.dot[i][j] = thisP.dot[j]
            else:
                P[i, np.concatenate((np.r_[0:i], np.r_[i + 1:n]))] = thisP

        self.beta = beta
        # Return final P-matrix
        # print( "Mean value of sigma: ", np.mean(np.sqrt(1 / beta)))
        return P

    # tsne with dual numbers
    def dNumtsne(self,X = np.array([]), no_dims = 2, initial_dims = 50, perplexity = 30.0,initY = None,maxIter = 999,initBeta = None, betaTries=50,initIY = None):
        """Runs t-SNE on the dataset in the NxD array X to reduce its dimensionality to no_dims dimensions.
        The syntax is of the function is Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array."""

        # Check inputs
        if isinstance(no_dims, float):
            print( "Error: array X should have type float.")
            return -1
        if round(no_dims) != no_dims:
            print( "Error: number of dimensions should be an integer.")
            return -1

        if self.dNum:
            (n,d) = X.val.shape
        else:
            X = np.array(X)
            (n, d) = X.shape
        max_iter = maxIter
        initial_momentum = 0.5
        final_momentum = 0.8
        eta = 500
        min_gain = 0.01

        Y =[]
        dY = []
        iY = []
        gains=[]
        Y = DualNum.DualNum(np.random.randn(n,no_dims),np.zeros((n,no_dims)))
        for i in range(n):
            dY.append(DualNum.DualNum(np.zeros(no_dims),np.zeros(no_dims)))
            gains.append([])
            iY.append([])
            for j in range(no_dims):
                gains[i].append(DualNum.DualNum(1, 0))
                iY[i].append(DualNum.DualNum(0, 0))

        dY = np.array(dY)
        gains = np.array(gains)
        iY = np.array(iY)

        if initY is not None:
            Y = initY

        if initIY is not None:
            for i in range(n):
                for j in range(no_dims):
                    iY[i][j]  = DualNum.DualNum(initIY[i][j],0)

        # Compute P-values
        P = self.x2p(X, 1e-5, perplexity,initBeta,betaTries)

        PT = DualNum.DualNum(P.val.T,P.dot.T)
        P = P + PT
        s = sum(sum(P))
        P = P / s
        P = P * 4
        P.val = np.maximum(P.val, pow(10,-12))

        # Run iterations
        for iter in range(max_iter):

            sum_Y = np.sum(np.square(Y), 1)
            num = 1 / (1 + np.add(np.add(-2 * np.dot(Y, Y.T), sum_Y).T, sum_Y))
            num[range(n), range(n)] = 0
            Q = num / np.sum(num)
            Q = np.maximum(Q,pow(10,-12))

            PQ = P - Q

            PQ2 = []

            iMax=len(PQ.val)
            jMax = len(PQ.val[0])

            for i in range(iMax):
                PQ2.append([])
                for j in range(jMax):
                    PQ2[i].append(DualNum.DualNum(np.array(PQ.val[i][j]),np.array(PQ.dot[i][j])))
            PQ= PQ2
            PQ = np.array(PQ)

            for i in range(n):
                temp = PQ[:,i] * num[:,i]
                temp2 = (Y[i,:] - Y)
                temp3 = np.tile(temp, (no_dims, 1))
                dY[i] = sum(temp3.T*temp2)

            # Perform the update
            if iter < 20:
                momentum = initial_momentum
            else:
                momentum = final_momentum
            dYNew =[]
            for i in range(len(dY)):
                dYNew.append([])
                for j in range(len(dY[i])):
                    dYNew[i].append(DualNum.DualNum(dY[i][j].val,dY[i][j].dot))
            dY = np.array(dYNew)
            t1 =  (gains + 0.2) * ((dY > 0) != (iY > 0))
            t2 = (gains * 0.8) * ((dY > 0) == (iY > 0))
            gains = t1+ t2
            gains[gains < min_gain] = min_gain

            iY = momentum * iY - eta * (gains * dY)

            Y = Y + iY
            Y = Y - np.tile(np.mean(Y, 0), (n, 1))

            # Compute current value of cost function
            if (iter + 1) % 10 == 0:
                pdq = P/Q
                C = np.sum(P * np.log(pdq))
                print( "Iteration ", (iter + 1), ": error is ", C)

            # Stop lying about P-values
            if iter == 100:
                P = P / 4
        self.iY = iY
        # Return solution
        return Y

    # tsne without dual numbers to be used for the initial run
    def tsne(self,X = np.array([]), no_dims = 2, initial_dims = 50, perplexity = 30.0,initY = None,maxIter = 1000,initBeta = None, betaTries=50,initIY = None):

        """Runs t-SNE on the dataset in the NxD array X to reduce its dimensionality to no_dims dimensions.
            The syntaxis of the function is Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array."""

        # Check inputs
        if isinstance(no_dims, float):
            print("Error: array X should have type float.");
            return -1;
        if round(no_dims) != no_dims:
            print("Error: number of dimensions should be an integer.")
            return -1;

        # Initialize variables
        (n, d) = X.shape;
        max_iter = maxIter;
        initial_momentum = 0.5;
        final_momentum = 0.8;
        eta = 500;
        min_gain = 0.01;
        Y = np.random.randn(n, no_dims);

        dY = np.zeros((n, no_dims));
        iY = np.zeros((n, no_dims));
        gains = np.ones((n, no_dims));

        # Compute P-values
        P = self.x2p(X, 1e-5, perplexity);
        P = P + np.transpose(P);
        if self.dNum:
            sumP = np.maximum(sum(sum(P)).val,1e-12)
        else:
            sumP = np.maximum(np.sum(P),1e-12)
        P = P / sumP
        P = P * 4;  # early exaggeration
        P = np.maximum(P, 1e-12);
        # Run iterations10
        for iter in range(max_iter):

            # Compute pairwise affinities
            sum_Y = np.sum(np.square(Y), 1);

            num = 1 / (1 + np.add(np.add(-2 * np.dot(Y, Y.T), sum_Y).T, sum_Y));
            num[range(n), range(n)] = 0;
            Q = num / np.sum(num);
            Q = np.maximum(Q, 1e-12);
            # Compute gradient
            PQ = P - Q

            for i in range(n):
                dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0);

            # Perform the update
            if iter < 20:
                momentum = initial_momentum
            else:
                momentum = final_momentum
            gains = (gains + 0.2) * ((dY > 0) != (iY > 0)) + (gains * 0.8) * ((dY > 0) == (iY > 0));
            gains[gains < min_gain] = min_gain;

            iY = momentum * iY - eta * (gains * dY);
            Y = Y + iY;
            Y = Y - np.tile(np.mean(Y, 0), (n, 1));

            # Compute current value of cost function
            if (iter + 1) % 10 == 0:
                C = np.sum(P * np.log(P / Q));
                print("Iteration ", (iter + 1), ": error is ", C)

            # Stop lying about P-values
            if iter == 100:
                P = P / 4;

        self.iY = iY
        # Return solution
        return Y;

    # determines whether to run tsne with dual numbers or without
    def runTSNE(self, dnum=False):
        self.dNum = dnum
        if type(self.X) is DualNum.DualNum:
            self.dNum = True
        if self.dNum:
            return self.dNumtsne(self.X,self.no_dims,self.initial_dims,self.perplexity,self.initY,self.maxIter,self.initBeta,self.betaTries,self.initIY)
        else:
            return self.tsne(self.X,self.no_dims,self.initial_dims,self.perplexity,self.initY,self.maxIter,self.initBeta,self.betaTries,self.initIY)

    # return parameters to seed future runs to have consistent results
    def getFinalParameters(self):
        return self.beta,self.iY
