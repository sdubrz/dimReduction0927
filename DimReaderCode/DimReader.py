import sys
from DimReaderCode import DualNum
import csv
import numpy as np
from DimReaderCode import tsne
import multiprocessing
import datetime
import time


class runTsne:
    def __init__(self,perplexity = 30):
        self.perplex = perplexity
        self.origY = None

    def calculateValues(self, points=None, perturbations=None):

        if points is None:
            points = self.origPoints
        else:
            self.points = points
            self.origPoints = points
        n = len(points)
        d = len(points[0])
        self.dualNumPts = DualNum.DualNum(np.array(points), np.zeros(np.shape(points)))

        self.perturb = perturbations
        if self.origY is None:
            # initial run of tsne to get seed parameters
            t = tsne.tSNE(points, initial_dims=d / 2, perplexity=self.perplex)
            self.origY = t.runTSNE()
            self.beta, self.iY = t.getFinalParameters()

        if (n != 0):
            procs = []
            cpus = multiprocessing.cpu_count()
            xOutArray = multiprocessing.Array('d', range(n))
            yOutArray = multiprocessing.Array('d', range(n))
            outDotArray = multiprocessing.Array('d', range(2 * n))

            if (cpus > n):
                cpus = 1
            chunksize = int(np.floor(float(n) / cpus))
            for i in range(cpus):
                minI = chunksize * i

                if (i < cpus - 1):
                    maxI = chunksize * (i + 1)
                else:
                    maxI = n

                procs.append(multiprocessing.Process(target=self.loopFunc,
                                                     args=(self.dualNumPts, minI, maxI, outDotArray,
                                                           self.origY, self.beta, self.iY, xOutArray, yOutArray)))

            for proc in procs:
                proc.start()

            for proc in procs:
                proc.join()

            points = []
            self.resultVect = [0] * (2 * n)
            for i in range(n):
                self.resultVect[i] = outDotArray[i]
                self.resultVect[i + n] = outDotArray[i + n]
                points.append([xOutArray[i], yOutArray[i]])

        self.points = points

    def loopFunc(self, pts, minI, maxI, dotArr, origY, beta, iY, xPts, yPts):
        for i in range(minI, maxI):

            if self.perturb is None:
                pts.dot[i][self.axis] = 1
            else:
                pts.dot[i] = self.perturb[i]
            m = len(self.origPoints[0])

            t = tsne.tSNE(self.dualNumPts, initial_dims=m / 2, perplexity=self.perplex, initY=origY,
                             maxIter=1, initBeta=beta, betaTries=1, initIY=iY)
            results = t.runTSNE(True)

            # print("Min: ", minI, "I: ", i, "max: ", maxI)

            dotArr[2 * i] = results[i][0].dot
            dotArr[2 * i + 1] = results[i][1].dot
            n = len(pts.val)

            if i == 0:
                for j in range(len(self.points)):
                    xPts[j] = results[j][0].val
                    yPts[j] = results[j][1].val
            pts.dot[i] = np.zeros(m)


projections = ["tsne"]
projectionClasses = [runTsne]


def readFile(filename):
    read = csv.reader(open(filename, 'rt'))

    points = []
    firstLine = next(read)
    headers = []
    rowDat = []
    head = False
    for i in range(0, len(firstLine)):
        try:
            rowDat.append(float(firstLine[i]))
        except:
            head = True
            break
    if head:
        headers = firstLine
    else:
        points.append(rowDat)

    for row in read:
        rowDat = []
        for i in range(0, len(row)):
            try:
                rowDat.append(float(row[i]))
            except:
                print("[125] invalid data type - must be numeric")
                print(i)
                print(row[i])
                exit(0)
        points.append(rowDat)
    return points


def runProjection(projection, points, perturbations):
    derivVects = []

    pertCount = 0

    for pert in perturbations:
        print('pert num', pertCount)
        projection.calculateValues(np.array(points), np.array(pert))
        derivVects.append(projection.resultVect)
        projPts = projection.points
        pertCount = pertCount + 1

    date = str(datetime.datetime.fromtimestamp(time.time())).replace(" ", "_")
    date = date.split(".")[0]
    fileName = date + "_output.csv"

    fileName = "output.csv"
    f = open(fileName, "w")

    n = len(projPts)

    headers = "ProjectedX,ProjectedY"
    if (len(perturbations) > 1):
        for i in range(len(perturbations)):
            headers += ",dx" + str(i) + ",dy" + str(i)
        headers += "\n"
    else:
        headers += ",dx,dy\n"
    f.write(headers)
    for i in range(n):
        row = str(projPts[i][0]) + "," + str(projPts[i][1])
        for j in range(len(perturbations)):
            row += "," + str(derivVects[j][2 * i]) + "," + str(derivVects[j][2 * i + 1])
        row += "\n"
        f.write(row)
    f.close()


def run_test():
    inputFile = "IRIS.csv"
    print(sys.argv)
    # perturbFile = sys.argv[2]
    perturbFile = "all"
    projection = "tsne"

    if str.lower(projection) not in map(str.lower, projections):
        print("Invalid Projection")
        print("Projection Options:")
        for opt in projections:
            print("\t" + opt)
        exit(0)

    projInd = list(map(str.lower, projections)).index(str.lower(projection))
    if (projInd < 5):
        print('小于5')
        inputPts = readFile(inputFile)
    else:
        projection = projectionClasses[projInd]()
        projection.loadMat(inputFile)
        inputPts = projection.origPoints

    if str.lower(perturbFile) == "all":
        perturbVects = []
        n, m = np.shape(inputPts)
        for i in range(m):
            currPert = np.zeros((n, m))
            for j in range(n):
                currPert[j][i] = 1
            perturbVects.append(currPert)
    else:
        perturbVects = [readFile(perturbFile)]

    points = DualNum.DualNum(inputPts, perturbVects)

    if (projInd < 5):
        if (len(sys.argv) == 5):
            projection = projectionClasses[projInd](float(sys.argv[4]))
        else:
            projection = projectionClasses[projInd]()

    np.savetxt("inputPts.csv", inputPts, fmt='%f', delimiter=",")
    # np.savetxt("perturbVects.csv", perturbVects, fmt='%f', delimiter=",")
    runProjection(projection, inputPts, perturbVects)


if __name__ == '__main__':
    run_test()


# if __name__ == "__main__":
#     if (len(sys.argv) >= 4):
#         inputFile = sys.argv[1]
#         perturbFile = sys.argv[2]
#         projection = sys.argv[3]
#
#         if str.lower(projection) not in map(str.lower, projections):
#             print("Invalid Projection")
#             print("Projection Options:")
#             for opt in projections:
#                 print("\t" + opt)
#             exit(0)
#
#         projInd = list(map(str.lower, projections)).index(str.lower(projection))
#         if (projInd < 5):
#             inputPts = readFile(inputFile)
#         else:
#             projection = projectionClasses[projInd]()
#             projection.loadMat(inputFile)
#             inputPts = projection.origPoints
#
#         if str.lower(perturbFile) == "all":
#             perturbVects = []
#             n, m = np.shape(inputPts)
#             for i in range(m):
#                 currPert = np.zeros((n, m))
#                 for j in range(n):
#                     currPert[j][i] = 1
#                 perturbVects.append(currPert)
#         else:
#             perturbVects = [readFile(perturbFile)]
#
#         points = DualNum.DualNum(inputPts, perturbVects)
#
#         if (projInd < 5):
#             if (len(sys.argv) == 5):
#                 projection = projectionClasses[projInd](float(sys.argv[4]))
#             else:
#                 projection = projectionClasses[projInd]()
#
#         runProjection(projection, inputPts, perturbVects)
#
#     else:
#         print("DimReaderScript [input file] [perturbation file] [Projection] [optional parameter]")
#         print("For all dimension perturbations, perturbation file = all")
#         print("Projection Options:")
#         for opt in projections:
#             print("\t" + opt)
#
#         exit(0)


