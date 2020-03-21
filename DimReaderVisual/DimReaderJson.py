# 根据我们的方法所求的导数，转化成DimReader系统可用的数据
import numpy as np
import json
from DimReaderVisual import Grid


def generateGrid(points):
    gridSize = 10
    points = np.array(points)
    xmax = max(points[:, 0])
    xmin = min(points[:, 0])
    ymax = max(points[:, 1])
    ymin = min(points[:, 1])

    gridCoord = []
    if (xmin == xmax):
        xmax += 1
        xmin -= 1
    if (ymin == ymax):
        ymin -= 1
        ymax += 1
    # if ymax > xmax:
    #     xmax = ymax
    # else:
    #     ymax = xmax
    # if ymin < xmin:
    #     xmin = ymin
    # else:
    #     ymin = xmin
    yrange = ymax - ymin
    xrange = xmax - xmin

    xstep = float(xrange) / (gridSize - 1)
    ystep = float(yrange) / (gridSize - 1)

    for i in range(gridSize + 1):
        gridCoord.append([])
        for j in range(gridSize + 1):
            gridCoord[i].append([(xmin - xstep / 2.0) + xstep * j, (ymax + ystep / 2.0) - ystep * i])
    return gridCoord


def calcGrid(points,dVects):  #date, grid, gridCoord, ind):
    # points 应该是降维后的坐标
    # dVects
    gridCoord = generateGrid(points)
    print(gridCoord)
    g = Grid.Grid(points, dVects, gridCoord)

    grid = g.calcGridPoints()

    n = len(gridCoord)
    m = len(gridCoord[0])
    gridPoints = []
    for i in range(n):
        gridPoints.append([])
        for j in range(m):
            gridPoints[i].append(grid[m * i + j])

    return grid.tolist()


def create_dimReader(X, Y, P, path=""):
    """
    生成DimReader系统可用的数据格式
    :param X: 高维数据矩阵
    :param Y: 降维结果矩阵
    :param P: 导数矩阵
    :param path: 文件存放的路径
    :return:
    """
    (n, m) = X.shape
    (pn, pm) = P.shape

    for attr in range(0, m):
        points = []
        perts = np.zeros((n, 2))
        if pn * pm == m*2:
            # 线性方法
            perts = np.tile(P[attr, :], (n, 1))
        else:
            for i in range(0, n):
                perts[i, 0] = P[2*i, m*i+attr]
                perts[i, 1] = P[2*i+1, m*i+attr]

        input_pert = [0] * m
        input_pert[attr] = 1

        for i in range(0, n):
            points.append({
                "domain": X[i, :].tolist(),
                "range": Y[i, :].tolist(),
                "inputPert": input_pert,
                "outputPert": perts[i, :].tolist()
            })

        perts_list = []
        for i in range(0, n):
            perts_list.append(perts[i, 0])
            perts_list.append(perts[i, 1])

        grid = calcGrid(Y.tolist(), perts_list)
        output = {"points": points, "scalarField": grid}
        f = open(path+"attr "+str(attr)+".dimreader", "w")
        f.write(json.dumps(output))
        f.close()


def test():
    # path = "E:\\文件\\IRC\\特征向量散点图项目\\result2020\\result0119\\MDS\\Iris3\\yita(0.102003062)nbrs_k(20)method_k(90)numbers(4)_b-spline_weighted\\"
    # path = "E:\\文件\\IRC\\特征向量散点图项目\\result2020\\result0119\\PCA\\Iris3\\yita(0.102003062)nbrs_k(20)method_k(90)numbers(4)_b-spline_weighted\\"
    # path = "E:\\文件\\IRC\\特征向量散点图项目\\result2020\\result0119\\cTSNE\\Iris3\\yita(0.102003062)nbrs_k(20)method_k(90)numbers(4)_b-spline_weighted\\"
    path = "E:\\文件\\IRC\\特征向量散点图项目\\result2020\\result0119\\PCA\\seeds\\yita(0.202003062)nbrs_k(25)method_k(90)numbers(4)_b-spline_weighted\\"
    X = np.loadtxt(path+"x.csv", dtype=np.float, delimiter=",")
    Y = np.loadtxt(path+"y.csv", dtype=np.float, delimiter=",")
    # P = np.loadtxt(path+"cTSNE_Pxy.csv", dtype=np.float, delimiter=",")
    P = np.loadtxt(path+"【weighted】P.csv", dtype=np.float, delimiter=",")

    create_dimReader(X, Y, P, path)


if __name__ == '__main__':
    test()
