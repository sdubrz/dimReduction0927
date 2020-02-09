# encoding:GBK
"""
Created on 2019/09/23 16:19:11
@author: Sirius_xuan
"""
'''
基于PCA的图像降维及重构
'''

import numpy as np

# 数据中心化
def Z_centered(dataMat):
    rows, cols = dataMat.shape
    meanVal = np.mean(dataMat, axis=0)  # 按列求均值，即求各个特征的均值
    meanVal = np.tile(meanVal, (rows, 1))
    newdata = dataMat - meanVal
    return newdata, meanVal


# 协方差矩阵
def Cov(dataMat):
    meanVal = np.mean(data, 0)  # 压缩行，返回1*cols矩阵，对各列求均值
    meanVal = np.tile(meanVal, (rows, 1))  # 返回rows行的均值矩阵
    Z = dataMat - meanVal
    Zcov = (1 / (rows - 1)) * Z.T * Z
    return Zcov


# 最小化降维造成的损失，确定k
def Percentage2n(eigVals, percentage):
    sortArray = np.sort(eigVals)  # 升序
    sortArray = sortArray[-1::-1]  # 逆转，即降序
    arraySum = sum(sortArray)
    tmpSum = 0
    num = 0
    for i in sortArray:
        tmpSum += i
        num += 1
        if tmpSum >= arraySum * percentage:
            return num


# 得到最大的k个特征值和特征向量
def EigDV(covMat, p):
    D, V = np.linalg.eig(covMat)  # 得到特征值和特征向量
    k = Percentage2n(D, p)  # 确定k值
    print("保留90%信息，降维后的特征个数：" + str(k) + "\n")
    eigenvalue = np.argsort(D)
    K_eigenValue = eigenvalue[-1:-(k + 1):-1]
    K_eigenVector = V[:, K_eigenValue]
    return K_eigenValue, K_eigenVector


# 得到降维后的数据
def getlowDataMat(DataMat, K_eigenVector):
    return DataMat * K_eigenVector


# 重构数据
def Reconstruction(lowDataMat, K_eigenVector, meanVal):
    reconDataMat = lowDataMat * K_eigenVector.T + meanVal
    return reconDataMat


# PCA算法
def PCA(data, p):
    dataMat = np.float32(np.mat(data))
    # 数据中心化
    dataMat, meanVal = Z_centered(dataMat)
    # 计算协方差矩阵
    # covMat = Cov(dataMat)
    covMat = np.cov(dataMat, rowvar=0)
    # 得到最大的k个特征值和特征向量
    D, V = EigDV(covMat, p)
    # 得到降维后的数据
    lowDataMat = getlowDataMat(dataMat, V)
    np.savetxt("C:\\Users\\Hayim\\Desktop\\testrun\\datasets\\MNIST\\new.csv", np.around(lowDataMat, decimals=6), delimiter=',')
    # 重构数据
    reconDataMat = Reconstruction(lowDataMat, V, meanVal)
    return reconDataMat


def main():
    data_path = 'C:\\Users\\Hayim\\Desktop\\testrun\\datasets\\MNIST\\data.csv'
    data_reader = np.loadtxt(data_path, dtype=np.int, delimiter=",")
    data = data_reader[:, :].astype(np.float)
    data_shape = data.shape
    rows,cols = data_shape
    print(data_shape)
    print("降维前的特征个数：" + str(cols) + "\n")
    print('----------------------------------------')
    reconImage = PCA(data, 0.80)
    reconImage = reconImage.astype(np.uint8)
    print(reconImage.shape)


if __name__ == '__main__':
    main()