# encoding:GBK
"""
Created on 2019/09/23 16:19:11
@author: Sirius_xuan
"""
'''
����PCA��ͼ��ά���ع�
'''

import numpy as np

# �������Ļ�
def Z_centered(dataMat):
    rows, cols = dataMat.shape
    meanVal = np.mean(dataMat, axis=0)  # �������ֵ��������������ľ�ֵ
    meanVal = np.tile(meanVal, (rows, 1))
    newdata = dataMat - meanVal
    return newdata, meanVal


# Э�������
def Cov(dataMat):
    meanVal = np.mean(data, 0)  # ѹ���У�����1*cols���󣬶Ը������ֵ
    meanVal = np.tile(meanVal, (rows, 1))  # ����rows�еľ�ֵ����
    Z = dataMat - meanVal
    Zcov = (1 / (rows - 1)) * Z.T * Z
    return Zcov


# ��С����ά��ɵ���ʧ��ȷ��k
def Percentage2n(eigVals, percentage):
    sortArray = np.sort(eigVals)  # ����
    sortArray = sortArray[-1::-1]  # ��ת��������
    arraySum = sum(sortArray)
    tmpSum = 0
    num = 0
    for i in sortArray:
        tmpSum += i
        num += 1
        if tmpSum >= arraySum * percentage:
            return num


# �õ�����k������ֵ����������
def EigDV(covMat, p):
    D, V = np.linalg.eig(covMat)  # �õ�����ֵ����������
    k = Percentage2n(D, p)  # ȷ��kֵ
    print("����90%��Ϣ����ά�������������" + str(k) + "\n")
    eigenvalue = np.argsort(D)
    K_eigenValue = eigenvalue[-1:-(k + 1):-1]
    K_eigenVector = V[:, K_eigenValue]
    return K_eigenValue, K_eigenVector


# �õ���ά�������
def getlowDataMat(DataMat, K_eigenVector):
    return DataMat * K_eigenVector


# �ع�����
def Reconstruction(lowDataMat, K_eigenVector, meanVal):
    reconDataMat = lowDataMat * K_eigenVector.T + meanVal
    return reconDataMat


# PCA�㷨
def PCA(data, p):
    dataMat = np.float32(np.mat(data))
    # �������Ļ�
    dataMat, meanVal = Z_centered(dataMat)
    # ����Э�������
    # covMat = Cov(dataMat)
    covMat = np.cov(dataMat, rowvar=0)
    # �õ�����k������ֵ����������
    D, V = EigDV(covMat, p)
    # �õ���ά�������
    lowDataMat = getlowDataMat(dataMat, V)
    np.savetxt("C:\\Users\\Hayim\\Desktop\\testrun\\datasets\\MNIST\\new.csv", np.around(lowDataMat, decimals=6), delimiter=',')
    # �ع�����
    reconDataMat = Reconstruction(lowDataMat, V, meanVal)
    return reconDataMat


def main():
    data_path = 'C:\\Users\\Hayim\\Desktop\\testrun\\datasets\\MNIST\\data.csv'
    data_reader = np.loadtxt(data_path, dtype=np.int, delimiter=",")
    data = data_reader[:, :].astype(np.float)
    data_shape = data.shape
    rows,cols = data_shape
    print(data_shape)
    print("��άǰ������������" + str(cols) + "\n")
    print('----------------------------------------')
    reconImage = PCA(data, 0.80)
    reconImage = reconImage.astype(np.uint8)
    print(reconImage.shape)


if __name__ == '__main__':
    main()