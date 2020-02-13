import cv2 as cv
import os
import numpy as np



path = 'D:\\Exp\\datasets\\coil-20-proc'
file_list = os.listdir(path)

data = np.empty([len(file_list),64], dtype = float)
label = []

for i, file in enumerate(file_list):
    img = cv.imread(path+'\\'+file)

    img=cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    # 将图片高和宽分别赋值给x，y
    x, y = img.shape[0:2]
    # 最近邻插值法缩放
    img_test2 = cv.resize(img, (0, 0), fx=0.0625, fy=0.0625, interpolation=cv.INTER_NEAREST)

    data[i] = np.array(img_test2).reshape([64])
    print(file.split('__')[0])
    label.append(file.split('__')[0])

np.savetxt(path+'\\'+'data.csv', data, '%d', ',')
np.savetxt(path+'\\'+'label.csv', label, '%s', ',')

