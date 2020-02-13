import os
path = "C:/Users/Hayim/Desktop/testrun/datasets" #文件夹目录
files= os.listdir(path) #得到文件夹下的所有文件名称
s = []
for file in files: #遍历文件夹
    if file.startswith('mnist50mminiclass'):
        print(file)