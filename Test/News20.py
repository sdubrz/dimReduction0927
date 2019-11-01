# 对20组新闻的数据进行加工

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import SnowballStemmer
import re
import csv
import math
import os


# 预处理一个字符串，统一大小写，去除停用词、标点和数字等
def preprocess(sentence):
    s = SnowballStemmer('english')
    sentence = sentence.lower()
    tokens = nltk.word_tokenize(sentence)
    wordnet_lemmatizer = WordNetLemmatizer()
    tokens2 = []
    for word in tokens:
        if word not in stopwords.words('english') and not isSymbol(word) and not hasNumbers(word):
            word2 = wordnet_lemmatizer.lemmatize(word)
            word3 = s.stem(word2)
            tokens2.append(word3)
    return tokens2


# 检查是否是特殊符号
def isSymbol(string):
    return bool(re.match(r'[^\w]',string))


# 检查是否含有数字
def hasNumbers(string):
    return bool(re.search(r'\d',string))


# 对某一文件夹下的所有文件进行预处理，并存储起来
def dopreprocess(inpath,outpath):
    count = 0
    for maindir, subdir, filenamelist in os.walk(inpath):
        for filename in filenamelist:
            ainpath = os.path.join(maindir, filename)
            aoutpath = os.path.join(outpath, ainpath[41:len(ainpath)-len(filename)-1])
            # print("输入目录：", ainpath)
            # print("输出目录：", aoutpath)

            if not os.path.exists(aoutpath):
                os.makedirs(aoutpath)
            tempfilename = filename+".txt"
            aoutpath = os.path.join(aoutpath,tempfilename)
            # aoutpath = aoutpath.join('.txt')
            sentence = open(ainpath,'r',encoding="UTF-8")  # 读取文件没有问题
            sentence2 = []
            filewriter = open(''.join(aoutpath), 'w',encoding="UTF-8")
            while True:
                try:
                    line = sentence.readline()
                except:
                    continue
                if not line:
                    break
                line2 = preprocess(line)
                if len(line2) == 0:
                    continue
                for word in line2:
                    #print(word)
                    filewriter.write(word)
                    filewriter.write(" ")
                #filewriter.write('\n')
            filewriter.close()
            sentence.close()
            count = count+1
            if count.__mod__(1000) == 0:
                print(count)


def asentencetest():
    # nltk.download('punkt')
    str = 'In article <114152@bu.edu>, lcai@acs2.bu.edu says:'
    words = preprocess(str)
    print(words)


def preprocess_run():
    path1 = 'E:\\Project\\DataLab\\20news-18828\\old'
    path2 = 'E:\\Project\\DataLab\\20news-18828\\preprocess'
    dopreprocess(path1, path2)


dictionary = []  # 字典
idf = []  # 计算每个词的IDF

totalcount = 18828  # 所有的文档数目


# 构建字典
def getDictionary(inpath):
    print('正在构建字典...')
    count = 0
    for maindir,subdir,filenamelist in os.walk(inpath):
        for filename in filenamelist:
            apath = os.path.join(maindir, filename)
            reader = open(apath, 'r', encoding='UTF-8')#读取文件
            count = count+1
            while True:
                try:
                    line = reader.readline()
                except:
                    print('\t存在一个读取错误', apath)
                    continue
                if not line:
                    break
                if len(line)==0:
                    continue
                words = line.split(' ')
                for word in words:
                    if word=='\n':
                        # print('有换行符')
                        continue
                    if word not in dictionary:
                        dictionary.append(word)
        print(maindir)
    print('构建字典完成', count)


# 新版的构建字典函数，并在最后将字典输出
def getDictionary2(inpath,outpath):
    print('正在构建字典...')
    for maindir,subdir,filenamelist in os.walk(inpath):
        for filename in filenamelist:
            apath = os.path.join(maindir, filename)
            reader = open(apath, 'r', encoding='UTF-8')  # 读取文件
            while True:
                try:
                    line = reader.readline()
                    # words = nltk.word_tokenize()
                    words = line.split(' ')
                    for word in words:
                        if word=='\n':
                            continue
                        if word not in dictionary:
                            dictionary.append(word)
                except:
                    continue
                if not line:
                    break
            reader.close()
        print(maindir)
    print('构建字典完成')
    filewriter = open(''.join(outpath), 'w',encoding="UTF-8")
    for iword in dictionary:
        filewriter(iword)
        filewriter("\n")
    filewriter.close()


# 统计文档中每个词出现的次数，并储存到文件中
def textVector(inpath,outpath):
    print('统计文档中每个次出现的次数...')
    csvfile = open(outpath,'w',newline='')
    writer = csv.writer(csvfile)
    firstline = dictionary.copy()
    firstline.insert(0,'label')
    firstline.insert(0,'filename')
    writer.writerow(firstline)
    filecount = 0
    for maindir,subdir,filenamelist in os.walk(inpath):
        for filename in filenamelist:
            apath = os.path.join(maindir,filename)
            reader = open(apath,'r',encoding='UTF-8')
            filecount = filecount+1
            v = [0 for i in range(len(dictionary))]
            while True:
                try:
                    line = reader.readline()
                except:
                    print('\t存在一个读取错误')
                    continue
                if not line:
                    break
                if len(line)==0:
                    continue
                words = line.split(' ')
                for word in words:
                    if word=='\n':
                        continue
                    wordindex = dictionary.index(word)
                    v[wordindex] = v[wordindex]+words.count(word)
            v.insert(0,maindir[51:len(maindir)])
            v.insert(0,filename[0:len(filename)-4])
            writer.writerow(v)
            if filecount.__mod__(1000)==0:
                print(filecount)
        print(maindir)
    csvfile.close()
    print('统计文档中单词出现次数成功')


# 计算每个词的TF_IDF
# 要注意一下内存的问题，所以结果要及时保存
def TF_IDF(inpath,outpath):
    textsize = []  # 每篇文章的长度（所含词的个数）
    #wordcount = [];  # 每个词的文档数
    label = []#每篇文档所属的类
    csv_reader = csv.reader(open(inpath,encoding='UTF-8'))
    isfirst = 1
    rowindex = 0
    diclength = 0#字典的长度
    for row in csv_reader:
        diclength = len(row)-2
        break
    wordcount = [0 for _ in range(diclength)]# 每个词的文档数
    nowcount = 0
    for row in csv_reader:
        if isfirst==1:#主要是为了过滤掉词典
            print(len(row))
            isfirst=0
            continue
        label.append(row[1])
        tempdata = row[2:len(row)]
        colindex = 0
        thissize = 0
        for number in tempdata:
            numberint = int(number)
            if numberint>0:
                wordcount[colindex] = wordcount[colindex]+1
                thissize = thissize + numberint
            colindex = colindex+1
        textsize.append(thissize)
        nowcount = nowcount+1
        if nowcount.__mod__(500)==0:
            print(nowcount)
    print(nowcount)
    print('开始计算每个单词的IDF值...')
    nouseword =[]#标记相应的单词是否无用，只在一篇文章中出现的词，对于分类来讲是不起作用的
    for count in wordcount:
        if count<20:#出现次数小于20的词过滤掉
            nouseword.append(1)
        else:
            nouseword.append(0)
        thisidf = math.log(totalcount/(count+1))
        idf.append(thisidf)
    print('开始计算TF-IDF值...')
    csv_reader2 = csv.reader(open(inpath, encoding='UTF-8'))
    outfile = open(outpath,'w',newline='')#输出文件
    writer = csv.writer(outfile)
    isfirst = 1  # 标记是否为第一行，也就是字典行
    rowindex = 0
    nowcount = 0
    for row in csv_reader2:
        if isfirst==1:
            isfirst = 0
            continue
        tempdata = row[2:len(row)]
        v = []
        v.append(label2int(row[1]))
        colindex = 0
        for number in tempdata:
            if nouseword[colindex]==1:  # 表明这个单词对于分类没有作用，只在一篇文章中出现过
                colindex = colindex+1
                continue
            number2 = int(number)
            if number2==0:
                v.append(0)
                colindex = colindex+1
                continue
            avalue = (number2/(textsize[rowindex]+1))*idf[colindex]
            colindex = colindex+1
            v.append(avalue)
        writer.writerow(v)
        rowindex = rowindex+1
        nowcount = nowcount+1
        if nowcount.__mod__(500)==0:
            print(nowcount)
    print(nowcount)
    outfile.close()
    print('已经成功将所有文档转变成向量形式')
    nousecount = 0
    for use in nouseword:
        if use==0:
            nousecount = nousecount+1
    print(nousecount)


# 将文档的类别标签转换为整数形式
def label2int(label):
    intlabel = 0
    if label=='theism':
        intlabel = 1
    elif label=='graphics':
        intlabel = 2
    elif label=='os.ms-windows.misc':
        intlabel = 3
    elif label=='sys.ibm.pc.hardware':
        intlabel = 4
    elif label=='sys.mac.hardware':
        intlabel = 5
    elif label=='windows.x':
        intlabel = 6
    elif label=='forsale':
        intlabel = 7
    elif label=='utos':
        intlabel = 8
    elif label=='otorcycles':
        intlabel = 9
    elif label=='port.baseball':
        intlabel = 10
    elif label=='port.hockey':
        intlabel = 11
    elif label=='rypt':
        intlabel = 12
    elif label=='lectronics':
        intlabel = 13
    elif label=='ed':
        intlabel = 14
    elif label=='pace':
        intlabel = 15
    elif label=='eligion.christian':
        intlabel = 16
    elif label=='politics.guns':
        intlabel = 17
    elif label=='politics.mideast':
        intlabel = 18
    elif label=='politics.misc':
        intlabel = 19
    elif label=='religion.misc':
        intlabel = 20
    return intlabel


def vsm():
    print("vsm")
    readpath = 'E:\\Project\\DataLab\\20news-18828\\preprocess'
    countpath = 'E:\\Project\\DataLab\\20news-18828\\wordcount.csv'
    dictpath = 'E:\\Project\\DataLab\\20news-18828\\dict.txt'
    getDictionary(readpath)
    textVector(readpath, countpath)
    vectorpath = 'E:\\Project\\DataLab\\20news-18828\\vector.csv'
    TF_IDF(countpath, vectorpath)


if __name__ == '__main__':
    # preprocess_run()
    # asentencetest()
    vsm()
