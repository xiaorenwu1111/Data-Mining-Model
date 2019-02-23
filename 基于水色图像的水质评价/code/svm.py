#-*- coding: utf-8 -*-
import pandas as pd

inputfile = '../data/moment.csv' 
outputfile1 = '../tmp/cm_train.xls' 
outputfile2 = '../tmp/cm_test.xls' 
data = pd.read_csv(inputfile, encoding = 'gbk') #读取数据，指定编码为gbk
data = data.as_matrix()

from numpy.random import shuffle #引入随机函数
shuffle(data) #随机打乱数据
data_train = data[:int(0.8*len(data)), :] #选取训练集
data_test = data[int(0.8*len(data)):, :] #选取测试数据集

#构造特征和标签
x_train = data_train[:, 2:]*30 #放大数据特征，因为原来所有特征的值都在[0,1]之间，svm区分不明显，乘以系数k，放大数据特征
y_train = data_train[:, 0].astype(int) #系数k是通过反复测试得出的最佳结果
x_test = data_test[:, 2:]*30
y_test = data_test[:, 0].astype(int)

#导入模型相关的函数建立模型
from sklearn import svm
model = svm.SVC()
model.fit(x_train, y_train)
import pickle
pickle.dump(model, open('../tmp/svm.model', 'wb'))
#保存模型，以后可以通过下面的语句重新加载模型
#model = pickle.load(open('../tmp/svm.model', 'rb'))

#导入相关库，生成混淆矩阵
from sklearn import metrics
cm_train = metrics.confusion_matrix(y_train, model.predict(x_train)) #训练样本的混淆矩阵
cm_test = metrics.confusion_matrix(y_test, model.predict(x_test)) #测试样本的混淆矩阵

#保存结果
pd.DataFrame(cm_train, index = range(1, 6), columns = range(1, 6)).to_excel(outputfile1)
pd.DataFrame(cm_test, index = range(1, 6), columns = range(1, 6)).to_excel(outputfile2)
