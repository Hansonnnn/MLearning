# -*- coding:utf-8 -*-
from sklearn import datasets
iris = datasets.load_iris()

x = iris.data
y = iris.target

"""数据集被分成两部分，训练集和测试集"""
from sklearn.cross_validation import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size= .5)

from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier()
clf.fit(x_train,y_train)
predict = clf.predict(x_test)

"""使用测试集计算决策树分类器预测准确率"""
from sklearn.metrics import accuracy_score
print accuracy_score(y_test,predict)

