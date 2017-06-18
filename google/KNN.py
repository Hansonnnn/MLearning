# -*- coding:utf-8 -*-

"""use knn """
from scipy.spatial import distance
"""计算两点距离"""
def euc(a,b):
    return distance.euclidean(a,b)


class ScrappyKNN():

    def fit(self,x_train,y_train):

        self.x_train = x_train
        self.y_train = y_train

    def predict(self,x_test):
        predict = []
        for row in x_test:
            label = self.closest(row)
            predict.append(label)
        return predict

    """计算最近相邻点，row代表x，输出即为y"""
    def closest(self,row):
        best_dist = euc(row,self.x_train[0])#标记目前位置发现的最短距离，取第一个训练集为默认
        best_index = 0
        for i in range(1,len(self.x_train)):
            dist = euc(row,self.x_train[i])#计算每个训练集中点与训练集点的距离
            if dist < best_dist:
                best_dist = dist
                best_index = i
        return self.y_train[best_index]




from sklearn import datasets
iris = datasets.load_iris()

x = iris.data# features
y = iris.target# label

"""数据集被分成两部分，训练集和测试集"""
from sklearn.cross_validation import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.5)

"""使用自定义实现的KNN分类器"""
my_classifer = ScrappyKNN()
my_classifer.fit(x_train, y_train)
predict = my_classifer.predict(x_test)

"""使用测试集计算决策树分类器预测准确率"""
from sklearn.metrics import accuracy_score
print accuracy_score(y_test, predict)