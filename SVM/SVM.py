# -*- coding:utf-8 -*-
"""使用PCA降维算法"""

import matplotlib
import matplotlib.pyplot as plt
import mlpy
import matplotlib.cm as cm
import numpy as np

wine=np.loadtxt("/Users/hanzhao/PycharmProjects/MLstudy/file/wine.data",delimiter=',')

"""定义特征值"""
x,y=wine[:,1:4],wine[:,0].astype(np.int)
print(x.shape)
print(y.shape)

"""创建新的PCA实例，并根据x轴中选择的特征值来选择相应的算法"""
pca=mlpy.PCA()
pca.learn(x)

"""对x轴特征值进行降维，转化为二维子空间"""

z=pca.transform(x,k=2)
print (z.shape)

fig1=plt.figure(1)
title=plt.title("PCA on wine dataset")
plot = plt.scatter(z[:, 0], z[:, 1], c=y,s=90, cmap = cm.Reds)
labx = plt.xlabel("First component")
laby = plt.ylabel("Second component")
plt.show()

