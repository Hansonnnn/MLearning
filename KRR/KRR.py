#coding utf-8
import numpy as np
import mlpy
from pylab import *
from mlpy import KernelRidge
import matplotlib.pyplot as plt
import dateutil.parser as dparser

np.random.seed(10)

tartgetValue=np.genfromtxt("/Users/hanzhao/PycharmProjects/MLstudy/file/Gold.csv",
                           skip_header=1,
                           dtype=None,
                           delimiter=',',
                           usecols=(1))


trainingPoints=np.arange(125).reshape(-1,1)
testPoints=np.arange(126).reshape(-1,1)

##kernel matrix
knl=mlpy.kernel_gaussion(trainingPoints, trainingPoints, sigma=1)
knlTest=mlpy.kernel_gaussion(testPoints,testPoints,sigma=1)

knlRidge=KernelRidge(lmb=0.01,kernel=None)
knlRidge.learn(knl,tartgetValue)
