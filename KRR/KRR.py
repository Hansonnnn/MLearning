#coding utf-8
import numpy as np
from mlpy import KernelGaussian
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
kg=KernelGaussian()
knl=kg.kernel(trainingPoints,trainingPoints)
knlTest=kg.kernel(testPoints,trainingPoints)

knlRidge=KernelRidge(lmb=0.01,kernel=None)
knlRidge.learn(knl,tartgetValue)
