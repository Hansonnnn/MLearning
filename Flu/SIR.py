# -*- coding:utf-8 -*-
import scipy
import scipy.integrate
import pylab as plt

"""转化概率"""
beta = 0.003
"""感染周期"""
gamma = 0.1

def SIR_model(X,t=0):
    """X[0]代表易染人群,X[1]代表感染人群"""
    r=scipy.array([- beta*X[0]*X[1],beta*X[0]*X[1] - gamma*X[1],gamma*X[1]])
    return r


if __name__=="__main__":

    time=scipy.linspace(0,60,num=100)
    parameters=scipy.array([225,1,0])
    X=scipy.integrate.odeint(SIR_model,parameters,time)
    """三种趋势分别代表SRI三种人群的不同状态可视化"""

    plt.plot(range(0,100),X[:,0],'o',color="green")
    plt.plot(range(0,100),X[:,1],'x',color="red")
    plt.plot(range(0,100),X[:,2],'*',color="blue")
    plt.show()

