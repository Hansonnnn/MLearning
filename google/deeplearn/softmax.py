scores = [3.0,1.0,0.2]

import numpy as np

def softmax(score):


    return np.exp(score)/np.sum(np.exp(score),axis=0)
print (softmax(scores))

import matplotlib.pyplot as plt
x = np.arange(-0.2,6.0,0.1)
scores = np.vstack([x,np.ones_like(x),0.2 * np.ones_like(x)])
plt.plot(x,softmax(scores).T,linewidth=2)
plt.show()