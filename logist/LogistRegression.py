from matplotlib import pyplot as plt
from logist.load_data import load_data
import numpy as np


def plot_best_fit(weights):
    data_mat, label_mat = load_data()
    data_arr = np.array(data_mat)
    n = np.shape(data_arr)[0]  # the shape of feature
    x_cord1 = [];
    y_cord1 = []
    x_cord2 = [];
    y_cord2 = []
    for i in range(n):
        if int(label_mat[i]) == 1:
            x_cord1.append(data_arr[i, 1])
            y_cord1.append(data_arr[i, 2])
        else:
            x_cord2.append(data_arr[i, 1])
            y_cord2.append(data_arr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x_cord1, y_cord1, s=30, c='red', marker='s')
    ax.scatter(x_cord2, y_cord2, s=30, c='green')
    if weights is not None:
        y = (-weights[0] - weights[1] * x) / weights[2]
        ax.plot(x, y)
    plt.xlabel('X1');
    plt.ylabel('X2');
    plt.show()


plot_best_fit(None)
