from matplotlib import pyplot as plt
from matplotlib import animation
from logist.load_data import load_data
import numpy as np
import random


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
        x = np.arange(-3.0, 3.0, 0.1)
        y = (-weights[0] - weights[1] * x) / weights[2]
        ax.plot(x, y)
    plt.xlabel('X1');
    plt.ylabel('X2');
    plt.show()


def sigmoid(inx):
    return 1.0 / (1 + np.exp(-inx))


def grad_ascent(data_mat_in, class_labels):
    """@description the batch grad ascent of logist function
       @param data_mat_in the input of X matrix
       @param class_labels the output of Y matrix
       
       
       repeat {
       
        Wj := Wj + alpha * (y - f(x)) * Xj
       }
       """

    data_matrix = np.mat(data_mat_in)
    label_mat = np.mat(class_labels).transpose()
    m, n = np.shape(data_matrix)
    alpha = 0.003  # the learning speed
    max_cycles = 500
    weights = np.zeros((n, 1))
    for k in range(max_cycles):
        predict = sigmoid(data_matrix * weights)
        cost = (label_mat - predict)
        weights += alpha * data_matrix.transpose() * cost
    return weights


def soc_grad_ascent(data_mat_in, class_label):
    """@description the stochastic grad ascent of logist function 
        the main idea of algrithem:
        step 1 : init every regression's weight with 1
        step 2 : for every sample:
                     caculate sample's gradient
                     alpha * gradient and update weight
    
    """
    data_matrix = np.array(data_mat_in)
    m, n = np.shape(data_matrix)
    alpha = 0.007  # the learning speed
    weights = np.ones(n)
    for i in range(m):
        h = sigmoid(data_matrix[i] * weights)
        error = (class_label[i] - h)
        weights += alpha * data_matrix[i] * error
    return weights


def soc_grad_ascent1(data_mat_in, class_label, iter_nums=150):
    """@description the stochastic grad ascent with logist function after modify 
       @param data_mat_in the input of X matrix
       @param class_label the output of Y matrix
       @param iter_nums the bumbers of iterator
       
       loop{
       
         for i in range(m)
            
          Wj := Wj + alpha * (y - f(x)) * Xj
       
       }    
       
       """
    data_matrix = np.array(data_mat_in)
    m, n = np.shape(data_matrix)
    weights = np.ones(n)
    for j in range(iter_nums):
        data_index = range(m)
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.001
            rand_index = int(random.uniform(0, len(data_index)))
            h = sigmoid(sum(data_matrix[rand_index] * weights))
            error = class_label[rand_index] - h
            weights = weights + alpha * data_matrix[rand_index] * error
            np.delete(data_matrix[rand_index])
    return weights


data_arr, label_mat = load_data()
weights = grad_ascent(data_arr, label_mat)

plot_best_fit(weights)
