import random
from matplotlib import pyplot as plt
import numpy as np


# def SGD(self, training_data, epochs, mini_batch_size, eta,
#         test_data=None):
#     """Train the neural network using mini-batch stochastic
#     gradient descent.  The "training_data" is a list of tuples
#     "(x, y)" representing the training inputs and the desired
#     outputs.  The other non-optional parameters are
#     self-explanatory.  If "test_data" is provided then the
#     network will be evaluated against the test data after each
#     epoch, and partial progress printed out.  This is useful for
#     tracking progress, but slows things down substantially."""
#     if test_data: n_test = len(test_data)
#     n = len(training_data)
#     for j in range(epochs):
#         random.shuffle(training_data)
#         mini_batches = [
#             training_data[k:k + mini_batch_size]
#             for k in range(0, n, mini_batch_size)]
#         for mini_batch in mini_batches:
#             self.update_mini_batch(mini_batch, eta)
#         if test_data:
#             print("Epoch {0}: {1} / {2}".format(
#                 j, self.evaluate(test_data), n_test))
#
#         else:
#             print("Epoch {0} complete".format(j))
# Wrangle Data
n = 30000  # number of steps
m = 1000  # Samples per step
k = 2  # number of features
dims = {'numExperiments': 0, 'numSamps': 1}
class1 = np.random.uniform(10, 20, (k, m))
class2 = np.random.uniform(5, 15, (k, m))
x = np.concatenate((class1, class2), axis=1)
y = np.concatenate((np.ones((1, m)), np.zeros((1, m))), axis=1)

plt.show()
fig = plt.figure()
plt.title('x vs. y')
plt.ylabel('y')
plt.xlabel('x')
plt.scatter(class1[1], class1[0], label='Class 1')
plt.scatter(class2[1], class2[0], label='Class 2')
plt.legend()

eps = 1 / (10 ** 7)  # for numerical stability

# Initialize Model
weights = np.random.randn(k, 1)  # Initialize the weights to a random value
bias = 0  # Initialize bias parameter to 0
learningRate = .01
biasLog = []
weightLog1 = []
weightLog2 = []
costLog = []

for i in range(n):
    biasLog.append(bias)
    weightLog1.append(weights[0])
    weightLog2.append(weights[1])
    # Compute the model estimates recall z = w'x+b a = sigma(z)
    z = np.dot(weights.transpose(), x) + bias
    a = sigmoid(z)

    # Compute the Updates
    J = (y * np.log(a) + (1.0 - y) * np.log(1 - a))
    costLog.append(np.mean(J, axis=1))
    dyhat = (-y / (a + (eps))) + (1. - y) / ((1. - a) + (eps))
    dz = a * (1. - a)
    dwn = x

    dw = np.mean(dyhat * dz * x, axis=1, keepdims=True)
    db = np.mean(dyhat * dz, axis=1, keepdims=True)

    # Apply Updates
    weights = weights - learningRate * dw
    bias = bias - learningRate * db


def prediction(weights, bias, x):
    z = np.dot(weights.transpose(), x) + bias
    a = sigmoid(z) > .5
    return (a)


fig2 = plt.figure()
plt.title('Weights And Bias vs Iterations')
plt.ylabel('Weights')
plt.xlabel('Iterations')
plt.plot(range(n), biasLog, label='Bias')
plt.plot(range(n), weightLog1, label='Weight 1')
plt.plot(range(n), weightLog2, label='Weight 2')
plt.legend()
plt.show()

plt.title('Cost vs Iterations')
plt.ylabel('Cost')
plt.xlabel('Iterations')
plt.plot(range(n), costLog, label='Bias')
plt.legend()
plt.show()


def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)
    plt.show()


plot_decision_boundary(lambda xin: prediction(weights, bias, xin.T), x, y)



