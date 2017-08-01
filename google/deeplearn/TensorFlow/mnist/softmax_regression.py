# -*- coding:utf-8 -*-
import tensorflow as tf

from google.deeplearn.TensorFlow.mnist import input_data

mnist = input_data.read_data_sets("/Users/hanzhao/Downloads/MNIST_data/", one_hot=True)

"""x不是一个特定的值，而是一个占位符placeholder，TensorFlow运行计算时输入这个值。
   并且希望能够输入任意数量的MNIST图像，每一张图展平成784维的向量。
   然后用2维的浮点数张量来表示这些图，这个张量的形状是[None，784 ]"""
x = tf.placeholder("float", [None, 784])

"""使用tersorflow中的varaible来表示softmax模型当中的权重以及偏置量"""
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))


"""使用tensorflow建立softmax模型"""
y = tf.nn.softmax(tf.matmul(x,W) + b)

"""评估模型"""
"""用于计算成本函数'交叉熵（cross-entropy）'"""
y_ = tf.placeholder("float", [None,10])

"""计算'交叉熵'"""
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

"""用梯度下降算法（gradient descent algorithm）以0.01的学习速率最小化交叉熵"""
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print (sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))