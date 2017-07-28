### 基本介绍

##### TensorFlow

* 使用Graph计算
* 在会话(session)的上下文(context)中执行图(Graph)
* 使用tensor(张量)表示数据
* 通过变量(variable)维持状态
* 使用 feed 和 fetch 可以为任意的操作(arbitrary operation) 赋值或者从其中获取数据


###### 概述

TensorFlow使用图(Graph)来表示计算，图中的每个节点称为op(operation),每个op分配0个或者多个tensor(tensor在tensorflow中是一个类型化的多维数组)，执行计算后，产生0个或多个tensor。

图必须在会话中执行，会话将图中op上的tensor分发到各个cpu或gpu上，并且执行op的方法。方法执行后返回相应的tensor。

###### Tensor

TensorFlow中tensor代表所有的数据，在graph中，operation之间传递的都是tensor，可以将tensor理解为n维的数组或列表，一个静态的tensor包括了一个静态的rank，一个shape。

###### 变量

变量在TensorFlow中存放更新参数，在建立模型的时候需要将其显示初始化。在模型训练结束后需要将其存储。


```
# 构造op，并将其存放到变量当中
b = tf.Variable(tf.zeros([1]))
W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
y = tf.matmul(W, x_data) + b

# 初始化变量
init = tf.initialize_all_variables()

```
###### Fetch

取出计算结果tensor，可以取一个，也可以取出多个tensor。

###### Feed