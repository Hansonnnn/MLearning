import tensorflow as tf

"""manage variable with scope"""

with tf.variable_scope("home"):
    v = tf.get_variable("father", [1], initializer=tf.constant_initializer(1.0))
    print(v.name)

"""if tf.variable_scope use parameter 'reuse' and set to True ,
   we can get same variable from one variable scope """
with tf.variable_scope("home", reuse=True):
    v1 = tf.get_variable("father", [1])
    print(v1 == v)
