import tensorflow as tf


class Mnist:
    def __init__(self):
        self.INPUT_NODE = 784
        self.OUT_NODE = 10
        self.LAYER1_NODE = 500
        self.batch_size = 100
        self.learning_rate_base = 0.8  # base of learning rate
        self.learning_rate_decay = 0.99  # decay of learning rate
        self.regularization_rate = 0.0001
        self.training_step = 30000
        self.moving_average_rate = 0.99

    def inference(self, input_tensor, avg_class, l1_weight, l2_weight, l1_bais, l2_bais):
        if avg_class is None:
            layer1 = tf.nn.relu(tf.matmul(input_tensor, l1_weight) + l1_bais)

            return tf.nn.relu(tf.matmul(layer1, l2_weight) + l2_bais)
        else:
            layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(l1_weight)) + avg_class.average(l1_bais))

            return tf.nn.relu(tf.matmul(layer1, avg_class.average(l2_weight)) + avg_class.average(l2_bais))

    def train(self):

        x = tf.placeholder(tf.float32, [None, self.INPUT_NODE], name='x_input')
        y_ = tf.placeholder(tf.float32, [None, self.OUT_NODE], name='y_output')

        weight1 = tf.Variable(tf.truncated_normal([self.INPUT_NODE, self.LAYER1_NODE], stddev=0.1))
        bais1 = tf.Variable(tf.constant(0.1, shape=[self.LAYER1_NODE]))

        weight2 = tf.Variable(tf.truncated_normal([self.LAYER1_NODE, self.OUT_NODE], stddev=0.1))
        bais2 = tf.Variable(tf.constant(0.1, shape=[self.OUT_NODE]))

        y = self.inference(x, None, weight1, weight2, bais1, bais2)

        global_step = tf.Variable(0, trainable=False)

        variable_averages = tf.train.ExponentialMovingAverage(self.moving_average_rate, global_step)

        #TODO
