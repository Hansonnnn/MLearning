import tensorflow as tf
from tensorflow_example.mnist.get_input_data import load_data


def inference(input_tensor, avg_class, l1_weight, l2_weight, l1_bais, l2_bais):
    if avg_class is None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, l1_weight) + l1_bais)

        return tf.nn.relu(tf.matmul(layer1, l2_weight) + l2_bais)
    else:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(l1_weight)) + avg_class.average(l1_bais))

        return tf.nn.relu(tf.matmul(layer1, avg_class.average(l2_weight)) + avg_class.average(l2_bais))


class MnistClass:
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

    def train(self, mnist):
        x = tf.placeholder(tf.float32, [None, self.INPUT_NODE], name='x_input')
        y_ = tf.placeholder(tf.float32, [None, self.OUT_NODE], name='y_output')

        weight1 = tf.Variable(tf.truncated_normal([self.INPUT_NODE, self.LAYER1_NODE], stddev=0.1))
        bais1 = tf.Variable(tf.constant(0.1, shape=[self.LAYER1_NODE]))

        weight2 = tf.Variable(tf.truncated_normal([self.LAYER1_NODE, self.OUT_NODE], stddev=0.1))
        bais2 = tf.Variable(tf.constant(0.1, shape=[self.OUT_NODE]))

        y = inference(x, None, weight1, weight2, bais1, bais2)

        global_step = tf.Variable(0, trainable=False)

        variable_averages = tf.train.ExponentialMovingAverage(self.moving_average_rate, global_step)

        variable_averages_op = variable_averages.apply(tf.trainable_variables())

        average_y = inference(x, variable_averages, weight1, weight2, bais1, bais2)

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=tf.arg_max(y_, 1))
        cross_entropy_mean = tf.reduce_mean(cross_entropy)

        regularizer = tf.contrib.layers.l2_regularizer(self.regularization_rate)
        regularization = regularizer(weight1) + regularizer(weight2)

        loss = cross_entropy_mean + regularization  # the method with L2 plus cv to optimize our model

        decay_step = mnist.train.num_examples / self.batch_size
        learning_rate = tf.train.exponential_decay(self.learning_rate_base, global_step, decay_step,
                                                   self.learning_rate_decay)

        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

        with tf.control_dependencies([train_step, variable_averages_op]):
            train_op = tf.no_op(name='train')

        correct_predict = tf.equal(tf.arg_max(average_y, 1), tf.arg_max(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))

        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}

            test_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
            for i in range(self.training_step):
                if i % 1000 == 0:
                    validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                    print(
                        "After %d training step(s) , validation accuracy using average model is %g" % (i, validate_acc))
                    xs, ys = mnist.train.next_batch(self.batch_size)
                    sess.run(train_op, feed_dict={x: xs, y_: ys})
            test_acc = sess.run(accuracy, feed_dict=test_feed)
            print(
                "After %d training step(s) , test accuracy using average model is %g" % (i, test_acc))


def main(argv=None):
    mnist = load_data()
    mc = MnistClass()
    mc.train(mnist)


if __name__ == "__main__":
    tf.app.run()
