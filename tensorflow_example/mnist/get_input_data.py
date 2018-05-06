from tensorflow.examples.tutorials.mnist import input_data
import os


def load_data():
    path = os.path.abspath("dataset")
    mnist = input_data.read_data_sets(path, one_hot=True)
    assert isinstance(mnist, object)
    return mnist
