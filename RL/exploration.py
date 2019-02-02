import tensorflow as tf
import numpy as np


class Exploration(object):

    def __init__(self, action_dim, kernel_num, sample_size):

        self.g = tf.Graph()
        with self.g.as_default():
            # data format
            self.action_dim = action_dim
            self.mean = tf.placeholder(shape=[self.action_dim], dtype=tf.float32)
            self.stddev = tf.placeholder(shape=[self.action_dim], dtype=tf.float32)
            self.action = tf.placeholder(shape=[self.action_dim], dtype=tf.float32)
            self.weight = tf.placeholder(dtype=tf.float32)

            self.gaussian_exploration = None
            self.kernel_num = kernel_num
            self._sample_size = sample_size

            config = tf.ConfigProto(device_count={"CPU": self.kernel_num},
                                    inter_op_parallelism_threads=0,
                                    intra_op_parallelism_threads=0,
                                    log_device_placement=True)
            self.sess = tf.Session(config=config, graph=self.g)

            # for sample_index in range(self._sample_size):
            #     gaussian_noise = tf.random_normal(shape=[self.action_dim], mean=self.mean, stddev=self.stddev)
            #     self.gaussian_exploration.append(self.action + self.weight * gaussian_noise)
            gaussian_noise = tf.random_normal(shape=[self.action_dim], mean=self.mean, stddev=self.stddev)
            self.gaussian_exploration = self.action + self.weight * gaussian_noise

    def get_gaussian_exploration(self, action, mean, stddev, weight=0.01):
        return self.sess.run(self.gaussian_exploration, feed_dict={self.action: action,
                                                                   self.mean: mean,
                                                                   self.stddev: stddev,
                                                                   self.weight: weight})


class Histogram(object):
    def __init__(self, csv_path):
        self.df = np.array(pd.read_csv(csv_path, header=None)).astype('float')
        self.threshold = self.df[:, -1]

    def get_probability(self, x):
        # print('value'+str(self.df[np.arange(len(x[:-1])).astype('int'), (x[:-1] * 20).astype('int')]))
        prob = np.array(
            self.df[np.arange(len(x[:-1])).astype('int'), (x[:-1] * 20).astype('int')[:]] > self.threshold).astype(
            'int')
        return prob
