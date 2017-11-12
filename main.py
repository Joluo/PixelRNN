import os
import logging
logging.basicConfig(format="[%(asctime)s] %(message)s", datefmt="%m-%d %H:%M:%S")

import numpy as np
import tensorflow as tf

from pixel_rnn import PixelRNN

flags = tf.app.flags

import helper 

# network
#flags.DEFINE_string("model", "pixel_cnn", "name of model [pixel_rnn, pixel_cnn]")
flags.DEFINE_string("model", "pixel_rnn", "name of model [pixel_rnn, pixel_cnn]")
flags.DEFINE_integer("batch_size", 100, "size of a batch")
flags.DEFINE_integer("hidden_dims", 16, "dimesion of hidden states of LSTM or Conv layers")
flags.DEFINE_integer("recurrent_length", 7, "the length of LSTM or Conv layers")
flags.DEFINE_integer("out_hidden_dims", 32, "dimesion of hidden states of output Conv layers")
flags.DEFINE_integer("out_recurrent_length", 2, "the length of output Conv layers")
flags.DEFINE_boolean("use_residual", False, "whether to use residual connections or not")

# training
flags.DEFINE_integer("max_epoch", 100000, "# of step in an epoch")
flags.DEFINE_integer("test_step", 100, "# of step to test a model")
flags.DEFINE_integer("save_step", 1000, "# of step to save a model")
flags.DEFINE_float("learning_rate", 1e-3, "learning rate")
flags.DEFINE_float("grad_clip", 1, "value of gradient to be used for clipping")
flags.DEFINE_boolean("use_gpu", True, "whether to use gpu for training")

# data
flags.DEFINE_string("data", "mnist", "name of dataset [mnist, cifar]")
flags.DEFINE_string("data_dir", "data", "name of data directory")
flags.DEFINE_string("sample_dir", "samples", "name of sample directory")

flags.DEFINE_string("out_dir", "out", "output directory")

# Debug
flags.DEFINE_boolean("is_train", True, "training or testing")
flags.DEFINE_boolean("display", False, "whether to display the training results or not")
flags.DEFINE_string("log_level", "INFO", "log level [DEBUG, INFO, WARNING, ERROR, CRITICAL]")
flags.DEFINE_integer("random_seed", 123, "random seed for python")

conf = flags.FLAGS

# logging
logger = logging.getLogger()
logger.setLevel(conf.log_level)

# random seed
tf.set_random_seed(conf.random_seed)
np.random.seed(conf.random_seed)

def main(_):
    helper.check_and_create_dir(conf.out_dir)

    # 0. prepare datasets
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('../MNIST_data', one_hot=True)
    train_step_per_epoch = mnist.train.num_examples / conf.batch_size

    height, width, channel = 28, 28, 1

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False)
        session_conf.gpu_options.allow_growth = True
        with tf.Session(config=session_conf) as sess:
            pixel_rnn = PixelRNN(sess, conf, height, width, channel)
            sess.run(tf.global_variables_initializer())
            logger.info("Training starts!")

            for epoch in range(conf.max_epoch):
                total_train_costs = []
                for idx in range(int(train_step_per_epoch)):
                    x_mb, _ = mnist.train.next_batch(conf.batch_size)
                    x_mb = x_mb.reshape([conf.batch_size, height, width, channel])
                    x_mb = (np.random.uniform(size=x_mb.shape) < x_mb).astype('float32')
                    cost = pixel_rnn.test(x_mb, with_update=True)
                    total_train_costs.append(cost)

                    if idx % 100 == 0:
                        avg_train_cost = np.mean(total_train_costs)
                        samples = pixel_rnn.generate()
                        helper.save_images(samples, height, width, 10, 10, dir=conf.out_dir)
                        print("Epoch: %d, iteration:%d, train l: %f" % (epoch, idx, avg_train_cost))


if __name__ == "__main__":
    tf.app.run()
