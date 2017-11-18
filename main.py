import os
import logging
logging.basicConfig(format="[%(asctime)s] %(message)s", datefmt="%m-%d %H:%M:%S")

import numpy as np
import tensorflow as tf

from pixel_rnn import PixelRNN

flags = tf.app.flags

import helper 


def main():
    out_dir = './out'
    helper.check_and_create_dir(out_dir)
    batch_size = 64
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('../MNIST_data', one_hot=True)
    train_step_per_epoch = mnist.train.num_examples / batch_size

    height, width, channel = 28, 28, 1

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False)
        session_conf.gpu_options.allow_growth = True
        sess = tf.Session(config=session_conf) 
        with sess.as_default(), tf.device('/gpu:0'):
            pixel_rnn = PixelRNN(sess, 28, 28, 1, 32, 1., 'pixel_rnn')
            sess.run(tf.global_variables_initializer())

            for epoch in range(100000):
                total_train_costs = []
                for idx in range(int(train_step_per_epoch)):
                    x_mb, _ = mnist.train.next_batch(batch_size)
                    x_mb = x_mb.reshape([batch_size, height, width, channel])
                    x_mb = (np.random.uniform(size=x_mb.shape) < x_mb).astype('float32')
                    _, cost, outputs = sess.run([pixel_rnn.optim, pixel_rnn.loss, pixel_rnn.output], feed_dict={pixel_rnn.inputs:x_mb})
                    total_train_costs.append(cost)
                    test_mb, _ = mnist.test.next_batch(batch_size)
                    test_mb = test_mb.reshape([batch_size, height, width, channel])
                    test_mb = (np.random.uniform(size=test_mb.shape) < test_mb).astype('float32')
                    test_cost = sess.run(pixel_rnn.loss, feed_dict={pixel_rnn.inputs:test_mb})
                    print("Epoch: %d, iteration:%d, train loss: %f, test loss: %f" % (epoch, idx, cost, test_cost))

                    if idx % 100 == 0:
                        avg_train_cost = np.mean(total_train_costs)
                        #print("Epoch: %d, iteration:%d, train l: %f" % (epoch, idx, cost))
                        #print("Epoch: %d, iteration:%d, train l: %f" % (epoch, idx, avg_train_cost))
                        samples = pixel_rnn.generate()
                        helper.save_images(samples, height, width, 10, 10, dir=out_dir, prefix='test_')
                avg_train_cost = np.mean(total_train_costs)
                samples = pixel_rnn.generate()
                helper.save_images(samples, height, width, 10, 10, dir=out_dir, prefix='test_')
                print("Epoch: %d, iteration:%d, train l: %f" % (epoch, idx, avg_train_cost))


if __name__ == "__main__":
    main()
