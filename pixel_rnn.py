import tensorflow as tf
from utils import *


class PixelRNN:
    def __init__(self, sess, height, width, channel, hidden_dim, grad_clip, model):
        self.sess = sess
        self.height = height
        self.width = width
        self.channel = channel
        self.hidden_dim = hidden_dim
        self.grad_clip = grad_clip
        self.model = model
        
        self.inputs = tf.placeholder(tf.float32, [None, height, width, channel])

        output = conv2d(self.inputs, self.channel, hidden_dim, 7, 'a', 'input_conv')

        if self.model == 'pixel_rnn':
            lstm1_output = diagonal_bilstm(output, hidden_dim, hidden_dim, 'LSTM1')
            lstm2_output = diagonal_bilstm(lstm1_output, hidden_dim, hidden_dim, 'LSTM2')
            output = lstm2_output
        elif self.model == 'pixel_cnn':
            #TODO
            pass

        output = conv2d(output, hidden_dim, hidden_dim, 1, 'b', 'output_conv1', he_init=True)
        output = tf.nn.relu(output)
        
        output = conv2d(output, hidden_dim, hidden_dim, 1, 'b', 'output_conv2', he_init=True)
        output = tf.nn.relu(output)

        logits = conv2d(output, hidden_dim, 1, 1, 'b', 'output_conv3')
        self.output = tf.nn.sigmoid(logits)

        #self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        #    logits=logits, labels=self.inputs, name='loss'))
        #self.loss = tf.reduce_mean(tf.contrib.keras.metrics.binary_crossentropy(self.inputs, self.output))
        self.loss = tf.reduce_mean(-(tf.multiply(self.inputs, tf.log(self.output)) + tf.multiply(1-self.inputs, tf.log(1-self.output))))

        #optimizer = tf.train.RMSPropOptimizer(1e-3)
        optimizer = tf.train.AdamOptimizer(5e-3)
        self.grads_and_vars = optimizer.compute_gradients(self.loss)

        self.new_grads_and_vars = \
            [(tf.clip_by_value(gv[0], -self.grad_clip, self.grad_clip), gv[1]) for gv in self.grads_and_vars]
        self.optim = optimizer.apply_gradients(self.new_grads_and_vars)

        show_all_variables()
        
    def predict(self, images):
        return self.sess.run(self.output, {self.inputs: images})

    def generate(self):
        samples = np.zeros((100, self.height, self.width, 1), dtype='float32')
        for i in range(self.height):
            for j in range(self.width):
                for k in range(self.channel):
                    next_sample = binarize(self.predict(samples))
                    samples[:, i, j, k] = next_sample[:, i, j, k]
        return samples
