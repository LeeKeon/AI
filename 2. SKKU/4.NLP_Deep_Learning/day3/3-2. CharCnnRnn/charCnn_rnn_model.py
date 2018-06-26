# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.contrib import rnn

# Write your code here
class Model(object):
    def __init__(self, max_seq_len=200, max_word_len=20,
                 emb_dim=128, rnn_hidden_dim=128,
                 filter_sizes=[1], filter_nums=[100],
                 vocab_size=1000, char_size=50,
                 use_clip=True, learning_rate=0.001):

        self.initializer = tf.random_uniform_initializer(-0.05, 0.05)
        self.max_seq_len = max_seq_len
        self.max_word_len = max_word_len

        self.emb_dim = emb_dim
        self.rnn_hidden_dim = rnn_hidden_dim

        self.filter_sizes = filter_sizes
        self.filter_nums = filter_nums

        self.vocab_size = vocab_size
        self.char_size = char_size

        self.use_clip = use_clip
        self.learning_rate = learning_rate

        self.x_char = tf.placeholder(dtype=tf.int32, shape=(None, self.max_seq_len, self.max_word_len))
        self.x_word = tf.placeholder(dtype=tf.int32, shape=(None, self.max_seq_len))

        self.x_len = tf.placeholder(dtype=tf.int32, shape=(None, ))
        self.x_len = self.x_len - 1

        mask = tf.sequence_mask(lengths=self.x_len, maxlen=self.max_seq_len-1, dtype=tf.float32)
        self.mask_1d = tf.reshape(mask, [-1])

        self.target = self.x_word[:, 1:]
        self.keep_prob = tf.placeholder_with_default(1.0, shape=None)

        self.embW_char = self.get_var(name="emb_char", shape=[self.char_size, self.emb_dim])
        self.embW_word = self.get_var(name="emb_word", shape=[self.vocab_size, self.emb_dim])

        self.batch_size = tf.shape(self.x_char)[0]
        self.x_char_emb = tf.nn.embedding_lookup(self.embW_char, self.x_char[:, :-1])
        self.x_word_emb = tf.nn.embedding_lookup(self.embW_word, self.x_word[:, :-1])

        self.build_model()
        self.build_loss()
        self.build_opt()

    def build_model(self):
        # Dimension change for conv net
        print(self.x_char_emb)
        print(self.x_word_emb)
        x_char_4d = tf.reshape(self.x_char_emb, [-1, self.max_word_len, self.emb_dim, 1])
        print(x_char_4d)
        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for filter_size, filter_num in zip(self.filter_sizes, self.filter_nums):
            # Seperated name scope for variables (W, b) for each filters
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, self.emb_dim, 1, filter_num]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[filter_num]), name="b")

                # Apply nonlinearity
                conv = tf.nn.conv2d(x_char_4d, W, strides=[1, 1, 1, 1], padding="VALID", name="conv")
                h = self.leaky_relu(conv + b)

                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(h, ksize=[1, self.max_word_len - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1], padding='VALID', name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = sum(self.filter_nums)
        h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_3d = tf.reshape(h_pool, [-1, self.max_seq_len-1, num_filters_total])

        # vector of word & character
        self.word_char = tf.concat([self.x_word_emb, self.h_pool_3d], 2)

        # RNN for word+char
        lstm_cell = rnn.BasicLSTMCell(self.rnn_hidden_dim)
        output, states = tf.nn.dynamic_rnn(cell=lstm_cell, inputs=self.word_char,
                                           sequence_length=self.x_len, dtype=tf.float32)
        output_2d = tf.reshape(output, [-1, self.rnn_hidden_dim])

        out_W = self.get_var(name="out_W", shape=[self.rnn_hidden_dim, self.vocab_size])
        out_b = self.get_var(name="out_b", shape=[self.vocab_size])
        self.word_prob = tf.nn.softmax(tf.matmul(output_2d, out_W) + out_b)
        self.output = tf.argmax(tf.reshape(self.word_prob, [-1, self.max_seq_len-1, self.vocab_size]), 2)

    def build_loss(self):
        self.cross_entropy = -tf.reduce_sum(
            tf.one_hot(tf.to_int32(tf.reshape(self.target, [-1])), self.vocab_size, 1.0, 0.0)
            * tf.log(tf.clip_by_value(tf.reshape(self.word_prob, [-1, self.vocab_size]), 1e-20, 1.0)), 1)
        self.loss = tf.reduce_sum(self.cross_entropy * self.mask_1d) / tf.reduce_sum(self.mask_1d)

    def build_opt(self):
        # define optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        grad, var = zip(*optimizer.compute_gradients(self.loss))

        # gradient clipping
        def clipped_grad(grad):
            return [None if g is None else tf.clip_by_norm(g, 2.5) for g in grad]

        if self.use_clip:
            grad = clipped_grad(grad)

        self.update = optimizer.apply_gradients(zip(grad, var))

    def leaky_relu(self, x):
        return tf.maximum((x), 0.1*(x))

    def get_var(self, name='', shape=None, dtype=tf.float32):
        return tf.get_variable(name, shape, dtype=dtype, initializer=self.initializer)

    def save(self, sess, global_step=None):
        var_list = [var for var in tf.all_variables()]
        saver = tf.train.Saver(var_list)
        save_path = saver.save(sess, save_path="models/model", global_step=global_step)
        print(' * model saved at \'{}\''.format(save_path))

    # Load whole weights
    def restore(self, sess):
        print(' - Restoring variables...')
        var_list = [var for var in tf.all_variables()]
        saver = tf.train.Saver(var_list)
        saver.restore(sess, "models/model")
        print(' * model restored ')