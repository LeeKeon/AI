# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.contrib import rnn

# Write your code here
class Model(object):
    def __init__(self, max_len=200, emb_dim=128, hidden_dim=128, vocab_size=10000,
                 class_size=4, use_clip=True, learning_rate=0.01, end_token="<eos>"):
        self.initializer = tf.random_uniform_initializer(-0.1, 0.1)
        self.max_len = max_len
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.end_token = end_token
        self.layers = []

        self.vocab_size = vocab_size
        self.class_size = class_size
        self.use_clip = use_clip
        self.learning_rate = learning_rate

        self.keep_prob = tf.placeholder_with_default(1.0, shape=None)
        self.is_training = tf.placeholder(dtype=tf.bool, shape=None)

        #CNN inputs
        self.x_image = tf.placeholder(dtype=tf.float32, shape=(None, 134, 91, 3))

        self.y_label = tf.placeholder(dtype=tf.int32, shape=(None, ))

        # LSTM inputs
        self.x_ids = tf.placeholder(dtype=tf.int32, shape=(None, self.max_len))
        self.x_len = tf.placeholder(dtype=tf.int32, shape=(None, ))

        # Embedding
        self.emb_W = self.get_var(name='emb_W', shape=[self.vocab_size, self.emb_dim])
        self.batch_size = tf.shape(self.x_ids)[0]
        self.x_emb = tf.nn.embedding_lookup(self.emb_W,
                                            tf.concat([self.x_ids[:, :-1], tf.ones([self.batch_size, 1], dtype=tf.int32)], 1))

        self.build_model()
        self.build_loss()
        self.build_opt()
        self.build_acc()

    def build_model(self):
        conv1 = self.conv(name = 'c1', inputs = self.x_image, shape=[3,3,3,64], s = 1, padding = 'SAME')
        conv1 = tf.nn.relu(conv1)
        conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        self.con1 = conv1
        self.layers.append(conv1)

        conv2 = self.conv(name = 'c2', inputs = conv1, shape=[3,3,64,128], s = 1, padding = 'SAME')
        conv2 = tf.nn.relu(conv2)
        conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        conv2 = tf.nn.dropout(conv2, self.keep_prob)
        self.con2 = conv2

        conv3 = self.conv(name = 'c3', inputs = conv2, shape=[3,3,128,256], s = 1, padding = 'SAME')
        conv3 = tf.nn.relu(conv3)
        conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        conv3 = tf.nn.dropout(conv3, self.keep_prob)
        self.con3 = conv3

        self.layers.append(conv3)

        x_image = tf.reduce_mean(self.layers[-1], [1, 2])
#         print('x_image',x_image)

        fc_l1 = tf.contrib.layers.fully_connected(x_image, 128, activation_fn=tf.nn.relu,weights_regularizer=tf.contrib.layers.l2_regularizer(scale=0.2))

        # CNN의 CON3를 LSTM의 INPUT 값으로
        img_cell = self.conv(name = 'ic', inputs = conv3, shape=[1,1,256,32], s = 1, padding = 'SAME')
        _, w, h, d = img_cell.get_shape()
        length = int(w) * int(h) * int(d)

        img_vector = tf.reshape(img_cell, [-1, 1, length])
        mul = tf.constant([1, self.max_len, 1])
        img_feature = tf.tile(img_vector, mul)
        embedding = tf.concat([self.x_emb, img_feature], axis=2)
        # input_cell = tf.concat(self.x_emb, img_vector, 1)

        fw_cell = rnn.DropoutWrapper(rnn.BasicLSTMCell(self.hidden_dim), output_keep_prob=self.keep_prob)
        bw_cell = rnn.DropoutWrapper(rnn.BasicLSTMCell(self.hidden_dim), output_keep_prob=self.keep_prob)
        output, (fw_state, bw_state) = tf.nn.bidirectional_dynamic_rnn(
            fw_cell, bw_cell, inputs=embedding, sequence_length=self.x_len, dtype=tf.float32)

        # CNN의 결과와 RNN의 결과 합치기
        text_vec = tf.concat([fw_state.h, bw_state.h], 1)
#         out_W = self.get_var(name="out_W", shape=[self.hidden_dim*2, 128])
#         out_b = self.get_var(name="out_b", shape=[128])
#         lstm_out = tf.matmul(text_vec, out_W) + out_b
        
        lstm_out = tf.contrib.layers.fully_connected(text_vec, 128, activation_fn=tf.nn.relu,weights_regularizer=tf.contrib.layers.l2_regularizer(scale=0.2))
#         text_vec = tf.concat([fw_state.h, bw_state.h], 1)


        final_layer = tf.concat([fc_l1, lstm_out], axis=1)
        self.genre_prob = tf.contrib.layers.fully_connected(final_layer, self.class_size, activation_fn=tf.nn.softmax)
        self.y_pred = tf.to_int32(tf.argmax(self.genre_prob, 1))

    def build_loss(self):
        self.cross_entropy = -tf.reduce_sum(
            tf.one_hot(tf.to_int32(tf.reshape(self.y_label, [-1])), self.class_size, 1.0, 0.0)
            * tf.log(tf.clip_by_value(tf.reshape(self.genre_prob, [-1, self.class_size]), 1e-20, 1.0)), 1)
        self.loss = tf.reduce_mean(self.cross_entropy)

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

    def build_acc(self):
        self.prob = tf.to_int32(tf.argmax(self.genre_prob, 1))
        self.is_correct = tf.equal(self.prob, self.y_label)
        self.accuracy = tf.reduce_mean(tf.cast(self.is_correct, tf.float32))

    def get_var(self, name='', shape=None, dtype=tf.float32):
        return tf.get_variable(name, shape, dtype=dtype, initializer=self.initializer)

    def conv(self, name='', inputs=None, shape=[], s=None, padding='SAME'):
        w = self.get_var(name='w'+name, shape=shape)
        b = self.get_var(name='b'+name, shape=shape[-1])
        return tf.nn.conv2d(inputs, filter=w, strides=[1, s, s, 1], padding=padding) + b

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
