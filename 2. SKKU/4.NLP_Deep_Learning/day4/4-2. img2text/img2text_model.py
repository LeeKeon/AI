# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.python.layers.core import Dense

from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn, dynamic_rnn

# Write your code here
class Model(object):
    def __init__(self, emb_dim=128, hidden_dim=128,
                 vocab_size=40, max_len=50,
                 img_x=128, img_y=128,
                 use_clip=True, learning_rate=0.001):
        self.initializer = tf.random_uniform_initializer(-0.1, 0.1)

        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim

        self.vocab_size = vocab_size
        self.max_len = max_len

        self.use_clip = use_clip
        self.learning_rate = learning_rate

        # Placeholder
        self.img = tf.placeholder(dtype=tf.float32, shape=(None, img_x, img_y, 3))
        self.text = tf.placeholder(dtype=tf.int32, shape=(None, max_len))
        self.text_len = tf.placeholder(dtype=tf.int32, shape=(None, ))

        # sequence mask for different size
        self.batch_size = tf.shape(self.img)[0]
        self.masks = tf.sequence_mask(lengths=self.text_len, maxlen=self.max_len, dtype=tf.float32)

        # Embedding
        self.emb_W = tf.get_variable(name='emb_W', shape=[self.vocab_size, self.emb_dim], dtype=tf.float32, initializer=self.initializer)
        self.text_emb = tf.nn.embedding_lookup(self.emb_W, self.text)

        self.image_cnn()

        self.decoder_cell = LSTMCell(self.hidden_dim, state_is_tuple=True)
        self.out_layer = Dense(self.vocab_size, name='out_layer')
        self.decoder_train()
        self.decoder_infer()

        self.build_loss()
        self.build_opt()

    def image_cnn(self):
        conv_f1 = tf.get_variable(name='conv_f1', shape=[3, 3, 3, 128], dtype=tf.float32, initializer=self.initializer)
        conv_b1 = tf.get_variable(name='conv_b1', shape=[128], dtype=tf.float32, initializer=self.initializer)
        conv_l1 = self.leaky_relu(tf.nn.conv2d(self.img, filter=conv_f1, strides=[1, 2, 2, 1], padding='SAME') + conv_b1)
        pool_l1 = tf.nn.max_pool(conv_l1, [1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
        print(pool_l1 .shape)

        conv_f2 = tf.get_variable(name='conv_f2', shape=[3, 3, 128, 128], dtype=tf.float32, initializer=self.initializer)
        conv_b2 = tf.get_variable(name='conv_b2', shape=[128], dtype=tf.float32, initializer=self.initializer)
        conv_l2 = self.leaky_relu(tf.nn.conv2d(pool_l1, filter=conv_f2, strides=[1, 2, 2, 1], padding='SAME') + conv_b2)
        pool_l2 = tf.nn.max_pool(conv_l2, [1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
        print(pool_l2.shape)

        conv_f3 = tf.get_variable(name='conv_f3', shape=[3, 3, 128, 64], dtype=tf.float32, initializer=self.initializer)
        conv_b3 = tf.get_variable(name='conv_b3', shape=[64], dtype=tf.float32, initializer=self.initializer)
        conv_l3 = self.leaky_relu(tf.nn.conv2d(pool_l2, filter=conv_f3, strides=[1, 2, 2, 1], padding='SAME') + conv_b3)
        pool_l3 = tf.nn.max_pool(conv_l3, [1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
        print(pool_l3.shape)

        conv_f4 = tf.get_variable(name='conv_f4', shape=[3, 3, 64, 64], dtype=tf.float32, initializer=self.initializer)
        conv_b4 = tf.get_variable(name='conv_b4', shape=[64], dtype=tf.float32, initializer=self.initializer)
        conv_l4 = self.leaky_relu(tf.nn.conv2d(pool_l3, filter=conv_f4, strides=[1, 1, 1, 1], padding='SAME') + conv_b4)
        pool_l4 = tf.nn.max_pool(conv_l4, [1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
        print(pool_l4.shape)

        self.img_vector = tf.reshape(pool_l4, [-1, 2*2*64])

        converter_c = Dense(self.hidden_dim, activation=tf.nn.tanh, dtype=tf.float32)
        converter_h = Dense(self.hidden_dim, activation=tf.nn.tanh, dtype=tf.float32)
        self.init_state = LSTMStateTuple(converter_c(self.img_vector),
                                         converter_h(self.img_vector))
        print(self.init_state.h.shape)


    def decoder_train(self):
        # time-batch-dimension
        text_emb_tbd = tf.transpose(self.text_emb, [1, 0, 2])
        word_prob = tf.TensorArray(dtype=tf.float32, size=self.max_len)

        def body(step, state, word_prob):
            word_logit = self.out_layer(tf.concat([state.h, self.img_vector], 1))
            word_prob = word_prob.write(step, word_logit)

            token_emb = text_emb_tbd[step]
            next_out, next_state = self.decoder_cell(token_emb, state)

            return step + 1, next_state, word_prob

        _step, _state, _word_prob = tf.while_loop(
            cond=lambda t, _state, _word_prob: t < self.max_len,
            body=body,
            loop_vars=(0, self.init_state, word_prob))

        self.train_prob = tf.transpose(_word_prob.stack(), perm=[1, 0, 2])

    def decoder_infer(self):
        word_token = tf.TensorArray(dtype=tf.int32, size=self.max_len)

        def body(step, state, word_token):
            word_logit = self.out_layer(tf.concat([state.h, self.img_vector], 1))
            next_token = tf.cast(tf.reshape(tf.argmax(word_logit, 1), [-1]), tf.int32)
            word_token = word_token.write(step, next_token)

            token_emb = tf.nn.embedding_lookup(self.emb_W, next_token)
            next_out, next_state = self.decoder_cell(token_emb, state)

            return step + 1, next_state, word_token

        _step, _state, _word_token = tf.while_loop(
            cond=lambda t, _state, _word_token: t < self.max_len,
            body=body,
            loop_vars=(0, self.init_state, word_token))

        self.output_token = tf.transpose(_word_token.stack(), perm=[1, 0])

    def build_loss(self):
        self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.text, logits=self.train_prob)
        self.loss = tf.reduce_sum(self.cross_entropy * self.masks) / (tf.reduce_sum(self.masks) + 1e-10)

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