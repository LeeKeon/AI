#-*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

from tensorflow.python.layers.core import Dense
from tensorflow.contrib import layers

from tensorflow.contrib.rnn import LSTMCell, DropoutWrapper, LSTMStateTuple
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn, dynamic_rnn

class Model(object):

    def __init__(self, enc_len=400, dec_len=100,
                 emb_dim=128, hidden_dim=128,
                 enc_vocab=50000, dec_vocab=50000,
                 use_clip=True, learning_rate=0.001):
        self.initializer = tf.random_uniform_initializer(-0.05, 0.05)
        self.enc_len = enc_len
        self.dec_len = dec_len
        self.enc_vocab = enc_vocab
        self.dec_vocab = dec_vocab
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim

        self.learning_rate = learning_rate
        self.use_clip = use_clip

        # Define placeholders
        self.init_placeholder()

        # decoder mask는 서로 다른 길이의 학습을 위해 필요하며,
        # encoder mask는 pad 된 부분에 대해 attention 비율을 0으로 만들어주기 위해 필요
        self.enc_masks = tf.sequence_mask(lengths=self.x_len, maxlen=self.enc_len, dtype=tf.float32)
        self.dec_masks = tf.sequence_mask(lengths=self.y_len, maxlen=self.dec_len, dtype=tf.float32)
        self.dec_mask_sum = tf.reduce_sum(self.dec_masks)

        # rnn encoder-decoder
        self.rnn_encode()
        self.init_enc_attention()

        self.decoder_cell = LSTMCell(self.hidden_dim, state_is_tuple=True)
        self.out_layer = Dense(self.dec_vocab, dtype=tf.float32, name='out_layer')

        self.decode_train()
        self.decode_infer()

        # Define loss & optimizer
        self.build_loss()
        self.build_opt()

        print(' .. Finish Building Model ... ')

    def init_placeholder(self):
        # word embedding
        self.emb_W_enc = self.get_var(name='emb_W_enc', shape=[self.enc_vocab, self.emb_dim])
        self.emb_W_dec = self.get_var(name='emb_W_dec', shape=[self.dec_vocab, self.emb_dim])

        # encoder part
        self.x = tf.placeholder(dtype=tf.int32, shape=(None, self.enc_len))
        self.x_len = tf.placeholder(dtype=tf.int32, shape=(None,))
        self.x_emb = tf.nn.embedding_lookup(self.emb_W_enc , self.x)
        self.batch_size = tf.shape(self.x)[0]

        # decoder part
        self.y = tf.placeholder(dtype=tf.int32, shape=(None, self.dec_len))
        self.y_len = tf.placeholder(dtype=tf.int32, shape=(None,))
        self.y_emb = tf.concat([tf.nn.embedding_lookup(self.emb_W_dec, tf.zeros([self.batch_size, 1], dtype=tf.int32)),
                                tf.nn.embedding_lookup(self.emb_W_dec, self.y)], axis=1)


    def get_var(self, name='', shape=None, dtype=tf.float32):
        return tf.get_variable(name, shape, dtype=dtype, initializer=self.initializer)

    def rnn_encode(self):
        (self.enc_out, (fw_state, bw_state)) = bidirectional_dynamic_rnn(
            LSTMCell(self.hidden_dim), LSTMCell(self.hidden_dim),
            inputs=self.x_emb, sequence_length=self.x_len, dtype=tf.float32)
        self.enc_out = tf.concat(self.enc_out, 2)

        decoder_init_c = Dense(self.hidden_dim, dtype=tf.float32, name='decoder_c', activation=tf.nn.tanh)
        decoder_init_h = Dense(self.hidden_dim, dtype=tf.float32, name='decoder_h', activation=tf.nn.tanh)

        self.init_state = LSTMStateTuple(decoder_init_c(tf.concat([fw_state.c, bw_state.c], 1)),
                                         decoder_init_h(tf.concat([fw_state.h, bw_state.h], 1)))

    # Attention W for encoder
    def init_enc_attention(self):
        # self.hidden_dim = 300   # defined at init
        self.attWe = self.get_var(name='attnW_et', shape=[1, 1, self.hidden_dim, self.hidden_dim])
        self.attve = self.get_var(name='attnV_et', shape=[self.hidden_dim, 1])

        self.attUe = self.get_var(name='attnU_et', shape=[1, 1, self.hidden_dim * 2, self.hidden_dim])
        self.attbe = self.get_var(name='attnB_et', shape=[self.hidden_dim])

        # U_a*h_j
        att_e = tf.nn.conv2d(tf.reshape(self.enc_out, [-1, self.enc_len, 1, self.hidden_dim * 2]),
                             filter=self.attUe, strides=[1, 1, 1, 1], padding='VALID')
        self.attUeh = tf.reshape(att_e + self.attbe, [-1, self.hidden_dim])


    # Encoder attention softmax
    def encoder_attention(self, state):
        # decoder state
        st = tf.reshape(tf.tile(state, [1, self.enc_len]),
                        [-1, self.enc_len, 1, self.hidden_dim])
        # W_a*s_(i-1)
        attWes = tf.reshape(tf.nn.conv2d(st, filter=self.attWe, strides=[1, 1, 1, 1], padding='SAME'),
                            [-1, self.hidden_dim])

        # encoder attention distribution (logit), e_ij
        e_t = tf.reshape(tf.matmul(tf.nn.tanh(attWes + self.attUeh), self.attve),
                         [-1, self.enc_len]) * self.enc_masks

        return tf.nn.softmax(e_t)

    def get_word_prob(self, out, out_rank=1):
        logit = self.out_layer(self.out_linear(out))
        word_prob = tf.nn.softmax(logit)
        return word_prob

    def decode_train(self):
        y_emb_tbd = tf.transpose(self.y_emb, [1, 0, 2])
        arr_prob = tf.TensorArray(dtype=tf.float32, size=self.dec_len)

        def body(step, state, arr_prob):
            enc_softmax = self.encoder_attention(state.h)
            context_vector = tf.reduce_sum(self.enc_out * tf.expand_dims(enc_softmax, -1), axis=1)

            word_prob = tf.nn.softmax(self.out_layer(tf.concat([state.h, context_vector], 1)))
            arr_prob = arr_prob.write(step, word_prob)
            next_input = tf.concat([y_emb_tbd[step + 1], context_vector], 1)
            next_out, next_state = self.decoder_cell(next_input, state)

            return step+1, next_state, arr_prob

        _step, _state, _arr_prob = tf.while_loop(
            cond=lambda t, _1, _2: t < self.dec_len,
            body=body,
            loop_vars=(0, self.init_state, arr_prob))

        self.train_prob = tf.transpose(_arr_prob.stack(), perm=[1, 0, 2])

    def decode_infer(self):
        arr_token = tf.TensorArray(dtype=tf.int32, size=self.dec_len)

        def body(step, state, arr_token):
            enc_softmax = self.encoder_attention(state.h)
            context_vector = tf.reduce_sum(self.enc_out * tf.expand_dims(enc_softmax, -1), axis=1)


            word_prob = tf.nn.softmax(self.out_layer(tf.concat([state.h, context_vector], 1)))

            # Write token
            next_token = tf.cast(tf.reshape(tf.argmax(word_prob, 1), [self.batch_size]), tf.int32)
            arr_token = arr_token.write(step, next_token)
            token_emb = tf.nn.embedding_lookup(self.emb_W_dec, next_token)
            next_input = tf.concat([token_emb, context_vector], 1)
            next_out, next_state = self.decoder_cell(next_input, state)

            return step+1, next_state, arr_token

        _step, _state, _arr_token = tf.while_loop(
            cond=lambda t, _1, _2: t < self.dec_len,
            body=body,
            loop_vars=(0, self.init_state, arr_token))

        self.test_token = tf.transpose(_arr_token.stack(), perm=[1, 0])

    def build_loss(self):
        # cross-entropy loss
        self.cross_entropy = -tf.reduce_sum(
            tf.one_hot(tf.to_int32(tf.reshape(self.y, [-1])), self.dec_vocab, 1.0, 0.0)
            * tf.log(tf.clip_by_value(tf.reshape(self.train_prob, [-1, self.dec_vocab]), 1e-20, 1.0)), 1)

        self.loss = tf.reduce_sum(self.cross_entropy * tf.reshape(self.dec_masks, [-1])) / self.dec_mask_sum
        tf.summary.scalar('loss', self.loss)

    def build_opt(self):
        opt_adam = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        optimizer = opt_adam

        def clipped_grad(grad):
            return [None if g is None else tf.clip_by_norm(g, 2.5) for g in grad]

        # compute gradient & clipping
        grad, var = zip(*optimizer.compute_gradients(self.loss))

        if self.use_clip:
            grad = clipped_grad(grad)

        # update weights
        self.update = optimizer.apply_gradients(zip(grad, var))

    # Save whole weights
    def save_model(self, sess, path, global_step=None):
        var_list = [var for var in tf.all_variables()]
        saver = tf.train.Saver(var_list)
        save_path = saver.save(sess, save_path="models/model", global_step=global_step)
        print(' * model saved at \'{}\''.format(save_path))

    # Load whole weights
    def restore_model(self, sess, path):
        print(' - Restoring variables...')
        var_list = [var for var in tf.all_variables()]
        saver = tf.train.Saver(var_list)
        saver.restore(sess, "models/model")
        print(' * model restored ')
