# -*- coding: utf-8 -*-

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2"

import tensorflow as tf
import numpy as np
import time

from plot_loader import plot_data
data = plot_data()

def initialize_session():
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    return tf.Session(config=config)

train_acc_summary = [] # 학습용 데이터의 오류를 중간 중간 기록한다
test_acc_summary = []  # 테스트용 데이터의 오류를 중간 중간 기록한다

##################################################
max_len = 100           # sequence 단어 수 제한
max_vocab = 20000       # maximum 단어 개수
BATCH_SIZE = 10         # 배치 사이즈
emb_dim = 128            # 단어 embedding dimension
hidden_dim = 128        # RNN hidden dim
learning_rate = 0.001  # Learning rate
use_clip = True         # Gradient clipping 쓸지 여부
##################################################
from genre_movie_rnn import Model
model = Model(max_len=max_len,
              emb_dim=emb_dim,
              hidden_dim=hidden_dim,
              vocab_size=max_vocab,
              class_size=2,
              use_clip=True, learning_rate=learning_rate, end_token=data.w2idx[END_TOKEN])

sess = initialize_session()
sess.run(tf.global_variables_initializer())


def test_model():
    num_it = len(data.x_test) / BATCH_SIZE
    test_loss, test_cnt, test_right = 0, 0, .0

    loss = sess.run(model.loss, model.out_label, feed_dict={model.x: test_ids, model.x_len: length, model.y: label})
    for i, o in enumerate(out):
        if o == label[i]:
            same += 1
    test_loss += loss
    test_cnt += 1
print(" --> test_loss: {:.3f} | test_acc: {:.3f}".format(test_loss / test_cnt, same/test_cnt/BATCH_SIZE))

# 0: neg, 1: pos
avg_loss, it_cnt, same = 0, 0, .0
it_log, it_test, it_save, it_sample = 10, 100, 1000, 100
start_time = time.time()

for it in range(0, 500):
    train_ids, length, label = data.get_train(BATCH_SIZE)
    loss, _, out = sess.run([model.loss, model.update, model.out_label],
                            feed_dict={model.x: train_ids, model.x_len: length, model.y: label, model.keep_prob: 0.5})
    for i, o in enumerate(out):
        if o == label[i]:
            same += 1
    avg_loss += loss
    it_cnt += 1

    if it % it_log == 0 and it:
        print(" it: {:4d} | loss: {:.3f} | acc: {:.3f} - {:.2f}s".format(
            it, avg_loss / it_cnt, same/BATCH_SIZE/it_log, time.time() - start_time))
        avg_loss, it_cnt, same = 0, 0, .0

    if it % it_test == 0 and it > 0:
        test_model()
