# -*- coding: utf-8 -*-

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2"

import tensorflow as tf
import numpy as np
import time

from ensemble_loader import movie_data
data = movie_data()

def initialize_session():
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    return tf.Session(config=config)

##################################################
BATCH_SIZE = 64
class_size = 4
learning_rate = 0.001   # Learning rate
use_clip = True         # Gradient clipping 쓸지 여부
train_keep_prob = 0.7
max_len = 800           # sequence 단어 수 제한
max_vocab = 50000       # maximum 단어 개수
emb_dim = 256            # 단어 embedding dimension
hidden_dim = 128        # RNN hidden dim
train_total_batch = int(data.train_size / BATCH_SIZE)
test_total_batch = int(data.test_size / BATCH_SIZE)
train_acc_summary = [] # 학습용 데이터의 정확도를 중간 중간 기록한다
test_acc_summary = []  # 테스트용 데이터의 정확도를 중간 중간 기록한다
train_loss_summary = [] # 학습용 데이터의 정확도를 중간 중간 기록한다
test_loss_summary = []  # 테스트용 데이터의 정확도를 중간 중간 기록한다
hypothesis_summary = []
##################################################
from genre_ensemble_model import Model
END_TOKEN = "<eos>"
model = Model(max_len=max_len,
              emb_dim=emb_dim,
              hidden_dim=hidden_dim,
              vocab_size=max_vocab,
              class_size=2,
              use_clip=True, learning_rate=learning_rate, end_token=data.w2idx[END_TOKEN])

sess = initialize_session()
sess.run(tf.global_variables_initializer())

def test_model():
    test_loss, test_acc = 0, 0

    for _ in range(test_total_batch):
        test_x_img, test_ids, test_length, test_y = data.get_test(BATCH_SIZE)

        loss, acc = sess.run([model.loss, model.accuracy],
                              feed_dict={model.x_image: test_x_img, model.x_ids: test_ids, model.x_len: test_length,
                              model.y_label: test_y, model.keep_prob : 1.0, model.is_training:False})
        test_loss += loss
        test_acc += acc

    return test_loss/test_total_batch, test_acc/test_total_batch
    #print(" * test loss: {:.3f} | test acc: {:.3f}\n".format(test_loss / test_cnt, sess.run(model.accuracy, feed_dict={model.x_image: test_x, model.y_label: test_label})))

for epoch in range(10):

    train_avg_loss, train_avg_acc = 0, 0

    for it in range(train_total_batch):
        train_x_img, train_ids, train_length, train_y = data.get_train(BATCH_SIZE)

        loss, acc, _, hypothesis_ = sess.run([model.loss, model.accuracy, model.update, model.genre_prob],
                                              feed_dict={model.x_image: train_x_img, model.x_ids: train_ids, model.x_len: train_length,
                                              model.y_label: train_y, model.keep_prob : train_keep_prob, model.is_training:True})
        train_avg_loss += loss / train_total_batch
        train_avg_acc += acc / train_total_batch

        #if it % 10 == 0 and it > 0:
            #model.save(sess)
            #print("*Model Saved train_loss: {:.4f}, train_acc: {:.4f}".format(loss/BATCH_SIZE, acc))

    test_avg_loss, test_avg_acc = test_model()

    train_acc_summary.append(train_avg_acc)
    test_acc_summary.append(test_avg_acc)
    train_loss_summary.append(train_avg_loss)
    test_loss_summary.append(test_avg_loss)
    hypothesis_summary.append(hypothesis_)

    print("epoch {} - train_loss: {:.4f}, train_acc: {:.4f}, test_loss: {:.4f}, test_acc: {:.4f}"
          .format(epoch+1, train_avg_loss, train_avg_acc, test_avg_loss, test_avg_acc))
