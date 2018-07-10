# -*- coding: utf-8 -*-

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2"

import tensorflow as tf
import numpy as np
import time

from genre_model import Model
from data_loader import movie_data

def initialize_session():
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    return tf.Session(config=config)

##################################################
BATCH_SIZE = 20
class_size = 24
learning_rate = 0.005   # Learning rate
use_clip = True         # Gradient clipping 쓸지 여부
keep_prob = 1.0
##################################################

data = movie_data()
model = Model(use_clip=use_clip, class_size=class_size, learning_rate=learning_rate)

sess = initialize_session()
sess.run(tf.global_variables_initializer())

def test_model():
    num_it = 10
    test_loss, test_cnt, test_right = 0, 0, .0

    for _ in range(num_it):
        test_x, test_label = data.get_test(BATCH_SIZE)
        loss, out = sess.run([model.loss, model.y_pred],
                              feed_dict={model.x_image: test_x, model.y_label: test_label})
        test_loss += loss
        test_cnt += 1
        for i, o in enumerate(out):
            if o == test_label[i]:
                test_right += 1
    print(" * test loss: {:.3f} | acc: {:.3f}\n".format(test_loss / test_cnt, test_right / test_cnt / BATCH_SIZE))

# 0: neg, 1: pos
avg_loss, it_cnt, same = 0, 0, .0
it_log, it_test, it_save, it_sample = 10, 100, 1000, 100
start_time = time.time()

for it in range(0, 10000):
    train_x, label = data.get_train(BATCH_SIZE)
    loss, _, out = sess.run([model.loss, model.update, model.y_pred],
                            feed_dict={model.x_image: train_x, model.y_label: label})
    for i, o in range enumerate(out):

    avg_loss += loss
    it_cnt += 1
    if it % it_log == 0 and it:
        print(" it: {:4d} | loss: {:.3f} | acc: {:.3f} - {:.2f}s".format(
            it, avg_loss / it_cnt, same/BATCH_SIZE/it_log, time.time() - start_time))
        avg_loss, it_cnt, same = 0, 0, .0

    if it % it_test == 0 and it > 0:
        test_model()
    if it % it_save == 0 and it > 0:
        model.save(sess)
