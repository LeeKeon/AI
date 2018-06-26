# -*- coding: utf-8 -*- #

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import tensorflow as tf
import numpy as np
import time
import random

from img2text_model import Model
from pascal_loader import Data

def initialize_session():
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    return tf.Session(config=config)

##################################################
BATCH_SIZE = 10         # 배치 사이즈
emb_dim = 64            # 단어 embedding dimension
hidden_dim = 64         # RNN hidden dim
max_len = 30            # text 최대 길이

IMG_X_SIZE = 256
IMG_Y_SIZE = 256
learning_rate = 0.0005    # Learning rate
use_clip = True         # Gradient clipping 쓸지 여부
##################################################

data = Data(path='./dataset/pascal-sentences',
            max_vocab=5000, max_len=max_len,
            img_x_size=IMG_X_SIZE, img_y_size=IMG_Y_SIZE)

model = Model(emb_dim=emb_dim, hidden_dim=hidden_dim,
              vocab_size=data.vocab_size, max_len=max_len,
              img_x=IMG_X_SIZE, img_y=IMG_Y_SIZE,
              use_clip=True, learning_rate=learning_rate)

sess = initialize_session()
sess.run(tf.global_variables_initializer())

def id2text(idx2w, tokens):
    line = ""
    for t in tokens:
        line += idx2w[t] + " "
        if idx2w[t] == "<eos>":
            break
    return line

def sample_test():
    img, text, text_len = data.get_test(1)
    output_token = sess.run(model.output_token, feed_dict={model.img: img})

    for t in text[0]:
        true_line = id2text(data.idx2w, t)
        print(" - true: {}".format(true_line))

    out_line = id2text(data.idx2w, output_token[0])
    print(" --> result: {}".format(out_line))


def test_model():
    num_it = 10
    test_loss, test_cnt = 0, 0

    for _ in range(num_it):
        img, text, text_len = data.get_test(BATCH_SIZE)
        feed_text, feed_length = [], []

        for i in range(len(text)):
            idx = random.randint(0, len(text[i]) - 1)
            feed_text.append(text[i][idx])
            feed_length.append(text_len[i][idx])

        feed_dict = {model.img: img}
        feed_dict[model.text] = feed_text
        feed_dict[model.text_len] = feed_length
        loss, _ = sess.run([model.loss, model.update], feed_dict=feed_dict)

        test_loss += loss
        test_cnt += 1
    print("test loss: {:.3f}".format(test_loss / test_cnt))


avg_loss, it_cnt = 0, 0
it_log, it_test, it_save, it_sample = 10, 50, 1000, 50
start_time = time.time()

for it in range(0, 10000):
    img, text, text_len = data.get_train(BATCH_SIZE)

    feed_text, feed_length = [], []
    for i in range(len(text)):
        idx = random.randint(0, len(text[i])-1)
        feed_text.append(text[i][idx])
        feed_length.append(text_len[i][idx])

    feed_dict = {model.img: img}
    feed_dict[model.text] = feed_text
    feed_dict[model.text_len] = feed_length
    loss, _ = sess.run([model.loss, model.update], feed_dict=feed_dict)

    avg_loss += loss
    it_cnt += 1

    if it % it_log == 0:
        print(" it: {:4d} | loss: {:.3f} - {:.2f}s".format(it, avg_loss / it_cnt, time.time() - start_time))
        avg_loss, it_cnt = 0, 0

    if it % it_test == 0 and it > 0:
        test_model()
    if it % it_save == 0 and it > 0:
        model.save(sess)
    if it % it_sample == 0 and it > 0:
        sample_test()

sess.close()