# -*- coding: utf-8 -*-

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import tensorflow as tf
import numpy as np
import time

from charCnn_rnn_model import Model
from data_loader_char import text_data

def initialize_session():
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    return tf.Session(config=config)

##################################################
max_seq_len = 30       # sequence 단어 수 제한
max_word_len = 15      # 단어의 최대 길이

BATCH_SIZE = 10         # 배치 사이즈
emb_dim = 64            # 단어 embedding dimension
rnn_hidden_dim = 128    # RNN hidden dim

filter_sizes = [2, 3, 4]    # CNN filter size
filter_nums = [100, 100, 100]  # # of CNN filter

learning_rate = 0.001  # Learning rate
use_clip = True        # Gradient clipping 쓸지 여부
##################################################

END_TOKEN = "<eos>"
data = text_data("./dataset/ptb", max_seq_len=max_seq_len, max_word_len=max_word_len, end_token=END_TOKEN)
model = Model(max_seq_len=max_seq_len, max_word_len=max_word_len,
              emb_dim=emb_dim, rnn_hidden_dim=rnn_hidden_dim,
              filter_sizes=filter_sizes, filter_nums=filter_nums,
              vocab_size=data.vocab_size, char_size=data.char_size,
              use_clip=True, learning_rate=learning_rate)

sess = initialize_session()
sess.run(tf.global_variables_initializer())


def sample_test(test_input=""):
    # test_input = raw_input("test text: ") # input("test text: ") for python 2, 3
    words = test_input.split()
    x_word = np.zeros((1, max_seq_len), dtype=np.int32)
    x_char = np.zeros((1, max_seq_len, max_word_len), dtype=np.int32)

    for i, word in enumerate(words[:-1]):
        if i == max_seq_len:
            break
        x_word[0][i] = data.w2idx[word]

        for j, w in enumerate(word):
            if j == max_word_len:
                break
            x_char[0][i][j] = data.c2idx[w]

    output = sess.run(model.output,
                      feed_dict={model.x_word: x_word, model.x_char: x_char, model.x_len: [i+1]})
    print("{} --> {} (answer: {})".format(words[:-1], data.idx2w[output[0][i]], words[-1]))
    print()

def test_model():
    num_it = int(len(data.test_ids) / BATCH_SIZE)
    num_it = 10
    test_loss, test_cnt = 0, 0

    for _ in range(num_it):
        test_ids, test_chars, length = data.get_test(BATCH_SIZE)
        loss = sess.run(model.loss, feed_dict={model.x_word: test_ids, model.x_char: test_chars, model.x_len: length})

        test_loss += loss
        test_cnt += 1
    print("test loss: {:.3f}".format(test_loss / test_cnt))

avg_loss, it_cnt = 0, 0
it_log, it_test, it_save, it_sample = 10, 100, 1000, 100
start_time = time.time()

for it in range(0, 10000):
    train_ids, train_chars, length = data.get_train(BATCH_SIZE)
    loss, _ = sess.run([model.loss, model.update],
                       feed_dict={model.x_word: train_ids, model.x_char: train_chars, model.x_len: length})

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
        sample_test("we 're talking about years ago before ")

sess.close()