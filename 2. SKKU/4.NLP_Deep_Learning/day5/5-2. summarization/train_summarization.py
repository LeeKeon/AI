# -*- coding: utf-8 -*-
from __future__ import print_function

import os
#os.environ['CUDA_VISIBLE_DEVICES'] = "0"

from summ_loader import summ_data
from summarization_model import Model as Model

import tensorflow as tf
import numpy as np
import random, time

BATCH_SIZE = 20
MODEL_NAME = 'model_cr'

def initialize_session(ratio=0.1):
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    return tf.Session(config=config)

def rouge2str(score, mode):
    return '{}_p - rouge-1: {:.4f} | rouge-2: {:.4f} | rouge-L: {:.4f} | \n' \
                    '{}_r - rouge-1: {:.4f} | rouge-2: {:.4f} | rouge-L: {:.4f} | \n' \
                    '{}_f - rouge-1: {:.4f} | rouge-2: {:.4f} | rouge-L: {:.4f}'.\
        format(mode, score['rouge-1']['p'], score['rouge-2']['p'], score['rouge-l']['p'],
               mode, score['rouge-1']['r'], score['rouge-2']['r'], score['rouge-l']['r'],
               mode, score['rouge-1']['f'], score['rouge-2']['f'], score['rouge-l']['f'])


def ids2text(data, token):
    text = ""
    for x in token:
        text += data.idx2w[x] + " "
    return text.rstrip()


def toRougeStr(data, line, words, ref=False):
    text = ""
    if type(line) != str:
        line = ids2text(data, line)
    cnt = 0
    for i, word in enumerate(line.split()):
        if word == "<eos>" or cnt==data.dec_len:
            break
        elif word == "</s>" and ref==True:
            text += "\n"
        elif word == "." and ref==False:
            text += ". \n"
        elif word != "<s>":
            text += word + " "
            cnt += 1
    return text.rstrip()


def test_model(sess, model, data, mode="test"):
    start_time = time.time()
    avg_loss, ref, hyp = .0, [], []
    it_size = 10
    for it in range(it_size):
        idx, enc, dec, enc_len, dec_len = data.get_batch("test", BATCH_SIZE)
        feed_dict = {model.x: enc, model.x_len: enc_len, model.y: dec, model.y_len: dec_len}

        loss, token = sess.run([model.loss, model.test_token], feed_dict=feed_dict)
        avg_loss += loss

        for i in range(BATCH_SIZE):
            ref.append(toRougeStr(data, data.text[idx[i]*2+1], True))
            hyp.append(toRougeStr(data, token[i], False))

    print('{} batches - test loss: {:.3f} '.format(it_size, avg_loss/it_size))
    print(u"ref----------------------------------------------------------\n" + ref[0])
    print(u"hyp----------------------------------------------------------\n" + hyp[0])


def train_model(sess, model, data):
    print(' . Start training - batch_size: {}'.format(BATCH_SIZE))
    global min_it, min_loss

    cur_time = time.time()
    avg_loss, loss_cov, avg_cov, loss_attn, avg_attn = .0, .0, .0, .0, .0
    it_log, it_test, it_save = 10, 50, 1000

    idxs = [i for i in range(0, data.train_size)]
    random.shuffle(idxs)

    for it in range(1, 10000):
        idx, enc, dec, enc_len, dec_len =  data.get_batch('train', BATCH_SIZE)
        feed_dict = {model.x: enc, model.x_len: enc_len, model.y: dec, model.y_len: dec_len}

        loss, _, = sess.run([model.loss, model.update], feed_dict=feed_dict)
        avg_loss+=loss

        if it % it_log is 0:
            log_str = "{:5d} : {:.4f}- {:.2f}s".format(it, avg_loss / it_log, time.time() - cur_time)
            print(log_str)
            avg_loss, cur_time = .0, time.time()

        if it % it_save is 0 and it:
            model.save_model(sess, 'saved_models/')

        if it % it_test is 0 and it:
            test_model(sess, model, data, 'test')


print('------- START -------')
data = summ_data('./dataset/cnn_daily', './preprocessed_data', refresh=False,
                 enc_vocab=50000, enc_len=400,
                 dec_vocab=50000, dec_len=100)

model = Model(enc_len=data.enc_len, dec_len=data.dec_len, emb_dim=128, hidden_dim=128,
              enc_vocab=data.enc_vocab, dec_vocab=data.dec_vocab, learning_rate=0.001, use_clip=True)

sess = initialize_session()
sess.run(tf.global_variables_initializer())

train_model(sess, model, data)


