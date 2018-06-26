#-*- coding: utf-8 -*-
import os, nltk, time

import numpy as np
import random

class summ_data(object):

    def __init__(self, path, save_name, refresh=False,
                  enc_vocab=50000, enc_len=400,
                  dec_vocab=50000, dec_len=100):

        self.start_time = time.time()
        self.path = path + ('/' if path[-1] != '/' else '')
        self.name = save_name
        self.save_name = save_name + '.npy'

        self.enc_vocab = enc_vocab
        self.dec_vocab = dec_vocab
        self.enc_len = enc_len
        self.dec_len = dec_len

        self.pad_token, self.pad_idx = "<pad>", 0
        self.unk_token, self.unk_idx = "<unk>", 1
        self.eos_token, self.eos_idx = "<eos>", 2
        self.w2idx = {self.pad_token: 0, self.unk_token: 1, self.eos_token: 2}
        self.idx2w = {}

        print(' *---- Data Loading ----*')
        print(' | path: {}'.format(path))

        if os.path.exists(self.save_name) and not refresh:
            self.load()
        else:
            self.read_vocab()
            self.text, self.size = self.read_text()
            self.enc_ids, self.dec_ids, self.length = self.build_data()
            self.save()

        for word in self.w2idx:
            self.idx2w[self.w2idx[word]] = word
        self.size = int(len(self.text)/2)
        self.train_size, self.test_size = len(self.train_idx), len(self.test_idx)

        print(' | Size: {}, {}'.format(self.train_size, self.test_size))
        print(' | Vocab: {}'.format(len(self.w2idx)))
        print(' | Avg length: {:.1f} / {:.1f} words'.format(self.length.mean(0)[0], self.length.mean(0)[1]))
        print(' | Building time: {:.2f}s'.format(time.time() - self.start_time))
        print(' *---- Dataset Intialized ----\n')

    # 토큰들을 문장으로 바꿈
    def ids2sent(self, ids):
        sent = ''
        for i in ids:
            word = self.idx2w[i]
            sent += word + ' '
            if word == self.pad_token:
                break
        return sent.rstrip()

    # 단어의 index 구함 (없는경우 unk)
    def word2idx(self, word, limit=1):
        if word in self.w2idx and self.w2idx[word] < limit:
            return self.w2idx[word]
        else:
            return self.w2idx[self.unk_token]

    #  단어사전 구축
    def read_vocab(self):
        with open(self.path + 'vocab', 'r', encoding='utf-8') as fvocab:      # python 3
            lines = fvocab.readlines()

        for line in lines:
            word = line.rstrip().split()[0]
            if word not in self.w2idx:
                self.w2idx[word] = len(self.w2idx)
            if len(self.w2idx) == self.enc_vocab:
                break
        print(' | Read vocab: {} - {:.2f}s'.format(len(self.w2idx), time.time()-self.start_time))

    # 데이터 로드 (형태: 원문 \n 요약문 \n 원문 \n 요약문 -> 홀수번째 줄: 원문, 짝수번째 줄: 요약문)
    def read_text(self):
        with open(self.path+'train.enc_dec', 'r', encoding='utf-8') as fin:
            train_lines = fin.readlines()
        with open(self.path+'test.enc_dec', 'r', encoding='utf-8') as fin:
            test_lines = fin.readlines()

        text = train_lines + test_lines
        n1, n2 = int(len(train_lines)/2), int(len(test_lines)/2)

        self.train_idx = list(np.arange(0, n1))
        self.test_idx = list(np.arange(n1, n1+n2))

        print(' | info. {} {}'.format(n1, n2))
        print(' | Read text: {} - {:.2f}s'.format(n1+n2, time.time() - self.start_time))

        return text, n1+n2

    def build_data(self):
        enc_ids = np.zeros((self.size, self.enc_len), dtype=np.int32)
        dec_ids = np.zeros((self.size, self.dec_len), dtype=np.int32)
        length = np.zeros((self.size, 2), dtype=np.int32)

        idx = 0
        for i in range(0, self.size*2, 2):
            self.text[i] += ' <eos>'
            self.text[i+1] += ' <eos>'

            enc = self.text[i].split()
            dec = self.text[i+1].split()

            length[idx][0] = min(len(enc), self.enc_len)
            length[idx][1] = min(len(dec), self.dec_len)
            dec_cnt = 0

            for j in range(length[idx][0]):
                enc_ids[idx][j] = self.word2idx(enc[j], self.enc_vocab)

            # 데이터에 있는 문장의 시작, 끝(<s>, '</s>')를 제거
            for j in range(length[idx][1]):
                if dec[j] != '<s>' and dec[j] != '</s>':
                    dec_ids[idx][dec_cnt] = self.word2idx(dec[j], self.dec_vocab)
                    dec_cnt += 1
            length[idx][1] = dec_cnt
            idx += 1
        return enc_ids, dec_ids, length

    def get_batch(self, mode, batch_size):
        if mode is 'train':
            idx = random.sample(self.train_idx, batch_size)
        else:
            idx = random.sample(self.test_idx, batch_size)

        return idx, self.enc_ids[idx], self.dec_ids[idx], self.length[idx, 0], self.length[idx, 1]


    def save(self):
        total = {'w2idx': self.w2idx, 'text': self.text,
                 'enc_ids': self.enc_ids, 'dec_ids': self.dec_ids, 'length': self.length,
                 'train_idx': self.train_idx, 'test_idx': self.test_idx}
        np.save(self.save_name, total)

    def load(self):
        print(' | Load data: {}'.format(self.save_name))
        total = np.load(self.save_name).item()
        self.enc_ids, self.dec_ids, self.length = total['enc_ids'], total['dec_ids'], total['length']
        self.train_idx, self.test_idx = total['train_idx'], total['test_idx']
        self.w2idx = total['w2idx']
        self.text = total['text']


