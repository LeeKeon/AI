import random
import numpy as np

class text_data(object):
    def __init__(self, path="./dataset/ptb", max_seq_len=40, max_word_len=15, end_token="<eos>"):
        self.train_pt, self.val_pt, self.test_pt = 0, 0, 0
        self.path = path
        self.max_seq_len = max_seq_len
        self.max_word_len = max_word_len

        self.w2idx = {end_token: 0}
        self.c2idx = {}

        self.train_ids, self.train_chars, self.train_len = self.file_to_ids(path+"/ptb.train.txt")
        self.val_ids, self.val_chars, self.val_len = self.file_to_ids(path + "/ptb.valid.txt")
        self.test_ids, self.test_chars, self.test_len = self.file_to_ids(path + "/ptb.test.txt")
        self.vocab_size = len(self.w2idx)
        self.char_size = len(self.c2idx)

        self.train_size = len(self.train_ids)
        self.val_size = len(self.val_ids)
        self.test_size = len(self.test_ids)

        self.idx2w, self.idx2c = {}, {}
        for word in self.w2idx:
            self.idx2w[self.w2idx[word]] = word
        for char in self.c2idx:
            self.idx2c[self.c2idx[char]] = char

    def file_to_ids(self, file_name):
        with open(file_name, "r") as fin:
            lines = fin.readlines()

        max_size = 20
        length, ids, chars = [], [], []

        for num, line in enumerate(lines):
            id = np.zeros(self.max_seq_len, dtype=np.int32)
            char = np.zeros((self.max_seq_len, self.max_word_len), dtype=np.int32)

            # line += " <e>"
            words = line.split()
            for i, word in enumerate(words):
                if i == self.max_seq_len:
                    break
                if word not in self.w2idx:
                    self.w2idx[word] = len(self.w2idx)
                id[i] = self.w2idx[word]

                for j, w in enumerate(word):
                    if j == self.max_word_len:
                        break
                    if w not in self.c2idx:
                        self.c2idx[w] = len(self.c2idx)
                    char[i][j] = self.c2idx[w]

            ids.append(id)
            chars.append(char)
            length.append(i+1)

            if num+1 == max_size:
                break

        return np.array(ids), np.array(chars), np.array(length)

    def get_train(self, batch_size=20):
        pt = self.train_pt
        self.train_pt = (self.train_pt + batch_size) % self.train_size
        return self.train_ids[pt: pt+batch_size], self.train_chars[pt: pt+batch_size], self.train_len[pt: pt+batch_size]

    def get_val(self, batch_size=20):
        pt = self.val_pt
        self.val_pt = (self.val_pt + batch_size) % self.val_size
        return self.val_ids[pt: pt+batch_size], self.val_chars[pt: pt+batch_size], self.val_len[pt: pt+batch_size]

    def get_test(self, batch_size=20):
        pt = self.test_pt
        self.test_pt = (self.test_pt + batch_size) % self.test_size
        return self.test_ids[pt: pt+batch_size], self.test_chars[pt: pt+batch_size], self.test_len[pt: pt+batch_size]