import os
import random
import numpy as np
import pandas as pd

class plot_data(object):
    def __init__(self, path="./dataset/poster_txt", max_vocab=30000, max_len=500, end_token="<eos>"):
        self.train_pt, self.val_pt, self.test_pt = 0, 0, 0
        self.path = path
        self.label_path = "./dataset/MovieGenre_fix_leekeon_total6000.csv"
        self.max_len = max_len
        self.max_vocab = max_vocab

        self.movie_df = pd.read_csv(self.label_path, encoding="ISO-8859-1")

        self.w2idx = {end_token: 0, "<unk>": 1}
        self.x_ids, self.x_len, self.x_label = self.files_to_ids(path)
        self.vocab_size = len(self.w2idx)


        self.train_size = int(len(self.x_ids) * 0.8)
        self.test_size = len(self.x_ids) - self.train_size

        self.train_ids, self.train_len, self.train_label = \
        self.x_ids[0:self.train_size], self.x_len[0:self.train_size], self.x_label[0:self.train_size]

        self.test_ids, self.test_len, self.test_label = \
        self.x_ids[self.train_size:-1], self.x_len[self.train_size:-1], self.x_label[self.train_size:-1]

        self.idx2w = {}
        for word in self.w2idx:
            self.idx2w[self.w2idx[word]] = word

    def get_w2idx(self, word):
        return 1 if word not in self.w2idx else self.w2idx[word]

    def files_to_ids(self, path):
        self.adventure_list = os.listdir(path + "/Adventure")
        self.documentary_list = os.listdir(path + "/Documentary")
        self.horror_list = os.listdir(path + "/Horror")
        self.romance_list = os.listdir(path + "/Romance")

        size = min(len(self.adventure_list), len(self.documentary_list), len(self.horror_list), len(self.romance_list))
        lines = []
        for i in range(size):
            with open(path + "/Adventure/" + self.adventure_list[i], "r", encoding="utf-8") as fin:
                lines.append(fin.readline())
            with open(path + "/Documentary/" + self.documentary_list[i], "r", encoding="utf-8") as fin:
                lines.append(fin.readline())
            with open(path + "/Horror/" + self.horror_list[i], "r", encoding="utf-8") as fin:
                lines.append(fin.readline())
            with open(path + "/Romance/" + self.romance_list[i], "r", encoding="utf-8") as fin:
                lines.append(fin.readline())

        cnt = {}
        for line in lines:
            for word in line.split():
                if word in cnt:
                    cnt[word] += 1
                else:
                    cnt[word] = 1
        cnt_sort = sorted(cnt.items(), key=lambda cnt:cnt[1], reverse=True)
        for word, count in cnt_sort:
            self.w2idx[word] = len(self.w2idx)
            if self.w2idx == self.max_vocab:
                break

        length, ids, label = [], [], []
        for num, line in enumerate(lines):
            id = np.zeros(self.max_len, dtype=np.int32)
            line += " <eos>"
            words = line.split()
            for i, word in enumerate(words):
                if i == self.max_len:
                    break
                if word not in self.w2idx and len(self.w2idx) < self.max_vocab:
                    self.w2idx[word] = len(self.w2idx)
                id[i] = self.get_w2idx(word)
            ids.append(id)
            length.append(i)
            n = num%4
            if n == 0 : #Adventure : 2
                label.append(2)
            elif n == 1 : #Documentary : 3
                label.append(3)
            elif n == 2 : #Horror : 1
                label.append(1)
            else : #Romance : 0
                label.append(0)

        return np.array(ids), np.array(length), np.array(label)

    def get_train(self, batch_size=20):
        pt = self.train_pt
        self.train_pt = (self.train_pt + batch_size) % self.train_size
        return self.train_ids[pt: pt+batch_size], self.train_len[pt: pt+batch_size], self.train_label[pt: pt+batch_size]

    def get_test(self, batch_size=20):
        pt = self.test_pt
        self.test_pt = (self.test_pt + batch_size) % self.test_size
        return self.test_ids[pt: pt+batch_size], self.test_len[pt: pt+batch_size], self.test_label[pt: pt+batch_size]
