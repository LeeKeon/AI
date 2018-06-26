import os
import random
import numpy as np
import cv2

class Data(object):
    def __init__(self, path="../../dataset/pacscal-sentences",
                 max_vocab=100, max_len=100,
                 img_x_size=256, img_y_size=256, end_token="<eos>"):
        self.path = path
        self.max_len = max_len
        self.max_vocab = max_vocab

        self.img_x_size = img_x_size
        self.img_y_size = img_y_size

        self.w2idx = {end_token: 0, "<unk>": 1}
        img_list, text_list= self.read_img(path), self.read_text(path)
        img_text = [(img_list[i], text_list[i]) for i in range(len(img_list))]
        random.shuffle(img_text)

        img_list, text_list = [], []
        for img, text in img_text:
            img_list.append(img)
            text_list.append(text)

        self.imgs = self.img_files(img_list)
        self.ids, self.lengths = self.text_files(text_list)
        self.vocab_size = len(self.w2idx)
        print(" - vocab_size: {}".format(self.vocab_size))

        self.total_size = len(self.imgs)
        self.train_pt, self.test_pt = 0, int(self.total_size * 0.9)

        self.idx2w = {}
        for word in self.w2idx:
            self.idx2w[self.w2idx[word]] = word

    def get_w2idx(self, word):
        return 1 if word not in self.w2idx else self.w2idx[word]

    def read_img(self, path):
        file_list = []
        for (path, dir, files) in os.walk(path + "/dataset"):
            for filename in files:
                file_list.append("/".join([path, filename]))
        file_list.sort()
        return file_list

    def read_text(self, path):
        file_list = []
        for (path, dir, files) in os.walk(path + "/sentence"):
            for filename in files:
                file_list.append("/".join([path, filename]))
        file_list.sort()
        return file_list

    def img_files(self, file_list):
        images = []
        for file_name in file_list:
            img = cv2.imread(file_name, cv2.IMREAD_COLOR)
            img_resized = cv2.resize(img, dsize=(self.img_x_size, self.img_y_size))
            images.append((img_resized - 128.0) / 128.0)

        return np.array(images)

    def text_files(self, file_list):
        ids = []
        lengths = []

        for file_name in file_list:
            with open(file_name, "r") as fin:
                lines = fin.readlines()

            id = np.zeros((len(lines), self.max_len), dtype=np.int32)
            length = np.zeros(len(lines), dtype=np.int32)

            for i, line in enumerate(lines):
                line += "<eos>"
                for j, word in enumerate(line.split()):
                    if j == self.max_len:
                        break
                    if word not in self.w2idx:
                        self.w2idx[word] = len(self.w2idx)
                    id[i][j] = self.w2idx[word]
                length[i] = j+1
            ids.append(id)
            lengths.append(length)

        return np.array(ids), np.array(lengths)


    def get_train(self, batch_size=20):
        pt = self.train_pt
        self.train_pt = (self.train_pt + batch_size) % int(self.total_size * 0.9)
        return self.imgs[pt: pt+batch_size], self.ids[pt: pt+batch_size], self.lengths[pt: pt+batch_size]

    def get_test(self, batch_size=20):
        pt = self.test_pt
        self.test_pt = (self.test_pt + batch_size) % int(self.total_size * 0.1) + int(self.total_size * 0.9)
        return self.imgs[pt: pt + batch_size], self.ids[pt: pt + batch_size], self.lengths[pt: pt + batch_size]