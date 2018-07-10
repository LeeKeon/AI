import numpy as np
from scipy import misc
import scipy
import os
import pylab as plt
import pandas as pd
import random
from keras.utils import to_categorical

class movie_data(object):

    def __init__(self, max_vocab=30000, max_len=500, end_token="<eos>", data_type="both"):
        self.train_pt, self.test_pt = 0, 0
        self.img_path = "../dataset/poster_image/new/"
        self.label_path = "../dataset/MovieGenre_fix_leekeon_total6000.csv"
        self.plot_path = "../dataset/poster_txt/new/"
        self.max_len = max_len
        self.max_vocab = max_vocab
        self.data_type = data_type

        self.genre_2_labelid = {}
        self.movieid_2_genre = {}
        self.labelid_2_genre = {}

        self.movie_df = pd.read_csv(self.label_path, encoding="ISO-8859-1")

        self.dict_init()

        files = [file for file in os.listdir(self.img_path)
             if os.path.isfile(os.path.join(self.img_path,file))]

        random.seed(777)
        random.shuffle(files)

        self.train_size = int(len(files) * 0.8)
        self.test_size = len(files) - self.train_size

        train_files = files[0:self.train_size]
        test_files = files[self.train_size:len(files)]



        if(self.data_type == "both"):
            # plot
            #단어 전 처리
            self.w2idx = {end_token: 0, "<unk>": 1}
            self.files_to_word(files)

            #return movieid, x, ids, length, y
            self.movieid_train, self.x_img_train, self.x_ids_train, self.x_len_train, self.y_train = self.init_data(train_files)
            self.movieid_test, self.x_img_test, self.x_ids_test, self.x_len_test, self.y_test = self.init_data(test_files)

        elif(self.data_type == "image"):
            print('No more Support')
            #self.movieid_train, self.x_img_train, self.y_train = self.init_data(train_files)
            #self.movieid_test,self.x_img_test, self.y_test = self.init_data(test_files)

        self.data_summary()

    def data_summary(self):
        print('x_train size', len(self.x_img_train))
        print('y_train size', len(self.y_train))
        print('x_test  size', len(self.x_img_test))
        print('y_test  size', len(self.y_test))
        print('x_ids_train  size', len(self.x_ids_train))
        print('x_len_train  size', len(self.x_len_train))
        print('x_ids_test  size', len(self.x_ids_test))
        print('x_len_test  size', len(self.x_len_test))

    def preprocess(self, img, size=(134, 91)):
        img = scipy.misc.imresize(img, size)
        img = img.astype(np.float32)
        img = (img / 127.5) - 1.

        return img

    def min_max_scaling(self, x, size=(134, 91)):
        x = scipy.misc.imresize(x, size)
        x_np = np.asarray(x)

        return (x_np - x_np.min()) / (x_np.max() - x_np.min() + 1e-7)

    def reverse_min_max_scaling(org_x, x):
        org_x_np = np.asarray(org_x)
        x_np = np.asarray(x)
        return (x_np * (org_x_np.max() - org_x_np.min() + 1e-7)) + org_x_np.min()

    def dict_init(self):
        for movie in self.movie_df.values:
            self.movieid_2_genre[movie[0]] = movie[7]

            if movie[7] not in self.genre_2_labelid:
                self.genre_2_labelid[movie[7]] = len(self.genre_2_labelid)

        for i, genre in enumerate(self.genre_2_labelid):
            self.labelid_2_genre[i] = genre

    def get_train(self, batch_size=20):
        pt = self.train_pt
        self.train_pt = (self.train_pt + batch_size) % self.train_size
        return self.x_img_train[pt: pt+batch_size], self.x_ids_train[pt: pt+batch_size], self.x_len_train[pt: pt+batch_size], self.y_train[pt: pt+batch_size]

    def get_test(self, batch_size=20):
        pt = self.test_pt
        self.test_pt = (self.test_pt + batch_size) % self.test_size
        return self.movieid_test[pt: pt+batch_size], self.x_img_test[pt: pt+batch_size], self.x_ids_test[pt: pt+batch_size], self.x_len_test[pt: pt+batch_size],self.y_test[pt: pt+batch_size]
    
    def get_test_movieid(self, batch_size=20):
        pt = self.test_pt
        return self.movieid_test[pt: pt+batch_size]
    
    def get_test_ids(self, batch_size=20):
        pt = self.test_pt
        return self.movieid_test[pt-batch_size: pt]

    def get_train_dataset(self):
        return self.x_img_train, to_categorical(self.y_train,4)

    def get_test_dataset(self):
        return self.x_img_test, to_categorical(self.y_test,4)

    def to_rgb2(self, im):
    # as 1, but we use broadcasting in one line
        w, h = im.shape
        ret = np.empty((w, h, 3), dtype=np.uint8)
        ret[:, :, :] = im[:, :, np.newaxis]
        return ret

    def init_data(self, files, plot_path="../dataset/poster_txt/new/", img_path="../dataset/poster_image/new/"):
        movieid = []
        x = []
        y = []
        lines = []
        length, ids = [], []
        for file in files:
            filename = file.split('.')[0] + '.txt'
            with open(plot_path + filename, "r", encoding="utf-8") as fin:
                lines.append(fin.readline())

            fid = int(file.split('.')[0])
            genre = self.movieid_2_genre[fid]
            img = self.min_max_scaling(misc.imread(img_path+file))
            if(img.ndim==2):
                img = self.to_rgb2(img)
            img = img[...,:3]
            x.append(img)
            # fid = int(file.split('.')[0])
            lid = self.genre_2_labelid[self.movieid_2_genre[fid]]
            movieid.append(fid)
            y.append(lid)

        for line in lines:
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
            
        return movieid, x, ids, length, y

    def get_w2idx(self, word):
        return 1 if word not in self.w2idx else self.w2idx[word]

    def files_to_word(self, files):
        lines = []
        for file in files:
            filename = file.split('.')[0] + '.txt'
            with open(self.plot_path + filename, "r", encoding="utf-8") as fin:
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

    def init_plot_data(self, files):
        lines = []
        for file in files:
            filename = file.split('.')[0] + '.txt'
            with open(self.plot_path + filename, "r", encoding="utf-8") as fin:
                lines.append(fin.readline())
        
        length, ids = [], []
        for line in lines:
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
        return np.array(ids), np.array(length)
    
    def init_data_test(self, files, plot_path="../dataset/poster_txt/new/", img_path="../dataset/poster_image/new/"):
        movieid = []
        x = []
        y = []
        lines = []
        length, ids = [], []
        for file in files:
            filename = file.split('.')[0] + '.txt'
            with open(plot_path + filename, "r", encoding="utf-8") as fin:
                lines.append(fin.readline())

            #fid = int(file.split('.')[0])
            genre = 'Adventure'
            img = self.min_max_scaling(misc.imread(img_path+file))
            if(img.ndim==2):
                img = self.to_rgb2(img)
            img = img[...,:3]
            x.append(img)

        for line in lines:
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
            
        return x, ids, length
