import numpy as np
from scipy import misc
import scipy
import os
import pylab as plt
import pandas as pd
import random
from keras.utils import to_categorical

class movie_data(object):
    
    def __init__(self):
        self.train_pt, self.test_pt = 0, 0
        self.img_path = "./dataset/poster_image/new/"
        self.label_path = "./dataset/MovieGenre_fix_leekeon_total6000.csv"
        
        self.genre_2_labelid = {}
        self.movieid_2_genre = {}
        self.labelid_2_genre = {}

        self.movie_df = pd.read_csv(self.label_path, encoding="ISO-8859-1")
        
        self.dict_init()
        
        files = [file for file in os.listdir(self.img_path) 
             if os.path.isfile(os.path.join(self.img_path,file))]
        
        random.shuffle(files)

        self.train_size = int(len(files) * 0.8)
        self.test_size = len(files) - self.train_size

        train_files = files[0:self.train_size]
        test_files = files[self.train_size:len(files)]
                
        self.x_train, self.y_train = self.init_data(train_files)
        self.x_test, self.y_test = self.init_data(test_files)
        
        self.data_summary()
        
    def data_summary(self):
        print('x_train size', len(self.x_train))
        print('y_train size', len(self.y_train))
        print('x_test  size', len(self.x_test))
        print('y_test  size', len(self.y_test))
            
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
                 if(movie[7]=='Adventure' 
                    or movie[7]=='Documentary'
                    or movie[7]=='Romance'
                    or movie[7]=='Horror'):
                      self.genre_2_labelid[movie[7]] = len(self.genre_2_labelid)

        for i, genre in enumerate(self.genre_2_labelid):
            self.labelid_2_genre[i] = genre
            
    def get_train(self, batch_size=20):
        pt = self.train_pt
        self.train_pt = (self.train_pt + batch_size) % self.train_size
        return self.x_train[pt: pt+batch_size], self.y_train[pt: pt+batch_size]

    def get_test(self, batch_size=20):
        pt = self.test_pt
        self.test_pt = (self.test_pt + batch_size) % self.test_size
        return self.x_test[pt: pt+batch_size], self.y_test[pt: pt+batch_size]
    
    def get_train_dataset(self):
        return self.x_train, to_categorical(self.y_train,4)
    
    def get_test_dataset(self):
        return self.x_test, to_categorical(self.y_test,4)
    
    def to_rgb2(self, im):
    # as 1, but we use broadcasting in one line
        w, h = im.shape
        ret = np.empty((w, h, 3), dtype=np.uint8)
        ret[:, :, :] = im[:, :, np.newaxis]    
        return ret
    
    def init_data(self, files):
        x = []
        y = []
        
        for file in files:
            fid = int(file.split('.')[0])
            genre = self.movieid_2_genre[fid]
            if(genre=='Adventure' 
               or genre=='Documentary'
               or genre=='Romance'
               or genre=='Horror'):
                    img = self.min_max_scaling(misc.imread(self.img_path+file))
                    if(img.ndim==2):
                        img = self.to_rgb2(img)
                    img = img[...,:3]
                    x.append(img)
                    fid = int(file.split('.')[0])
                    lid = self.genre_2_labelid[self.movieid_2_genre[fid]]
                    y.append(lid)
        return x, y