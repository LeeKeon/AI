from bs4 import BeautifulSoup
import urllib.request as req
import pandas as pd
import os
label_path = "../../dataset/MovieGenre_fix_leekeon_total6000.csv"
movie_df = pd.read_csv(label_path, encoding="ISO-8859-1")

for movie in movie_df.values:
    #print(movie[3]) #link
    #print(movie[6]) #genre
    #print(movie[1]) #id
    
    # movie[0] ID
    # movie[7] Genre
    # movie[8] Image link 
    try:
        if(movie[7]=='Documentary'):
            url = str(movie[3]) + '/plotsummary?ref_=tt_ov_pl'
            res = req.urlopen(url)
            soup = BeautifulSoup(res, "html.parser")   
            synop = soup.find('li', 'ipl-zebra-list__item')
            synop = synop.find('p')
            synop = str(synop).replace('</p>', '')
            synop = synop.replace('<p>', '')
    
            save_path = '../../dataset/poster_txt/new/Documentary/' + str(movie[0]) + '.txt'
            file = open(save_path,'w')
            file.write(synop)
            file.flush()
    except:
        print(movie[1])
