from bs4 import BeautifulSoup
import urllib.request as req
import pandas as pd
import sys
label_path = "./dataset/MovieGenre_genre500_final.csv"
movie_df = pd.read_csv(label_path, encoding="ISO-8859-1")

for movie in movie_df.values:
    #print(movie[3]) #link
    #print(movie[6]) #genre
    #print(movie[1]) #id
    try:
        url = str(movie[3]) + '/plotsummary?ref_=tt_ov_pl'
        res = req.urlopen(url)
        soup = BeautifulSoup(res, "html.parser")   
        synop = soup.find('li', 'ipl-zebra-list__item')
        synop = synop.find('p')
        synop = str(synop).replace('</p>', '')
        synop = synop.replace('<p>', '')
    
        save_path = './dataset/poster_txt/' + str(movie[7]) + '/' + str(movie[1]) + '.txt'
        file = open(save_path,'w')
        file.write(synop)
        file.flush()
    except KeyboardInterrupt:
        sys.exit()
    except:
        print(movie[1])
