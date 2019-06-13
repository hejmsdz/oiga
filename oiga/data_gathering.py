import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .io import read_mp3
from .spectro import generate_spectrograms
import pickle

def collect_data(genres=['Hip-Hop','Rock']):
    path_to_meta = '/home/imegirin/Documents/Studia/TechProgr/fma_metadata/fma_metadata/'
    path_to_tracks = '/home/imegirin/Documents/Studia/TechProgr/fma_small/'
    df = pd.read_csv(path_to_meta+'top_level_genres.csv', dtype={'track_id':str})

    trn_x = []
    trn_Y = []
    tst_x = []
    tst_Y = []

    for i, g in enumerate(genres):
        genre_df = df[df[g]==1]
        for track_id in genre_df[genre_df['split']=='training']['track_id']:
            track_name = track_id.zfill(6)
            sr, td = read_mp3(path_to_tracks+track_name[:3]+'/'+track_name+'.mp3')
            if len(td.shape)!=1:
                td = np.average(td, axis=1)
            for subtrack in generate_spectrograms(td, sr, 12):
                if subtrack.shape==(65,99):
                    trn_x.append(subtrack)
                    label = np.zeros(len(genres))
                    label[i] = 1.
                    trn_Y.append(label)

    for i, g in enumerate(genres):
        genre_df = df[df[g]==1]
        for track_id in genre_df[genre_df['split']=='test']['track_id']:
            track_name = track_id.zfill(6)
            sr, td = read_mp3(path_to_tracks+track_name[:3]+'/'+track_name+'.mp3')
            if len(td.shape)!=1:
                td = np.average(td, axis=1)
            for subtrack in generate_spectrograms(td, sr, 12):
                if subtrack.shape==(65,99):
                    tst_x.append(subtrack)
                    label = np.zeros(len(genres))
                    label[i] = 1.
                    tst_Y.append(label)
    
    print(trn_x)
    print(trn_Y)
    print(tst_x)
    print(tst_Y)
    pickle.dump((np.array(trn_x),
                 np.array(trn_Y), 
                 np.array(tst_x), 
                 np.array(tst_Y)), open('dataset.p', 'wb'))

def load_dataset_from_pckl(filename):
    return pickle.load(open(filename, 'rb'))

if __name__=="__main__":
    collect_data()