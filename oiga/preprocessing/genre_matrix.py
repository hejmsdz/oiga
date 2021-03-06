import os.path

import numpy as np
import pandas as pd

class GenreMatrix:
    def __init__(self, root_dir, subset='large'):
        self.root_dir = root_dir
        self.subset = subset
        self.tracks = self.get_tracks()
        self.genres = self.get_genres()
        self.limit_subset()
        self.process_genres()
    
    def get_tracks(self):
        return pd.read_csv(
            os.path.join(self.root_dir, 'tracks.csv'),
            skiprows=[0, 1, 2],
            usecols=[0, 31, 32, 41],
            names=['track_id', 'split', 'subset', 'genres']
        )
    
    def get_genres(self):
        return pd.read_csv(
            os.path.join(self.root_dir, 'genres.csv'),
            usecols=['genre_id', 'title', 'top_level']
        )

    def process_genres(self):
        genres_dict = {}
        top_level_genres = self.genres[self.genres.top_level == self.genres.genre_id]
        subgenres = self.genres[self.genres.top_level != self.genres.genre_id]
        for _idx, genre in top_level_genres.iterrows():
            genres_dict[genre.genre_id] = genre.title
            self.tracks[genre.title] = 0
        
        for _idx, genre in subgenres.iterrows():
            genres_dict[genre.genre_id] = genres_dict[genre.top_level]
        
        for idx, track in self.tracks[self.tracks.genres != '[]'].iterrows():
            genres = map(int, track.genres[1:-1].split(', '))
            for genre_id in genres:
                self.tracks.at[idx, genres_dict[genre_id]] = 1
        
        del self.tracks['genres']
    
    def limit_subset(self):
        if self.subset == 'medium' or self.subset == 'small':
            self.tracks = self.tracks[self.tracks.subset != 'large']
        if self.subset == 'small':
            self.tracks = self.tracks[self.tracks.subset != 'medium']
        del self.tracks['subset']
        
    
    def save(self):
        target = os.path.realpath(os.path.join(self.root_dir, 'top_level_genres.csv'))
        self.tracks.to_csv(target, index=False)
        return target, len(self.tracks)
