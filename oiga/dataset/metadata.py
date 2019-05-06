import os.path

import numpy as np
import pandas as pd

class Metadata:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.tracks = self.get_tracks()
        self.genres = self.get_genres()
        self.add_genres_to_tracks()
    
    def get_tracks(self):
        return pd.read_csv(
            os.path.join(self.root_dir, 'tracks.csv'),
            skiprows=[0, 1, 2],
            usecols=[0, 26, 41, 52],
            names=['track_id', 'artist', 'genres', 'title']
        )
    
    def get_genres(self):
        return pd.read_csv(
            os.path.join(self.root_dir, 'genres.csv'),
            usecols=['genre_id', 'title', 'top_level']
        )

    def add_genres_to_tracks(self):
        genres_dict = {}
        top_level_genres = self.genres[self.genres.top_level == self.genres.genre_id]
        subgenres = self.genres[self.genres.top_level != self.genres.genre_id]
        for _idx, genre in top_level_genres.iterrows():
            genres_dict[genre.genre_id] = genre.title
            self.tracks[genre.title] = 0
        
        for _idx, genre in subgenres.iterrows():
            genres_dict[genre.genre_id] = genres_dict[genre.top_level]
        
        for idx, track in self.tracks[self.tracks.genres != '[]'].iterrows():
            genres = map(int, track.genres[1:-1].split(','))
            for genre_id in genres:
                self.tracks.at[idx, genres_dict[genre_id]] = 1
        
    def query(self, track_id):
        row = track_id[self.tracks.track_id == track_id]
        return row

if __name__ == '__main__':
    metadata = Metadata('/mnt/data/Data/fma_metadata')
    print(metadata.tracks)
