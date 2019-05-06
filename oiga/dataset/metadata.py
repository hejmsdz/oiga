import os.path

import numpy as np
import pandas as pd

class Metadata:
    def __init__(self, root_dir, subset='small'):
        self.root_dir = root_dir
    
    def tracks(self):
        return pd.read_csv(
            os.path.join(self.root_dir, 'tracks.csv'),
            skiprows=[0, 1, 2],
            usecols=[0, 26, 40, 52],
            names=['track_id', 'artist', 'genre', 'title']
        )
    
    def query(self, track_id):
        pass

if __name__ == '__main__':
    metadata = Metadata('/mnt/data/Data/fma_metadata')
    tracks = metadata.tracks()
    print(tracks)
