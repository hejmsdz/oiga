import os.path

import numpy as np

from .cnn_model import create_model
from .spectro import generate_spectrograms
from .io import read_mp3

class GenreClassifier:
    def __init__(self, weights_file=None):
        if weights_file is None:
            weights_file = os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                '..', 'assets', 'model.h5'
            )
        self.model = create_model((1,65,99), 2)
        self.model.load_weights(weights_file)

    def predict(self, path):
        sample_rate, track = read_mp3(path)
        track = np.average(track, axis=1)
        for spec in generate_spectrograms(track, sample_rate, 12):
            print(self.model.predict(spec))

if __name__ == '__main__':
    import sys
    path = sys.argv[1]    
    print(GenreClassifier().predict(path))
