import argparse
import sys

from .preprocessing import GenreMatrix
from .analysis.beat import Beat
from .analysis.key import Key
from .genre import GenreClassifier

class CLI:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Music analysis tool')
        self.parser.add_argument('command', help='')

    def run(self):
        args = self.parser.parse_args(sys.argv[1:2])
        commands = {
            'preprocess': self.preprocess,
            'analyse': self.analyse,
        }
        commands[args.command](sys.argv[2:])
    
    def preprocess(self, args):
        parser = argparse.ArgumentParser(description='Preprocess the dataset')
        parser.add_argument('metadata_root')
        parser.add_argument('--subset', choices=['small', 'medium', 'large'], default='large')
        args = parser.parse_args(args)

        genre_matrix = GenreMatrix(args.metadata_root, args.subset)
        target, nrows = genre_matrix.save()
        print(f"Successfully written {nrows} rows to {target}")
    
    def analyse(self, args):
        parser = argparse.ArgumentParser(description='Analyse a track')
        parser.add_argument('track')
        args = parser.parse_args(args)

        beat = Beat(args.track)
        key = Key(args.track)
        genre = GenreClassifier()

        print(f"Tempo: {beat.bpm():.1f} BPM")
        print(f"Key signature: {key.key_signature()}")
        print("Note distribution:")
        for note, val in key.note_distribution().items():
            bar = '#' * int(val * 20)
            print(f"* {note:3}: {val:.03f} [{bar:_<20}]")
        
        print(f"Detected genre: {genre.predict(args.track)}")
