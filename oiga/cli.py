import argparse
import sys

from .preprocessing import GenreMatrix

class CLI:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Music analysis tool')
        self.parser.add_argument('command', help='')

    def run(self):
        args = self.parser.parse_args(sys.argv[1:2])
        commands = {
            'preprocess': self.preprocess,
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
