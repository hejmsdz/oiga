import numpy as np
import aubio
import matplotlib.pyplot as plt

class Key:
    NOTES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
    NOTES_SHARP = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    NOTES_FLAT = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']

    KEYS_WITH_SHARPS = {
        'major': ['G', 'D', 'A', 'E', 'B'],
        'minor': ['E', 'B', 'F#', 'C#', 'G#']
    }

    def __init__(self, path):
        self.src = aubio.source(path, channels=1)
        self.find_note = aubio.notes('default', samplerate=self.src.samplerate)
    
    def find_notes(self):
        return np.array([
            int(note) % 12 for note in (
                self.find_note(block)[0]
                for block in self.src
                if len(block) == self.find_note.hop_size
            ) if note != 0
        ])

    def key_signature(self):
        notes = self.find_notes()
        self.distribution = np.bincount(notes)

        tonic = np.argmax(self.distribution)
        tonic_note = self.NOTES[tonic]
        minor_third = (tonic + 3) % 12
        major_third = (tonic + 4) % 12
        
        if self.distribution[major_third] >= self.distribution[minor_third]:
            mode = 'major'
        else:
            mode = 'minor'
        
        self.detected = (tonic_note, mode)
        return self.detected
    
    def note_distribution(self):
        hist = self.distribution / self.distribution.max()

        if self.detected[0] in self.KEYS_WITH_SHARPS[self.detected[1]]:
            notes = self.NOTES_SHARP
        else:
            notes = self.NOTES_FLAT

        return dict(zip(notes, hist))

if __name__ == '__main__':
    import sys
    path = sys.argv[1]    
    print(Key(path).key_signature())
