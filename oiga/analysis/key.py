import numpy as np
import aubio
import matplotlib.pyplot as plt

class Key:
    def __init__(self, path):
        self.src = aubio.source(path, channels=1)
        self.find_pitch = aubio.pitch('fcomb', samplerate=self.src.samplerate)
        self.find_note = aubio.notes('default', samplerate=self.src.samplerate)
    
    def find_notes(self):
        return np.array([
            self.find_pitch(block)[0]
            for block in self.src
            if len(block) == self.find_pitch.hop_size and self.find_pitch(block)
        ])
    
    def frequency_to_pitch(self, frequency):
        return ((np.rint(12 * np.log2(frequency / 440)) - 3) % 12).astype(int)

    def key_signature(self):
        notes = self.frequency_to_pitch(self.find_notes())
        distribution = np.bincount(notes)

        pitches = np.arange(12)
        labels = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
        plt.bar(pitches, distribution)
        plt.xticks(pitches, labels)
        plt.show()

if __name__ == '__main__':
    import sys
    path = sys.argv[1]    
    print(Key(path).key_signature())
