import numpy as np
import aubio
import matplotlib.pyplot as plt

class Key:
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
        distribution = np.bincount(notes)

        labels = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
        tonic = np.argmax(distribution)
        tonic_note = labels[tonic]
        minor_third = (tonic + 3) % 12
        major_third = (tonic + 4) % 12
        
        if distribution[major_third] > distribution[minor_third]:
            mode = 'major'
        else:
            mode = 'minor'
        
        pitches = np.arange(12)
        plt.bar(pitches, distribution)
        plt.xticks(pitches, labels)
        plt.show()
        
        return (tonic_note, mode)

if __name__ == '__main__':
    import sys
    path = sys.argv[1]    
    print(Key(path).key_signature())
