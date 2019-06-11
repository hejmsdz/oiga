import numpy as np
import aubio

class Beat:
    def __init__(self, path, min_bpm=60, max_bpm=180):
        self.src = aubio.source(path, channels=1)
        self.check_beat = aubio.tempo(samplerate=self.src.samplerate)
        self.min_bpm = min_bpm
        self.max_bpm = max_bpm
    
    def find_beats(self):
        return [
            self.check_beat.get_last_s()
            for block in self.src
            if len(block) == self.check_beat.hop_size and self.check_beat(block)
        ]
    
    def bpm(self):
        beats = self.find_beats()
        diffs = np.diff(beats)
        diffs = self.normalize_diffs(diffs)
        return 60 / np.median(diffs)
        
    def normalize_diffs(self, diffs):
        max_diff = 60 / self.min_bpm
        min_diff = 60 / self.max_bpm
        
        while np.any(diffs > max_diff):
            diffs[diffs > max_diff] /= 2

        while np.any(diffs < min_diff):
            diffs[diffs < min_diff] *= 2
        
        return diffs


if __name__ == '__main__':
    import sys
    path = sys.argv[1]    
    print(Beat(path).bpm())
