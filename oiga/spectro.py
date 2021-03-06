from scipy import signal
from scipy.io import wavfile
from scipy.ndimage.interpolation import zoom
from skimage.measure import block_reduce
import matplotlib.pyplot as plt
from .io import read_mp3
import numpy as np
import sys
import os

def generate_spectrograms(track, sr, div):
    subtrack_len = len(track)//div
    for d in range(div):
        subtrack = track[d*subtrack_len : (d+1)*subtrack_len]
        freq, times, spec = signal.spectrogram(subtrack, fs = sr)
        spec = 10*np.log(spec)

        spec = block_reduce(spec,(2,5),func=np.mean)
        yield spec

if __name__=='__main__':
        print(sys.argv[1])
        files = os.listdir(sys.argv[1])
        files = [f for f in files if f.split('.')[-1]=='mp3']
        for f in files:
                sr, x = read_mp3(sys.argv[1]+'/'+f)

                x = np.average(x, axis=1)
                
                for i,s in enumerate(generate_spectrograms(x, sr, 12)):
                        plt.pcolormesh(s)
                plt.show()
                break


