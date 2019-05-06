import numpy as np
import pydub

def read_mp3(filename, normalized=False):
    a = pydub.AudioSegment.from_mp3(filename)
    y = np.array(a.get_array_of_samples())
    if a.channels == 2:
        y = y.reshape((-1, 2))
    if normalized:
        return a.frame_rate, np.float32(y) / 2**15
    else:
        return a.frame_rate, y
