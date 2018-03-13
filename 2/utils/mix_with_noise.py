import numpy as np
import librosa
from os import listdir, walk
from numpy.random import randint

def noise_file_generator(folders, sr):
    list_files = []

    for folder in folders:
        for path, subdirs, files in walk(folder):
            for name in files:
                if name[-4:] == '.wav':
                    list_files.append(path + '/' + name)

    while True:
        idx = randint(0, len(list_files))
        noise = librosa.load(list_files[idx], sr=sr)[0]
        yield noise

def preload_noise_files(folders, sr):
    list_files = []

    for folder in folders:
        for path, subdirs, files in walk(folder):
            for name in files:
                if name[-4:] == '.wav':
                    list_files.append(path + '/' + name)

    noises = []

    for noise_file in list_files:
        noise = librosa.load(noise_file, sr=sr)[0]
        noises.append(noise)

    return noises


def mix_with_noise(track, noise):
    # trim silence
    y_noise, _ = librosa.effects.trim(noise)

    # normalize volume
    med_val = np.percentile(np.abs(y_noise), 95)
    dst_med = np.percentile(np.abs(track), 95)
    y_noise = y_noise * (dst_med / med_val)

    # align files
    noise_len = y_noise.shape[0]
    track_len = track.shape[0]
    if noise_len > track_len:
        start_idx = randint(0, noise_len - track_len)
        y_noise = y_noise[start_idx: start_idx + track_len]
        return track + y_noise
    elif noise_len == track_len:
        return track + y_noise
    else:
        dst_len = (track_len + noise_len - 1) // noise_len
        y_noise = np.tile(y_noise, dst_len)[:track_len]
        return track + y_noise
