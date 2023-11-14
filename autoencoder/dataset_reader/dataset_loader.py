import os
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import librosa
from torch.utils.data import Dataset
from random import randrange

from dataset_reader.data_aug import process_Rawboost_feature


def genSpoof_list(dir_meta, is_train=False, is_eval=False):
    d_meta = {}
    file_list = []
    with open(dir_meta, "r") as f:
        l_meta = f.readlines()

    if is_train:
        for line in l_meta:
            _, key, _, _, label = line.strip().split()

            file_list.append(key)
            d_meta[key] = 1 if label == "bonafide" else 0
        return d_meta, file_list

    elif is_eval:
        for line in l_meta:
            key = line.strip()
            file_list.append(key)
        return file_list
    else:
        for line in l_meta:
            _, key, _, _, label = line.strip().split()

            file_list.append(key)
            d_meta[key] = 1 if label == "bonafide" else 0
        return d_meta, file_list


def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


class Dataset_ASVspoof2019_train(Dataset):
    def __init__(self, args, list_IDs, labels, base_dir, algo, ae=False, ae_detector=False):
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.algo = algo
        self.args = args
        self.cut = 64600  # take ~4 sec audio (64600 samples)
        self.ae_mode = ae
        self.ae_detector_mode = ae_detector
        self.sr = 16000
        self.n_fft = 2048
        self.hop_size = 1024//4
        self.n_band = 128

    def __len__(self):
        return len(self.list_IDs)

    def get_audio_spectrogram(self, x):
        S = librosa.feature.melspectrogram(
            y=x,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_size,
            n_mels=self.n_band,
        )
        # melspectrogram has raised np.abs(S)**power, default power=2
        # so power_to_db is directly applicable
        S = librosa.core.power_to_db(S, ref=np.max)

        return S

    def get_min_max_scaled(self, x):
        x -= x.mean()
        x_min = x.min()
        x_max = x.max()
        nom = x - x_min
        den = x_max - x_min

        max_val = 1
        min_val = 0

        if abs(den) > 1e-4:
            return (max_val - min_val) * (nom / den) + min_val
        else:
            return nom

    def __getitem__(self, index):
        utt_id = self.list_IDs[index]
        need_to_load_audio = True
        need_to_create_spectrogram = self.ae_mode or self.ae_detector_mode
        
        if self.ae_mode or self.ae_detector_mode: # try to load spectrogram directly
            try:
                spectrogram_scaled = np.load(self.base_dir + "flac/" + utt_id + ".npy")
                spectrogram_scaled = Tensor(spectrogram_scaled)
                need_to_load_audio = False # if loaded spectrogram, no need to open audio
                need_to_create_spectrogram = False # and no need to save spectrogram
            except:
                pass

        if need_to_load_audio: # will load if no spectrogram loaded, or not ae mode
            X, fs = librosa.load(self.base_dir + "flac/" + utt_id + ".flac", sr=self.sr)
            X_pad = pad(X, self.cut)

        if need_to_create_spectrogram: # enters when cannot load spectrogram and ae_mode
            spectrogram = self.get_audio_spectrogram(X_pad)
            spectrogram_scaled = self.get_min_max_scaled(spectrogram)
            spectrogram_scaled = np.expand_dims(spectrogram_scaled, axis=0)
            np.save(self.base_dir + "flac/" + utt_id + ".npy", spectrogram_scaled)
            spectrogram_scaled = Tensor(spectrogram_scaled)
        
        if self.ae_mode:
            return spectrogram_scaled, spectrogram_scaled
        
        if self.ae_detector_mode:
            target = self.labels[utt_id]
            return spectrogram_scaled, target

        # if no ae mode, add rawboost and return waveform and label
        Y = process_Rawboost_feature(X, fs, self.args, self.algo)
        x_inp = Tensor(X_pad)
        target = self.labels[utt_id]
        return x_inp, target


class Dataset_ASVspoof2021_eval(Dataset):
    def __init__(self, list_IDs, base_dir, ae=False, ae_detector=False):
        """self.list_IDs    : list of strings (each string: utt key),"""

        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.cut = 64600  # take ~4 sec audio (64600 samples)
        self.ae_mode = ae
        self.ae_detector_mode = ae_detector
        self.sr = 16000
        self.n_fft = 2048
        self.hop_size = 1024//4
        self.n_band = 128

    def __len__(self):
        return len(self.list_IDs)

    def get_audio_spectrogram(self, x):
        S = librosa.feature.melspectrogram(
            y=x,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_size,
            n_mels=self.n_band,
        )
        # melspectrogram has raised np.abs(S)**power, default power=2
        # so power_to_db is directly applicable
        S = librosa.core.power_to_db(S, ref=np.max)

        return S

    def get_min_max_scaled(self, x):
        x -= x.mean()
        x_min = x.min()
        x_max = x.max()
        nom = x - x_min
        den = x_max - x_min

        max_val = 1
        min_val = 0

        if abs(den) > 1e-4:
            return (max_val - min_val) * (nom / den) + min_val
        else:
            return nom

    def __getitem__(self, index):
        utt_id = self.list_IDs[index]
        need_to_load_audio = True
        need_to_create_spectrogram = self.ae_mode or self.ae_detector_mode
        
        if self.ae_mode or self.ae_detector_mode: # try to load spectrogram directly
            try:
                spectrogram_scaled = np.load(self.base_dir + "flac/" + utt_id + ".npy")
                spectrogram_scaled = Tensor(spectrogram_scaled)
                need_to_load_audio = False # if loaded spectrogram, no need to open audio
                need_to_create_spectrogram = False # and no need to save spectrogram
            except:
                pass

        if need_to_load_audio: # will load if no spectrogram loaded, or not ae mode
            X, fs = librosa.load(self.base_dir + "flac/" + utt_id + ".flac", sr=self.sr)
            X_pad = pad(X, self.cut)

        if need_to_create_spectrogram: # enters when cannot load spectrogram and ae_mode
            spectrogram = self.get_audio_spectrogram(X_pad)
            spectrogram_scaled = self.get_min_max_scaled(spectrogram)
            spectrogram_scaled = np.expand_dims(spectrogram_scaled, axis=0)
            np.save(self.base_dir + "flac/" + utt_id + ".npy", spectrogram_scaled)
            spectrogram_scaled = Tensor(spectrogram_scaled)
        
        if self.ae_mode:
            return spectrogram_scaled, spectrogram_scaled
        
        if self.ae_detector_mode:
            return spectrogram_scaled, utt_id

        # if no ae mode, return waveform
        x_inp = Tensor(X_pad)
        return x_inp, utt_id
