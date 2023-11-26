import os
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import librosa
from torch.utils.data import Dataset
from random import randrange
import pandas as pd
from torchaudio.functional import highpass_biquad


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


class Dataset_ASVspoof(Dataset):
    def __init__(self, list_IDs, base_dir, ae, ae_detector, noise_ae, noise_detector):
        super().__init__()
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.ae_mode = ae
        self.ae_detector_mode = ae_detector
        self.noise_ae = noise_ae
        self.noise_detector = noise_detector
        self._init_defaults()

    def _init_defaults(self):
        self.cut = 64600  # take ~4 sec audio (64600 samples)
        self.sr = 16000
        self.n_fft = 2048
        self.hop_size = 1024 // 4
        self.n_band = 128

    def __len__(self):
        return len(self.list_IDs)

    def pad(self, x, max_len=64600):
        x_len = x.shape[0]
        if x_len >= max_len:
            return x[:max_len]
        # need to pad
        num_repeats = int(max_len / x_len) + 1
        padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
        return padded_x

    def load_spectrogram(self, utt_id):
        spectrogram_scaled = np.load(self.base_dir + "flac/" + utt_id + ".npy")
        return Tensor(spectrogram_scaled)

    def create_and_save_spectrogram(self, utt_id, X_pad):
        spectrogram = self.get_audio_spectrogram(X_pad)
        spectrogram_scaled = self.get_min_max_scaled(spectrogram)
        spectrogram_scaled = np.expand_dims(spectrogram_scaled, axis=0)
        np.save(self.base_dir + "flac/" + utt_id + ".npy", spectrogram_scaled)
        return Tensor(spectrogram_scaled)

    def create_and_save_noise_spectrogram(self, utt_id, wave):
        spectrogram_scaled = self.get_noise_processed_spectrogram(wave)
        np.save(self.base_dir + "flac/" + utt_id + "_noise.npy", spectrogram_scaled)
        return Tensor(spectrogram_scaled)

    def get_noise_processed_spectrogram(self, wave):
        noise_audio = highpass_biquad(Tensor(wave), sample_rate=self.sr, cutoff_freq=7e2).numpy()
        spectrogram = self.get_audio_spectrogram(noise_audio)
        spectrogram_scaled = self.get_min_max_scaled(spectrogram)
        spectrogram_scaled = np.expand_dims(spectrogram_scaled, axis=0)
        return spectrogram_scaled

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

    def load_audio_and_pad(self, utt_id):
        X, fs = librosa.load(self.base_dir + "flac/" + utt_id + ".flac", sr=self.sr)
        X_pad = self.pad(X, self.cut)
        return X_pad


class Dataset_ASVspoof2019_train(Dataset_ASVspoof):
    def __init__(
        self,
        args,
        list_IDs,
        labels,
        base_dir,
        algo,
        ae=False,
        ae_detector=False,
        noise_ae=False,
        noise_detector=False,
    ):
        super().__init__(
            list_IDs,
            base_dir,
            ae,
            ae_detector,
            noise_ae,
            noise_detector,
        )
        self.labels = labels
        self.algo = algo
        self.args = args

    def __getitem__(self, index):
        utt_id = self.list_IDs[index]
        spectrogram_scaled = None

        if self.ae_mode:
            try:
                spectrogram_scaled = self.load_spectrogram(utt_id)
            except FileNotFoundError:
                audio = self.load_audio_and_pad(utt_id)
                spectrogram_scaled = self.create_and_save_spectrogram(utt_id, audio)
            return spectrogram_scaled, spectrogram_scaled

        if self.ae_detector_mode:
            try:
                spectrogram_scaled = self.load_spectrogram(utt_id)
            except FileNotFoundError:
                audio = self.load_audio_and_pad(utt_id)
                spectrogram_scaled = self.create_and_save_spectrogram(utt_id, audio)
            target = self.labels[utt_id]
            return spectrogram_scaled, target

        if self.noise_ae:
            try:
                noise_spectrogram = self.load_spectrogram(utt_id + "_noise")
            except FileNotFoundError:
                audio = self.load_audio_and_pad(utt_id)
                noise_spectrogram = self.create_and_save_noise_spectrogram(utt_id, audio)
            return noise_spectrogram, noise_spectrogram

        if self.noise_detector:
            try:
                spectrogram_scaled = self.load_spectrogram(utt_id)
            except FileNotFoundError:
                audio = self.load_audio_and_pad(utt_id)
                spectrogram_scaled = self.create_and_save_spectrogram(utt_id, audio)

            try:
                noise_spectrogram = self.load_spectrogram(utt_id + "_noise")
            except FileNotFoundError:
                audio = self.load_audio_and_pad(utt_id)
                noise_spectrogram = self.create_and_save_noise_spectrogram(utt_id, audio)

            target = self.labels[utt_id]
            return (spectrogram_scaled, noise_spectrogram), target

        # If no ae mode, return waveform and label
        X_pad = self.load_audio_and_pad(utt_id)
        x_inp = Tensor(X_pad)
        target = self.labels[utt_id]
        return x_inp, target


class Dataset_ASVspoof2021_eval(Dataset_ASVspoof):
    def __init__(
        self,
        list_IDs,
        base_dir,
        ae=False,
        ae_detector=False,
        noise_ae=False,
        noise_detector=False,
    ):
        super().__init__(
            list_IDs,
            base_dir,
            ae,
            ae_detector,
            noise_ae,
            noise_detector,
        )

    def __getitem__(self, index):
        utt_id = self.list_IDs[index]
        spectrogram_scaled = None

        if self.ae_mode:
            try:
                spectrogram_scaled = self.load_spectrogram(utt_id)
            except FileNotFoundError:
                audio = self.load_audio_and_pad(utt_id)
                spectrogram_scaled = self.create_and_save_spectrogram(utt_id, audio)
            return spectrogram_scaled, spectrogram_scaled

        if self.ae_detector_mode:
            try:
                spectrogram_scaled = self.load_spectrogram(utt_id)
            except FileNotFoundError:
                audio = self.load_audio_and_pad(utt_id)
                spectrogram_scaled = self.create_and_save_spectrogram(utt_id, audio)
            return spectrogram_scaled, utt_id

        if self.noise_ae:
            try:
                noise_spectrogram = self.load_spectrogram(utt_id + "_noise")
            except FileNotFoundError:
                audio = self.load_audio_and_pad(utt_id)
                noise_spectrogram = self.create_and_save_noise_spectrogram(utt_id, audio)
            return noise_spectrogram, noise_spectrogram

        if self.noise_detector:
            try:
                spectrogram_scaled = self.load_spectrogram(utt_id)
            except FileNotFoundError:
                audio = self.load_audio_and_pad(utt_id)
                spectrogram_scaled = self.create_and_save_spectrogram(utt_id, audio)

            try:
                noise_spectrogram = self.load_spectrogram(utt_id + "_noise")
            except FileNotFoundError:
                audio = self.load_audio_and_pad(utt_id)
                noise_spectrogram = self.create_and_save_noise_spectrogram(utt_id, audio)

            return (spectrogram_scaled, noise_spectrogram), utt_id

        # If no ae mode, return waveform and label
        X_pad = self.load_audio_and_pad(utt_id)
        x_inp = Tensor(X_pad)
        return x_inp, utt_id


def genSpoof_list_inthewild(dir_meta):
    df = pd.read_csv(dir_meta)
    d_meta = {}
    file_list = []

    for index, row in df.iterrows():
        key = row["file"]
        label = row["label"]
        file_list.append(key)
        d_meta[key] = 1 if "bona-fide" in label else 0

    return d_meta, file_list


class Dataset_InTheWild_eval(Dataset_ASVspoof):
    def __init__(self, list_IDs, labels, base_dir, ae=False, ae_detector=False):
        """self.list_IDs	: list of strings (each string: utt key),
        self.labels      : dictionary (key: utt key, value: label integer)"""
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.cut = 64600  # take ~4 sec audio (64600 samples)
        self.ae_mode = ae
        self.ae_detector_mode = ae_detector
        self.sr = 16000
        self.n_fft = 2048
        self.hop_size = 1024 // 4
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

    def pad(self, x, max_len=64600):
        x_len = x.shape[0]
        if x_len >= max_len:
            return x[:max_len]
        # need to pad
        num_repeats = int(max_len / x_len) + 1
        padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
        return padded_x

    def __getitem__(self, index):
        utt_id = self.list_IDs[index]
        need_to_load_audio = True
        need_to_create_spectrogram = self.ae_mode or self.ae_detector_mode

        if self.ae_mode or self.ae_detector_mode:  # try to load spectrogram directly
            try:
                spectrogram_scaled = np.load(self.base_dir + utt_id + ".npy")
                spectrogram_scaled = Tensor(spectrogram_scaled)
                need_to_load_audio = False  # if loaded spectrogram, no need to open audio
                need_to_create_spectrogram = False  # and no need to save spectrogram
            except:
                pass

        if need_to_load_audio:  # will load if no spectrogram loaded, or not ae mode
            X, fs = librosa.load(self.base_dir + utt_id, sr=self.sr)
            X_pad = self.pad(X, self.cut)

        if need_to_create_spectrogram:  # enters when cannot load spectrogram and ae_mode
            spectrogram = self.get_audio_spectrogram(X_pad)
            spectrogram_scaled = self.get_min_max_scaled(spectrogram)
            spectrogram_scaled = np.expand_dims(spectrogram_scaled, axis=0)
            np.save(self.base_dir + utt_id.replace(".wav", "") + ".npy", spectrogram_scaled)
            spectrogram_scaled = Tensor(spectrogram_scaled)

        if self.ae_mode:
            return spectrogram_scaled, spectrogram_scaled

        key = self.list_IDs[index]
        label = self.labels[key]
        if self.ae_detector_mode:
            return spectrogram_scaled, label

        # if no ae mode, return waveform
        x_inp = Tensor(X_pad)
        return x_inp, label
