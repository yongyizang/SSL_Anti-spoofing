import os
import numpy as np
import torch
from torch import Tensor
import librosa
from torch.utils.data import Dataset
from RawBoost import ISD_additive_noise, LnL_convolutive_noise, SSI_additive_noise, normWav

def genSpoof_list(dir_meta, is_train=False, is_eval=False):
    d_meta = {}
    file_list = []
    
    with open(dir_meta, 'r') as f:
        l_meta = f.readlines()

    if is_eval:
        for line in l_meta:
            key = line.strip()
            file_list.append(key)
        return file_list
    else:
        for line in l_meta:
            _, key, _, target_type, label = line.strip().split()
            file_list.append(key)
            d_meta[key] = {
                'label': 1 if label == 'bonafide' else 0,
                'target_type': target_type
            }
        return d_meta, file_list

def parse_protocol_file(file_path):
    files = []
    
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split('-')
            if len(parts) >= 6:
                utt_id = parts[1].strip()
                label = parts[4].strip()
                phase = parts[5].strip()
                
                if label == 'spoof':
                    label = 0
                elif label == 'bonafide':
                    label = 1
                
                files.append([utt_id, label, phase])
    
    return files

def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x

class Dataset_ASVspoof2019_train(Dataset):
    def __init__(self, args, list_IDs, labels, base_dir, algo):
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.algo = algo
        self.args = args
        self.cut = 64600  # take ~4 sec audio (64600 samples)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        utt_id = self.list_IDs[index]
        X, fs = librosa.load(os.path.join(self.base_dir, 'flac', f'{utt_id}.flac'), sr=16000)
        
        if self.args.rawboost_on:
            Y = process_Rawboost_feature(X, fs, self.args, self.algo)
        else:
            Y = X
        
        X_pad = pad(Y, self.cut)
        x_inp = Tensor(X_pad)
        
        target = self.labels[utt_id]['label']
        target_type = self.labels[utt_id]['target_type']
        
        return x_inp, target

class Dataset_ASVspoof2021_eval(Dataset):
    def __init__(self, list_IDs, base_dir):
        self.list_IDs = list_IDs
        self.labels = parse_protocol_file(list_IDs)
        self.base_dir = base_dir
        self.cut = 64600  # take ~4 sec audio (64600 samples)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        utt_id, label, phase = self.labels[index]
        X, fs = librosa.load(os.path.join(self.base_dir, 'flac', f'{utt_id}.flac'), sr=16000)
        X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)
        return x_inp, label

def process_Rawboost_feature(feature, sr, args, algo):
    if algo == 1:
        feature = LnL_convolutive_noise(feature, args.N_f, args.nBands, args.minF, args.maxF, args.minBW, args.maxBW, args.minCoeff, args.maxCoeff, args.minG, args.maxG, args.minBiasLinNonLin, args.maxBiasLinNonLin, sr)
    elif algo == 2:
        feature = ISD_additive_noise(feature, args.P, args.g_sd)
    elif algo == 3:
        feature = SSI_additive_noise(feature, args.SNRmin, args.SNRmax, args.nBands, args.minF, args.maxF, args.minBW, args.maxBW, args.minCoeff, args.maxCoeff, args.minG, args.maxG, sr)
    elif algo == 4:
        feature = LnL_convolutive_noise(feature, args.N_f, args.nBands, args.minF, args.maxF, args.minBW, args.maxBW, args.minCoeff, args.maxCoeff, args.minG, args.maxG, args.minBiasLinNonLin, args.maxBiasLinNonLin, sr)
        feature = ISD_additive_noise(feature, args.P, args.g_sd)
        feature = SSI_additive_noise(feature, args.SNRmin, args.SNRmax, args.nBands, args.minF, args.maxF, args.minBW, args.maxBW, args.minCoeff, args.maxCoeff, args.minG, args.maxG, sr)
    elif algo == 5:
        feature = LnL_convolutive_noise(feature, args.N_f, args.nBands, args.minF, args.maxF, args.minBW, args.maxBW, args.minCoeff, args.maxCoeff, args.minG, args.maxG, args.minBiasLinNonLin, args.maxBiasLinNonLin, sr)
        feature = ISD_additive_noise(feature, args.P, args.g_sd)
    elif algo == 6:
        feature = LnL_convolutive_noise(feature, args.N_f, args.nBands, args.minF, args.maxF, args.minBW, args.maxBW, args.minCoeff, args.maxCoeff, args.minG, args.maxG, args.minBiasLinNonLin, args.maxBiasLinNonLin, sr)
        feature = SSI_additive_noise(feature, args.SNRmin, args.SNRmax, args.nBands, args.minF, args.maxF, args.minBW, args.maxBW, args.minCoeff, args.maxCoeff, args.minG, args.maxG, sr)
    elif algo == 7:
        feature = ISD_additive_noise(feature, args.P, args.g_sd)
        feature = SSI_additive_noise(feature, args.SNRmin, args.SNRmax, args.nBands, args.minF, args.maxF, args.minBW, args.maxBW, args.minCoeff, args.maxCoeff, args.minG, args.maxG, sr)
    elif algo == 8:
        feature1 = LnL_convolutive_noise(feature, args.N_f, args.nBands, args.minF, args.maxF, args.minBW, args.maxBW, args.minCoeff, args.maxCoeff, args.minG, args.maxG, args.minBiasLinNonLin, args.maxBiasLinNonLin, sr)
        feature2 = ISD_additive_noise(feature, args.P, args.g_sd)
        feature_para = feature1 + feature2
        feature = normWav(feature_para, 0)
    
    return feature