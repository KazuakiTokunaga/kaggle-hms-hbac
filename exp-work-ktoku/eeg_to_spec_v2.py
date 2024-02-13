import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt, gc

import albumentations as A
import gc
import librosa
import matplotlib.pyplot as plt
import math
import multiprocessing
import numpy as np
import os
import pandas as pd
import pywt
import random
import time
import timm
import torch
import torch.nn as nn
from torch.nn.functional import log_softmax

from albumentations.pytorch import ToTensorV2
from glob import glob
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from typing import Dict, List

FEATS_V2 = [[('Cz', 'C4'), ('Cz', 'Fz'), ('C4', 'F4'), ('Fz', 'F4')],
         [('Cz', 'C4'), ('Cz', 'Pz'), ('C4', 'P4'), ('Pz', 'P4')],
         [('Cz', 'C3'), ('Cz', 'Fz'), ('C3', 'F3'), ('Fz', 'F3')],
         [('Cz', 'C3'), ('Cz', 'Pz'), ('C3', 'P3'), ('Pz', 'P3')]]

def spectrogram_from_eeg_v2(parquet_path, display=False):
    # LOAD MIDDLE 50 SECONDS OF EEG SERIES
    eeg = pd.read_parquet(parquet_path)
    middle = (len(eeg)-10_000)//2
    eeg = eeg.iloc[middle:middle+10_000]
    
    # VARIABLE TO HOLD SPECTROGRAM
    img = np.zeros((128,256,4),dtype='float32')
    
    signals = []
    for k in range(4):
        COLS = FEATS_V2[k]
        
        for a, b in COLS:
        
            # COMPUTE PAIR DIFFERENCES
            x = eeg[a].values - eeg[b].values

            # FILL NANS
            m = np.nanmean(x)
            if np.isnan(x).mean()<1: x = np.nan_to_num(x,nan=m)
            else: x[:] = 0

            signals.append(x)

            # RAW SPECTROGRAM
            mel_spec = librosa.feature.melspectrogram(y=x, sr=200, hop_length=len(x)//256, 
                  n_fft=1024, n_mels=128, fmin=0, fmax=20, win_length=128)

            # LOG TRANSFORM
            width = (mel_spec.shape[1]//32)*32
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max).astype(np.float32)[:,:width]

            # STANDARDIZE TO -1 TO 1
            mel_spec_db = (mel_spec_db+40)/40 
            img[:,:,k] += mel_spec_db
                
        # AVERAGE THE 4 MONTAGE DIFFERENCES
        img[:,:,k] /= 4.0
    
    return img