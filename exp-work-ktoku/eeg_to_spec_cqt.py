import torch
import numpy as np
import pandas as pd
from scipy.ndimage import zoom
from nnAudio.Spectrogram import CQT1992v2


FEATS = [['Fp1','F7','T3','T5','O1'],
         ['Fp1','F3','C3','P3','O1'],
         ['Fp2','F8','T4','T6','O2'],
         ['Fp2','F4','C4','P4','O2']]

CQT = CQT1992v2(sr=200, fmin=1, fmax=50, hop_length=10000//256, bins_per_octave=24, n_bins=7*24)
TARGET_SHAPE = (128, 256)

def spectrogram_from_eeg(parquet_path):

    eeg = pd.read_parquet(parquet_path)
    middle = (len(eeg)-10_000)//2
    eeg = eeg.iloc[middle:middle+10_000]
    
    img = np.zeros((TARGET_SHAPE[0], TARGET_SHAPE[1], 4),dtype='float32')
    for k in range(4):
        COLS = FEATS[k]
        
        for kk in range(4):
        
            x = eeg[COLS[kk]].values - eeg[COLS[kk+1]].values
            m = np.nanmean(x)
            if np.isnan(x).mean()<1: x = np.nan_to_num(x,nan=m)
            else: x[:] = 0
                
            image = CQT(torch.from_numpy(x).float().unsqueeze(dim=0)).numpy()
            zoom_factors = [TARGET_SHAPE[0] / image.shape[1], TARGET_SHAPE[1] / image.shape[2]]

            # 補間を使用してリサイズ
            cqt_resized_interpolated = zoom(image[0], zoom_factors)
            img[:, :, k] += cqt_resized_interpolated
                
        # AVERAGE THE 4 MONTAGE DIFFERENCES
        img[:,:,k] /= 4.0
    
    return img