import numpy as np
import pandas as pd
import pywt
import os
from scipy.signal import butter, filtfilt
from scipy.ndimage import zoom
from glob import glob
from typing import Dict, List


FEATS = [['Fp1','F7','T3','T5','O1'],
         ['Fp1','F3','C3','P3','O1'],
         ['Fp2','F8','T4','T6','O2'],
         ['Fp2','F4','C4','P4','O2']]

WAVELET = 'cmor3.0-3.0'
DOWNSAMPLE_RATE = 2
DT = DOWNSAMPLE_RATE * (1 / 200)
FREQUENCIES = pywt.scale2frequency(WAVELET, np.linspace(1.5, 20, num=64)) / DT

# 低域通過フィルタの設定
nyquist = 0.5 * 200  # ナイキスト周波数（元のサンプリングレートの半分）
cutoff = nyquist / 2  # カットオフ周波数をナイキスト周波数の半分に設定
B, A = butter(5, cutoff / nyquist, btype='low')  # 5次のバターワースフィルタ


def spectrogram_from_eeg_cwt(parquet_path):

    eeg = pd.read_parquet(parquet_path)
    middle = (len(eeg)-10_000)//2
    eeg = eeg.iloc[middle:middle+10_000]
    
    img = np.zeros((64,256,4),dtype='float32')
    for k in range(4):
        COLS = FEATS[k]
        
        for kk in range(4):
        
            x = eeg[COLS[kk]].values - eeg[COLS[kk+1]].values
            m = np.nanmean(x)
            if np.isnan(x).mean()<1: x = np.nan_to_num(x,nan=m)
            else: x[:] = 0
                
            downsampled_x = filtfilt(B, A, x)[::DOWNSAMPLE_RATE]
            coefficients, frequencies = pywt.cwt(downsampled_x, FREQUENCIES, WAVELET, sampling_period=1/200)
#             print(coefficients.shape)
            resized_coefficients = zoom(np.abs(coefficients), (1, 256/5000))
#             print(resized_coefficients.shape)
            img[:, :, k] += resized_coefficients
                
        # AVERAGE THE 4 MONTAGE DIFFERENCES
        img[:,:,k] /= 4.0
    
    return img