import albumentations as A
import librosa
import numpy as np
import pandas as pd
import pywt
import torch
import os
from scipy.ndimage import gaussian_filter, zoom
from scipy.signal import butter, filtfilt, iirnotch
from nnAudio.Spectrogram import CQT1992v2

FEAT_V1 = [['Fp1','F7','T3','T5','O1'],
        ['Fp1','F3','C3','P3','O1'],
        ['Fp2','F8','T4','T6','O2'],
        ['Fp2','F4','C4','P4','O2']]
FEAT_V2 = ['Fp1', 'F7', 'T3', 'T5', 'O1', 'F3', 'C3', 'P3', 'Fp2', 'F8', 'T4', 'T6', 'O2', 'F4', 'C4', 'O2']
FEAT_V2_P1 = ['Fp1', 'F7', 'T3', 'T5', 'O1', 'F3', 'C3', 'P3']
FEAT_V2_P2 = ['Fp2', 'F8', 'T4', 'T6', 'O2', 'F4', 'C4', 'O2']
FEAT_V3 = [[('Cz', 'C4'), ('Cz', 'Fz'), ('C4', 'F4'), ('Fz', 'F4')],
         [('Cz', 'C4'), ('Cz', 'Pz'), ('C4', 'P4'), ('Pz', 'P4')],
         [('Cz', 'C3'), ('Cz', 'Fz'), ('C3', 'F3'), ('Fz', 'F3')],
         [('Cz', 'C3'), ('Cz', 'Pz'), ('C3', 'P3'), ('Pz', 'P3')]]

CQT = CQT1992v2(sr=200, fmin=1, fmax=50, hop_length=10000//384, bins_per_octave=24, n_bins=7*24)


######################################################
# ヘルパー関数
######################################################


def eeg_fill_na(x):
    m = np.nanmean(x)
    if np.isnan(x).mean()<1: 
        x = np.nan_to_num(x,nan=m)
    else: 
        x[:] = 0

    return x


def lowpass_filter(x, sr=200):
    
    nyquist = 0.5 * sr
    cutoff = nyquist // 2
    B, A = butter(5, cutoff / nyquist, btype='low')
    return filtfilt(B, A, x)


def bandpass_filter(data, low=0.1, high=30, sr=200):

    nyquist_freq = 0.5 * sr
    low_cut_freq_normalized = low / nyquist_freq
    high_cut_freq_normalized = high / nyquist_freq

    bandpass_coefficients = butter(5, [low_cut_freq_normalized, high_cut_freq_normalized], btype='band')
    signal_filtered = filtfilt(*bandpass_coefficients, data)

    return signal_filtered


def notch_filter(data, w0=60, Q=20, fs=200):

    notch_coefficients = iirnotch(w0=w0, Q=Q, fs=fs)
    signal_filtered = filtfilt(*notch_coefficients, data)
    return signal_filtered


def standardize_melspec(data):

    # LOG TRANSFORM
    width = (data.shape[1]//32)*32
    mel_spec_db = librosa.power_to_db(data, ref=np.max).astype(np.float32)[:,:width]
    mel_spec_db = (mel_spec_db+40)/40 
    
    return mel_spec_db


def standardize_img(img):

    img = np.clip(img,np.exp(-4),np.exp(8))
    img = np.log(img)
    ep = 1e-6
    m = np.nanmean(img.flatten())
    s = np.nanstd(img.flatten())
    img = (img-m)/(s+ep)
    img = np.nan_to_num(img, nan=0.0)

    return img
    
######################################################
# スペクトログラム作成関数
######################################################

def etos(eeg, denoise=False, gaussian=False):

    img = np.zeros((128,256,4),dtype='float32')
    for k in range(4):
        COLS = FEAT_V1[k]
        
        for kk in range(4):
        
            x = eeg[COLS[kk]].values - eeg[COLS[kk+1]].values
            x = eeg_fill_na(x)

            if denoise:
                x = bandpass_filter(x)
                x = notch_filter(x)
        
            mel_spec = librosa.feature.melspectrogram(y=x, sr=200, hop_length=len(x)//256, n_fft=1024, n_mels=128, fmin=0, fmax=20, win_length=128)
            mel_spec_db = standardize_melspec(mel_spec)

            if gaussian:
                mel_spec_db = gaussian_filter(mel_spec_db, sigma=0.7)

            img[:,:,k] += mel_spec_db

        img[:,:,k] /= 4.0

    return img

def etos_v3(eeg, denoise=False, gaussian=False):

    img = np.zeros((128,256,4),dtype='float32')
    for k in range(4):
        COLS = FEAT_V3[k]
        
        for a, b in COLS:
        
            # COMPUTE PAIR DIFFERENCES
            x = eeg[a].values - eeg[b].values
            x = eeg_fill_na(x)

            if denoise:
                x = bandpass_filter(x)
                x = notch_filter(x)
        
            mel_spec = librosa.feature.melspectrogram(y=x, sr=200, hop_length=len(x)//256, n_fft=1024, n_mels=128, fmin=0, fmax=20, win_length=128)
            mel_spec_db = standardize_melspec(mel_spec)

            if gaussian:
                mel_spec_db = gaussian_filter(mel_spec_db, sigma=0.7)

            img[:,:,k] += mel_spec_db

        img[:,:,k] /= 4.0

    return img
    
    
def etos_cwt_mexh(eeg, denoise=False, gaussian=False):

    img = np.zeros((64,512,4),dtype='float32')
    for k in range(4):
        COLS = FEAT_V1[k]
        
        for kk in range(4):
        
            x = eeg[COLS[kk]].values - eeg[COLS[kk+1]].values
            x = eeg_fill_na(x)
            
            if denoise:
                x = bandpass_filter(x)
                x = notch_filter(x).copy()
                
            frequencies = pywt.scale2frequency('mexh', np.linspace(0.1, 20, num=64))  * 200
            coefficients, frequencies = pywt.cwt(x, frequencies, 'mexh', sampling_period=1/200)
            x = zoom(np.abs(coefficients), (1, 512/10000))
            x = standardize_img(x)

            if gaussian:
                x = gaussian_filter(x, sigma=0.7)

            img[:, :, k] += x
                
        img[:,:,k] /= 4.0
        
    return img


def etos_cwt_cmor(eeg, gaussian=False):
    
    img = np.zeros((64, 512,4),dtype='float32')
    for k in range(4):
        COLS = FEAT_V1[k]
        
        for kk in range(4):
        
            x = eeg[COLS[kk]].values - eeg[COLS[kk+1]].values
            x = eeg_fill_na(x)
            x = lowpass_filter(x, sr=100)
                
            downsampled_x = x[::2]
            wavelet = 'cmor1.5-1.5'
            frequencies = pywt.scale2frequency(wavelet, np.linspace(0.1, 20, num=64)) * 100
            coefficients, frequencies = pywt.cwt(downsampled_x, frequencies, wavelet, sampling_period=100)
            resized_coefficients = zoom(np.abs(coefficients), (1, 512/5000))

            if gaussian:
                resized_coefficients = gaussian_filter(resized_coefficients, sigma=0.7)

            img[:, :, k] += resized_coefficients
                
        # AVERAGE THE 4 MONTAGE DIFFERENCES
        img[:,:,k] /= 4.0
    
    return img


def etos_cqt(eeg, denoise=False, gaussian=False):

    img = np.zeros((128, 384, 4),dtype='float32')
    for k in range(4):
        COLS = FEAT_V1[k]
        
        for kk in range(4):
        
            x = eeg[COLS[kk]].values - eeg[COLS[kk+1]].values
            x = eeg_fill_na(x)
            
            if denoise:
                x = bandpass_filter(x)
                x = notch_filter(x).copy()
                
            image = CQT(torch.from_numpy(x).float().unsqueeze(dim=0)).numpy()
            zoom_factors = [128 / image.shape[1], 384 / image.shape[2]]
            cqt_resized_interpolated = zoom(image[0], zoom_factors)
            cqt_resized_interpolated = standardize_img(cqt_resized_interpolated)

            if gaussian:
                cqt_resized_interpolated = gaussian_filter(cqt_resized_interpolated, sigma=0.7)
            
            img[:, :, k] += cqt_resized_interpolated
                
        img[:,:,k] /= 4.0
    
    return img


def etos_common(eeg, feats="P1", denoise=False, gaussian=False):

    # 共通平均参照
    common_feats = [c for c in eeg.columns if c != 'EKG']
    m = eeg[common_feats].mean(axis=1)
    for column in eeg.columns:
        eeg[column] = eeg[column] - m

    if feats == "ALL": 
        target_feats = FEAT_V2
    elif feats == "P1":
        target_feats = FEAT_V2_P1
    else:
        target_feats = FEAT_V2_P2
    
    img = np.zeros((128,128,len(target_feats)),dtype='float32')
    for k, feat in enumerate(target_feats):

        x = eeg[feat].values
        x = eeg_fill_na(x)
        
        if denoise:
            x = bandpass_filter(x)
            x = notch_filter(x)
        
        # RAW SPECTROGRAM
        mel_spec = librosa.feature.melspectrogram(y=x, sr=200, hop_length=len(x)//128,
                n_fft=1024, n_mels=128, fmin=0, fmax=30, win_length=128)

        mel_spec_db = standardize_melspec(mel_spec)

        if gaussian:
            mel_spec_db = gaussian_filter(mel_spec_db, sigma=0.7)

        img[:,:,k] += mel_spec_db

    return img


def etos_common_cwt_mexh(eeg, feats="P1", denoise=False, gaussian=False):

    # 共通平均参照
    common_feats = [c for c in eeg.columns if c != 'EKG']
    m = eeg[common_feats].mean(axis=1)
    for column in eeg.columns:
        eeg[column] = eeg[column] - m

    if feats == "ALL": 
        target_feats = FEAT_V2
    elif feats == "P1":
        target_feats = FEAT_V2_P1
    else:
        target_feats = FEAT_V2_P2
    
    img = np.zeros((128,128,len(target_feats)),dtype='float32')
    for k, feat in enumerate(target_feats):

        x = eeg[feat].values
        x = eeg_fill_na(x)
        
        if denoise:
            x = bandpass_filter(x)
            x = notch_filter(x)
        
        # RAW SPECTROGRAM
        frequencies = pywt.scale2frequency('mexh', np.linspace(0.1, 20, num=128))  * 200
        coefficients, frequencies = pywt.cwt(x, frequencies, 'mexh', sampling_period=1/200)
        x = zoom(np.abs(coefficients), (1, 128/10000))
        x = standardize_img(x)

        if gaussian:
            x = gaussian_filter(x, sigma=0.7)

        img[:,:,k] += x

    return img
