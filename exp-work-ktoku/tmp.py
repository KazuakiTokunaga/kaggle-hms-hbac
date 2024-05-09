import numpy as np
import pandas as pd
import polars as pl
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 5000)

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn

pl.Config.set_tbl_cols(-1)
pl.Config.set_tbl_rows(4000)

import hashlib

###########################
# オリジナル版
###########################

df = pd.read_csv('/kaggle/input/hms-harmful-brain-activity-classification/train.csv')
TARGETS = ["seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"]
df['total_evaluators'] = df[['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']].sum(axis=1)

def get_train_df(df_tmp):
    train = df_tmp.groupby('eeg_id').agg(
        spectrogram_id= ('spectrogram_id','first'),
        min = ('spectrogram_label_offset_seconds','min'),
        max = ('spectrogram_label_offset_seconds','max'),
        eeg_min = ('eeg_label_offset_seconds','min'),
        eeg_max = ('eeg_label_offset_seconds','max'),
        patient_id = ('patient_id','first'),
        total_evaluators = ('total_evaluators','mean'),
        target = ('expert_consensus','first'),
        seizure_vote = ('seizure_vote','sum'),
        lpd_vote = ('lpd_vote','sum'),
        gpd_vote = ('gpd_vote','sum'),
        lrda_vote = ('lrda_vote','sum'),
        grda_vote = ('grda_vote','sum'),
        other_vote = ('other_vote','sum'),
    ).reset_index()
    y_data = train[TARGETS].values
    y_data = y_data / y_data.sum(axis=1,keepdims=True)
    train[TARGETS] = y_data
    
    train['spec_offset_second'] = (train['max'] + train['min']) // 2 
    train['eeg_offset_second'] = (train['eeg_max'] + train['eeg_min']) // 2
    return train

eeg_low = df[df['total_evaluators']<10]['eeg_id'].unique()
eeg_high = df[df['total_evaluators']>=10]['eeg_id'].unique()
eeg_both = [eeg_id for eeg_id in eeg_high if eeg_id in eeg_low]

# low, highについてはそれぞれ集計
df_not_both = df[~df['eeg_id'].isin(eeg_both)].copy()
train_not_both = get_train_df(df_not_both)

# 両方に含まれるeeg_idについては、total_evaluatorsが10以上のもののみを集計
df_both = df[df['eeg_id'].isin(eeg_both)].copy()
df_both = df_both[df_both['total_evaluators']>=10]
train_both = get_train_df(df_both)

train = pd.concat([train_not_both, train_both]).reset_index(drop=True)
train['stage'] = train['total_evaluators'].apply(lambda x: 2 if x >= 10.0 else 1)


###########################
# ラベルごとに分ける
###########################

ROOT_PATH = '/content/drive/MyDrive/HMS'
df = pd.read_csv(ROOT_PATH + '/input/hms-harmful-brain-activity-classification/train.csv')
TARGETS = ["seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"]
df['total_evaluators'] = df[['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']].sum(axis=1)

def get_train_df(df_tmp):
    train = df_tmp.groupby('eeg_id').agg(
        spectrogram_id= ('spectrogram_id','first'),
        min = ('spectrogram_label_offset_seconds','min'),
        max = ('spectrogram_label_offset_seconds','max'),
        eeg_min = ('eeg_label_offset_seconds','min'),
        eeg_max = ('eeg_label_offset_seconds','max'),
        patient_id = ('patient_id','first'),
        total_evaluators = ('total_evaluators','mean'),
        target = ('expert_consensus','first'),
        seizure_vote = ('seizure_vote','sum'),
        lpd_vote = ('lpd_vote','sum'),
        gpd_vote = ('gpd_vote','sum'),
        lrda_vote = ('lrda_vote','sum'),
        grda_vote = ('grda_vote','sum'),
        other_vote = ('other_vote','sum'),
    ).reset_index()
    y_data = train[TARGETS].values
    y_data = y_data / y_data.sum(axis=1,keepdims=True)
    train[TARGETS] = y_data
    
    train['spec_offset_second'] = (train['max'] + train['min']) // 2 
    train['eeg_offset_second'] = (train['eeg_max'] + train['eeg_min']) // 2
    train['stage'] = train['total_evaluators'].apply(lambda x: 2 if x >= 10.0 else 1)
    return train

eeg_low = df[df['total_evaluators']<10]['eeg_id'].unique()
eeg_high = df[df['total_evaluators']>=10]['eeg_id'].unique()
eeg_both = [eeg_id for eeg_id in eeg_high if eeg_id in eeg_low]

# low, highについてはそれぞれ集計
df_not_both = df[~df['eeg_id'].isin(eeg_both)].copy()
train_not_both = get_train_df(df_not_both)

# 両方に含まれるeeg_idについては、total_evaluatorsが10以上のもののみを集計
df_both = df[df['eeg_id'].isin(eeg_both)].copy()
df_both = df_both[df_both['total_evaluators']>=10]
train_both = get_train_df(df_both)

train = pd.concat([train_not_both, train_both]).reset_index(drop=True)

# Otherを1stに移す処理
train_1st = train[train['total_evaluators']<10].copy()
train_2nd = train[train['total_evaluators']>=10].copy()
train_2nd_other = train_2nd[train_2nd['target'] == 'Other'].copy()
patient_id_list = train_2nd_other['patient_id'].unique()
rng = np.random.default_rng(4946)
patient_id_list_updated = rng.choice(patient_id_list, len(patient_id_list)-300, replace=False)

# 300人を1stに移す
train_1st_other = train_2nd_other[~train_2nd_other['patient_id'].isin(patient_id_list_updated)].copy()
train_1st_other['stage'] = 1
train_2nd_other = train_2nd_other[train_2nd_other['patient_id'].isin(patient_id_list_updated)].copy()
train_2nd_iiic = train_2nd[train_2nd['target'] != 'Other'].copy()
train_2nd = pd.concat([train_2nd_other, train_2nd_iiic]).reset_index(drop=True)

# カラムの順番を揃える
train = pd.concat([train_1st, train_1st_other, train_2nd]).reset_index(drop=True)


###########################
# ラベルごとに分け,Otherを捨てる
###########################

ROOT_PATH = '/content/drive/MyDrive/HMS'
df = pd.read_csv(ROOT_PATH + '/input/hms-harmful-brain-activity-classification/train.csv')
TARGETS = ["seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"]
df['total_evaluators'] = df[['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']].sum(axis=1)

def string_to_11_digit_hash(input_string):
    hash_object = hashlib.sha256(input_string.encode())
    hex_dig = hash_object.hexdigest()
    hash_int = int(hex_dig, 16)
    hash_11_digit = hash_int % (10**11)
    return hash_11_digit

def get_train_df(df_tmp):
    train = df_tmp.groupby('eeg_id').agg(
        spectrogram_id= ('spectrogram_id','first'),
        min = ('spectrogram_label_offset_seconds','min'),
        max = ('spectrogram_label_offset_seconds','max'),
        eeg_min = ('eeg_label_offset_seconds','min'),
        eeg_max = ('eeg_label_offset_seconds','max'),
        patient_id = ('patient_id','first'),
        total_evaluators = ('total_evaluators','mean'),
        target = ('expert_consensus','first'),
        seizure_vote = ('seizure_vote','sum'),
        lpd_vote = ('lpd_vote','sum'),
        gpd_vote = ('gpd_vote','sum'),
        lrda_vote = ('lrda_vote','sum'),
        grda_vote = ('grda_vote','sum'),
        other_vote = ('other_vote','sum'),
    ).reset_index()
    y_data = train[TARGETS].values
    y_data = y_data / y_data.sum(axis=1,keepdims=True)
    train[TARGETS] = y_data
    train['eeg_id_original'] = train['eeg_id'].copy()
    train['stage'] = 1

    return train

def get_train_df_high(df_tmp):

    train = df_tmp.groupby(['eeg_id']+TARGETS).agg(
        spectrogram_id= ('spectrogram_id','first'),
        min = ('spectrogram_label_offset_seconds','min'),
        max = ('spectrogram_label_offset_seconds','max'),
        eeg_min = ('eeg_label_offset_seconds','min'),
        eeg_max = ('eeg_label_offset_seconds','max'),
        patient_id = ('patient_id','first'),
        total_evaluators = ('total_evaluators','mean'),
        target = ('expert_consensus','first')
    ).reset_index()

    train['eeg_id_rank'] = train.groupby('eeg_id')['eeg_id'].cumcount()+1
    train['eeg_id_original'] = train['eeg_id'].copy()
    train['eeg_id'] = (train['eeg_id_original'] + train['eeg_id_rank']).apply(lambda x: string_to_11_digit_hash(str(x)))

    y_data = train[TARGETS].values
    y_data = y_data / y_data.sum(axis=1,keepdims=True)
    train[TARGETS] = y_data
    train['stage'] = 2

    return train.drop(['eeg_id_rank'], axis=1)

print('Create labels considering 2nd stage learning.')
eeg_low = df[df['total_evaluators']<10]['eeg_id'].unique()
eeg_high = df[df['total_evaluators']>=10]['eeg_id'].unique()
eeg_both = [eeg_id for eeg_id in eeg_high if eeg_id in eeg_low]

# low, highについてはそれぞれ集計
df_low = df[(df['eeg_id'].isin(eeg_low))&(~df['eeg_id'].isin(eeg_both))].copy()
train_low = get_train_df(df_low)

df_high = df[(df['eeg_id'].isin(eeg_high))&(~df['eeg_id'].isin(eeg_both))].copy()
train_high = get_train_df_high(df_high)

# 両方に含まれるeeg_idについては、total_evaluatorsが10以上のもののみを集計
df_both = df[df['eeg_id'].isin(eeg_both)].copy()
df_both = df_both[df_both['total_evaluators']>=10]
train_both = get_train_df_high(df_both)

# Otherについて、patient_id 300人分を無作為に削除する
# 無作為抽出を行う
train_2nd = pd.concat([train_high, train_both] ).reset_index(drop=True)
train_2nd_other = train_2nd[train_2nd['target'] == 'Other'].copy()
patient_id_list = train_2nd_other['patient_id'].unique()
patient_id_list_updated = np.random.choice(patient_id_list, len(patient_id_list)-300, replace=False)

# 300人を1stに移す
train_1st_other = train_2nd_other[~train_2nd_other['patient_id'].isin(patient_id_list_updated)].copy()
train_1st_other['stage'] = 1
train_2nd_other = train_2nd_other[train_2nd_other['patient_id'].isin(patient_id_list_updated)].copy()
train_2nd_iiic = train_2nd[train_2nd['target'] != 'Other'].copy()
train_2nd = pd.concat([train_2nd_other, train_2nd_iiic]).reset_index(drop=True)

# カラムの順番を揃える
columns = train_low.columns
train_1st_other = train_1st_other[columns]
train_2nd = train_2nd[columns]

train = pd.concat([train_low, train_1st_other, train_2nd]).reset_index(drop=True)
train['spec_offset_second'] = (train['max'] + train['min']) // 2 
train['eeg_offset_second'] = (train['eeg_max'] + train['eeg_min']) // 2