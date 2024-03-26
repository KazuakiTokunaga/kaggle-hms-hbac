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
# ラベルごとに分ける場合 (2nd:7978個)
###########################

df = pd.read_csv('/kaggle/input/hms-harmful-brain-activity-classification/train.csv')
TARGETS = ["seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"]
df['total_evaluators'] = df[['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']].sum(axis=1)

def create_offsets(row):
    if row.eeg_len < 40:
        return [(row.eeg_max + row.eeg_min)//2]
    if row.eeg_len < 80:
        return [row.eeg_min+10, row.eeg_max-10]
    else:
        return [row.eeg_min+10, (row.eeg_max+row.eeg_min)//2, row.eeg_max-10]
    
def get_hash_id(row, offset):
    s = str(row.eeg_id_original)+str(offset)
    return string_to_11_digit_hash(s)
    
def create_eeg_ids(row):
    if len(row.offsets) <= 1:
        return None
    else:
        return [get_hash_id(row, offset) for offset in row.offsets]

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
    
    train['offset'] = (train['eeg_max'] + train['eeg_min']) // 2
    train['offsets'] = train['offset'].apply(lambda x: [x])
    train['augment_flag'] = False
    train['augmented_ids'] = None

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
    
    train['eeg_len'] = train['eeg_max'] - train['eeg_min']
    train['offset'] = (train['eeg_max'] + train['eeg_min']) // 2
    train['offsets'] = train.apply(create_offsets, axis=1)
    train['augment_flag'] = train['offsets'].apply(lambda x: True if len(x) > 1 else False)
    train['augmented_ids'] = train.apply(create_eeg_ids, axis=1)

    return train.drop(['eeg_id_rank', 'eeg_len'], axis=1)

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

# カラムの順番を揃える
columns = train_low.columns
train_high = train_high[columns]
train_both = train_both[columns]

train = pd.concat([train_low, train_high, train_both]).reset_index(drop=True)
train['stage'] = train['total_evaluators'].apply(lambda x: 2 if x >= 10.0 else 1)


###########################
# ラベルごとに分けない場合 (2nd: 7715個)
###########################


df = pd.read_csv('/kaggle/input/hms-harmful-brain-activity-classification/train.csv')
TARGETS = ["seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"]
df['total_evaluators'] = df[['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']].sum(axis=1)

def create_offsets(row):
    if row.total_evaluators < 10.0:
        return [(row.eeg_max + row.eeg_min)//2]
    if row.eeg_len < 40:
        return [(row.eeg_max + row.eeg_min)//2]
    if row.eeg_len < 80:
        return [row.eeg_min+10, row.eeg_max-10]
    else:
        return [row.eeg_min+10, (row.eeg_max+row.eeg_min)//2, row.eeg_max-10]
    
def get_hash_id(row, offset):
    s = str(row.eeg_id) +'49464946' + str(offset)
    return string_to_11_digit_hash(s)
    
def create_eeg_ids(row):
    if len(row.offsets) <= 1:
        return None
    else:
        return [get_hash_id(row, offset) for offset in row.offsets]

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
    
    train['eeg_len'] = train['eeg_max'] - train['eeg_min']
    train['offset'] = (train['eeg_max'] + train['eeg_min']) // 2
    train['offsets'] = train.apply(create_offsets, axis=1)
    train['augment_flag'] = train['offsets'].apply(lambda x: True if len(x) > 1 else False)
    train['augmented_ids'] = train.apply(create_eeg_ids, axis=1)

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
# 2ndは最初をとる
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

def get_train_df_high(df_tmp):

    train = df_tmp.groupby('eeg_id').agg(
        spectrogram_id= ('spectrogram_id','first'),
        min = ('spectrogram_label_offset_seconds','first'),
        max = ('spectrogram_label_offset_seconds','max'),
        eeg_min = ('eeg_label_offset_seconds','first'),
        eeg_max = ('eeg_label_offset_seconds','max'),
        patient_id = ('patient_id','first'),
        total_evaluators = ('total_evaluators','first'),
        target = ('expert_consensus','first'),
        seizure_vote = ('seizure_vote','first'),
        lpd_vote = ('lpd_vote','first'),
        gpd_vote = ('gpd_vote','first'),
        lrda_vote = ('lrda_vote','first'),
        grda_vote = ('grda_vote','first'),
        other_vote = ('other_vote','first'),
    ).reset_index()

    y_data = train[TARGETS].values
    y_data = y_data / y_data.sum(axis=1,keepdims=True)
    train[TARGETS] = y_data

    train['spec_offset_second'] = train['min'] // 2
    train['eeg_offset_second'] = train['eeg_min'].copy()
    return train

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

# カラムの順番を揃える
columns = train_low.columns
train_high = train_high[columns]
train_both = train_both[columns]

train = pd.concat([train_low, train_high, train_both]).reset_index(drop=True)
train['stage'] = train['total_evaluators'].apply(lambda x: 2 if x >= 10.0 else 1)