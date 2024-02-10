import torch
import torch.nn as nn
import os, gc
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import timm
import datetime
import random
import warnings
import albumentations as A
import pathlib
from pathlib import Path
from torch import optim
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.model_selection import KFold, GroupKFold
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import log_softmax, softmax

import sys
sys.path.append('/content/drive/MyDrive/HMS/input/kaggle-kl-div')
from kaggle_kl_div import score

from utils import set_random_seed, to_device
from utils import WriteSheet, Logger, class_vars_to_dict

warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

logger = Logger()
TARGETS = ["seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"]
TARS = {'Seizure': 0, 'LPD': 1, 'GPD': 2, 'LRDA': 3, 'GRDA': 4, 'Other': 5}
TARS2 = {x:y for y,x in TARS.items()}


class RCFG:
    """実行に関連する設定"""
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ROOT_PATH = '/content/drive/MyDrive/HMS'
    DEBUG = True
    DEBUG_SIZE = 300

class ENV:
    """実行環境に関連する設定"""
    env = "kaggle"
    commit_hash = ""
    save_to_sheet = True
    sheet_json_key = RCFG.ROOT_PATH + '/input/ktokunagautils/ktokunaga-4094cf694f5c.json'
    sheet_key = '1Wcg2EvlDgjo0nC-qbHma1LSEAY_OlS50mJ-yI4QI-yg'

class CFG:
    """モデルに関連する設定"""
    EPOCHS = 4
    N_SPLITS = 5
    BATCH_SIZE = 32
    AUGMENT = True


class EfficentNetDataset(Dataset):
    def __init__(
        self, 
        data, 
        specs, 
        eeg_specs,
        augment=CFG.AUGMENT, 
        mode='train',
    ):

        self.data = data
        self.augment = augment
        self.mode = mode
        self.specs = specs
        self.eeg_specs = eeg_specs
        self.indexes = np.arange( len(self.data))

    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        indexes = self.indexes[idx]
        X, y = self.__data_generation(indexes)
        if self.augment:
            X = self._augment_batch(X)
        if self.mode != 'test':
            return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
        else:
            return torch.tensor(X, dtype=torch.float32)

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples'

        X = np.zeros((128,256,8),dtype='float32')
        y = np.zeros((6),dtype='float32')
        img = np.ones((128,256),dtype='float32')

        row = self.data.iloc[indexes]
        if self.mode=='test':
            r = 0
        else:
            r = int( (row['min'] + row['max'])//4 )

        for k in range(4):
            # EXTRACT 300 ROWS OF SPECTROGRAM(4種類抜いてくる)
            img = self.specs[row.spec_id][r:r+300,k*100:(k+1)*100].T

            # LOG TRANSFORM SPECTROGRAM
            img = np.clip(img,np.exp(-4),np.exp(8))
            img = np.log(img)

            # STANDARDIZE PER IMAGE
            ep = 1e-6
            m = np.nanmean(img.flatten())
            s = np.nanstd(img.flatten())
            img = (img-m)/(s+ep)
            img = np.nan_to_num(img, nan=0.0)

            # CROP TO 256 TIME STEPS
            X[14:-14,:,k] = img[:,22:-22] / 2.0

        # EEG SPECTROGRAMS
        img = self.eeg_specs[row.eeg_id]
        X[:,:,4:] = img

        if self.mode!='test':
            y = row.loc[TARGETS]

        return X,y

    def _augment_batch(self, img):
        transforms = A.Compose([
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0, rotate_limit=0, p=0.3, border_mode=0), # 時間軸方向のシフト
            A.GaussNoise(var_limit=(10, 50), p=0.3) # ガウス雑音
        ])
        return transforms(image=img)['image']


class CustomInputTransform(nn.Module):
    def __init__(self, use_kaggle=True, use_eeg=True):
        super(CustomInputTransform, self).__init__()
        self.use_kaggle = use_kaggle
        self.use_eeg = use_eeg

    def forward(self, x):
        # Kaggleスペクトログラム
        if self.use_kaggle:
            x1 = torch.cat([x[:, :, :, i:i+1] for i in range(4)], dim=1)

        # EEGスペクトログラム
        if self.use_eeg:
            x2 = torch.cat([x[:, :, :, i+4:i+5] for i in range(4)], dim=1)

        # 結合
        if self.use_kaggle and self.use_eeg:
            x = torch.cat([x1, x2], dim=2)
        elif self.use_eeg:
            x = x2
        else:
            x = x1

        # 3チャンネルに複製
        x = x.repeat(1, 1, 1, 3)
        x = x.permute(0, 3, 1, 2)

        return x

class CustomEfficientNet(nn.Module):
    def __init__(self, num_classes=6):
        super(CustomEfficientNet, self).__init__()
        self.input_transform = CustomInputTransform(use_kaggle=True, use_eeg=True)
        self.base_model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0, in_chans=3)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_features=1280, out_features=num_classes) #1280 #1408 #1792
        self.base_model.classifier = self.fc

    def forward(self, x):
        x = self.input_transform(x)
        x = self.base_model(x)
        return x
    

def calc_cv_score(oof,true):
    oof = pd.DataFrame(np.concatenate(oof).copy())
    oof['id'] = np.arange(len(oof))

    true = pd.DataFrame(np.concatenate(true).copy())
    true['id'] = np.arange(len(true))

    cv = score(solution=true, submission=oof, row_id_column_name='id')
    return cv


# モデル訓練関数
def train_model(model, train_loader, valid_loader, optimizer, scheduler, criterion):
    model.train()
    train_loss = []
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(RCFG.DEVICE), labels.to(RCFG.DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(log_softmax(outputs, dim = 1), labels)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
        if CFG.DEBUG:
            print(f'train_loss: {loss.item()}')
    scheduler.step()
    # 検証ループ
    model.eval()
    oof = []
    true = []
    valid_loss = []
    with torch.no_grad():
        for inputs, labels in valid_loader:
            true.append(labels)
            inputs, labels = inputs.to(RCFG.DEVICE), labels.to(RCFG.DEVICE)
            outputs = model(inputs)
            loss = criterion(log_softmax(outputs, dim = 1), labels)
            valid_loss.append(loss.item())
            oof.append(softmax(outputs,dim=1).to('cpu').numpy())

    # モデルの重みを保存
    cv=calc_cv_score(oof,true)
    return model,np.mean(train_loss),np.mean(valid_loss),cv


######################################################
# Runner
######################################################


class Runner():
    def __init__(self, commit_hash=""):

        set_random_seed()
        logger.info('Initializing Runner.')
        start_dt_jst = str(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).strftime('%Y-%m-%d %H:%M:%S'))
        self.info =  {"start_dt_jst": start_dt_jst}

        ENV.commit_hash=commit_hash
        
        if ENV.env == "kaggle":
            from kaggle_secrets import UserSecretsClient
            self.user_secrets = UserSecretsClient()

        if ENV.save_to_sheet:
            logger.info('Initializing Google Sheet.')
            self.sheet = WriteSheet(
                sheet_json_key = ENV.sheet_json_key,
                sheet_key = ENV.sheet_key
            )

        if RCFG.DEBUG:
            logger.info('DEBUG MODE: Decrease N_SPLITS and EPOCHS.')
            CFG.N_SPLITS = 2
            CFG.EPOCHS = 3

    
    def load_dataset(self, ):
        
        df = pd.read_csv(RCFG.ROOT_PATH + '/train.csv')
        TARGETS = df.columns[-6:]
        train = df.groupby('eeg_id')[['spectrogram_id','spectrogram_label_offset_seconds']].agg(
            {'spectrogram_id':'first','spectrogram_label_offset_seconds':'min'})
        train.columns = ['spec_id','min']

        tmp = df.groupby('eeg_id')[['spectrogram_id','spectrogram_label_offset_seconds']].agg(
            {'spectrogram_label_offset_seconds':'max'})
        train['max'] = tmp

        tmp = df.groupby('eeg_id')[['patient_id']].agg('first')
        train['patient_id'] = tmp

        tmp = df.groupby('eeg_id')[TARGETS].agg('sum')
        for t in TARGETS:
            train[t] = tmp[t].values

        y_data = train[TARGETS].values
        y_data = y_data / y_data.sum(axis=1,keepdims=True)
        train[TARGETS] = y_data

        tmp = df.groupby('eeg_id')[['expert_consensus']].agg('first')
        train['target'] = tmp

        if RCFG.DEBUG:
            train = train.iloc[:RCFG.DEBUG_SIZE]

        self.train = train.reset_index()
        print('Train non-overlapp eeg_id shape:', train.shape )

        # READ ALL SPECTROGRAMS
        self.spectrograms = np.load(RCFG.ROOT_PATH  + '/specs.npy',allow_pickle=True).item()
        self.all_eegs = np.load(RCFG.ROOT_PATH + '/eeg_specs.npy',allow_pickle=True).item()


    def run_train(self, ):

        gkf = GroupKFold(n_splits=CFG.N_SPLITS)
        for i, (train_index, valid_index) in enumerate(gkf.split(self.train, self.train.target, self.train.patient_id)):
            print(f'### Fold {i+1}')
            # データローダーの作成
            train_dataset = EfficentNetDataset(
                self.train.iloc[train_index],
                self.spectrograms, 
                self.all_eegs,
            )
            train_loader = DataLoader(train_dataset, batch_size=CFG.BATCH_SIZE, shuffle=True, num_workers=2,pin_memory=True)

            valid_dataset = EfficentNetDataset(
                self.train.iloc[valid_index],
                self.spectrograms, 
                self.all_eegs,
                augment=False
            )
            valid_loader = DataLoader(valid_dataset, batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=2,pin_memory=True)

            # モデルの構築
            model = CustomEfficientNet().to(RCFG.DEVICE)
            optimizer = optim.AdamW(model.parameters(),lr=0.001)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4, eta_min=1e-4)
            criterion = nn.KLDivLoss(reduction='batchmean')  # 適切な損失関数を選択

            # トレーニングループ
            for epoch in range(CFG.EPOCHS):
                model, tr_loss, val_loss, cv = train_model(
                    model, 
                    train_loader, 
                    valid_loader,
                    optimizer,
                    scheduler,
                    criterion
                )
                # エポックごとのログを出力
                print(f'Epoch {epoch+1}, Train Loss: {tr_loss}, Valid Loss: {val_loss}')
                print('CV Score KL-Div for EfficientNetB2 =',cv)
                torch.save(model.state_dict(), RCFG.ROOT_PATH + f'/model/fold{i}_Eff_net_snapshot_epoch_{epoch}.pickle')
            del model
            gc.collect()
            torch.cuda.empty_cache()

    def main(self):
        self.load_dataset()
        self.run_train()