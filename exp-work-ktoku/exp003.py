# EfficientNet or Resnet

import sys
import torch
import torch.nn as nn
import os, gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import timm
import datetime
import random
import warnings
import albumentations as A
import pathlib
import hashlib
import torch.nn.functional as F
from tqdm import tqdm
from glob import glob
from pathlib import Path
from scipy.ndimage import zoom
from torch import optim
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.model_selection import KFold, GroupKFold, StratifiedGroupKFold
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import log_softmax, softmax
from torch.nn.parameter import Parameter


from utils import set_random_seed, create_random_id
from utils import WriteSheet, Logger, class_vars_to_dict
from eeg_to_spec import spectrogram_from_eeg
from eeg_to_spec_cwt import spectrogram_from_eeg_cwt

warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

class RCFG:
    """実行に関連する設定"""
    RUN_NAME = ""
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    DEBUG = False
    DEBUG_SIZE = 300
    PREDICT = False
    COMMIT_HASH = ""
    USE_FOLD = [] # 空のときは全fold、0-4で指定したfoldのみを使う
    SAVE_TO_SHEET = True
    SHEET_KEY = '1Wcg2EvlDgjo0nC-qbHma1LSEAY_OlS50mJ-yI4QI-yg'
    PSEUDO_LABELLING = True
    LABELS_V2 = False
    LABELS_V3 = True
    # USE_SPECTROGRAMS = ['kaggle']
    USE_SPECTROGRAMS = ['kaggle', 'cwt_mexh_20sec_v105', 'cwt_mexh_10sec_v105', 'cwt_mexh_20sec_last_v105']
    CREATE_SPECS = True
    USE_ALL_LOW_QUALITY = False
    ADD_MIXUP_DATA = False

class CFG:
    """モデルに関連する設定"""
    MODEL_NAME = 'efficientnet_b0'
    IN_CHANS = 3
    EPOCHS = 3
    N_SPLITS = 5
    BATCH_SIZE = 32
    AUGMENT = False
    EARLY_STOPPING = -1
    TWO_STAGE_THRESHOLD = 10.0 # 2nd stageのデータとして使うためのtotal_evaluatorsの閾値
    TWO_STAGE_EPOCHS = 3 # 0のときは1stのみ
    SAVE_BEST = False # Falseのときは最後のモデルを保存
    SMOOTHING = False

RCFG.RUN_NAME = create_random_id()
TARGETS = ["seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"]
TARS = {'Seizure': 0, 'LPD': 1, 'GPD': 2, 'LRDA': 3, 'GRDA': 4, 'Other': 5}
TARS2 = {x:y for y,x in TARS.items()}
EFFICIENTNET_SIZE = {
    "efficientnet_b0": 1280, 
    "efficientnet_b2": 1408, 
    "efficientnet_b4": 1792, 
    "efficientnet_b5": 2048, 
    "efficientnet_b6": 2304, 
    "efficientnet_b7": 2560
}

def string_to_11_digit_hash(input_string):
    hash_object = hashlib.sha256(input_string.encode())
    hex_dig = hash_object.hexdigest()
    hash_int = int(hex_dig, 16)
    hash_11_digit = hash_int % (10**11)
    return hash_11_digit

def eeg_fill_na(x):
    m = np.nanmean(x)
    if np.isnan(x).mean()<1: 
        x = np.nan_to_num(x,nan=m)
    else: 
        x[:] = 0

    return x

def standardize_img(img):

    img = np.clip(img,np.exp(-4),np.exp(8))
    img = np.log(img)
    
    ep = 1e-6
    m = np.nanmean(img.flatten())
    s = np.nanstd(img.flatten())
    img = (img-m)/(s+ep)
    img = np.nan_to_num(img, nan=0.0)

    return img

class HMSDataset(Dataset):
    def __init__(
        self, 
        data, 
        all_spectrograms,
        augment=CFG.AUGMENT, 
        mode='train',
        smoothing=False
    ):

        self.data = data
        self.augment = augment
        self.mode = mode
        self.specs = all_spectrograms
        self.indexes = np.arange( len(self.data))
        self.smoothing = smoothing

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

        row = self.data.iloc[indexes]
        y = np.zeros((6),dtype='float32')
        
        if self.mode!='test':
            y = row.loc[TARGETS]
            if self.smoothing:
                y = self.__apply_label_smoothing(y)

        if self.mode=='test':
            r = 0
        else:
            r = int( row['spec_offset_second'] //2 )
            # r = int( (row['min'] + row['max'])//4 )

        img = self.specs['kaggle'][row.spectrogram_id]
        img = eeg_fill_na(img)
        img = standardize_img(img)

        x_tmp = np.zeros((128, 256, 4), dtype='float32')
        for k in range(4):
            img_t = img[r:r+300,k*100:(k+1)*100].T
            x_tmp[14:-14,:,k] = img_t[:,22:-22]

        x1 = np.concatenate([x_tmp[:, :, i:i+1] for i in range(4)], axis=0) # (512, 256, 1)

        # # v11
        # img = self.specs['cwt_v11'][row.eeg_id] # (64, 512, 4)
        # img = standardize_img(img)
        # img = np.concatenate([img[:, :, i:i+1] for i in range(4)], axis=0) # (256, 512, 1)
        # x2 = img.transpose(1, 0, 2) # (512, 256, 1)
        # img = np.vstack((img[:, :256, :], img[:, 256:, :])) # (64, 512, 4) -> (128, 256, 4)に変換
        # x2 = np.concatenate([img[:, :, i:i+1] for i in range(4)], axis=0) # (512, 256, 1)

        # (64, 512, 4)型
        img = self.specs['cwt_mexh_20sec_v105'][row.eeg_id] # (64, 512, 4)
        img = np.concatenate([img[:, :, i:i+1] for i in range(4)], axis=0) # (256, 512, 1)
        x2 = img.transpose(1, 0, 2) # (512, 256, 1)

        # (64, 512, 4)型
        img = self.specs['cwt_mexh_10sec_v105'][row.eeg_id] # (64, 512, 4))
        img = np.concatenate([img[:, :, i:i+1] for i in range(4)], axis=0) # (256, 512, 1)
        x3 = img.transpose(1, 0, 2) # (512, 256, 1)

        # (64, 512, 4)型
        img = self.specs['cwt_mexh_20sec_last_v105'][row.eeg_id] # (64, 512, 4))
        img = np.concatenate([img[:, :, i:i+1] for i in range(4)], axis=0) # (256, 512, 1)
        x4 = img.transpose(1, 0, 2) # (512, 256, 1)

        X = np.concatenate([x1, x2, x3, x4], axis=1) # (512, 768, 1)

        return X, y # (), (6)
        # return x1, y

    def _augment_batch(self, img):
        transforms = A.Compose([
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0, rotate_limit=0, p=0.3, border_mode=0), # 時間軸方向のシフト
            A.GaussNoise(var_limit=(10, 50), p=0.3) # ガウス雑音
        ])
        return transforms(image=img)['image']
    
    def __apply_label_smoothing(self, labels, smoothing=0.1):

        labels = labels * (1 - smoothing) + (smoothing / labels.shape[0])
        labels /= labels.sum()  # 再正規化
        return labels


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, device='cuda:0'):
        super(GeM, self).__init__()
        # pを固定値として定義
        self.p = torch.ones(1, device=device) * p
        self.eps = eps

    def forward(self, x):
        return F.avg_pool2d(
            x.clamp(min=self.eps).pow(self.p), 
            (x.size(-2), x.size(-1))).pow(1. / self.p)

    def __repr__(self):
        return f'GeM(p={self.p.item()}, eps={self.eps})'


class HMSModel(nn.Module):
    def __init__(self, pretrained=True, num_classes=6):
        super(HMSModel, self).__init__()
        in_features = EFFICIENTNET_SIZE[CFG.MODEL_NAME]
        self.fc = nn.Linear(in_features=in_features, out_features=num_classes)

        # conv2d
        # self.conv2d = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding=0)
        # self.relu = nn.ReLU()

        # GeM Pooling
        # self.gem = GeM(p=3, eps=1e-6)
        # self.base_model = timm.create_model(CFG.MODEL_NAME, pretrained=pretrained, num_classes=0, global_pool='', in_chans=CFG.IN_CHANS)

        # Baseline
        self.base_model = timm.create_model(CFG.MODEL_NAME, pretrained=pretrained, num_classes=num_classes, in_chans=CFG.IN_CHANS)
        self.base_model.classifier = self.fc

    def forward(self, x):
        x = x.repeat(1, 1, 1, 3) 
        x = x.permute(0, 3, 1, 2)

        # conv2d
        # x = self.conv2d(x)
        # x = self.relu(x)

        x = self.base_model(x)

        # Gem Pooling
        # x = self.gem(x) # (batch_size, 1280, 1, 1)
        # x = x.view(x.size(0), -1) # (batch_size, 1280)
        # x = self.fc(x)

        return x
    

def calc_cv_score(oof,true):
    from kaggle_kl_div import score

    oof = pd.DataFrame(np.concatenate(oof).copy())
    oof['id'] = np.arange(len(oof))
    true = pd.DataFrame(np.concatenate(true).copy())
    true['id'] = np.arange(len(true))
    cv = score(solution=true, submission=oof, row_id_column_name='id')
    return cv

def get_cv_score(oof,true):
    from kaggle_kl_div import score

    oof = pd.DataFrame(oof)
    oof['id'] = np.arange(len(oof))
    true = pd.DataFrame(true)
    true['id'] = np.arange(len(true))
    cv = score(solution=true, submission=oof, row_id_column_name='id')
    return cv


# モデル訓練関数
def train_model(model, train_loader, valid_loader, optimizer, scheduler, criterion):
    model.train()
    train_loss = []
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(torch.device(RCFG.DEVICE)), labels.to(torch.device(RCFG.DEVICE))
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(log_softmax(outputs, dim = 1), labels)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
        if RCFG.DEBUG:
            logger.info(f'train_loss: {loss.item()}')
    scheduler.step()
    # 検証ループ
    model.eval()
    oof = []
    true = []
    valid_loss = []
    with torch.no_grad():
        for inputs, labels in valid_loader:
            true.append(labels)
            inputs, labels = inputs.to(torch.device(RCFG.DEVICE)), labels.to(torch.device(RCFG.DEVICE))
            outputs = model(inputs)
            loss = criterion(log_softmax(outputs, dim = 1), labels)
            valid_loss.append(loss.item())
            oof.append(softmax(outputs,dim=1).to('cpu').numpy())

    return model, oof, np.mean(train_loss),np.mean(valid_loss)


def inference_function(test_loader, model, device):
    model.eval()
    softmax = nn.Softmax(dim=1)
    prediction_dict = {}
    preds = []
    with tqdm(test_loader, unit="test_batch", desc='Inference') as tqdm_test_loader:
        for step, X in enumerate(tqdm_test_loader):
            X = X.to(device)
            with torch.no_grad():
                y_preds = model(X)
            y_preds = softmax(y_preds)
            preds.append(y_preds.to('cpu').numpy()) 
                
    prediction_dict["predictions"] = np.concatenate(preds) 
    return prediction_dict


######################################################
# Runner
######################################################


class Runner():
    def __init__(self, env="colab", commit_hash=""):

        global ENV, ROOT_PATH, OUTPUT_PATH, MODEL_PATH
        ENV = env
        ROOT_PATH = '/content/drive/MyDrive/HMS' if ENV == "colab" else '/kaggle'
        OUTPUT_PATH = ROOT_PATH if ENV == "colab" else '/kaggle/working'
        MODEL_PATH = '/content/drive/MyDrive/HMS/model' if ENV == "colab" else '/kaggle/input/hms-hbac-model'
        if ENV == "kaggle":
            (Path(OUTPUT_PATH) / 'log').mkdir(exist_ok=True)
            (Path(OUTPUT_PATH) / 'model').mkdir(exist_ok=True)
            (Path(OUTPUT_PATH) / 'data').mkdir(exist_ok=True)
        sys.path.append(f'{ROOT_PATH}/input/kaggle-kl-div') # score関数のために必要

        set_random_seed()
        global logger
        logger = Logger(log_path=f'{OUTPUT_PATH}/log/', filename_suffix=RCFG.RUN_NAME, debug=RCFG.DEBUG)
        logger.info(f'Initializing Runner.　Run Name: {RCFG.RUN_NAME}')
        start_dt_jst = str(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).strftime('%Y-%m-%d %H:%M:%S'))
        self.info =  {"start_dt_jst": start_dt_jst}
        self.info['fold_cv'] = [0 for _ in range(5)]

        logger.info(f'commit_hash: {commit_hash}')
        RCFG.COMMIT_HASH=commit_hash
        
        if ENV == "kaggle":
            from kaggle_secrets import UserSecretsClient
            self.user_secrets = UserSecretsClient()

        if RCFG.SAVE_TO_SHEET:
            sheet_json_key = ROOT_PATH + '/input/ktokunagautils/ktokunaga-4094cf694f5c.json'
            logger.info('Initializing Google Sheet.')
            self.sheet = WriteSheet(
                sheet_json_key = sheet_json_key,
                sheet_key = RCFG.SHEET_KEY
            )

        if RCFG.DEBUG:
            logger.info('DEBUG MODE: Decrease N_SPLITS, EPOCHS, BATCH_SIZE.')
            CFG.N_SPLITS = 2
            CFG.EPOCHS = 2
            CFG.BATCH_SIZE = 8

        self.MODEL_FILES =[MODEL_PATH + f"/{RCFG.RUN_NAME}_fold{k}_{CFG.MODEL_NAME}.pickle" for k in range(CFG.N_SPLITS)]

    
    def load_dataset(self, ):
        
        df = pd.read_csv(ROOT_PATH + '/input/hms-harmful-brain-activity-classification/train.csv')
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

        logger.info('Create labels considering 2nd stage learning.')
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

        # Create Fold
        if RCFG.USE_ALL_LOW_QUALITY:
            logger.info('Use all low quality data.')
            train_2nd = train[train['stage']==2].copy().reset_index()
            train_2nd["fold"] = -1
            sgkf = StratifiedGroupKFold(n_splits=5)
            for fold_id, (_, val_idx) in enumerate(
                sgkf.split(train_2nd, y=train_2nd['target'], groups=train_2nd["patient_id"])
            ):
                train_2nd.loc[val_idx, "fold"] = fold_id
            df_patient_id_fold = train_2nd[['patient_id', 'fold']].drop_duplicates()
            train = train.merge(df_patient_id_fold, on='patient_id', how='left')
            train.loc[train['fold'].isnull(), 'fold'] = -1
        else:
            sgkf = StratifiedGroupKFold(n_splits=CFG.N_SPLITS, shuffle=True, random_state=34)
            train["fold"] = -1
            for fold_id, (_, val_idx) in enumerate(
                sgkf.split(train, y=train["target"], groups=train["patient_id"])
            ):
                train.loc[val_idx, "fold"] = fold_id

        if RCFG.ADD_MIXUP_DATA:
            logger.info('Add external data.')
            mixup_data = pd.read_csv(ROOT_PATH + '/input/hms-harmful-brain-activity-classification/df_mixup_v2.csv')
            mixup_data = mixup_data[train.columns]
            mixup_data['target'] = 'Ext'
            if RCFG.DEBUG:
                mixup_data = mixup_data.iloc[:100]

            train = pd.concat([train, mixup_data]).reset_index()
            logger.info(f'Train shape after adding external data: {train.shape}')

        if RCFG.PSEUDO_LABELLING:
            logger.info('Load pseudo labelling data.')
            train = pd.read_csv(ROOT_PATH + '/data/dkyibdx_train_oof.csv')
            targets_oof = [f"{c}_oof" for c in TARGETS]
            pseudo_labels = train.loc[train['total_evaluators']<10.0, targets_oof]
            train.loc[pseudo_labels.index, TARGETS] = pseudo_labels.values

        self.train = train

    def load_spectrograms(self, ):
        self.all_spectrograms = {}
        for name in RCFG.USE_SPECTROGRAMS:
            logger.info(f'Loading spectrograms eeg_spec_{name}.py')
            self.all_spectrograms[name] = np.load(ROOT_PATH + f'/input/hms-hbac-data/eeg_specs_{name}.npy',allow_pickle=True).item()

        if RCFG.ADD_MIXUP_DATA:
            for name in RCFG.USE_SPECTROGRAMS:
                if name == 'kaggle': continue
                logger.info(f'Loading mixup spectrograms eeg_spec_{name}.py')
                self.all_spectrograms[name].update(np.load(ROOT_PATH + f'/input/hms-hbac-data/eeg_specs_mixup_v2_{name}.npy',allow_pickle=True).item())

    def run_train(self, ):

        TARGETS_OOF = [f"{c}_oof" for c in TARGETS]
        self.train[TARGETS_OOF] = 0

        fold_lists = RCFG.USE_FOLD if len(RCFG.USE_FOLD) > 0 else list(range(CFG.N_SPLITS))
        train_2nd = self.train[self.train['stage']==2]
        
        for fold_id in fold_lists:

            logger.info(f'###################################### Fold {fold_id+1}')
            train_index = self.train[self.train.fold != fold_id].index
            valid_index = self.train[self.train.fold == fold_id].index
            
            valid_df = self.train[self.train.fold == fold_id].reset_index().copy()
            true = valid_df[TARGETS].values
            valid_2nd_index = valid_df[valid_df['stage']==2].index
            train_2nd_index = train_2nd[train_2nd.fold != fold_id].index
            
            # データローダーの作成
            train_dataset = HMSDataset(
                self.train.iloc[train_index],
                self.all_spectrograms,
                smoothing = CFG.SMOOTHING
            )
            logger.info(f'Length of train_dataset: {len(train_dataset)}')
            train_loader = DataLoader(train_dataset, batch_size=CFG.BATCH_SIZE, shuffle=True, num_workers=2,pin_memory=True)

            valid_dataset = HMSDataset(
                self.train.iloc[valid_index],
                self.all_spectrograms
            )
            valid_loader = DataLoader(valid_dataset, batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=2,pin_memory=True)

            # モデルの構築
            model = HMSModel().to(torch.device(RCFG.DEVICE))
            optimizer = optim.AdamW(model.parameters(),lr=0.001)
            # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.EPOCHS, eta_min=1e-6)
            # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2, 5], gamma=0.1)
            # 学習率スケジュールを定義
            lr_schedule = {0: 1e-3, 1: 1e-3, 2: 1e-4, 3: 1e-4}
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: lr_schedule[epoch] / lr_schedule[0])
            criterion = nn.KLDivLoss(reduction='batchmean')  # 適切な損失関数を選択

            # トレーニングループ
            best_valid_loss = np.inf
            best_cv = np.inf
            best_epoch = 0
            best_oof = None
            for epoch in range(1, CFG.EPOCHS+1):
                model, oof, tr_loss, val_loss = train_model(
                    model, 
                    train_loader, 
                    valid_loader,
                    optimizer,
                    scheduler,
                    criterion
                )
                
                oof = np.concatenate(oof).copy()
                valid_2nd_loss = get_cv_score(oof[valid_2nd_index], true[valid_2nd_index])

                # エポックごとのログを出力
                logger.info(f'Epoch {epoch}, Train Loss: {np.round(tr_loss, 6)}, Valid Loss: {np.round(val_loss, 6)}, Valid 2nd Loss: {np.round(valid_2nd_loss, 6)}')

                if not CFG.SAVE_BEST or val_loss < best_valid_loss:
                    best_oof = oof
                    best_epoch = epoch
                    best_cv = valid_2nd_loss
                    best_valid_loss = val_loss
                    self.info['fold_cv'][fold_id] = valid_2nd_loss
                    if not RCFG.DEBUG:
                        torch.save(model.state_dict(), OUTPUT_PATH + f'/model/{RCFG.RUN_NAME}_fold{fold_id}_{CFG.MODEL_NAME}.pickle')


            if CFG.TWO_STAGE_EPOCHS > 0:
                
                logger.info(f'############ Second Stage')
                # データローダーの作成
                train_dataset = HMSDataset(
                    train_2nd.loc[train_2nd_index],
                    self.all_spectrograms
                )
                logger.info(f'Length of train_dataset: {len(train_dataset)}')
                train_loader = DataLoader(train_dataset, batch_size=CFG.BATCH_SIZE, shuffle=True, num_workers=2,pin_memory=True)

                optimizer = optim.AdamW(model.parameters(),lr=1e-4)
                lr_schedule = {0: 1e-4, 1: 1e-5, 2: 1e-5, 3: 1e-6}
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: lr_schedule[epoch] / lr_schedule[0])
                # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2, 4], gamma=0.1)
                criterion = nn.KLDivLoss(reduction='batchmean') 

                # トレーニングループ
                best_valid_loss = np.inf
                best_cv = np.inf
                best_epoch = 0
                best_oof = None
                for epoch in range(1, CFG.TWO_STAGE_EPOCHS+1):
                    model, oof, tr_loss, val_loss = train_model(
                        model, 
                        train_loader, 
                        valid_loader,
                        optimizer,
                        scheduler,
                        criterion
                    )

                    oof = np.concatenate(oof).copy()
                    valid_2nd_loss = get_cv_score(oof[valid_2nd_index], true[valid_2nd_index])

                    # エポックごとのログを出力
                    logger.info(f'Epoch {epoch}, Train Loss: {np.round(tr_loss, 6)}, Valid Loss: {np.round(val_loss, 6)}, Valid 2nd Loss: {np.round(valid_2nd_loss, 6)}')

                    if not CFG.SAVE_BEST or val_loss < best_valid_loss:
                        best_oof = oof
                        best_epoch = epoch
                        best_cv = valid_2nd_loss
                        best_valid_loss = val_loss
                        self.info['fold_cv'][fold_id] = valid_2nd_loss
                        if not RCFG.DEBUG:
                            torch.save(model.state_dict(), OUTPUT_PATH + f'/model/{RCFG.RUN_NAME}_fold{fold_id}_{CFG.MODEL_NAME}.pickle')

            self.train.loc[valid_index, TARGETS_OOF] = best_oof
            self.train.to_csv(OUTPUT_PATH + f'/data/{RCFG.RUN_NAME}_train_oof.csv', index=False)
            logger.info(f'CV Score KL-Div for {CFG.MODEL_NAME} fold_id {fold_id}: {best_cv} (Epoch {best_epoch})')

            del model
            gc.collect()
            torch.cuda.empty_cache()


    def write_sheet(self, ):
        logger.info('Write info to google sheet.')
        write_dt_jst = str(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).strftime('%Y-%m-%d %H:%M:%S'))

        data = [
            RCFG.RUN_NAME,
            self.info['start_dt_jst'],
            write_dt_jst,
            RCFG.COMMIT_HASH,
            ENV,
            CFG.MODEL_NAME,
            class_vars_to_dict(RCFG),
            class_vars_to_dict(CFG),
            *self.info['fold_cv'],
            np.mean(self.info['fold_cv'][:CFG.N_SPLITS])
        ]
    
        self.sheet.write(data, sheet_name='cv_scores')


    def inference(self, all_infer_spectrograms = {}):
        
        logger.info('Start inference.')
        test_df = pd.read_csv(ROOT_PATH + '/input/hms-harmful-brain-activity-classification/test.csv')

        paths_spectrograms = glob(ROOT_PATH + '/input/hms-harmful-brain-activity-classification/test_spectrograms/*.parquet')
        logger.info(f'There are {len(paths_spectrograms)} spectrogram parquets')

        paths_eegs = glob(ROOT_PATH + '/input/hms-harmful-brain-activity-classification/test_eegs/*.parquet')
        logger.info(f'There are {len(paths_eegs)} EEG spectrograms')
        
        all_infer_spectrograms = {}
        for name in RCFG.USE_SPECTROGRAMS:
            all_infer_spectrograms[name] = {}

        for file_path in tqdm(paths_spectrograms):
            aux = pd.read_parquet(file_path)
            name = int(file_path.split("/")[-1].split('.')[0])
            all_infer_spectrograms['kaggle'] [name] = aux.iloc[:,1:].values
            del aux
        
        gc.collect()
        TMP = Path(ROOT_PATH) / "tmp"
        if RCFG.CREATE_SPECS:
            TMP.mkdir(exist_ok=True)
            
            converted_specs = {}            
            for file_path in tqdm(paths_eegs):
                eeg_id = file_path.split("/")[-1].split(".")[0]
                converted_specs[int(eeg_id)] = spectrogram_from_eeg(file_path)
            np.save(TMP / f'eeg_specs_v2.npy', converted_specs)
            del converted_specs
            gc.collect()
            
            converted_specs = {}            
            for file_path in tqdm(paths_eegs):
                eeg_id = file_path.split("/")[-1].split(".")[0]
                converted_specs[int(eeg_id)] = spectrogram_from_eeg(file_path)
            np.save(TMP / f'eeg_specs_cwt_v5.npy', converted_specs)
            del converted_specs
            gc.collect()
            
            converted_specs = {}            
            for file_path in tqdm(paths_eegs):
                eeg_id = file_path.split("/")[-1].split(".")[0]
                converted_specs[int(eeg_id)] = spectrogram_from_eeg(file_path)
            np.save(TMP / f'eeg_specs_cwt_v11.npy', converted_specs)
            del converted_specs
            gc.collect()
            
            converted_specs = {}            
            for file_path in tqdm(paths_eegs):
                eeg_id = file_path.split("/")[-1].split(".")[0]
                converted_specs[int(eeg_id)] = spectrogram_from_eeg(file_path)
            np.save(TMP / f'eeg_specs_cqt.npy', converted_specs)
            del converted_specs
            gc.collect()
        
        for name in RCFG.USE_SPECTROGRAMS:
            if name == 'kaggle': continue
            all_infer_spectrograms[name] = np.load(TMP / f'eeg_specs_{name}.npy', allow_pickle=True).item()

        test_dataset = HMSDataset(
            data = test_df, 
            all_spectrograms = all_infer_spectrograms,
            mode="test"
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=CFG.BATCH_SIZE,
            shuffle=False,
            num_workers=0, 
            pin_memory=True, 
            drop_last=False
        )

        predictions = []
        for model_weight in self.MODEL_FILES:
            logger.info(f'model weight: {model_weight}')
            model = HMSModel(pretrained=False)
            checkpoint = torch.load(model_weight)
            model.load_state_dict(checkpoint)
            model.to(torch.device(RCFG.DEVICE))
            prediction_dict = inference_function(test_loader, model, torch.device(RCFG.DEVICE))
            predictions.append(prediction_dict["predictions"])
            del model
            torch.cuda.empty_cache()
            gc.collect()
            
        predictions = np.array(predictions)
        predictions = np.mean(predictions, axis=0)

        self.sub = pd.DataFrame({'eeg_id': test_df.eeg_id.values})
        self.sub[TARGETS] = predictions
        self.sub.to_csv('submission.csv',index=False)
        

    def main(self):
        self.load_dataset()
        self.load_spectrograms()
        self.run_train()

        if RCFG.SAVE_TO_SHEET:
            self.write_sheet()
        
        if RCFG.PREDICT:
            self.inference()