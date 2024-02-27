# EfficientNet or Resnet

import sys
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
from tqdm import tqdm
from glob import glob
from pathlib import Path
from torch import optim
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.model_selection import KFold, GroupKFold, StratifiedGroupKFold
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import log_softmax, softmax

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

class CFG:
    """モデルに関連する設定"""
    MODEL_NAME = 'efficientnet_b0'
    IN_CHANS = 3
    EPOCHS = 3
    N_SPLITS = 5
    BATCH_SIZE = 32
    AUGMENT = False
    EARLY_STOPPING = -1
    TWO_STAGE_EPOCHS = 3 # 0のときは1stのみ
    SAVE_BEST = False # Falseのときは最後のモデルを保存

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

class HMSDataset(Dataset):
    def __init__(
        self, 
        data, 
        all_spectrograms,
        augment=CFG.AUGMENT, 
        mode='train',
    ):

        self.data = data
        self.augment = augment
        self.mode = mode
        self.specs = all_spectrograms
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

        X = np.zeros((128,256,12),dtype='float32')
        y = np.zeros((6),dtype='float32')
        img = np.ones((128,256),dtype='float32')

        row = self.data.iloc[indexes]
        if self.mode=='test':
            r = 0
        else:
            r = int( (row['min'] + row['max'])//4 )

        for k in range(4):
            # EXTRACT 300 ROWS OF SPECTROGRAM(4種類抜いてくる)
            img = self.specs['kaggle'][row.spectrogram_id][r:r+300,k*100:(k+1)*100].T

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

        # # Chris
        # img = self.specs['chris'][row.eeg_id] # (128, 256, 4)
        # X[:,:,4:8] = img

        # v2
        img = self.specs['v2'][row.eeg_id] # (128, 256, 4)
        X[:,:,4:8] = img

        # # cqt
        # img = self.specs['cqt'][row.eeg_id] # (128, 256, 4)
        # img = np.clip(img,np.exp(-4),np.exp(8))
        # img = np.log(img)
        # ep = 1e-6
        # m = np.nanmean(img.flatten())
        # s = np.nanstd(img.flatten())
        # img = (img-m)/(s+ep)
        # img = np.nan_to_num(img, nan=0.0)
        # X[:,:,12:16] = img

        # v11
        img = self.specs['v11'][row.eeg_id] # (64, 256, 4)
        img = np.clip(img,np.exp(-4),np.exp(8))
        img = np.log(img)
        ep = 1e-6
        m = np.nanmean(img.flatten())
        s = np.nanstd(img.flatten())
        img = (img-m)/(s+ep)
        img = np.nan_to_num(img, nan=0.0)
        # img = np.vstack((img[:, :, :2], img[:, :, 2:])) # (64, 256, 4) -> (128, 256, 2)に変換
        img = np.vstack((img[:, :256, :], img[:, 256:, :])) # (64, 512, 2) -> (128, 256, 4)に変換
        X[:,:,8:12] = img

        
        if self.mode!='test':
            y = row.loc[TARGETS]

        return X,y # (128,256,12), (6)

    def _augment_batch(self, img):
        transforms = A.Compose([
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0, rotate_limit=0, p=0.3, border_mode=0), # 時間軸方向のシフト
            A.GaussNoise(var_limit=(10, 50), p=0.3) # ガウス雑音
        ])
        return transforms(image=img)['image']


class CustomInputTransform(nn.Module):
    def __init__(self, ):
        super(CustomInputTransform, self).__init__()

    def forward(self, x): 
        x1 = torch.cat([x[:, :, :, i:i+1] for i in range(4)], dim=1) # (batch_size, 512, 256, 1)
        x2 = torch.cat([x[:, :, :, i+4:i+5] for i in range(4)], dim=1) # (batch_size, 512, 256, 1)
        x3 = torch.cat([x[:, :, :, i+8:i+9] for i in range(4)], dim=1) # (batch_size, 512, 256, 1)
        # x4 = torch.cat([x[:, :, :, i+12:i+13] for i in range(4)], dim=1) # (batch_size, 512, 256, 1)
        # x5 = torch.cat([x[:, :, :, i+16:i+17] for i in range(2)], dim=1) # (batch_size, 256, 256, 1)

        # x_t = torch.cat([x1, x2, x3], dim=2) # (batch_size, 512, 768, 1)
        x = torch.cat([x1, x2, x3], dim=2) # (batch_size, 512, 768, 1)
        # x_t2 = torch.cat([x4, x5], dim=1) #(batch_size, 768, 256, 1)
        # x_t2 = x_t2.permute(0, 2, 1, 3) # (batch_size, 256, 768, 1)
        # x = torch.cat([x_t, x_t2], dim=1) # (batch_size, 768, 768, 1)
    
        x = x.repeat(1, 1, 1, 3) 
        x = x.permute(0, 3, 1, 2)
        return x

class HMSModel(nn.Module):
    def __init__(self, pretrained=True, num_classes=6):
        super(HMSModel, self).__init__()
        self.input_transform = CustomInputTransform()
        self.base_model = timm.create_model(CFG.MODEL_NAME, pretrained=pretrained, num_classes=num_classes, in_chans=CFG.IN_CHANS)

        # EfficientNetで必要
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        in_features = EFFICIENTNET_SIZE[CFG.MODEL_NAME]
        self.fc = nn.Linear(in_features=in_features, out_features=num_classes)
        self.base_model.classifier = self.fc

    def forward(self, x):
        x = self.input_transform(x)
        x = self.base_model(x)
        return x
    

def calc_cv_score(oof,true):
    from kaggle_kl_div import score

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

    # モデルの重みを保存
    cv=calc_cv_score(oof,true)
    return model, oof, np.mean(train_loss),np.mean(valid_loss),cv


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
        train = df.groupby('eeg_id').agg(
            spectrogram_id= ('spectrogram_id','first'),
            min = ('spectrogram_label_offset_seconds','min'),
            max = ('spectrogram_label_offset_seconds','max'),
            patient_id = ('patient_id','first'),
            total_evaluators = ('total_evaluators','mean'),
            target = ('expert_consensus','first'),
            seizure_vote = ('seizure_vote','sum'),
            lpd_vote = ('lpd_vote','sum'),
            gpd_vote = ('gpd_vote','sum'),
            lrda_vote = ('lrda_vote','sum'),
            grda_vote = ('grda_vote','sum'),
            other_vote = ('other_vote','sum'),
        )
        y_data = train[TARGETS].values
        y_data = y_data / y_data.sum(axis=1,keepdims=True)
        train[TARGETS] = y_data

        # compute kl-loss with uniform distribution by pytorch
        labels = train[TARGETS].values + 1e-5
        train['kl'] = torch.nn.functional.kl_div(
            torch.log(torch.tensor(labels)),
            torch.tensor([1 / 6] * 6),
            reduction='none'
        ).sum(dim=1).numpy()

        if RCFG.DEBUG:
            train = train.iloc[:RCFG.DEBUG_SIZE]

        self.train = train.reset_index()
        logger.info(f'Train non-overlapp eeg_id shape: {train.shape}')

        # Create Fold
        sgkf = StratifiedGroupKFold(n_splits=CFG.N_SPLITS, shuffle=True, random_state=34)
        self.train["fold"] = -1
        for fold_id, (_, val_idx) in enumerate(
            sgkf.split(self.train, y=self.train["target"], groups=self.train["patient_id"])
        ):
            self.train.loc[val_idx, "fold"] = fold_id

        # READ ALL SPECTROGRAMS
        self.all_spectrograms = {}
        logger.info('Loading spectrograms specs.py')
        self.all_spectrograms['kaggle'] = np.load(ROOT_PATH  + '/input/hms-hbac-data/specs.npy',allow_pickle=True).item()
        # logger.info('Loading spectrograms eeg_spec.py')
        # self.all_spectrograms['chris'] = np.load(ROOT_PATH + '/input/hms-hbac-data/eeg_specs.npy',allow_pickle=True).item()
        logger.info('Loading spectrograms eeg_spec_v2.py')
        self.all_spectrograms['v2'] = np.load(ROOT_PATH + '/input/hms-hbac-data/eeg_specs_v2.npy',allow_pickle=True).item()
        logger.info('Loading spectrograms eeg_spec_cwt_v11.py')
        self.all_spectrograms['v11'] = np.load(ROOT_PATH + '/input/hms-hbac-data/eeg_specs_cwt_v11.npy',allow_pickle=True).item()
        # logger.info('Loading spectrograms eeg_spec_cqt.py')
        # self.all_spectrograms['cqt'] = np.load(ROOT_PATH + '/input/hms-hbac-data/eeg_specs_cqt.npy',allow_pickle=True).item()

    def run_train(self, ):

        TARGETS_OOF = [f"{c}_oof" for c in TARGETS]
        self.train[TARGETS_OOF] = 0

        fold_lists = RCFG.USE_FOLD if len(RCFG.USE_FOLD) > 0 else list(range(CFG.N_SPLITS))
        for fold_id in fold_lists:

            logger.info(f'###################################### Fold {fold_id+1}')
            train_index = self.train[self.train.fold != fold_id].index
            valid_index = self.train[self.train.fold == fold_id].index
            
            # データローダーの作成
            train_dataset = HMSDataset(
                self.train.iloc[train_index],
                self.all_spectrograms
            )
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
            lr_schedule = {0: 1e-3, 1: 1e-3, 2: 1e-4}
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: lr_schedule[epoch] / lr_schedule[0])
            criterion = nn.KLDivLoss(reduction='batchmean')  # 適切な損失関数を選択

            # トレーニングループ
            best_valid_loss = np.inf
            best_cv = np.inf
            best_epoch = 0
            best_oof = None
            for epoch in range(1, CFG.EPOCHS+1):
                model, oof, tr_loss, val_loss, cv = train_model(
                    model, 
                    train_loader, 
                    valid_loader,
                    optimizer,
                    scheduler,
                    criterion
                )
                # エポックごとのログを出力
                logger.info(f'Epoch {epoch}, Train Loss: {tr_loss}, Valid Loss: {val_loss}')

                if not CFG.SAVE_BEST or val_loss < best_valid_loss:
                    best_oof = np.concatenate(oof).copy()
                    best_epoch = epoch
                    best_cv = cv
                    best_valid_loss = val_loss
                    self.info['fold_cv'][fold_id] = cv
                    if not RCFG.DEBUG:
                        torch.save(model.state_dict(), OUTPUT_PATH + f'/model/{RCFG.RUN_NAME}_fold{fold_id}_{CFG.MODEL_NAME}.pickle')


            if CFG.TWO_STAGE_EPOCHS > 0:
                
                logger.info(f'############ Second Stage')
                # データローダーの作成
                train_2nd = self.train[self.train['total_evaluators']>= 10]
                train_index = train_2nd[train_2nd.fold != fold_id].index
                train_dataset = HMSDataset(
                    train_2nd.loc[train_index],
                    self.all_spectrograms
                )
                train_loader = DataLoader(train_dataset, batch_size=CFG.BATCH_SIZE, shuffle=True, num_workers=2,pin_memory=True)

                optimizer = optim.AdamW(model.parameters(),lr=1e-4)
                lr_schedule = {0: 1e-4, 1: 1e-5, 2: 1e-5}
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: lr_schedule[epoch] / lr_schedule[0])
                # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2, 4], gamma=0.1)
                criterion = nn.KLDivLoss(reduction='batchmean') 

                # トレーニングループ
                best_valid_loss = np.inf
                best_cv = np.inf
                best_epoch = 0
                best_oof = None
                for epoch in range(1, CFG.TWO_STAGE_EPOCHS+1):
                    model, oof, tr_loss, val_loss, cv = train_model(
                        model, 
                        train_loader, 
                        valid_loader,
                        optimizer,
                        scheduler,
                        criterion
                    )
                    # エポックごとのログを出力
                    logger.info(f'Epoch {epoch}, Train Loss: {tr_loss}, Valid Loss: {val_loss}')

                    if not CFG.SAVE_BEST or val_loss < best_valid_loss:
                        best_oof = np.concatenate(oof).copy()
                        best_epoch = epoch
                        best_cv = cv
                        best_valid_loss = val_loss
                        self.info['fold_cv'][fold_id] = cv
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
        
        all_infer_spectrograms['kaggle'] = {}
        for file_path in tqdm(paths_spectrograms):
            aux = pd.read_parquet(file_path)
            name = int(file_path.split("/")[-1].split('.')[0])
            all_infer_spectrograms['kaggle'] [name] = aux.iloc[:,1:].values
            del aux
        
        for file_path in tqdm(paths_eegs):
            eeg_id = file_path.split("/")[-1].split(".")[0]
            eeg_spectrogram = spectrogram_from_eeg(file_path)
            all_infer_spectrograms['v2'][int(eeg_id)] = eeg_spectrogram
            all_infer_spectrograms['v5'][int(eeg_id)] = spectrogram_from_eeg_cwt(file_path)

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
            torch.cuda.empty_cache()
            gc.collect()
            
        predictions = np.array(predictions)
        predictions = np.mean(predictions, axis=0)

        self.sub = pd.DataFrame({'eeg_id': test_df.eeg_id.values})
        self.sub[TARGETS] = predictions
        self.sub.to_csv('submission.csv',index=False)

        return all_infer_spectrograms
        

    def main(self):
        self.load_dataset()
        self.run_train()

        if RCFG.SAVE_TO_SHEET:
            self.write_sheet()
        
        if RCFG.PREDICT:
            self.inference()