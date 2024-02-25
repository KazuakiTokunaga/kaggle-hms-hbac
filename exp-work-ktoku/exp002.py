# 1dCNN
# https://www.kaggle.com/code/medali1992/hms-resnet1d-gru-train

import sys
import torch
import torch.nn as nn
import os, gc
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import datetime
import random
import warnings
import albumentations as A
import pathlib
from scipy.signal import butter, lfilter
from tqdm import tqdm
from glob import glob
from pathlib import Path
from torch import optim
from sklearn.model_selection import KFold, GroupKFold, StratifiedGroupKFold
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import log_softmax, softmax

from utils import set_random_seed, create_random_id
from utils import WriteSheet, Logger, class_vars_to_dict

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
    MODEL_NAME = '1dCNN'
    EPOCHS = 10
    IN_CHANNELS = 8
    TARGET_SIZE = 6
    N_SPLITS = 5
    BATCH_SIZE = 64
    EARLY_STOPPING = -1
    COSANNEAL_RES_PARAMS = {
        'T_0':20,
        'eta_min':1e-6,
        'T_mult':1,
        'last_epoch':-1
    }
    TWO_STAGE_EPOCHS = 6 # 0のときは1stのみ

RCFG.RUN_NAME = create_random_id()
TARGETS = ["seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"]
EEG_FEATURES = ['Fp1','T3','C3','O1','Fp2','C4','T4','O2']
FEATURE2INDEX = {x:y for x,y in zip(EEG_FEATURES, range(len(EEG_FEATURES)))}


def butter_lowpass_filter(data, cutoff_freq=20, sampling_rate=200, order=4):
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = lfilter(b, a, data, axis=0)
    return filtered_data

def eeg_from_parquet(parquet_path: str, display: bool = False) -> np.ndarray:
    """
    This function reads a parquet file and extracts the middle 50 seconds of readings. Then it fills NaN values
    with the mean value (ignoring NaNs).
    :param parquet_path: path to parquet file.
    :param display: whether to display EEG plots or not.
    :return data: np.array of shape  (time_steps, EEG_FEATURES) -> (10_000, 8)
    """
    # === Extract middle 50 seconds ===
    eeg = pd.read_parquet(parquet_path, columns=EEG_FEATURES)
    rows = len(eeg)
    offset = (rows - 10_000) // 2 # 50 * 200 = 10_000
    eeg = eeg.iloc[offset:offset+10_000] # middle 50 seconds, has the same amount of readings to left and right
    if display: 
        plt.figure(figsize=(10,5))
        offset = 0
    # === Convert to numpy ===
    data = np.zeros((10_000, len(EEG_FEATURES))) # create placeholder of same shape with zeros
    for index, feature in enumerate(EEG_FEATURES):
        x = eeg[feature].values.astype('float32') # convert to float32
        mean = np.nanmean(x) # arithmetic mean along the specified axis, ignoring NaNs
        nan_percentage = np.isnan(x).mean() # percentage of NaN values in feature
        # === Fill nan values ===
        if nan_percentage < 1: # if some values are nan, but not all
            x = np.nan_to_num(x, nan=mean)
        else: # if all values are nan
            x[:] = 0
        data[:, index] = x
        if display: 
            if index != 0:
                offset += x.max()
            plt.plot(range(10_000), x-offset, label=feature)
            offset -= x.min()
    if display:
        plt.legend()
        name = parquet_path.split('/')[-1].split('.')[0]
        plt.yticks([])
        plt.title(f'EEG {name}',size=16)
        plt.show()    
    return data


class HMSDataset(Dataset):
    def __init__(
        self, 
        df, 
        eegs, 
        mode: str = 'train',
        downsample = None
    ): 
        self.df = df
        self.mode = mode
        self.eegs = eegs
        self.downsample = downsample
        self.batch_size = CFG.BATCH_SIZE
        
    def __len__(self):
        """
        Length of dataset.
        """
        return len(self.df)
        
    def __getitem__(self, index):
        """
        Get one item.
        """
        X, y = self.__data_generation(index)
        if self.downsample is not None:
            X = X[::self.downsample,:]
        if self.mode != 'test':
            return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
        else:
            return torch.tensor(X, dtype=torch.float32)
                        
    def __data_generation(self, index):
        row = self.df.iloc[index]
        X = np.zeros((10_000, 8), dtype='float32')
        y_prob = np.zeros(6, dtype='float32')
        data = self.eegs[row.eeg_id]

        # === Feature engineering ===
        X[:,0] = data[:,FEATURE2INDEX['Fp1']] - data[:,FEATURE2INDEX['T3']]
        X[:,1] = data[:,FEATURE2INDEX['T3']] - data[:,FEATURE2INDEX['O1']]

        X[:,2] = data[:,FEATURE2INDEX['Fp1']] - data[:,FEATURE2INDEX['C3']]
        X[:,3] = data[:,FEATURE2INDEX['C3']] - data[:,FEATURE2INDEX['O1']]

        X[:,4] = data[:,FEATURE2INDEX['Fp2']] - data[:,FEATURE2INDEX['C4']]
        X[:,5] = data[:,FEATURE2INDEX['C4']] - data[:,FEATURE2INDEX['O2']]

        X[:,6] = data[:,FEATURE2INDEX['Fp2']] - data[:,FEATURE2INDEX['T4']]
        X[:,7] = data[:,FEATURE2INDEX['T4']] - data[:,FEATURE2INDEX['O2']]

        # === Standarize ===
        X = np.clip(X,-1024, 1024)
        X = np.nan_to_num(X, nan=0) / 32.0

        # === Butter Low-pass Filter ===
        X = butter_lowpass_filter(X)
        if self.mode != 'test':
            y_prob = row[TARGETS].values.astype(np.float32)
        return X, y_prob

class ResNet_1D_Block(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, downsampling):
        super(ResNet_1D_Block, self).__init__()
        self.bn1 = nn.BatchNorm1d(num_features=in_channels)
        self.relu = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(p=0.0, inplace=False)
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(num_features=out_channels)
        self.conv2 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=False)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.downsampling = downsampling

    def forward(self, x):
        identity = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)

        out = self.maxpool(out)
        identity = self.downsampling(x)

        out += identity
        return out


class HMSModel(nn.Module):

    def __init__(self, 
        kernels = [3, 5, 7, 9], 
        in_channels=CFG.IN_CHANNELS, 
        fixed_kernel_size=5, 
        num_classes=CFG.TARGET_SIZE
    ):
        
        super(HMSModel, self).__init__()
        self.kernels = kernels
        self.planes = 24
        self.parallel_conv = nn.ModuleList()
        self.in_channels = in_channels
        
        for i, kernel_size in enumerate(list(self.kernels)):
            sep_conv = nn.Conv1d(in_channels=in_channels, out_channels=self.planes, kernel_size=(kernel_size),
                               stride=1, padding=0, bias=False,)
            self.parallel_conv.append(sep_conv)

        self.bn1 = nn.BatchNorm1d(num_features=self.planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv1 = nn.Conv1d(in_channels=self.planes, out_channels=self.planes, kernel_size=fixed_kernel_size,
                               stride=2, padding=2, bias=False)
        self.block = self._make_resnet_layer(kernel_size=fixed_kernel_size, stride=1, padding=fixed_kernel_size//2)
        self.bn2 = nn.BatchNorm1d(num_features=self.planes)
        self.avgpool = nn.AvgPool1d(kernel_size=6, stride=6, padding=2)
        self.rnn = nn.GRU(input_size=self.in_channels, hidden_size=128, num_layers=1, bidirectional=True)
        self.fc = nn.Linear(in_features=424, out_features=num_classes)

    def _make_resnet_layer(self, kernel_size, stride, blocks=9, padding=0):
        layers = []
        downsample = None
        base_width = self.planes

        for i in range(blocks):
            downsampling = nn.Sequential(
                    nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
                )
            layers.append(ResNet_1D_Block(in_channels=self.planes, out_channels=self.planes, kernel_size=kernel_size,
                                       stride=stride, padding=padding, downsampling=downsampling))

        return nn.Sequential(*layers)
    def extract_features(self, x):
        x = x.permute(0, 2, 1)
        out_sep = []

        for i in range(len(self.kernels)):
            sep = self.parallel_conv[i](x)
            out_sep.append(sep)

        out = torch.cat(out_sep, dim=2)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv1(out)  

        out = self.block(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.avgpool(out)  
        
        out = out.reshape(out.shape[0], -1)  
        rnn_out, _ = self.rnn(x.permute(0, 2, 1))
        new_rnn_h = rnn_out[:, -1, :]  

        new_out = torch.cat([out, new_rnn_h], dim=1) 
        return new_out
    
    def forward(self, x):
        new_out = self.extract_features(x)
        result = self.fc(new_out)  

        return result
    

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
        logger.info('Loading spectrograms eegs.npy')
        self.all_eegs = np.load(ROOT_PATH  + '/input/hms-hbac-data/eegs.npy',allow_pickle=True).item()
        # self.all_eegs = np.load('/kaggle/input/brain-eegs/eegs.npy',allow_pickle=True).item()
        

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
                self.all_eegs
            )
            train_loader = DataLoader(train_dataset, batch_size=CFG.BATCH_SIZE, shuffle=True, num_workers=2,pin_memory=True)

            valid_dataset = HMSDataset(
                self.train.iloc[valid_index],
                self.all_eegs
            )
            valid_loader = DataLoader(valid_dataset, batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=2,pin_memory=True)

            # モデルの構築
            model = HMSModel().to(torch.device(RCFG.DEVICE))
            optimizer = optim.AdamW(model.parameters(),lr=0.008)
            # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.EPOCHS, eta_min=1e-6)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **CFG.COSANNEAL_RES_PARAMS)
            # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2, 5], gamma=0.1)
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

                if val_loss < best_valid_loss:
                    best_oof = np.concatenate(oof).copy()
                    best_epoch = epoch
                    best_cv = cv
                    best_valid_loss = val_loss
                    self.info['fold_cv'][fold_id] = cv
                    if not RCFG.DEBUG:
                        torch.save(model.state_dict(), OUTPUT_PATH + f'/model/{RCFG.RUN_NAME}_fold{fold_id}_{CFG.MODEL_NAME}.pickle')


            if CFG.TWO_STAGE_EPOCHS > 0:
                
                # データローダーの作成
                train_dataset = HMSDataset(
                    self.train.iloc[train_index][self.train['total_evaluators'] >= 10],
                    self.all_eegs
                )
                train_loader = DataLoader(train_dataset, batch_size=CFG.BATCH_SIZE, shuffle=True, num_workers=2,pin_memory=True)

                del model
                gc.collect()
                torch.cuda.empty_cache()

                # 1st stageでベストのモデルをロード
                model = HMSModel()
                model_weight = OUTPUT_PATH + f'/model/{RCFG.RUN_NAME}_fold{fold_id}_{CFG.MODEL_NAME}.pickle'
                checkpoint = torch.load(model_weight)
                model.load_state_dict(checkpoint)
                model.to(torch.device(RCFG.DEVICE))

                optimizer = optim.AdamW(model.parameters(),lr=0.001)
                scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2, 4], gamma=0.1)
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

                    if val_loss < best_valid_loss:
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


    def inference(self,):
        
        logger.info('Start inference.')
        test_df = pd.read_csv(ROOT_PATH + '/input/hms-harmful-brain-activity-classification/test.csv')

        paths_eegs = glob(ROOT_PATH + '/input/hms-harmful-brain-activity-classification/test_eegs/*.parquet')
        logger.info(f'There are {len(paths_eegs)} EEG spectrograms')

        all_eegs = {}
        for file_path in tqdm(paths_eegs):
            eeg_id = int(file_path.split("/")[-1].split(".")[0])
            all_eegs[eeg_id] = eeg_from_parquet(file_path)

        test_dataset = HMSDataset(
            test_df, 
            all_eegs,
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
            model = HMSModel()
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

        return all_eegs
    
    def main(self):
        self.load_dataset()
        self.run_train()

        if RCFG.SAVE_TO_SHEET:
            self.write_sheet()
        
        if RCFG.PREDICT:
            self.inference()