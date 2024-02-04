# https://www.kaggle.com/code/nischaydnk/lightning-1d-eegnet-training-pipeline-hbs/notebook
# https://www.kaggle.com/code/nischaydnk/hms-submission-1d-eegnet-pipeline-lightning

import numpy as np
import pandas as pd
import os
import torch
import datetime
import gc
from tqdm import tqdm
from pathlib import Path
from scipy.signal import butter, lfilter
from sklearn.model_selection import GroupKFold
import pytorch_lightning as pl
import torch
import torch.nn as nn
import albumentations as A
import torch_audiomentations as tA
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, BackboneFinetuning, EarlyStopping
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from torchtoolbox.tools import mixup_data, mixup_criterion

import sys
sys.path.append('/kaggle/input/kaggle-kl-div')
from kaggle_kl_div import score

from utils import set_random_seed, to_device
from utils import WriteSheet, Logger, class_vars_to_dict

torch.set_float32_matmul_precision('high')
tqdm.pandas()

logger = Logger()
TARGETS = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']
TARS = {'Seizure':0, 'LPD':1, 'GPD':2, 'LRDA':3, 'GRDA':4, 'Other':5}
TARS_INV = {x:y for y,x in TARS.items()}

ROOT = Path.cwd().parent
TMP = ROOT / "tmp"
EEG_SPLIT = TMP / "eeg_split"

TMP.mkdir(exist_ok=True)
EEG_SPLIT.mkdir(exist_ok=True)


class RCFG:
    debug = False
    debug_size = 1000

class ENV:
    on_kaggle = True
    commit_hash = ""
    save_to_sheet = True
    sheet_json_key = '/kaggle/input/ktokunagautils/ktokunaga-4094cf694f5c.json'
    sheet_key = '1Wcg2EvlDgjo0nC-qbHma1LSEAY_OlS50mJ-yI4QI-yg'
    data_root = "/kaggle/input/"
    output_dir = '.'
    model_dir = "/kaggle/input/hms-hbac-1dcnn"

class Config:
    batch_size = 64 # 88
    test_batch_size = 32
    epochs = 20 # 20
    PRECISION = 16    
    PATIENCE = 20    
    seed = 2024
    weight_decay = 1e-2
    use_mixup = False
    mixup_alpha = 0.1   
    num_channels = 20
    LR = 8e-3
    trn_folds = [0,1,2,3,4] #[0,1,2,3,4]
    num_classes = len(TARS.keys())
    

def eeg_from_parquet(parquet_path, feats):
    
    # EXTRACT MIDDLE 50 SECONDS
    eeg = pd.read_parquet(parquet_path, columns=feats)
    rows = len(eeg)
    offset = (rows-10_000)//2
    eeg = eeg.iloc[offset:offset+10_000]
    
    # CONVERT TO NUMPY
    data = np.zeros((10_000,len(feats)))
    for j,col in enumerate(feats):
        
        # FILL NAN
        x = eeg[col].values.astype('float32')
        m = np.nanmean(x)
        if np.isnan(x).mean()<1: x = np.nan_to_num(x,nan=m)
        else: x[:] = 0
            
        data[:,j] = x
        
    return data


def get_transforms(*, data):

    if data == 'train':
        return tA.Compose(
                transforms=[
                     # tA.ShuffleChannels(p=0.25,mode="per_channel",p_mode="per_channel",),
                     tA.AddColoredNoise(p=0.15,mode="per_channel",p_mode="per_channel", max_snr_in_db = 15, sample_rate=200),
                ])

    elif data == 'valid':
        return tA.Compose([])


def quantize_data(data, classes):
    mu_x = mu_law_encoding(data, classes)
    # bins = np.linspace(-1, 1, classes)
    # quantized = np.digitize(mu_x, bins) - 1
    return mu_x #quantized

def mu_law_encoding(data, mu):
    mu_x = np.sign(data) * np.log(1 + mu * np.abs(data)) / np.log(mu + 1)
    return mu_x

def mu_law_expansion(data, mu):
    s = np.sign(data) * (np.exp(np.abs(data) * np.log(mu + 1)) - 1) / mu
    return s

def butter_lowpass_filter(data, cutoff_freq=20, sampling_rate=200, order=4):
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = lfilter(b, a, data, axis=0)
    return filtered_data

def get_optimizer(lr, params):
    model_optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, params), 
            lr=lr,
            weight_decay=Config.weight_decay
        )
    interval = "epoch"
    
    lr_scheduler = CosineAnnealingWarmRestarts(
                            model_optimizer, 
                            T_0=Config.epochs, 
                            T_mult=1, 
                            eta_min=1e-6, 
                            last_epoch=-1
                        )

    return {
        "optimizer": model_optimizer, 
        "lr_scheduler": {
            "scheduler": lr_scheduler,
            "interval": interval,
            "monitor": "val_loss",
            "frequency": 1
        }
    }


class EEGDataset(torch.utils.data.Dataset):

    def __init__(self, data, augmentations = None, test = False): 

        self.data = data
        self.augmentations = augmentations
        self.test = test
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):

        row = self.data.iloc[index]      
        eeg_path = EEG_SPLIT / f'eegs_{row.eeg_id}.npy'
        data = np.load(eeg_path)

        data = np.clip(data,-1024,1024)
        data = np.nan_to_num(data, nan=0) / 32.0
        
        data = butter_lowpass_filter(data)
        data = quantize_data(data,1)

        samples = torch.from_numpy(data).float()
        if self.augmentations:
            samples = self.augmentations(samples.unsqueeze(0), None)
        samples = samples.squeeze()

        samples = samples.permute(1,0)
        if not self.test:
            label = row[TARGETS] 
            label = torch.tensor(label).float()  
            return samples, label
        else:
            return samples


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


class EEGNet(nn.Module):

    def __init__(self, kernels, in_channels=20, fixed_kernel_size=17, num_classes=6):
        super(EEGNet, self).__init__()
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
        self.rnn1 = nn.GRU(input_size=156, hidden_size=156, num_layers=1, bidirectional=True)

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

    def forward(self, x):
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

        rnn_out, _ = self.rnn(x.permute(0,2, 1))
        new_rnn_h = rnn_out[:, -1, :]  

        new_out = torch.cat([out, new_rnn_h], dim=1)  
        result = self.fc(new_out)  

        return result
    
class KLDivLossWithLogits(nn.KLDivLoss):

    def __init__(self):
        super().__init__(reduction="batchmean")

    def forward(self, y, t):
        y = nn.functional.log_softmax(y,  dim=1)
        loss = super().forward(y, t)

        return loss
    

class EEGModel(pl.LightningModule):
    def __init__(self, num_classes = Config.num_classes, fold = 0):
        super().__init__()
        self.num_classes = num_classes
        self.fold = fold
        self.backbone = EEGNet(kernels=[3,5,7,9], in_channels=Config.num_channels, fixed_kernel_size=5, num_classes=Config.num_classes)
        self.loss_function = KLDivLossWithLogits() #nn.KLDivLoss() #nn.BCEWithLogitsLoss() 
        self.validation_step_outputs = []
        self.training_step_outputs = []
        self.lin = nn.Softmax(dim=1)

    def forward(self,images):
        logits = self.backbone(images)
        # logits = self.lin(logits)
        return logits
        
    def configure_optimizers(self):
        return get_optimizer(lr=Config.LR, params=self.parameters())

    def train_with_mixup(self, X, y):
        X, y_a, y_b, lam = mixup_data(X, y, alpha=Config.mixup_alpha)
        y_pred = self(X)
        loss_mixup = mixup_criterion(KLDivLossWithLogits(), y_pred, y_a, y_b, lam)
        return loss_mixup

    def training_step(self, batch, batch_idx):
        image, target = batch        
        if Config.use_mixup:
            loss = self.train_with_mixup(image, target)
        else:
            y_pred = self(image)
            loss = self.loss_function(y_pred,target)

        self.training_step_outputs.append({"loss": loss})
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss        

    def validation_step(self, batch, batch_idx):
        image, target = batch 
        y_pred = self(image)
        val_loss = self.loss_function(y_pred, target)
        self.log("val_loss", val_loss, on_step=True, on_epoch=True, logger=True, prog_bar=True)
        self.validation_step_outputs.append({"val_loss": val_loss, "logits": y_pred, "targets": target})

        return {"val_loss": val_loss, "logits": y_pred, "targets": target}
    
    def train_dataloader(self):
        return self._train_dataloader 
    
    def on_train_epoch_end(self,):
        outputs = self.training_step_outputs
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        logger.info(f'Fold {self.fold}: Epoch {self.current_epoch} training loss {avg_loss}')
        return {"loss": avg_loss}

    def validation_dataloader(self):
        return self._validation_dataloader
    
    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        output_val = nn.Softmax(dim=1)(torch.cat([x['logits'] for x in outputs],dim=0)).cpu().detach().numpy()
        target_val = torch.cat([x['targets'] for x in outputs],dim=0).cpu().detach().numpy()
        self.validation_step_outputs = []

        val_df = pd.DataFrame(target_val, columns = list(TARGETS))
        pred_df = pd.DataFrame(output_val, columns = list(TARGETS))

        val_df['id'] = [f'id_{i}' for i in range(len(val_df))] 
        pred_df['id'] = [f'id_{i}' for i in range(len(pred_df))] 

        avg_score = score(val_df, pred_df, row_id_column_name = 'id')

        logger.info(f'Fold {self.fold}: Epoch {self.current_epoch} validation loss {avg_loss}')
        logger.info(f'Fold {self.fold}: Epoch {self.current_epoch} validation KDL score {avg_score}')
        
        return {'val_loss': avg_loss,'val_cmap':avg_score}
    

def get_fold_dls(df_train, df_valid):

    ds_train = EEGDataset(
        df_train, 
        augmentations = get_transforms(data='valid'),
        test = False
    )
    
    ds_val = EEGDataset(
        df_valid, 
        augmentations = get_transforms(data='valid'),
        test = False
    )
    dl_train = DataLoader(ds_train, batch_size=Config.batch_size , shuffle=True, num_workers = 2)    
    dl_val = DataLoader(ds_val, batch_size=Config.batch_size, num_workers = 2)
    return dl_train, dl_val, ds_train, ds_val

def get_test_dls(df_test):
    ds_test = EEGDataset(
        df_test, 
        augmentations = None,
        test = True
    )
    dl_test = DataLoader(ds_test, batch_size=Config.test_batch_size , shuffle=False, num_workers = 2)    
    return dl_test, ds_test

def predict(data_loader, model, mode='train'):
        
    model.to('cuda')
    model.eval()    
    predictions = []
    for batch in tqdm(data_loader):

        with torch.no_grad():
            if mode == 'train':
                x, y = batch
            else:
                x = batch
            x = x.cuda()
            # inputs = {key:val.reshape(val.shape[0], -1).to(config.device) for key,val in batch.items()}
            outputs = model(x)
            outputs = nn.Softmax(dim=1)(outputs)
        predictions.extend(outputs.detach().cpu().numpy())
    predictions = np.vstack(predictions)
    return predictions


def run_training(train, fold_id, Config):
    logger.info(f"Running training for fold {fold_id}...")
    pred_cols = [f'pred_{t}' for t in TARGETS]
    
    df_train = train[train['fold']!=fold_id].copy()
    df_valid = train[train['fold']==fold_id].copy()
    
    dl_train, dl_val, ds_train, ds_val = get_fold_dls(df_train, df_valid)
    
    eeg_model = EEGModel(num_classes = Config.num_classes, fold = fold_id)
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=Config.PATIENCE, verbose= True, mode="min")
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath= f"{ENV.output_dir}/",
        save_top_k=1,
        save_last= True,
        save_weights_only=False,
        filename= f'eegnet_best_loss_fold{fold_id}',
        verbose= True,
        mode='min'
    )    
    callbacks_to_use = [checkpoint_callback,early_stop_callback]

    trainer = pl.Trainer(
        devices=[0],
        val_check_interval=1.0,
        deterministic=True,
        max_epochs=Config.epochs,        
        logger=None,
        callbacks=callbacks_to_use,
        precision=Config.PRECISION*2,
        accelerator="gpu" 
    )

    logger.info("Running trainer.fit")
    trainer.fit(eeg_model, train_dataloaders = dl_train, val_dataloaders = dl_val)                

    model = EEGModel.load_from_checkpoint(f'{ENV.output_dir}/eegnet_best_loss_fold{fold_id}.ckpt',train_dataloader=None,validation_dataloader=None,config=Config)    
    preds = predict(dl_val, model, mode='train')  
    df_valid[pred_cols] = preds
    df_valid.to_csv(f'{ENV.output_dir}/pred_df_f{fold_id}.csv',index=False)
    gc.collect()
    # torch.cuda.empty_cache()
    return preds


def run_inference(test, fold_id, Config):
    logger.info(f"Running training for fold {fold_id}...")
    df_test = test.copy()
    dl_test, ds_test = get_test_dls(df_test)
    
    logger.info(f"Running inference model '{ENV.model_dir}/eegnet_best_loss_fold{fold_id}.ckpt'..")
    model = EEGModel.load_from_checkpoint(
        f'{ENV.model_dir}/eegnet_best_loss_fold{fold_id}.ckpt',
        map_location='cuda:0',
        train_dataloader=None,
        validation_dataloader=None,
        config=Config
    )
    
    preds = predict(dl_test, model, mode='test')  
    logger.info(preds.shape)
    gc.collect()
    # torch.cuda.empty_cache()
    return preds


######################################################
# Runner
######################################################


class Runner():
    def __init__(self,):

        logger.info('Initializing Runner.')
        start_dt_jst = str(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).strftime('%Y-%m-%d %H:%M:%S'))
        self.info =  {"start_dt_jst": start_dt_jst}
        
        if ENV.on_kaggle:
            from kaggle_secrets import UserSecretsClient
            self.user_secrets = UserSecretsClient()

        if ENV.save_to_sheet:
            logger.info('Initializing Google Sheet.')
            self.sheet = WriteSheet(
                sheet_json_key = ENV.sheet_json_key,
                sheet_key = ENV.sheet_key
            )

        self.folds = [0, 1, 2, 3, 4]
        self.n_folds = len(self.folds)
        if RCFG.debug:
            logger.info(f'DEBUG MODE!!! sample size is reduced to {RCFG.debug_size}')
            logger.info(f'FOLDS will be [0, 1]')
            Config.epochs = 3
            Config.trn_folds = [0,1] #[0,1,2,3,4]

        pl.seed_everything(Config.seed, workers=True)
        if not os.path.exists(ENV.output_dir):
            os.makedirs(ENV.output_dir)


    def _create_eeg(self, mode='train' ):

        df = pd.read_parquet(f'{ENV.data_root}hms-harmful-brain-activity-classification/train_eegs/1000913311.parquet')
        feats = df.columns
        if mode == 'train':
            self.eeg_ids = self.train.eeg_id.unique()
            PATH = f'{ENV.data_root}hms-harmful-brain-activity-classification/train_eegs/'
        else:
            self.eeg_ids = self.test.eeg_id.unique()
            PATH = f'{ENV.data_root}hms-harmful-brain-activity-classification/test_eegs/'

        for i,eeg_id in enumerate(self.eeg_ids):
            if (i%100==0)&(i!=0): print(i,', ',end='') 
            
            # SAVE EEG TO PYTHON DICTIONARY OF NUMPY ARRAYS
            data = eeg_from_parquet(f'{PATH}{eeg_id}.parquet', feats = feats)              
            np.save(EEG_SPLIT / f'eegs_{eeg_id}.npy', data)


    def load_dataset(self, ):

        df = pd.read_csv(f'{ENV.data_root}hms-harmful-brain-activity-classification/train.csv')
        self.eeg_ids = df.eeg_id.unique()
        self.train = df.groupby('eeg_id')[['patient_id']].agg('first')

        tmp = df.groupby('eeg_id')[TARGETS].agg('sum')
        for t in TARGETS:
            self.train[t] = tmp[t].values
            
        y_data = self.train[TARGETS].values
        y_data = y_data / y_data.sum(axis=1,keepdims=True)
        self.train[TARGETS] = y_data

        tmp = df.groupby('eeg_id')[['expert_consensus']].agg('first')
        self.train['target'] = tmp

        self.train = self.train.reset_index()
        self.train = self.train.loc[self.train.eeg_id.isin(self.eeg_ids)]
        logger.info(f'Train Data with unique eeg_id shape: {self.train.shape}')

        if RCFG.debug:
            logger.info(f'DEBUG MODE!!! sample size is reduced to {RCFG.debug_size}')
            self.train = self.train.sample(RCFG.debug_size).reset_index()

        self._create_eeg(mode='train')


    def run_train(self, ):
        
        iot = torch.randn(2, Config.num_channels, 10000)#.cuda()
        model = EEGNet(kernels=[3,5,7,9], in_channels=Config.num_channels, fixed_kernel_size=5, num_classes=6)#.cuda()
        output = model(iot)

        del iot, model
        gc.collect()

        gkf = GroupKFold(n_splits=len(Config.trn_folds))
        self.train['fold'] = 0
        for fold, (tr_idx, val_idx) in enumerate(gkf.split(self.train, self.train.target, self.train.patient_id)):   
            self.train.loc[val_idx, 'fold'] = fold

        oof_df = self.train.copy()
        pred_cols = [f'pred_{t}' for t in TARGETS]
        oof_df[pred_cols] = 0.0
        for f in Config.trn_folds:
            val_idx = list(self.train[self.train['fold']==f].index)
            val_preds = run_training(self.train, f, Config)    
            # val_df = pd.read_csv(f'{ENV.output_dir}/val_df_f{f}.csv')
            # pred_df = pd.read_csv(f'{ENV.output_dir}/pred_df_f{f}.csv')
            oof_df.loc[val_idx, pred_cols] = val_preds

        oof_pred_df= oof_df[['eeg_id'] + list(['pred_'+i for i in TARGETS])]
        oof_pred_df.columns = ['eeg_id'] + list(TARGETS)

        oof_true_df = oof_df[oof_pred_df.columns].copy()

        oof_score = score(solution=oof_true_df, submission=oof_pred_df, row_id_column_name='eeg_id')
        self.info['cv_score'] = oof_score
        logger.info(f'OOF Score for solution = {oof_score}')

        oof_df.to_csv(f'{ENV.output_dir}/oof.csv',index=False)

        val_idx = list(self.train[self.train['fold']==0].index)
        oof_df.loc[val_idx, TARGETS]    


    def write_sheet(self, ):
        logger.info('Write info to google sheet.')
        write_dt_jst = str(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).strftime('%Y-%m-%d %H:%M:%S'))

        data = [
            self.info['start_dt_jst'],
            write_dt_jst,
            ENV.commit_hash,
            class_vars_to_dict(RCFG),
            class_vars_to_dict(Config),
            self.info['cv_score']
        ]
    
        self.sheet.write(data, sheet_name='cv_scores')


    def get_prediction(self, ):
        
        self.test = pd.read_csv('/kaggle/input/hms-harmful-brain-activity-classification/test.csv')
        self._create_eeg(mode='test')

        self.sub_df = self.test[['eeg_id']].copy()
        self.sub_df[TARGETS] = 0.0

        test_preds = []
        for en,f in enumerate(Config.trn_folds):
            preds = run_inference(self.test, f, Config)
            test_preds.append(preds)

        test_preds = np.mean(test_preds, 0)
        self.sub_df[TARGETS] = test_preds
        self.sub_df.to_csv('submission.csv',index=False)
        