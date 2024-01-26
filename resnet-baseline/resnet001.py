# https://www.kaggle.com/code/ttahara/hms-hbac-resnet34d-baseline-training/
# https://www.kaggle.com/code/ttahara/hms-hbac-resnet34d-baseline-inference

import sys
import os
import gc
import shutil
import datetime
import numpy as np
import pandas as pd
from time import time
import typing as tp
import pathlib
from pathlib import Path
from tqdm.notebook import tqdm
from sklearn.model_selection import StratifiedGroupKFold
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.cuda import amp
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from utils import set_random_seed, to_device
from utils import WriteSheet, Logger, class_vars_to_dict

logger = Logger()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

ROOT = Path.cwd().parent
INPUT = ROOT / "input"
OUTPUT = ROOT / "output"
SRC = ROOT / "src"

DATA = INPUT / "hms-harmful-brain-activity-classification"
TRAIN_SPEC = DATA / "train_spectrograms"
TEST_SPEC = DATA / "test_spectrograms"

TMP = ROOT / "tmp"
TRAIN_SPEC_SPLIT = TMP / "train_spectrograms_split"
TEST_SPEC_SPLIT = TMP / "test_spectrograms_split"
TMP.mkdir(exist_ok=True)
TRAIN_SPEC_SPLIT.mkdir(exist_ok=True)
TEST_SPEC_SPLIT.mkdir(exist_ok=True)
# TRAINED_MODEL = INPUT / "hms-hbac-resnet001"
TRAINED_MODEL = pathlib.Path().resolve()

RANDAM_SEED = 1086
CLASSES = ["seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"]
N_CLASSES = len(CLASSES)

FilePath = tp.Union[str, Path]
Label = tp.Union[int, float, np.ndarray]

class RCFG:
    debug = True
    debug_size = 100

class ENV:
    on_kaggle = True
    commit_hash = ""
    save_to_sheet = True
    sheet_json_key = '/kaggle/input/ktokunagautils/ktokunaga-4094cf694f5c.json'
    sheet_key = '1Wcg2EvlDgjo0nC-qbHma1LSEAY_OlS50mJ-yI4QI-yg'

class CFG:
    model_name = "resnet34d"
    img_size = 512
    max_epoch = 9
    batch_size = 32
    lr = 1.0e-03
    weight_decay = 1.0e-02
    es_patience =  5
    seed = 1086
    deterministic = True
    enable_amp = True
    device = "cuda"


class HMSHBACSpecModel(nn.Module):

    def __init__(
            self,
            model_name: str,
            pretrained: bool,
            in_channels: int,
            num_classes: int,
        ):
        super().__init__()
        self.model = timm.create_model(
            model_name=model_name, pretrained=pretrained,
            num_classes=num_classes, in_chans=in_channels
        )

    def forward(self, x):
        h = self.model(x)      

        return h
    
class HMSHBACSpecDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        image_paths: tp.Sequence[FilePath],
        labels: tp.Sequence[Label],
        transform: A.Compose,
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index: int):
        img_path = self.image_paths[index]
        label = self.labels[index]

        img = np.load(img_path)  # shape: (Hz, Time) = (400, 300)
        
        # log transform
        img = np.clip(img,np.exp(-4), np.exp(8))
        img = np.log(img)
        
        # normalize per image
        eps = 1e-6
        img_mean = img.mean(axis=(0, 1))
        img = img - img_mean
        img_std = img.std(axis=(0, 1))
        img = img / (img_std + eps)

        img = img[..., None] # shape: (Hz, Time) -> (Hz, Time, Channel)
        img = self._apply_transform(img)

        return {"data": img, "target": label}

    def _apply_transform(self, img: np.ndarray):
        """apply transform to image and mask"""
        transformed = self.transform(image=img)
        img = transformed["image"]
        return img
    

class KLDivLossWithLogits(nn.KLDivLoss):

    def __init__(self):
        super().__init__(reduction="batchmean")

    def forward(self, y, t):
        y = nn.functional.log_softmax(y,  dim=1)
        loss = super().forward(y, t)

        return loss


class KLDivLossWithLogitsForVal(nn.KLDivLoss):
    
    def __init__(self):
        """"""
        super().__init__(reduction="batchmean")
        self.log_prob_list  = []
        self.label_list = []

    def forward(self, y, t):
        y = nn.functional.log_softmax(y, dim=1)
        self.log_prob_list.append(y.numpy())
        self.label_list.append(t.numpy())
        
    def compute(self):
        log_prob = np.concatenate(self.log_prob_list, axis=0)
        label = np.concatenate(self.label_list, axis=0)
        final_metric = super().forward(
            torch.from_numpy(log_prob),
            torch.from_numpy(label)
        ).item()
        self.log_prob_list = []
        self.label_list = []
        
        return final_metric

def get_path_label(val_fold, train_all: pd.DataFrame):
    """Get file path and target info."""
    
    train_idx = train_all[train_all["fold"] != val_fold].index.values
    val_idx   = train_all[train_all["fold"] == val_fold].index.values
    img_paths = []
    labels = train_all[CLASSES].values
    for label_id in train_all["label_id"].values:
        img_path = TRAIN_SPEC_SPLIT / f"{label_id}.npy"
        img_paths.append(img_path)

    train_data = {
        "image_paths": [img_paths[idx] for idx in train_idx],
        "labels": [labels[idx].astype("float32") for idx in train_idx]}

    val_data = {
        "image_paths": [img_paths[idx] for idx in val_idx],
        "labels": [labels[idx].astype("float32") for idx in val_idx]}
    
    return train_data, val_data, train_idx, val_idx


def get_test_path_label(test: pd.DataFrame):
    """Get file path and dummy target info."""
    
    img_paths = []
    labels = np.full((len(test), 6), -1, dtype="float32")
    for spec_id in test["spectrogram_id"].values:
        img_path = TEST_SPEC_SPLIT / f"{spec_id}.npy"
        img_paths.append(img_path)
        
    test_data = {
        "image_paths": img_paths,
        "labels": [l for l in labels]}
    
    return test_data


def get_transforms(CFG):
    train_transform = A.Compose([
        A.Resize(p=1.0, height=CFG.img_size, width=CFG.img_size),
        ToTensorV2(p=1.0)
    ])
    val_transform = A.Compose([
        A.Resize(p=1.0, height=CFG.img_size, width=CFG.img_size),
        ToTensorV2(p=1.0)
    ])
    return train_transform, val_transform

def get_test_transforms(CFG):
    test_transform = A.Compose([
        A.Resize(p=1.0, height=CFG.img_size, width=CFG.img_size),
        ToTensorV2(p=1.0)
    ])
    return test_transform


def train_one_fold(CFG, val_fold, train_all, output_path):
    """Main"""
    torch.backends.cudnn.benchmark = True
    set_random_seed(CFG.seed, deterministic=CFG.deterministic)
    device = torch.device(CFG.device)
    
    train_path_label, val_path_label, _, _ = get_path_label(val_fold, train_all)
    train_transform, val_transform = get_transforms(CFG)
    
    train_dataset = HMSHBACSpecDataset(**train_path_label, transform=train_transform)
    val_dataset = HMSHBACSpecDataset(**val_path_label, transform=val_transform)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=CFG.batch_size, num_workers=4, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=CFG.batch_size, num_workers=4, shuffle=False, drop_last=False)
    
    model = HMSHBACSpecModel(
        model_name=CFG.model_name, pretrained=True, num_classes=6, in_channels=1)
    model.to(device)
    
    optimizer = optim.AdamW(params=model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    scheduler = lr_scheduler.OneCycleLR(
        optimizer=optimizer, epochs=CFG.max_epoch,
        pct_start=0.0, steps_per_epoch=len(train_loader),
        max_lr=CFG.lr, div_factor=25, final_div_factor=4.0e-01
    )
    
    loss_func = KLDivLossWithLogits()
    loss_func.to(device)
    loss_func_val = KLDivLossWithLogitsForVal()
    
    use_amp = CFG.enable_amp
    scaler = amp.GradScaler(enabled=use_amp)
    
    best_val_loss = 1.0e+09
    best_epoch = 0
    train_loss = 0
    
    for epoch in range(1, CFG.max_epoch + 1):
        epoch_start = time()
        model.train()
        for batch in train_loader:
            batch = to_device(batch, device)
            x, t = batch["data"], batch["target"]
                
            optimizer.zero_grad()
            with amp.autocast(use_amp):
                y = model(x)
                loss = loss_func(y, t)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
            
        model.eval()
        for batch in val_loader:
            x, t = batch["data"], batch["target"]
            x = to_device(x, device)
            with torch.no_grad(), amp.autocast(use_amp):
                y = model(x)
            y = y.detach().cpu().to(torch.float32)
            loss_func_val(y, t)
        val_loss = loss_func_val.compute()        
        if val_loss < best_val_loss:
            best_epoch = epoch
            best_val_loss = val_loss
            # logger.info("save model")
            torch.save(model.state_dict(), str(output_path / f'snapshot_epoch_{epoch}.pth'))
        
        elapsed_time = time() - epoch_start
        logger.info(
            f"[epoch {epoch}] train loss: {train_loss: .6f}, val loss: {val_loss: .6f}, elapsed_time: {elapsed_time: .3f}"
        )
        
        if epoch - best_epoch > CFG.es_patience:
            logger.info("Early Stopping!")
            break
            
        train_loss = 0
            
    return val_fold, best_epoch, best_val_loss


def run_inference_loop(model, loader, device):
    model.to(device)
    model.eval()
    pred_list = []
    with torch.no_grad():
        for batch in tqdm(loader):
            x = to_device(batch["data"], device)
            y = model(x)
            pred_list.append(y.softmax(dim=1).detach().cpu().numpy())
        
    pred_arr = np.concatenate(pred_list)
    del pred_list
    return pred_arr


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
            self.folds = [0, 1]
            self.n_folds = len(self.folds)
            CFG.max_epoch = 2

    def load_dataset(self, ):

        self.train = pd.read_csv(DATA / "train.csv")
        self.train[CLASSES] /= self.train[CLASSES].sum(axis=1).values[:, None] # convert vote to probability
        self.train = self.train.groupby("spectrogram_id").head(1).reset_index(drop=True)

        if RCFG.debug:
            logger.info(f'DEBUG MODE!!! sample size is reduced to {RCFG.debug_size}')
            self.train = self.train.sample(n=RCFG.debug_size, random_state=40).reset_index(drop=True)
        
        sgkf = StratifiedGroupKFold(n_splits=self.n_folds, shuffle=True, random_state=RANDAM_SEED)
        self.train["fold"] = -1

        for fold_id, (_, val_idx) in enumerate(
            sgkf.split(self.train, y=self.train["expert_consensus"], groups=self.train["patient_id"])
        ):
            self.train.loc[val_idx, "fold"] = fold_id

        logger.info('Get and split spectrogram files.')
        for spec_id, df in tqdm(self.train.groupby("spectrogram_id")):
            spec = pd.read_parquet(TRAIN_SPEC / f"{spec_id}.parquet")
            
            spec_arr = spec.fillna(0).values[:, 1:].T.astype("float32")  # (Hz, Time) = (400, 300)
            
            for spec_offset, label_id in df[
                ["spectrogram_label_offset_seconds", "label_id"]
            ].astype(int).values:
                spec_offset = spec_offset // 2
                split_spec_arr = spec_arr[:, spec_offset: spec_offset + 300]
                np.save(TRAIN_SPEC_SPLIT / f"{label_id}.npy" , split_spec_arr)


    def run_train(self, ):

        self.score_list = []
        for fold_id in self.folds:
            output_path = Path(f"fold{fold_id}")
            output_path.mkdir(exist_ok=True)
            logger.info(f"[fold{fold_id}]---------------------------------------------")
            self.score_list.append(train_one_fold(CFG, fold_id, self.train, output_path))

        # logger.info(score_list)


    def infer_oof(self, ):
    
        logger.info('Execute infer_oof.')
        for (fold_id, best_epoch, _) in self.score_list:
            
            exp_dir_path = Path(f"fold{fold_id}")
            best_model_path = exp_dir_path / f"snapshot_epoch_{best_epoch}.pth"
            copy_to = f"./best_model_fold{fold_id}.pth"
            shutil.copy(best_model_path, copy_to)
            
            for p in exp_dir_path.glob("*.pth"):
                p.unlink()

        self.oof_pred_arr = np.zeros((len(self.train), N_CLASSES))

        for fold_id in range(self.n_folds):
            logger.info(f"\n[fold {fold_id}]-------------------------------------------")
            device = torch.device(CFG.device)

            # # get_dataloader
            _, val_path_label, _, val_idx = get_path_label(fold_id, self.train)
            _, val_transform = get_transforms(CFG)
            val_dataset = HMSHBACSpecDataset(**val_path_label, transform=val_transform)
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=CFG.batch_size, num_workers=4, shuffle=False, drop_last=False)
            
            # # get model
            model_path = f"./best_model_fold{fold_id}.pth"
            model = HMSHBACSpecModel(
                model_name=CFG.model_name, pretrained=False, num_classes=6, in_channels=1)
            model.load_state_dict(torch.load(model_path, map_location=device))
            
            # # inference
            val_pred = run_inference_loop(model, val_loader, device)
            self.oof_pred_arr[val_idx] = val_pred
            
            del val_idx, val_path_label
            del model, val_loader
            torch.cuda.empty_cache()
            gc.collect()

    
    def get_cv_score(self, ):

        logger.info('Execute get_cv_score.')
        import sys
        sys.path.append('/kaggle/input/kaggle-kl-div')
        from kaggle_kl_div import score

        true = self.train[["label_id"] + CLASSES].copy()
        oof = pd.DataFrame(self.oof_pred_arr, columns=CLASSES)
        oof.insert(0, "label_id", self.train["label_id"])

        cv_score = score(solution=true, submission=oof, row_id_column_name='label_id')
        self.info['cv_score'] = cv_score
        logger.info(f'CV Score KL-Div for ResNet34d: {cv_score}')


    def get_prediction(self, ):

        logger.info('Execute get_prediction.')
        self.test = pd.read_csv(DATA / "test.csv")
        for spec_id in self.test["spectrogram_id"]:
            spec = pd.read_parquet(TEST_SPEC / f"{spec_id}.parquet")
            spec_arr = spec.fillna(0).values[:, 1:].T.astype("float32")  # (Hz, Time) = (400, 300)
            np.save(TEST_SPEC_SPLIT / f"{spec_id}.npy", spec_arr)

        self.test_preds_arr = np.zeros((self.n_folds, len(self.test), N_CLASSES))

        test_path_label = get_test_path_label(self.test)
        test_transform = get_test_transforms(CFG)
        test_dataset = HMSHBACSpecDataset(**test_path_label, transform=test_transform)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=CFG.batch_size, num_workers=4, shuffle=False, drop_last=False)

        device = torch.device(CFG.device)

        for fold_id in range(self.n_folds):
            print(f"\n[fold {fold_id}]-------------------------------------------")
            
            # # get model
            model_path = TRAINED_MODEL / f"best_model_fold{fold_id}.pth"
            model = HMSHBACSpecModel(
                model_name=CFG.model_name, pretrained=False, num_classes=6, in_channels=1)
            model.load_state_dict(torch.load(model_path, map_location=device))
            
            # # inference©
            test_pred = run_inference_loop(model, test_loader, device)
            self.test_preds_arr[fold_id] = test_pred
            
            del model
            torch.cuda.empty_cache()
            gc.collect()


    def create_submission(self, ):

        logger.info('Create submission file.')
        test_pred = self.test_preds_arr.mean(axis=0)
        test_pred_df = pd.DataFrame(
            test_pred, columns=CLASSES
        )
        test_pred_df = pd.concat([self.test[["eeg_id"]], test_pred_df], axis=1)
        smpl_sub = pd.read_csv(DATA / "sample_submission.csv")

        sub = pd.merge(
            smpl_sub[["eeg_id"]], test_pred_df, on="eeg_id", how="left")

        sub.to_csv("submission.csv", index=False)
        sub.head()


    def write_sheet(self, ):
        logger.info('Write info to google sheet.')
        write_dt_jst = str(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).strftime('%Y-%m-%d %H:%M:%S'))

        data = [
            self.info['start_dt_jst'],
            write_dt_jst,
            ENV.commit_hash,
            class_vars_to_dict(RCFG),
            class_vars_to_dict(CFG),
            self.info['cv_score']
        ]
    
        self.sheet.write(data, sheet_name='cv_scores')