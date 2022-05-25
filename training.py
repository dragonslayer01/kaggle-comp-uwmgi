
from pathlib import Path
from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple

import albumentations as A
import cupy as cp
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import segmentation_models_pytorch as smp
import torch
from albumentations.pytorch import ToTensorV2
from monai.metrics.utils import get_mask_edges
from monai.metrics.utils import get_surface_distance
from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics import Metric
from torchmetrics import MetricCollection
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from data_load import *
from config import *
from metrics import *
from train import *

def train(
    random_seed: int = RANDOM_SEED,
    train_csv_path: str = str(INPUT_DATA_NPY_DIR / "train_preprocessed.csv"),
    img_size: int = IMG_SIZE,
    use_augs: bool = USE_AUGS,
    val_fold: int = VAL_FOLD,
    load_images: bool = LOAD_IMAGES,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    arch: str = ARCH,
    encoder_name: str = ENCODER_NAME,
    encoder_weights: str = ENCODER_WEIGHTS,
    loss: str = LOSS,
    optimizer: str = OPTIMIZER,
    learning_rate: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    scheduler: str = SCHEDULER,
    min_lr: float = MIN_LR,
    gpus: int = GPUS,
    fast_dev_run: bool = FAST_DEV_RUN,
    max_epochs: int = MAX_EPOCHS,
    precision: int = PRECISION,
    debug: bool = DEBUG,
):
    pl.seed_everything(random_seed)

    if debug:
        num_workers = 0
        max_epochs = 2

    data_module = LitDataModule(
        train_csv_path=train_csv_path,
        test_csv_path=None,
        img_size=img_size,
        use_augs=use_augs,
        val_fold=val_fold,
        load_images=load_images,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    module = LitModule(
        arch=arch,
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        loss=loss,
        optimizer=optimizer,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        scheduler=scheduler,
        T_max=int(30_000 / batch_size * max_epochs) + 50,
        T_0=25,
        min_lr=min_lr,
    )

    trainer = pl.Trainer(
        fast_dev_run=fast_dev_run,
        gpus=gpus,
        limit_train_batches=0.02 if debug else 1.0,
        limit_val_batches=0.02 if debug else 1.0,
        limit_test_batches=0.02 if debug else 0.5, # Metric computation takes too much memory otherwise
        logger=pl.loggers.CSVLogger(save_dir='logs/'),
        log_every_n_steps=10,
        max_epochs=max_epochs,
        precision=precision,
    )

    trainer.fit(module, datamodule=data_module)
    
    
    if not fast_dev_run:
        trainer.test(module, datamodule=data_module)
    
    return trainer