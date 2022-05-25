
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

class LitModule(pl.LightningModule):
    LOSS_FNS = {
        "bce": smp.losses.SoftBCEWithLogitsLoss(),
        "dice": smp.losses.DiceLoss(mode="multilabel"),
        "focal": smp.losses.FocalLoss(mode="multilabel"),
        "jaccard": smp.losses.JaccardLoss(mode="multilabel"),
        "lovasz": smp.losses.LovaszLoss(mode="multilabel"),
        "tversky": smp.losses.TverskyLoss(mode="multilabel"),
    }

    def __init__(
        self,
        arch: str,
        encoder_name: str,
        encoder_weights: str,
        loss: str,
        optimizer: str,
        learning_rate: float,
        weight_decay: float,
        scheduler: Optional[str],
        T_max: int,
        T_0: int,
        min_lr: int,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.model = self._init_model()

        self.loss_fn = self._init_loss_fn()

        self.metrics = self._init_metrics()

    def _init_model(self):
        return smp.create_model(
            self.hparams.arch,
            encoder_name=self.hparams.encoder_name,
            encoder_weights=self.hparams.encoder_weights,
            in_channels=3,
            classes=3,
            activation=None,
        )

    def _init_loss_fn(self):
        losses = self.hparams.loss.split("_")
        loss_fns = [self.LOSS_FNS[loss] for loss in losses]

        def criterion(y_pred, y_true):
            return sum(loss_fn(y_pred, y_true) for loss_fn in loss_fns) / len(loss_fns)

        return criterion

    def _init_metrics(self):
        val_metrics = MetricCollection({"val_dice": DiceMetric(), "val_iou": IOUMetric()})
        test_metrics = MetricCollection({"test_comp_metric": CompetitionMetric()})

        return torch.nn.ModuleDict(
            {
                "val_metrics": val_metrics,
                "test_metrics": test_metrics,
            }
        )

    def configure_optimizers(self):
        optimizer_kwargs = dict(
            params=self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay
        )
        if self.hparams.optimizer == "Adadelta":
            optimizer = torch.optim.Adadelta(**optimizer_kwargs)
        elif self.hparams.optimizer == "Adagrad":
            optimizer = torch.optim.Adagrad(**optimizer_kwargs)
        elif self.hparams.optimizer == "Adam":
            optimizer = torch.optim.Adam(**optimizer_kwargs)
        elif self.hparams.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(**optimizer_kwargs)
        elif self.hparams.optimizer == "Adamax":
            optimizer = torch.optim.Adamax(**optimizer_kwargs)
        elif self.hparams.optimizer == "SGD":
            optimizer = torch.optim.SGD(**optimizer_kwargs)
        else:
            raise ValueError(f"Unknown optimizer: {self.hparams.optimizer}")

        if self.hparams.scheduler is not None:
            if self.hparams.scheduler == "CosineAnnealingLR":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=self.hparams.T_max, eta_min=self.hparams.min_lr
                )
            elif self.hparams.scheduler == "CosineAnnealingWarmRestarts":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer, T_0=self.hparams.T_0, eta_min=self.hparams.min_lr
                )
            elif self.hparams.scheduler == "ExponentialLR":
                scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
            elif self.hparams.scheduler == "ReduceLROnPlateau":
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
            else:
                raise ValueError(f"Unknown scheduler: {self.hparams.scheduler}")

            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
        else:
            return {"optimizer": optimizer}

    def forward(self, images):
        return self.model(images)
    
    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self.shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self.shared_step(batch, "test")

    def shared_step(self, batch, stage, log=True):
        images, masks = batch
        y_pred = self(images)

        loss = self.loss_fn(y_pred, masks)

        if stage != "train":
            metrics = self.metrics[f"{stage}_metrics"](y_pred, masks)
        else:
            metrics = None

        if log:
            self._log(loss, metrics, stage)

        return loss

    def _log(self, loss, metrics, stage):
        on_step = True if stage == "train" else False

        self.log(f"{stage}_loss", loss, on_step=on_step, on_epoch=True, prog_bar=True)

        if metrics is not None:
            self.log_dict(metrics, on_step=on_step, on_epoch=True)

    @classmethod
    def load_eval_checkpoint(cls, checkpoint_path, device):
        module = cls.load_from_checkpoint(checkpoint_path=checkpoint_path).to(device)
        module.eval()

        return module