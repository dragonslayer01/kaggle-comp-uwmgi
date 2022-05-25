
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



class DiceMetric(Metric):
    def __init__(self, thr=0.5, dim=(2, 3), epsilon=0.001):
        super().__init__(compute_on_cpu=True)

        self.thr = thr
        self.dim = dim
        self.epsilon = epsilon

        self.add_state("dice", default=[])

    def update(self, y_pred, y_true):
        self.dice.append(dice_metric_update(y_pred, y_true, self.thr, self.dim, self.epsilon))

    def compute(self):
        if len(self.dice) == 1:
            return self.dice[0]
        else:
            return torch.mean(torch.stack(self.dice))


def dice_metric_update(y_pred, y_true, thr=0.5, dim=(2, 3), epsilon=0.001):
    y_pred = torch.nn.Sigmoid()(y_pred)
    y_pred = (y_pred > thr).detach().to(torch.float32)

    y_true = y_true.detach().to(torch.float32)

    inter = (y_true * y_pred).sum(dim=dim)
    den = y_true.sum(dim=dim) + y_pred.sum(dim=dim)

    dice = ((2 * inter + epsilon) / (den + epsilon)).mean(dim=(1, 0))

    return dice


class IOUMetric(Metric):
    def __init__(self, thr=0.5, dim=(2, 3), epsilon=0.001):
        super().__init__(compute_on_cpu=True)

        self.thr = thr
        self.dim = dim
        self.epsilon = epsilon

        self.add_state("iou", default=[])

    def update(self, y_pred, y_true):
        self.iou.append(iou_metric_update(y_pred, y_true, self.thr, self.dim, self.epsilon))

    def compute(self):
        if len(self.iou) == 1:
            return self.iou[0]
        else:
            return torch.mean(torch.stack(self.iou))


def iou_metric_update(y_pred, y_true, thr=0.5, dim=(2, 3), epsilon=0.001):
    y_pred = torch.nn.Sigmoid()(y_pred)
    y_pred = (y_pred > thr).detach().to(torch.float32)

    y_true = y_true.detach().to(torch.float32)

    inter = (y_true * y_pred).sum(dim=dim)
    union = (y_true + y_pred - y_true * y_pred).sum(dim=dim)

    iou = ((inter + epsilon) / (union + epsilon)).mean(dim=(1, 0))

    return iou


class CompetitionMetric(Metric):
    def __init__(self, thr=0.5):
        super().__init__(compute_on_step=False)

        self.thr = thr

        self.add_state("y_pred", default=[])
        self.add_state("y_true", default=[])

    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        y_pred = torch.nn.Sigmoid()(y_pred)
        y_pred = (y_pred > self.thr).to("cpu").detach().to(torch.float32)

        y_true = y_true.to("cpu").detach().to(torch.float32)

        self.y_pred.append(y_pred)
        self.y_true.append(y_true)

    def compute(self):
        y_pred = torch.cat(self.y_pred).numpy()
        y_true = torch.cat(self.y_true).numpy()

        return compute_competition_metric(y_pred, y_true)[0]


def compute_competition_metric(preds: np.ndarray, targets: np.ndarray) -> float:
    dice_ = compute_dice(preds, targets)
    hd_dist_ = compute_hd_dist(preds, targets)
    return 0.4 * dice_ + 0.6 * hd_dist_, dice_, hd_dist_


# Slightly adapted from https://www.kaggle.com/code/carnozhao?scriptVersionId=93589877&cellId=2
def compute_dice(preds: np.ndarray, targets: np.ndarray) -> float:
    preds = preds.astype(np.uint8)
    targets = targets.astype(np.uint8)

    I = (targets & preds).sum((2, 3))  # noqa: E741
    U = (targets | preds).sum((2, 3))  # noqa: E741

    return np.mean((2 * I / (U + I + 1) + (U == 0)).mean(1))


def compute_hd_dist(preds: np.ndarray, targets: np.ndarray) -> float:
    return 1 - np.mean([hd_dist_batch(preds[:, i, ...], targets[:, i, ...]) for i in range(3)])


def hd_dist_batch(preds: np.ndarray, targets: np.ndarray) -> float:
    return np.mean([hd_dist(pred, target) for pred, target in zip(preds, targets)])


# From https://www.kaggle.com/code/yiheng?scriptVersionId=93883465&cellId=4
def hd_dist(pred: np.ndarray, target: np.ndarray) -> float:
    if np.all(pred == target):
        return 0.0

    edges_pred, edges_gt = get_mask_edges(pred, target)
    surface_distance = get_surface_distance(edges_pred, edges_gt, distance_metric="euclidean")

    if surface_distance.shape == (0,):
        return 0.0

    dist = surface_distance.max()
    max_dist = np.sqrt(np.sum(np.array(pred.shape) ** 2))

    if dist > max_dist:
        return 1.0

    return dist / max_dist