
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
from config import *



class Dataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, load_images: bool, load_mask: bool, transforms: Optional[Callable] = None):
        self.df = df
        self.load_images = load_images
        self.load_mask = load_mask
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        if self.load_images:
            image = self._load_images(eval(row["image_paths"]))
        else:
            image = self._load_image(row["image_path"])

        if self.load_mask:
            mask = self._load_mask(row["mask_path"])

            if self.transforms:
                data = self.transforms(image=image, mask=mask)
                image, mask = data["image"], data["mask"]

            return image, mask
        else:
            id_ = row["id"]
            h, w = image.shape[:2]

            if self.transforms:
                data = self.transforms(image=image)
                image = data["image"]

            return image, id_, h, w
        
    def _load_images(self, paths):
        images = [self._load_image(path, tile=False) for path in paths]
        image = np.stack(images, axis=-1)
        return image

    @staticmethod
    def _load_image(path, tile: bool = True):
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        image = image.astype("float32")  # original is uint16
        
        if tile:
            image = np.tile(image[..., None], [1, 1, 3])  # gray to rgb
            
        image /= image.max()

        return image
    

    @staticmethod
    def _load_mask(path):
        return np.load(path).astype("float32") / 255.0




class LitDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_csv_path: str,
        test_csv_path: Optional[str],
        img_size: int,
        use_augs: bool,
        val_fold: int,
        load_images: bool,
        batch_size: int,
        num_workers: int,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.train_df = pd.read_csv(train_csv_path)

        if test_csv_path is not None:
            self.test_df = pd.read_csv(test_csv_path)
        else:
            self.test_df = None

        self.train_transforms, self.val_test_transforms = self._init_transforms()

    def _init_transforms(self):
        img_size = self.hparams.img_size

        train_transforms = [
            A.Resize(img_size, img_size, interpolation=cv2.INTER_NEAREST),
        ]
        if self.hparams.use_augs:
            train_transforms += [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.OneOf(
                    [
                        A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
                        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0),
                    ],
                    p=0.25,
                ),
            ]
        train_transforms += [ToTensorV2(transpose_mask=True)]
        train_transforms = A.Compose(train_transforms, p=1.0)

        val_test_transforms = [
            A.Resize(img_size, img_size, interpolation=cv2.INTER_NEAREST),
            ToTensorV2(transpose_mask=True),
        ]
        val_test_transforms = A.Compose(val_test_transforms, p=1.0)

        return train_transforms, val_test_transforms

    def setup(self, stage: Optional[str] = None):
        train_df = self.train_df[self.train_df.fold != self.hparams.val_fold].reset_index(drop=True)
        val_df = self.train_df[self.train_df.fold == self.hparams.val_fold].reset_index(drop=True)

        if stage == "fit" or stage is None:
            self.train_dataset = self._dataset(train_df, load_mask=True, transform=self.train_transforms)
            self.val_dataset = self._dataset(val_df, load_mask=True, transform=self.val_test_transforms)

        if stage == "test" or stage is None:
            if self.test_df is not None:
                self.test_dataset = self._dataset(self.test_df, load_mask=False, transform=self.val_test_transforms)
            else:
                self.test_dataset = self._dataset(val_df, load_mask=True, transform=self.val_test_transforms)

    def _dataset(self, df: pd.DataFrame, load_mask: bool, transform: Callable) -> Dataset:
        return Dataset(df=df, load_images=self.hparams.load_images, load_mask=load_mask, transforms=transform)

    def train_dataloader(self) -> DataLoader:
        return self._dataloader(self.train_dataset, train=True)

    def val_dataloader(self) -> DataLoader:
        return self._dataloader(self.val_dataset)

    def test_dataloader(self) -> DataLoader:
        return self._dataloader(self.test_dataset)

    def _dataloader(self, dataset: Dataset, train: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            shuffle=train,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )
