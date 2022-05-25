import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.model_selection import StratifiedGroupKFold
from tqdm.notebook import tqdm


def load_mask(row):
    shape = (row.height, row.width, 3)
    mask = np.zeros(shape, dtype=np.uint8)

    for i, rle in enumerate(row.segmentation):
        if rle:
            mask[..., i] = rle_decode(rle, shape[:2])

    return mask * 255


# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_decode(mask_rle, shape):
    """
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    """
    s = np.asarray(mask_rle.split(), dtype=int)
    starts = s[0::2] - 1
    lengths = s[1::2]
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)  # Needed to align to RLE direction


def save_array(file_path, array):
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(file_path, array)



def load_image(path):
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    image = image.astype("float32")  # original is uint16
    image /= image.max()
    return image


def load_images(paths):
    images = [load_image(path) for path in paths]
    images = np.stack(images, axis=-1)
    return images


def load_mask(path):
    mask = np.load(path)
    mask = mask.astype("float32")
    mask /= 255.0
    return mask
    
    
def show_image(image, mask=None):
    plt.imshow(image, cmap="bone")

    if mask is not None:
        plt.imshow(mask, alpha=0.5)

        handles = [
            Rectangle((0, 0), 1, 1, color=_c) for _c in [(0.667, 0.0, 0.0), (0.0, 0.667, 0.0), (0.0, 0.0, 0.667)]
        ]
        labels = ["Stomach", "Large Bowel", "Small Bowel"]

        plt.legend(handles, labels)

    plt.axis("off")
    

def show_grid(train_df, nrows, ncols):
    fig, _ = plt.subplots(figsize=(5 * ncols, 5 * nrows))

    train_df_sampled = train_df[~train_df["empty"]].sample(n=nrows * ncols)
    for i, row in enumerate(train_df_sampled.itertuples()):

        image = load_images(row.image_paths)
        #image = load_image(row.image_path)

        mask = load_mask(row.mask_path)

        plt.subplot(nrows, ncols, i + 1)
        plt.tight_layout()
        plt.title(row.id)

        show_image(image, mask)