from pathlib import Path

KAGGLE_DIR = Path("/") / "kaggle"
INPUT_DIR = KAGGLE_DIR / "input"
OUTPUT_DIR = KAGGLE_DIR / "working"

INPUT_DATA_DIR = INPUT_DIR / "uw-madison-gi-tract-image-segmentation"
INPUT_DATA_NPY_DIR = INPUT_DIR / "uw-madison-gi-tract-image-segmentation-masks"

N_SPLITS = 5
RANDOM_SEED = 2022
IMG_SIZE = 224
VAL_FOLD = 0
LOAD_IMAGES = True # True for 2.5D data
USE_AUGS = True
BATCH_SIZE = 128
NUM_WORKERS = 2
ARCH = "Unet"
ENCODER_NAME = "efficientnet-b1"
ENCODER_WEIGHTS = "imagenet"
LOSS = "bce_tversky"
OPTIMIZER = "Adam"
LEARNING_RATE = 2e-3
WEIGHT_DECAY = 1e-6
SCHEDULER = "CosineAnnealingLR"
MIN_LR = 1e-6

FAST_DEV_RUN = False # Debug training
GPUS = 1
MAX_EPOCHS = 15
PRECISION = 16

CHANNELS = 3
STRIDE = 2
DEVICE = "cuda"
THR = 0.45

DEBUG = False # Debug complete pipeline