from pathlib import Path


DATASET_CSV = Path("dataset/FINAL_DATASET.csv")
IMAGE_ROOT = Path("dataset/images")
BEST_MODEL_PATH = Path("best_resnet_deepfake.pth")
TRAINING_CURVES_PATH = Path("training_curves.png")

DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 20
DEFAULT_NUM_WORKERS = 4
RANDOM_STATE = 42
NUM_CLASSES = 2
IMAGE_SIZE = 224