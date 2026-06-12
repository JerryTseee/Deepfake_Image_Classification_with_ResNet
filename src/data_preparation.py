from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

from .config import IMAGE_ROOT, IMAGE_SIZE, RANDOM_STATE


@dataclass(frozen=True)
class SplitData:
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    train_image_dir: Path
    val_image_dir: Path
    test_image_dir: Path


def load_metadata(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def download_image(url: str, save_path: Path, max_retries: int = 3) -> bool:
    for _ in range(max_retries):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                image = Image.open(BytesIO(response.content)).convert("RGB")
                image.save(save_path)
                return True
        except Exception:
            continue
    return False


def prepare_images(df: pd.DataFrame, image_root: Path = IMAGE_ROOT) -> None:
    for split_name in ["train", "val", "test"]:
        (image_root / split_name).mkdir(parents=True, exist_ok=True)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Downloading images"):
        split_name = str(row["dataset_split"])
        image_id = row["image_id"]
        url = row["image_url"]
        save_dir = image_root / split_name
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{image_id}.jpg"

        if not save_path.exists():
            download_image(url, save_path)


def create_splits(df: pd.DataFrame) -> SplitData:
    train_df = df[df["dataset_split"] == "train"].reset_index(drop=True)
    test_df = df[df["dataset_split"] == "test"].reset_index(drop=True)

    explicit_val_df = df[df["dataset_split"] == "val"].reset_index(drop=True)
    if len(explicit_val_df) > 0:
        val_df = explicit_val_df
        val_image_dir = IMAGE_ROOT / "val"
    else:
        train_df, val_df = train_test_split(
            train_df,
            test_size=0.15,
            random_state=RANDOM_STATE,
            stratify=train_df["label_numeric"],
        )
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        val_image_dir = IMAGE_ROOT / "train"

    return SplitData(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        train_image_dir=IMAGE_ROOT / "train",
        val_image_dir=val_image_dir,
        test_image_dir=IMAGE_ROOT / "test",
    )


# data augmentation
def get_train_transform():
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomResizedCrop(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

# evaluation transform (no augmentation, just resizing and normalization)
def get_eval_transform():
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class DeepfakeImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_dir: Path, transform=None):
        self.df = df
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_path = self.image_dir / f"{row['image_id']}.jpg"

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color="black")

        label = int(row["label_numeric"])

        if self.transform is not None:
            image = self.transform(image)

        return image, label