from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from .config import DATASET_CSV, DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS, DEFAULT_NUM_WORKERS, IMAGE_ROOT, NUM_CLASSES
from .data_preparation import DeepfakeImageDataset, create_splits, get_eval_transform, get_train_transform, load_metadata, prepare_images
from .inference import evaluate_model, load_trained_model
from .model import build_model
from .training import train_model


def build_dataloaders(batch_size: int = DEFAULT_BATCH_SIZE, num_workers: int = DEFAULT_NUM_WORKERS):
    df = load_metadata(DATASET_CSV)
    splits = create_splits(df)

    train_dataset = DeepfakeImageDataset(splits.train_df, splits.train_image_dir, transform=get_train_transform())
    val_dataset = DeepfakeImageDataset(splits.val_df, splits.val_image_dir, transform=get_eval_transform())
    test_dataset = DeepfakeImageDataset(splits.test_df, splits.test_image_dir, transform=get_eval_transform())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return splits, train_loader, val_loader, test_loader


def prepare_data():
    df = load_metadata(DATASET_CSV)
    prepare_images(df, IMAGE_ROOT)


def run_training(epochs: int = DEFAULT_EPOCHS):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, train_loader, val_loader, _ = build_dataloaders()
    model = build_model(num_classes=NUM_CLASSES).to(device)
    trained_model, train_losses, val_accs = train_model(model, train_loader, val_loader, device, epochs=epochs)
    return trained_model, train_losses, val_accs, device


def run_evaluation(device: torch.device | None = None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, _, test_loader = build_dataloaders()
    model = load_trained_model(device)
    accuracy, report, matrix = evaluate_model(model, test_loader, device)

    print(f"\nTest Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)
    print("\nConfusion Matrix:")
    print(matrix)
    print(f"\nResNet50 Test Accuracy: {accuracy:.4f}")
    print("Previous Tabular Model Accuracy: 0.8224")
    print(f"Improvement: {(accuracy - 0.8224) * 100:.2f}%")


def run_full_pipeline(epochs: int = DEFAULT_EPOCHS):
    prepare_data()
    _, _, _, device = run_training(epochs=epochs)
    return run_evaluation(device=device)