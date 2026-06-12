from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import BEST_MODEL_PATH, NUM_CLASSES
from .model import build_model
from .training import evaluate_predictions


def load_trained_model(device: torch.device, model_path: Path = BEST_MODEL_PATH, num_classes: int = NUM_CLASSES):
    model = build_model(num_classes=num_classes)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def evaluate_model(model: torch.nn.Module, test_loader: DataLoader, device: torch.device):
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    return evaluate_predictions(all_labels, all_preds)


def predict_single_image(model: torch.nn.Module, image_tensor: torch.Tensor, device: torch.device):
    model.eval()
    with torch.no_grad():
        logits = model(image_tensor.unsqueeze(0).to(device))
        return torch.argmax(logits, dim=1).item()