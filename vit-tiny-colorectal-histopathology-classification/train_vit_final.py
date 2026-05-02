import os
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import timm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


# =========================================================
# CONFIG
# =========================================================
BATCH_SIZE = 8
EPOCHS = 5
LR = 1e-4
NUM_CLASSES = 13
IMAGE_SIZE = 224
DEVICE = "cpu"
SEED = 42

TRAIN_DIR = Path("dataset_split/train")
VAL_DIR = Path("dataset_split/val")
BEST_MODEL_PATH = Path("best_vit_tiny.pth")


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # CPU-only setup, but keep these for completeness.
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)


# =========================================================
# TRANSFORMS
# =========================================================
# Keep this simple first. We can add stronger augmentation later.
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


# =========================================================
# DATASET / DATALOADER
# =========================================================
def build_dataloaders():
    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Windows-safe
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,  # Windows-safe
    )

    return train_dataset, val_dataset, train_loader, val_loader


# =========================================================
# CLASS WEIGHTS
# =========================================================
def compute_class_weights(train_dataset: datasets.ImageFolder) -> torch.Tensor:
    class_counts = [0] * len(train_dataset.classes)
    for _, label in train_dataset.samples:
        class_counts[label] += 1

    total = sum(class_counts)
    # Inverse-frequency style weights.
    weights = [total / (len(class_counts) * count) for count in class_counts]
    weights_tensor = torch.tensor(weights, dtype=torch.float32)

    print("Class counts:", dict(zip(train_dataset.classes, class_counts)))
    print("Class weights:", {c: round(w, 4) for c, w in zip(train_dataset.classes, weights)})

    return weights_tensor


# =========================================================
# MODEL
# =========================================================
def build_model() -> nn.Module:
    print("Loading ViT-Tiny model...")
    model = timm.create_model(
        "vit_tiny_patch16_224",
        pretrained=True,
        num_classes=NUM_CLASSES,
    )
    return model


# =========================================================
# TRAIN / EVAL
# =========================================================
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(loader, desc="TRAIN", leave=False)
    for images, labels in loop:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        loop.set_postfix(
            loss=running_loss / max(1, (total / BATCH_SIZE)),
            acc=100.0 * correct / max(1, total),
        )

    avg_loss = running_loss / len(loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="VAL", leave=False):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


# =========================================================
# MAIN
# =========================================================
def main():
    set_seed(SEED)

    train_dataset, val_dataset, train_loader, val_loader = build_dataloaders()
    print("Classes:", train_dataset.classes)
    print("Train images:", len(train_dataset))
    print("Val images:", len(val_dataset))

    class_weights = compute_class_weights(train_dataset).to(DEVICE)

    model = build_model().to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=LR)

    best_val_acc = 0.0
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}%")
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "classes": train_dataset.classes,
                    "epoch": epoch + 1,
                    "val_acc": val_acc,
                    "train_loss_history": train_loss_history,
                    "train_acc_history": train_acc_history,
                    "val_loss_history": val_loss_history,
                    "val_acc_history": val_acc_history,
                    "config": {
                        "batch_size": BATCH_SIZE,
                        "epochs": EPOCHS,
                        "lr": LR,
                        "image_size": IMAGE_SIZE,
                        "num_classes": NUM_CLASSES,
                    },
                },
                BEST_MODEL_PATH,
            )
            print(f"Best model saved to {BEST_MODEL_PATH}")

    print(f"\nTraining finished. Best Val Acc = {best_val_acc:.2f}%")


if __name__ == "__main__":
    main()
