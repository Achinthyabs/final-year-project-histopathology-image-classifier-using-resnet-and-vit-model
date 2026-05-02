from pathlib import Path
import csv

import torch
import timm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# =========================================================
# CONFIG
# =========================================================
BATCH_SIZE = 8
NUM_CLASSES = 13
IMAGE_SIZE = 224
DEVICE = "cpu"

TEST_DIR = Path("dataset_split/test")
MODEL_PATH = Path("best_vit_tiny.pth")
CSV_OUTPUT = Path("vit_test_predictions.csv")


# =========================================================
# TEST TRANSFORM
# =========================================================
test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


# =========================================================
# LOAD MODEL
# =========================================================
def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    classes = checkpoint["classes"]

    model = timm.create_model(
        "vit_tiny_patch16_224",
        pretrained=False,
        num_classes=NUM_CLASSES,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    return model, classes


# =========================================================
# MAIN TEST LOOP
# =========================================================
def main():
    test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model, classes = load_model()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # ------------------- METRICS -------------------
    acc = accuracy_score(all_labels, all_preds)
    print(f"\nFinal Test Accuracy: {acc * 100:.2f}%\n")

    print("Classification Report:\n")
    print(classification_report(all_labels, all_preds, target_names=classes, digits=4))

    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:\n")
    print(cm)

    # ------------------- SAVE CSV -------------------
    with open(CSV_OUTPUT, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["True_Label", "Predicted_Label"])

        for t, p in zip(all_labels, all_preds):
            writer.writerow([classes[t], classes[p]])

    print(f"\nPredictions saved to {CSV_OUTPUT}")


if __name__ == "__main__":
    main()
