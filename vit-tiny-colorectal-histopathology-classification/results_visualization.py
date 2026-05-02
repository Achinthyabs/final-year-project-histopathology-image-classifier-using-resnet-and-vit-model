import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

MODEL_PATH = "best_vit_tiny.pth"
CSV_PATH = "vit_test_predictions.csv"


def main():
    # ---------------- LOAD SAVED HISTORY ----------------
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")

    train_loss = checkpoint["train_loss_history"]
    train_acc = checkpoint["train_acc_history"]
    val_loss = checkpoint["val_loss_history"]
    val_acc = checkpoint["val_acc_history"]
    classes = checkpoint["classes"]

    epochs = list(range(1, len(train_loss) + 1))

    # ---------------- ACCURACY GRAPH ----------------
    plt.figure()
    plt.plot(epochs, train_acc, marker='o', label='Train Accuracy')
    plt.plot(epochs, val_acc, marker='o', label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training vs Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('accuracy_curve.png')
    plt.close()

    # ---------------- LOSS GRAPH ----------------
    plt.figure()
    plt.plot(epochs, train_loss, marker='o', label='Train Loss')
    plt.plot(epochs, val_loss, marker='o', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curve.png')
    plt.close()

    # ---------------- CONFUSION MATRIX ----------------
    df = pd.read_csv(CSV_PATH)
    y_true = df['True_Label']
    y_pred = df['Predicted_Label']

    cm = confusion_matrix(y_true, y_pred, labels=classes)

    plt.figure(figsize=(12, 12))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(xticks_rotation=45)
    plt.title('ViT-Tiny Test Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()

    print('Saved: accuracy_curve.png')
    print('Saved: loss_curve.png')
    print('Saved: confusion_matrix.png')


if __name__ == '__main__':
    main()
